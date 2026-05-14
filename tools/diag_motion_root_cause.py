"""Pin down WHY motion's diagonal violates descent.

The previous diagnostic (diag_motion_descent.py) showed motion descent fails
in EVERY wiring config. Magnitude (+1.77 at α=1e-3 normalized step) is 600×
the expected -α‖g‖=−0.003, so the loss surface is either jagged OR the
"loss" being descended is not the same "loss" being measured.

This script tests the second hypothesis directly by HOOKING the motion
sampler and logging exactly what it returns on every forward inside the
probe sequence (baseline → grad-fwd → stepped). If the returned tensors
DRIFT across forwards, the freeze is fake-broken — the gradient is computed
wrt one objective and re-evaluated against another.

Tests:

  [P1] Per-forward determinism: 2 baselines in freeze ctx → freeze must pin
       motion `cls_target`, `reg_target`, `cls_weight` bit-exactly.

  [P2] Step invariance: baseline → grad fwd → α-step → stepped fwd. The
       motion sampler must return BIT-EXACT cls_target across all 3 forwards
       (params change but mode_idx is supposed to be frozen).

  [P3] If P1/P2 fail: localize where divergence enters. Likely places:
       (a) `motion_loss_cache['indices']` source — det matching that feeds
           into motion. If det indices drift (different number of pos
           samples), motion's reg_target shape changes.
       (b) `cls_target` cache — does session._take return the right entry?
           Counts sampler.sample calls per forward and prints queue depth
           used.

  [P4] α-sweep at very small α (1e-6 to 1e-3). True descent must satisfy
       ΔL/α ≈ -‖g‖ in the limit α→0. Plotting reveals if motion's "loss"
       even has a well-defined gradient at this point.

  [P5] Freeze ON vs freeze OFF, source=motion at α=1e-3 normalized. If
       freeze OFF gives DIFFERENT Δ than freeze ON, freeze is changing the
       effective objective (which is wrong — freeze should only pin
       matching/state, not the loss value).

Run::
    cd /home/yongjae/e2e/HiP-AD && \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/yongjae/miniconda3/envs/hipad/bin/python tools/diag_motion_root_cause.py
"""
from __future__ import annotations

import contextlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


@contextlib.contextmanager
def per_forward_seed(seed: int):
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    npx = np.random.get_state()
    py = random.getstate()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(npx)
        random.setstate(py)


PREFIX = {
    "det":    ["det_loss", "loss_det_cls", "loss_det_reg"],
    "map":    ["map_loss", "loss_map_cls", "loss_map_reg"],
    "motion": ["motion_loss", "loss_motion_cls", "loss_motion_reg"],
    "plan":   ["plan_loss", "loss_plan_cls", "loss_plan_reg"],
}


def _split(losses, prefix_list):
    out = None
    for k, v in losses.items():
        if not isinstance(v, torch.Tensor):
            continue
        if any(k.startswith(p) for p in prefix_list):
            out = v if out is None else out + v
    return out


def main() -> int:
    from tools.gradient_analysis.collector import (
        GradientCollector, build_dataloader, compute_task_full_gradient,
    )
    from tools.gradient_analysis.probe import (
        apply_virtual_step, snapshot_params, restore_params,
    )
    from tools.run_gradient_analysis import (
        _import_runtime, _resolve_ckpt_path, build_adapter, set_seeds,
    )

    with open("configs/gradient_analysis.yaml") as f:
        cfg = yaml.safe_load(f)
    set_seeds(cfg["seed"], False)

    rt = _import_runtime()
    adapter = build_adapter(rt, cfg, "hipad")
    device = cfg["device"]
    model = adapter.build_model(
        ckpt=_resolve_ckpt_path(cfg, "1ep"), device=device,
    )
    collector = GradientCollector(
        model=model, adapter=adapter,
        shared_layer_names=cfg["shared_param_groups"], device=device,
    )

    dl = build_dataloader(adapter, batch_size=1, shuffle=True, seed=cfg["seed"])
    data = next(iter(dl))
    seed = cfg["seed"]
    full_params = collector.full_params
    raw = collector.model.module if hasattr(collector.model, "module") else collector.model

    # ----- Sampler classes (HiP-AD samplers are not nn.Module — discover by
    # class import).
    from projects.mmdet3d_plugin.models.motion.target import SparseMotionTarget  # type: ignore
    from projects.mmdet3d_plugin.models.det.target import SparseBox3DTarget  # type: ignore

    # ----- Logger that records what the (potentially patched) sample returns --
    motion_log: list = []
    det_log: list = []

    def make_motion_logger(orig):
        def logged(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos = out
            motion_log.append({
                "cls_target": cls_target.detach().cpu().clone(),
                "cls_weight": cls_weight.detach().cpu().clone(),
                "reg_target": reg_target.detach().cpu().clone(),
                "reg_weight": reg_weight.detach().cpu().clone(),
                "best_reg":   best_reg.detach().cpu().clone(),
                "num_pos":    num_pos.detach().cpu().clone() if torch.is_tensor(num_pos) else num_pos,
            })
            return out
        return logged

    def make_det_logger(orig):
        def logged(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            indices_snapshot = []
            for tup in (self.indices or []):
                indices_snapshot.append(tuple(
                    (None if x is None else x.detach().cpu().tolist()) for x in tup
                ))
            det_log.append({"indices": indices_snapshot})
            return out
        return logged

    @contextlib.contextmanager
    def install_loggers():
        # Wrap whatever class.sample is currently bound (so freeze patch sits
        # underneath our logger).
        cur_motion = SparseMotionTarget.sample
        cur_det = SparseBox3DTarget.sample
        SparseMotionTarget.sample = make_motion_logger(cur_motion)
        SparseBox3DTarget.sample = make_det_logger(cur_det)
        try:
            yield
        finally:
            SparseMotionTarget.sample = cur_motion
            SparseBox3DTarget.sample = cur_det

    def reset_logs():
        motion_log.clear()
        det_log.clear()

    def diff_motion(a_list, b_list, label):
        if len(a_list) != len(b_list):
            print(f"  {label}: CALL-COUNT DIFF {len(a_list)} vs {len(b_list)}")
            return False
        all_ok = True
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            for k in a:
                ta, tb = a[k], b[k]
                if isinstance(ta, torch.Tensor):
                    if ta.shape != tb.shape:
                        print(f"  {label}#{i}.{k}: shape diff {ta.shape} vs {tb.shape}")
                        all_ok = False
                    else:
                        d = (ta.float() - tb.float()).abs().max().item()
                        flag = " OK" if d < 1e-7 else f" DIFF absmax={d:.4e}"
                        print(f"  {label}#{i}.{k:11s} shape={tuple(ta.shape)}{flag}")
                        if d >= 1e-7:
                            all_ok = False
        return all_ok

    # ============================================================== TEST [P1]
    print("=" * 78)
    print("[P1] 2 baseline forwards in freeze ctx — motion sampler outputs identical?")
    print("=" * 78)
    snap = adapter.snapshot_temporal_state(model)
    with adapter.freeze_stochastic_state():
        with install_loggers():
            reset_logs()
            adapter.restore_temporal_state(model, snap)
            with per_forward_seed(seed):
                with torch.no_grad():
                    _ = collector.forward_losses(data)
            f0_motion = list(motion_log)
            f0_det = list(det_log)

            reset_logs()
            adapter.restore_temporal_state(model, snap)
            with per_forward_seed(seed):
                with torch.no_grad():
                    _ = collector.forward_losses(data)
            f1_motion = list(motion_log)
            f1_det = list(det_log)
    print(f"  forward 0: motion.sample called {len(f0_motion)}× | det.sample called {len(f0_det)}×")
    print(f"  forward 1: motion.sample called {len(f1_motion)}× | det.sample called {len(f1_det)}×")
    diff_motion(f0_motion, f1_motion, "f0 vs f1 motion")

    # ============================================================== TEST [P2]
    print()
    print("=" * 78)
    print("[P2] baseline → motion grad fwd → α-step → stepped fwd")
    print("     Motion sampler MUST return identical cls_target across all 3.")
    print("=" * 78)
    param_snap = snapshot_params(full_params)
    try:
        snap = adapter.snapshot_temporal_state(model)
        with adapter.freeze_stochastic_state():
            with install_loggers():
                # Baseline (no_grad)
                reset_logs()
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        bl = collector.forward_losses(data)
                bl_motion = list(motion_log)
                bl_det = list(det_log)
                bl_motion_loss = float(_split(bl, PREFIX["motion"]).item())

                # Gradient forward (with grad)
                reset_logs()
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    losses = collector.forward_losses(data)
                gd_motion = list(motion_log)
                gd_det = list(det_log)
                tl = _split(losses, PREFIX["motion"])
                g = compute_task_full_gradient(tl, full_params, retain_graph=False)
                gn = float(torch.cat([gi.flatten() for gi in g]).norm())
                apply_virtual_step(full_params, g, alpha=1e-3, normalize=True)
                del losses, tl, g
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stepped forward (no_grad)
                reset_logs()
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        st = collector.forward_losses(data)
                st_motion = list(motion_log)
                st_det = list(det_log)
                st_motion_loss = float(_split(st, PREFIX["motion"]).item())
        print(f"  motion calls: baseline={len(bl_motion)}  grad={len(gd_motion)}  stepped={len(st_motion)}")
        print(f"  det    calls: baseline={len(bl_det)}     grad={len(gd_det)}     stepped={len(st_det)}")
        print(f"  motion loss: baseline={bl_motion_loss:.6f}  stepped={st_motion_loss:.6f}  Δ={st_motion_loss-bl_motion_loss:+.6f}")
        print(f"  ‖g_motion‖={gn:.4e}  expected Δ≈-α‖g‖={-1e-3*gn:+.6f}")
        print()
        print("  --- baseline vs grad-fwd (motion sampler) ---")
        diff_motion(bl_motion, gd_motion, "bl vs grad")
        print("  --- baseline vs stepped-fwd (motion sampler) ---")
        diff_motion(bl_motion, st_motion, "bl vs stepped")
        print("  --- baseline vs stepped (det indices) ---")
        det_drift = []
        for i, (a, b) in enumerate(zip(bl_det, st_det)):
            same = (a["indices"] == b["indices"])
            det_drift.append(same)
            print(f"    det.sample#{i} indices match: {same}")
            if not same and len(a["indices"]) > 0:
                # Show first batch element diff
                a0, b0 = a["indices"][0], b["indices"][0]
                print(f"      a={a0[:2]}\n      b={b0[:2]}")
    finally:
        restore_params(full_params, param_snap)

    # ============================================================== TEST [P4]
    print()
    print("=" * 78)
    print("[P4] α-sweep: ΔL_motion / α as α → 0 must approach -‖g‖")
    print("=" * 78)
    snap = adapter.snapshot_temporal_state(model)
    param_snap = snapshot_params(full_params)
    try:
        with adapter.freeze_stochastic_state():
            # Baseline + gradient (compute once, reuse)
            adapter.restore_temporal_state(model, snap)
            with per_forward_seed(seed):
                with torch.no_grad():
                    bl = collector.forward_losses(data)
            bl_motion = float(_split(bl, PREFIX["motion"]).item())

            adapter.restore_temporal_state(model, snap)
            with per_forward_seed(seed):
                losses = collector.forward_losses(data)
            tl = _split(losses, PREFIX["motion"])
            g = compute_task_full_gradient(tl, full_params, retain_graph=False)
            gn = float(torch.cat([gi.flatten() for gi in g]).norm())
            del losses, tl
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"  baseline motion loss = {bl_motion:.6f}, ‖g‖ = {gn:.4e}")
            print(f"  α          ΔL_motion        ΔL/α          expected -‖g‖={-gn:.4e}")
            for alpha in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]:
                apply_virtual_step(full_params, g, alpha=alpha, normalize=True)
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        st = collector.forward_losses(data)
                st_motion = float(_split(st, PREFIX["motion"]).item())
                d = st_motion - bl_motion
                print(f"  {alpha:.0e}   {d:+.6e}   {d/alpha:+.4e}")
                restore_params(full_params, param_snap)
    finally:
        restore_params(full_params, param_snap)

    # ============================================================== TEST [P5]
    print()
    print("=" * 78)
    print("[P5] freeze ON vs OFF, source=motion, α=1e-3 normalized")
    print("     If freeze ON differs from OFF → freeze changes the objective")
    print("=" * 78)

    def measure_motion_descent(use_freeze):
        snap = adapter.snapshot_temporal_state(model)
        param_snap = snapshot_params(full_params)
        try:
            ctx = adapter.freeze_stochastic_state() if use_freeze else contextlib.nullcontext()
            with ctx:
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        bl = collector.forward_losses(data)
                bl_motion = float(_split(bl, PREFIX["motion"]).item())

                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    losses = collector.forward_losses(data)
                tl = _split(losses, PREFIX["motion"])
                g = compute_task_full_gradient(tl, full_params, retain_graph=False)
                gn = float(torch.cat([gi.flatten() for gi in g]).norm())
                apply_virtual_step(full_params, g, alpha=1e-3, normalize=True)
                del losses, tl, g
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        st = collector.forward_losses(data)
                st_motion = float(_split(st, PREFIX["motion"]).item())
            return bl_motion, st_motion, gn
        finally:
            restore_params(full_params, param_snap)

    on_bl, on_st, on_gn = measure_motion_descent(True)
    off_bl, off_st, off_gn = measure_motion_descent(False)
    print(f"  freeze ON : base={on_bl:.6f}  step={on_st:.6f}  Δ={on_st-on_bl:+.4e}  ‖g‖={on_gn:.4e}")
    print(f"  freeze OFF: base={off_bl:.6f}  step={off_st:.6f}  Δ={off_st-off_bl:+.4e}  ‖g‖={off_gn:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
