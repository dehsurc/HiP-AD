"""Diagnostic — is the probe forward actually deterministic given fixed
weights, fixed input, freeze_matching, reset_temporal_state, selective_eval?

If yes (3 consecutive baselines give identical det loss to ~1e-6) → ΔL_det
explosion is a real Lipschitz/cliff issue (α too aggressive); answer is to
shrink alpha.

If no (det loss varies across consecutive identical forwards) → some
stochastic / mutable state is escaping our context managers; we need to
find and pin it.

Run::
    cd /home/yongjae/e2e/HiP-AD && \
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
    /home/yongjae/miniconda3/envs/hipad/bin/python tools/diag_forward_determinism.py
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


def main() -> int:
    from tools.gradient_analysis.collector import GradientCollector, build_dataloader
    from tools.run_gradient_analysis import (
        _import_runtime,
        _resolve_ckpt_path,
        build_adapter,
        set_seeds,
    )

    with open("configs/gradient_analysis.yaml") as f:
        cfg_ana = yaml.safe_load(f)
    set_seeds(cfg_ana["seed"], False)

    rt = _import_runtime()
    adapter = build_adapter(rt, cfg_ana, "hipad")
    device = cfg_ana["device"]
    model = adapter.build_model(
        ckpt=_resolve_ckpt_path(cfg_ana, "1ep"),
        device=device,
    )

    collector = GradientCollector(
        model=model,
        adapter=adapter,
        shared_layer_names=cfg_ana["shared_param_groups"], device=device,
    )

    dataloader = build_dataloader(
        adapter, batch_size=cfg_ana["primary"]["batch_size"],
        shuffle=True, seed=cfg_ana["seed"],
    )
    data = next(iter(dataloader))

    @contextlib.contextmanager
    def per_forward_seed(seed: int):
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        py_state = random.getstate()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            np.random.set_state(np_state)
            random.setstate(py_state)

    class ModelStateSnapshot:
        def __init__(self, model):
            self._model = model
            self._snap = adapter.snapshot_temporal_state(model)

        def restore(self) -> None:
            adapter.restore_temporal_state(self._model, self._snap)

    class FrozenMatching:
        def __enter__(self):
            self._ctx = adapter.freeze_stochastic_state()
            self._ctx.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._ctx.__exit__(exc_type, exc, tb)

        def next_forward(self) -> None:
            pass

    # ---------- Diagnostic A: 3 consecutive forwards, no intervention ----------
    print("\n[A] 3 consecutive forwards (no freeze/no temporal/no seed) — train mode")
    losses_per_call = []
    for i in range(3):
        with torch.no_grad():
            losses = collector.forward_losses(data)
        det = sum(v.item() for k, v in losses.items() if k.startswith("det_loss"))
        map_ = sum(v.item() for k, v in losses.items() if k.startswith("map_loss"))
        losses_per_call.append((det, map_))
        print(f"   call {i}: det={det:.6f}  map={map_:.6f}")
    deltas = [(losses_per_call[i+1][0] - losses_per_call[0][0]) for i in range(2)]
    print(f"   det drift call0→call1,call0→call2: {deltas}")

    # ---------- Diagnostic B: with freeze + temporal + seed, 3 forwards ----------
    print("\n[B] 3 consecutive forwards WITH freeze_matching + temporal_reset + per_forward_seed")
    fm = FrozenMatching()
    state_snap = ModelStateSnapshot(collector.model)
    losses_per_call = []
    with fm:
        for i in range(3):
            state_snap.restore()
            fm.next_forward()
            with per_forward_seed(cfg_ana["seed"]):
                with torch.no_grad():
                    losses = collector.forward_losses(data)
            det = sum(v.item() for k, v in losses.items() if k.startswith("det_loss"))
            map_ = sum(v.item() for k, v in losses.items() if k.startswith("map_loss"))
            losses_per_call.append((det, map_))
            print(f"   call {i}: det={det:.6f}  map={map_:.6f}")
    deltas = [(losses_per_call[i+1][0] - losses_per_call[0][0]) for i in range(2)]
    print(f"   det drift call0→call1,call0→call2: {deltas}")

    # ---------- Diagnostic C: per-decoder-layer breakdown for one stepped forward ----------
    print("\n[C] Per-decoder-layer det loss breakdown (single forward, deterministic context)")
    fm = FrozenMatching()
    state_snap = ModelStateSnapshot(collector.model)
    with fm:
        state_snap.restore(); fm.next_forward()
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                losses = collector.forward_losses(data)
        for k in sorted(losses.keys()):
            if k.startswith("det_loss"):
                print(f"   {k}: {losses[k].item():.6f}")

    # ---------- Diagnostic D: Apply tiny step to dec5_gnn_0 in det's gradient direction,
    # then re-forward and report per-decoder-layer Δ ----------
    print("\n[D] Apply normalized α=0.001 step to dec5_gnn_0 in source=det direction")
    print("    Expected: det_loss_cls_5 / det_loss_box_5 etc. decrease, dec0..4 unchanged")
    fm = FrozenMatching()
    state_snap = ModelStateSnapshot(collector.model)
    with fm:
        # Baseline
        state_snap.restore(); fm.next_forward()
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                bl_losses = collector.forward_losses(data)
        baseline = {k: v.item() for k, v in bl_losses.items() if k.startswith("det_loss")}

        # Build det grad on dec5_gnn_0
        from tools.gradient_analysis.collector import compute_task_full_gradient
        from analyze_gradient_conflict import _sum_task_loss
        from tools.gradient_analysis.probe import (
            apply_virtual_step, snapshot_params, restore_params,
        )
        groups = collector.shared_param_groups
        update_params = list(groups["dec5_gnn_0"]) if not hasattr(groups["dec5_gnn_0"], "values") \
                        else list(groups["dec5_gnn_0"].values())
        snap = snapshot_params(update_params)

        state_snap.restore(); fm.next_forward()
        with per_forward_seed(cfg_ana["seed"]):
            losses = collector.forward_losses(data)
        tl = _sum_task_loss(losses, "det")
        g = compute_task_full_gradient(tl, update_params, retain_graph=False)
        gn = float(torch.cat([gi.flatten() for gi in g]).norm())
        print(f"   ‖g_dec5_gnn_0(L_det)‖ = {gn:.6f}")
        apply_virtual_step(update_params, g, alpha=0.001, normalize=True)

        state_snap.restore(); fm.next_forward()
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                stepped_losses = collector.forward_losses(data)
        print("   per-decoder-layer ΔL_det (stepped - baseline):")
        for k in sorted(baseline.keys()):
            db = stepped_losses[k].item() - baseline[k]
            print(f"     {k:35s} baseline={baseline[k]:9.4f}  stepped={stepped_losses[k].item():9.4f}  Δ={db:+9.4f}")
        restore_params(update_params, snap)

    # ---------- Diagnostic F: isolate which safety mechanism causes drift ----------
    # [B] showed that all three safety knobs ON together produce ~80 drift across
    # repeated forwards on the same data. [A] without any of them produced ~0.5.
    # So one of {freeze_matching, temporal_reset, per_forward_seed} introduces
    # the drift. Test all 6 non-empty combinations and report drift per case.
    print("\n[F] Isolating which safety knob makes drift worse")
    import contextlib
    scenarios = [
        ("none (sanity)",      dict(freeze=False, temp=False, seed=False)),
        ("freeze ONLY",        dict(freeze=True,  temp=False, seed=False)),
        ("temporal ONLY",      dict(freeze=False, temp=True,  seed=False)),
        ("seed ONLY",          dict(freeze=False, temp=False, seed=True)),
        ("freeze+temporal",    dict(freeze=True,  temp=True,  seed=False)),
        ("freeze+seed",        dict(freeze=True,  temp=False, seed=True)),
        ("temporal+seed",      dict(freeze=False, temp=True,  seed=True)),
        ("all three",          dict(freeze=True,  temp=True,  seed=True)),
    ]
    for name, opts in scenarios:
        fm = FrozenMatching() if opts["freeze"] else None
        snap = ModelStateSnapshot(collector.model) if opts["temp"] else None
        losses_seq = []
        ctx = fm if fm is not None else contextlib.nullcontext()
        with ctx:
            for i in range(3):
                if snap is not None:
                    snap.restore()
                if fm is not None:
                    fm.next_forward()
                seed_ctx = (per_forward_seed(cfg_ana["seed"])
                            if opts["seed"] else contextlib.nullcontext())
                with seed_ctx:
                    with torch.no_grad():
                        ll = collector.forward_losses(data)
                det = sum(v.item() for k, v in ll.items() if k.startswith("det_loss"))
                losses_seq.append(det)
        drift = max(losses_seq) - min(losses_seq)
        print(f"   {name:18s}: det={losses_seq}  drift={drift:.4f}")

    # ---------- Diagnostic E: matching freeze identity check ----------
    # Three forwards through patched samplers + a control forward without
    # patches. After each forward we read the sampler's `self.indices` /
    # `cls_target` and check whether they really stayed pinned to the
    # baseline matching, or quietly recomputed when predictions changed.
    print("\n[E] Matching freeze verification — re-running Hungarian when params change?")
    raw = collector.model.module if hasattr(collector.model, "module") else collector.model

    # Locate live sampler instances inside the model graph.
    samplers = {"det": None, "map": None, "motion": None, "plan": None}
    for m in raw.modules():
        cn = type(m).__name__
        if cn == "SparseBox3DTarget" and samplers["det"] is None:
            samplers["det"] = m
        elif cn == "SparsePoint3DTarget" and samplers["map"] is None:
            samplers["map"] = m
        elif cn == "SparseMotionTarget" and samplers["motion"] is None:
            samplers["motion"] = m
        elif cn in ("SparsePlanTarget", "AlignPlanTarget", "PlanningTarget") and samplers["plan"] is None:
            samplers["plan"] = m
    print(f"   live samplers found: { {k: v is not None for k, v in samplers.items()} }")

    # Per-call output snapshot: for det/map we capture pred_idx + target_idx
    # tuples; for motion/plan we capture cls_target (mode_idx).
    call_log: dict = {k: [] for k in samplers if samplers[k] is not None}
    orig_methods: dict = {}

    def make_logger(tag: str, sampler):
        orig = type(sampler).sample
        orig_methods[tag] = orig

        def logger(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            if tag in ("det", "map"):
                # indices = list of (pred_idx, target_idx[, perm_idx]) per batch
                snapshot = []
                for tup in (self.indices or []):
                    snapshot.append(tuple(
                        (None if x is None else x.detach().cpu().tolist())
                        for x in tup
                    ))
                call_log[tag].append(snapshot)
            else:
                # motion/plan: cls_target (mode index) is the matching surrogate
                cls_target = out[0] if tag == "motion" else out[1]
                call_log[tag].append(cls_target.detach().cpu().tolist())
            return out

        # Patch via the existing class binding so our logger sees what
        # FrozenMatching's patched.sample actually returns (logger calls
        # `orig` which IS the freeze-patched method at this point in time).
        type(sampler).sample = logger

    def restore_loggers():
        for tag, sampler in samplers.items():
            if sampler is None:
                continue
            type(sampler).sample = orig_methods[tag]

    def run_one(label, perturb_step=False):
        # Clear logs for this run
        for k in call_log:
            call_log[k] = []
        if state_snap is not None:
            state_snap.restore()
        if fm is not None:
            fm.next_forward()
        if perturb_step:
            with torch.no_grad():
                for p in update_params:
                    # Large enough perturbation that a non-frozen Hungarian
                    # would almost certainly reshuffle.
                    p.data.add_(torch.randn_like(p) * 1e-2)
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                _ = collector.forward_losses(data)
        return {k: list(v) for k, v in call_log.items()}

    def compare(a, b, tag):
        if len(a) != len(b):
            return f"call-count diff ({len(a)} vs {len(b)})"
        for i, (xa, xb) in enumerate(zip(a, b)):
            if xa != xb:
                return f"FIRST DIFF at sample call #{i}"
        return "IDENTICAL"

    fm = FrozenMatching()
    state_snap = ModelStateSnapshot(collector.model)
    update_params, _ = _resolve_params(collector, "dec5_gnn_0")
    snap = snapshot_params(update_params)

    # Install logging wrappers (must be done while freeze patches are NOT
    # installed yet, so logger.orig captures the freeze-patched method when
    # we enter the FrozenMatching context below). To keep things simple, we
    # nest: enter FrozenMatching first (now class.sample = freeze patched),
    # then install loggers that call into it.
    with fm:
        for tag, sampler in samplers.items():
            if sampler is not None:
                make_logger(tag, sampler)
        try:
            base = run_one("baseline (freeze ON, no perturb)")
            print(f"   baseline calls: { {k: len(v) for k, v in base.items()} }")

            same = run_one("repeat (freeze ON, no perturb)")
            for k in base:
                print(f"     [{k:6s}] freeze-ON repeat               vs baseline: {compare(base[k], same[k], k)}")

            perturbed = run_one("freeze ON + perturb dec5_gnn_0", perturb_step=True)
            for k in base:
                print(f"     [{k:6s}] freeze-ON + perturb            vs baseline: {compare(base[k], perturbed[k], k)}")
            restore_params(update_params, snap)
        finally:
            restore_loggers()

    # Now without freeze. Re-install loggers (this time they wrap the
    # ORIGINAL sampler.sample, since we've left the FrozenMatching context).
    for tag, sampler in samplers.items():
        if sampler is not None:
            make_logger(tag, sampler)
    try:
        # Baseline without freeze
        no_freeze_base = run_one("baseline (freeze OFF, no perturb)")
        # Perturbed without freeze
        no_freeze_pert = run_one("freeze OFF + perturb dec5_gnn_0", perturb_step=True)
        for k in no_freeze_base:
            print(f"     [{k:6s}] freeze-OFF + perturb           vs no_freeze_baseline: {compare(no_freeze_base[k], no_freeze_pert[k], k)}")
            print(f"               (expect DIFF — Hungarian re-runs without freeze)")
        restore_params(update_params, snap)
    finally:
        restore_loggers()

    # Interpretation guide
    print("\n  Read:")
    print("   - 'freeze-ON repeat' must be IDENTICAL — same params, same input.")
    print("   - 'freeze-ON + perturb' must be IDENTICAL too — that's the freeze's whole job.")
    print("   - 'freeze-OFF + perturb' SHOULD differ; if it doesn't, even the original Hungarian isn't reshuffling, meaning matching never depended on the chosen layer's weights anyway.")
    print("   If freeze-ON + perturb is NOT IDENTICAL, FrozenMatching is failing to replay → that is the explosion's root cause.")
    return 0


def _resolve_params(collector, layer_key):
    """Mirror probe._resolve_param_set without circular import."""
    val = collector.shared_param_groups[layer_key]
    if hasattr(val, "values"):
        return list(val.values()), layer_key
    return list(val), layer_key


if __name__ == "__main__":
    raise SystemExit(main())
