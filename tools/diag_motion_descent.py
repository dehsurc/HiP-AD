"""Motion-descent diagnostic.

Question: in the new (Phase-2 adapter) freeze plumbing, why does
``probe_matrix_1step_normalized.csv`` show ``(motion, motion) = +0.022`` —
i.e. stepping in motion's gradient direction *increases* motion loss?

Tests, in order of cost:

  [A] Inside ``freeze_stochastic_state`` + outer state_snap + per_forward_seed,
      run 3 consecutive baseline forwards on the SAME data + SAME params.
      Expected: motion loss drift ~ 0 (down to ~1e-6). If it drifts, freeze
      is leaking even before any param mutation — the patch isn't really
      pinning motion's mode_idx.

  [B] Same context, but: baseline fwd → motion grad on full params →
      apply normalized α=1e-3 step → re-forward. Report ΔL_motion.
      Expected: NEGATIVE (descent). Observed in the probe CSV: POSITIVE.
      This script confirms it on a tiny batch.

  [C] Same as [B] but for det/map/plan as sanity. If only motion
      misbehaves, the bug is motion-specific (mode-argmin freeze vs
      Hungarian) — confirms our reading of the matrix.

  [D] Repeat [A] and [B] but DISABLE the inner ``_freeze_snapshot`` in
      ``adapter.forward_losses`` (keep only the probe's outer state_snap).
      If motion descent is restored → the inner snapshot is the culprit.

  [E] Repeat [A] and [B] with the hardcoded ``_per_forward_seed(42)`` in
      ``forward_losses`` removed. If descent is restored → the duplicate
      seed reset is the culprit.

Run::
    cd /home/yongjae/e2e/HiP-AD && \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/yongjae/miniconda3/envs/hipad/bin/python tools/diag_motion_descent.py
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


def _split(losses, prefix):
    """Sum loss values whose keys start with `prefix` (matches adapter.split_losses)."""
    out = None
    for k, v in losses.items():
        if not isinstance(v, torch.Tensor):
            continue
        if any(k.startswith(p) for p in prefix):
            out = v if out is None else out + v
    return out


# Prefixes mirror HipadAdapter._load_utils() task_groups.
PREFIX = {
    "det":    ["det_loss", "loss_det_cls", "loss_det_reg"],
    "map":    ["map_loss", "loss_map_cls", "loss_map_reg"],
    "motion": ["motion_loss", "loss_motion_cls", "loss_motion_reg"],
    "plan":   ["plan_loss", "loss_plan_cls", "loss_plan_reg"],
}


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

    # Tiny batch — we just need one batch to verify the descent property.
    dl = build_dataloader(adapter, batch_size=1, shuffle=True, seed=cfg["seed"])
    data = next(iter(dl))

    seed = cfg["seed"]
    full_params = collector.full_params

    # ------------------------------------------------------------------ helpers
    def fwd_baseline(label):
        """One no_grad forward inside freeze ctx, with outer snap+seed."""
        snap = adapter.snapshot_temporal_state(model)
        with adapter.freeze_stochastic_state():
            adapter.restore_temporal_state(model, snap)
            with per_forward_seed(seed):
                with torch.no_grad():
                    losses = collector.forward_losses(data)
            return {t: float(v.item()) if (v := _split(losses, p)) is not None else float("nan")
                    for t, p in PREFIX.items()}

    def three_baselines(label):
        """3 baselines in one freeze ctx — measures intra-context drift."""
        snap = adapter.snapshot_temporal_state(model)
        all_losses = []
        with adapter.freeze_stochastic_state():
            for i in range(3):
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        losses = collector.forward_losses(data)
                tasks = {t: float(v.item()) if (v := _split(losses, p)) is not None else float("nan")
                         for t, p in PREFIX.items()}
                all_losses.append(tasks)
        return all_losses

    def step_and_measure(source_task, label, alpha=1e-3, normalized=True):
        """Baseline → grad on source → α-step → restepped forward.
        Returns dict { target: (baseline, stepped, delta) } for all tasks."""
        snap = adapter.snapshot_temporal_state(model)
        param_snap = snapshot_params(full_params)
        try:
            with adapter.freeze_stochastic_state():
                # Baseline
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        bl = collector.forward_losses(data)
                baseline = {t: float(v.item()) if (v := _split(bl, p)) is not None else float("nan")
                            for t, p in PREFIX.items()}

                # Gradient on source task (with grad)
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    losses = collector.forward_losses(data)
                tl = _split(losses, PREFIX[source_task])
                if tl is None:
                    return None
                g = compute_task_full_gradient(tl, full_params, retain_graph=False)
                gn = float(torch.cat([gi.flatten() for gi in g]).norm())
                apply_virtual_step(full_params, g, alpha=alpha, normalize=normalized)
                del losses, tl, g
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Stepped forward
                adapter.restore_temporal_state(model, snap)
                with per_forward_seed(seed):
                    with torch.no_grad():
                        st = collector.forward_losses(data)
                stepped = {t: float(v.item()) if (v := _split(st, p)) is not None else float("nan")
                           for t, p in PREFIX.items()}
        finally:
            restore_params(full_params, param_snap)
        return {
            "grad_norm": gn,
            "deltas": {t: (baseline[t], stepped[t], stepped[t] - baseline[t])
                       for t in PREFIX},
        }

    # ============================================================== TEST [A]
    print("=" * 78)
    print("[A] Drift across 3 consecutive baseline forwards (same params, same data)")
    print("    Expected: ALL tasks ~0 drift if freeze is solid.")
    print("=" * 78)
    losses_a = three_baselines("A")
    for t in PREFIX:
        vs = [L[t] for L in losses_a]
        drift = max(vs) - min(vs)
        print(f"  {t:6s}: {vs[0]:.6f}  {vs[1]:.6f}  {vs[2]:.6f}    drift={drift:+.6e}")

    # ============================================================== TEST [B]+[C]
    print()
    print("=" * 78)
    print("[B]+[C] Diagonal descent: source=task → step → ΔL_task ?")
    print("    Expected: ΔL_self < 0 (stepping in -gradient direction descends).")
    print("=" * 78)
    diag_results = {}
    for source in ["det", "map", "motion", "plan"]:
        r = step_and_measure(source, label=source, alpha=1e-3, normalized=True)
        if r is None:
            print(f"  source={source}: SKIPPED (loss missing)")
            continue
        diag_results[source] = r
        print(f"  source={source:6s} ‖g‖={r['grad_norm']:.4e}")
        for t in PREFIX:
            b, s, d = r["deltas"][t]
            mark = ""
            if t == source:
                mark = "   <-- DIAGONAL " + ("OK (negative)" if d < 0 else "VIOLATION (positive!)")
            print(f"    target={t:6s}  base={b:9.4f}  step={s:9.4f}  Δ={d:+9.4e}{mark}")

    # ============================================================== TEST [D]
    print()
    print("=" * 78)
    print("[D] Disable INNER _freeze_snapshot in adapter.forward_losses, retry [A]+[B]")
    print("    If motion drift/descent recover → inner snapshot is the culprit.")
    print("=" * 78)
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    # Monkey-patch forward_losses to skip the inner _freeze_snapshot dance.
    _orig_fwd = HipadAdapter.forward_losses

    def fwd_no_inner_snapshot(self, model, data):
        raw = model.module if hasattr(model, "module") else model
        model.train()
        if hasattr(model, "module"):
            batch = data
        else:
            from mmcv.parallel import scatter
            dev = next(raw.parameters()).device
            did = dev.index if dev.type == "cuda" else -1
            batch = scatter(data, [did])[0] if did >= 0 else data
        # Skip _freeze_snapshot entirely. Still call next_forward() so
        # cached patches replay.
        if self._freeze_session is not None:
            self._freeze_session.next_forward()
        with self._selective_eval(raw):
            losses = model(**batch)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]
        return losses

    HipadAdapter.forward_losses = fwd_no_inner_snapshot
    try:
        losses_d = three_baselines("D-A")
        print("  [A] drift without inner snapshot:")
        for t in PREFIX:
            vs = [L[t] for L in losses_d]
            drift = max(vs) - min(vs)
            print(f"    {t:6s}: {vs[0]:.6f}  {vs[1]:.6f}  {vs[2]:.6f}    drift={drift:+.6e}")
        print("  [B] motion descent without inner snapshot:")
        r = step_and_measure("motion", "D-B", alpha=1e-3, normalized=True)
        for t in PREFIX:
            b, s, d = r["deltas"][t]
            mark = "   <-- DIAGONAL" if t == "motion" else ""
            print(f"    target={t:6s}  base={b:9.4f}  step={s:9.4f}  Δ={d:+9.4e}{mark}")
    finally:
        HipadAdapter.forward_losses = _orig_fwd

    # ============================================================== TEST [E]
    print()
    print("=" * 78)
    print("[E] Remove hardcoded _per_forward_seed(42) in forward_losses, retry [A]+[B]")
    print("    (inner snapshot reinstated; only the duplicate seed is removed)")
    print("=" * 78)

    def fwd_no_inner_seed(self, model, data):
        raw = model.module if hasattr(model, "module") else model
        model.train()
        if hasattr(model, "module"):
            batch = data
        else:
            from mmcv.parallel import scatter
            dev = next(raw.parameters()).device
            did = dev.index if dev.type == "cuda" else -1
            batch = scatter(data, [did])[0] if did >= 0 else data
        if self._freeze_session is not None:
            if self._freeze_snapshot is None:
                from tools.gradient_analysis.adapters.hipad import _ModelStateSnapshot
                self._freeze_snapshot = _ModelStateSnapshot(model)
            self._freeze_snapshot.restore()
            self._freeze_session.next_forward()
        # NO inner per_forward_seed; rely on outer.
        with self._selective_eval(raw):
            losses = model(**batch)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]
        return losses

    HipadAdapter.forward_losses = fwd_no_inner_seed
    try:
        losses_e = three_baselines("E-A")
        print("  [A] drift without inner per_forward_seed:")
        for t in PREFIX:
            vs = [L[t] for L in losses_e]
            drift = max(vs) - min(vs)
            print(f"    {t:6s}: {vs[0]:.6f}  {vs[1]:.6f}  {vs[2]:.6f}    drift={drift:+.6e}")
        print("  [B] motion descent without inner per_forward_seed:")
        r = step_and_measure("motion", "E-B", alpha=1e-3, normalized=True)
        for t in PREFIX:
            b, s, d = r["deltas"][t]
            mark = "   <-- DIAGONAL" if t == "motion" else ""
            print(f"    target={t:6s}  base={b:9.4f}  step={s:9.4f}  Δ={d:+9.4e}{mark}")
    finally:
        HipadAdapter.forward_losses = _orig_fwd

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
