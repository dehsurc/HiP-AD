"""Verify that monkey-patching CUDA cumsum with CPU-offload eliminates motion drift.

If yes → root cause is `torch.cumsum` CUDA nondeterminism. Fix: monkey-patch
inside the freeze context.

Run::
    cd /home/yongjae/e2e/HiP-AD && \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/yongjae/miniconda3/envs/hipad/bin/python tools/diag_motion_cumsum_fix.py
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


@contextlib.contextmanager
def deterministic_cumsum():
    """Monkey-patch torch.Tensor.cumsum + torch.cumsum to run on CPU."""
    _orig_tensor_cumsum = torch.Tensor.cumsum
    _orig_torch_cumsum = torch.cumsum

    def patched_tensor(self, *args, **kwargs):
        if self.is_cuda:
            return _orig_tensor_cumsum(self.cpu(), *args, **kwargs).to(self.device)
        return _orig_tensor_cumsum(self, *args, **kwargs)

    def patched_torch(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_cuda:
            return _orig_torch_cumsum(input.cpu(), *args, **kwargs).to(input.device)
        return _orig_torch_cumsum(input, *args, **kwargs)

    torch.Tensor.cumsum = patched_tensor
    torch.cumsum = patched_torch
    try:
        yield
    finally:
        torch.Tensor.cumsum = _orig_tensor_cumsum
        torch.cumsum = _orig_torch_cumsum


PREFIX = {"motion": ["motion_loss", "loss_motion_cls", "loss_motion_reg"]}


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
    from projects.mmdet3d_plugin.models.motion.target import SparseMotionTarget

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

    motion_log = []

    def make_motion_logger(orig):
        def logged(self, *args, **kwargs):
            out = orig(self, *args, **kwargs)
            motion_log.append({"best_reg": out[2].detach().cpu().clone()})
            return out
        return logged

    @contextlib.contextmanager
    def install_logger():
        cur = SparseMotionTarget.sample
        SparseMotionTarget.sample = make_motion_logger(cur)
        try:
            yield
        finally:
            SparseMotionTarget.sample = cur

    def p1_drift(use_cumsum_patch):
        ctx_cs = deterministic_cumsum() if use_cumsum_patch else contextlib.nullcontext()
        snap = adapter.snapshot_temporal_state(model)
        with ctx_cs:
            with adapter.freeze_stochastic_state():
                with install_logger():
                    motion_log.clear()
                    adapter.restore_temporal_state(model, snap)
                    with per_forward_seed(seed):
                        with torch.no_grad():
                            _ = collector.forward_losses(data)
                    f0 = list(motion_log)
                    motion_log.clear()
                    adapter.restore_temporal_state(model, snap)
                    with per_forward_seed(seed):
                        with torch.no_grad():
                            _ = collector.forward_losses(data)
                    f1 = list(motion_log)
        drifts = []
        for a, b in zip(f0, f1):
            d = (a["best_reg"].float() - b["best_reg"].float()).abs().max().item()
            drifts.append(d)
        return drifts

    def p5_descent(use_cumsum_patch, use_freeze):
        snap = adapter.snapshot_temporal_state(model)
        psnap = snapshot_params(full_params)
        ctx_cs = deterministic_cumsum() if use_cumsum_patch else contextlib.nullcontext()
        try:
            with ctx_cs:
                ctx_fr = adapter.freeze_stochastic_state() if use_freeze else contextlib.nullcontext()
                with ctx_fr:
                    adapter.restore_temporal_state(model, snap)
                    with per_forward_seed(seed):
                        with torch.no_grad():
                            bl = collector.forward_losses(data)
                    blm = float(_split(bl, PREFIX["motion"]).item())

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
                    stm = float(_split(st, PREFIX["motion"]).item())
            return blm, stm, gn
        finally:
            restore_params(full_params, psnap)

    print("=" * 78)
    print("[BASELINE] No cumsum patch")
    print("=" * 78)
    d = p1_drift(use_cumsum_patch=False)
    print(f"  [P1] best_reg drift per layer: {[f'{x:.2e}' for x in d]}  max={max(d):.4e}")
    on = p5_descent(False, True)
    off = p5_descent(False, False)
    print(f"  [P5] freeze ON  Δ={on[1]-on[0]:+.4e}  ‖g‖={on[2]:.4e}")
    print(f"  [P5] freeze OFF Δ={off[1]-off[0]:+.4e}  ‖g‖={off[2]:.4e}")

    print()
    print("=" * 78)
    print("[FIX] CUDA cumsum → CPU offload")
    print("=" * 78)
    d = p1_drift(use_cumsum_patch=True)
    print(f"  [P1] best_reg drift per layer: {[f'{x:.2e}' for x in d]}  max={max(d):.4e}")
    on = p5_descent(True, True)
    off = p5_descent(True, False)
    print(f"  [P5] freeze ON  Δ={on[1]-on[0]:+.4e}  ‖g‖={on[2]:.4e}")
    print(f"  [P5] freeze OFF Δ={off[1]-off[0]:+.4e}  ‖g‖={off[2]:.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
