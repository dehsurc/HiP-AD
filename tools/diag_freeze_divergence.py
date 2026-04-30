"""Find which nn.Module outputs diverge between F0 (record) and F1 (replay)
when freeze_matching+temporal_state+per_forward_seed are all on.

If predictions are perfectly deterministic, the only effect of `freeze_matching`
is to swap one set of GT-derived targets for another (cached vs current). Same
predictions × same targets = same loss. The fact that loss drifts ~100 means
SOME module is producing different outputs across the two forwards. This script
hooks every nn.Module, runs F0 then F1 with full restoration, and prints the
first divergent module along with diff magnitude.

Run::
    cd /home/yongjae/e2e/HiP-AD && \
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
    /home/yongjae/miniconda3/envs/hipad/bin/python tools/diag_freeze_divergence.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _summarize(t: torch.Tensor) -> str:
    return (f"shape={tuple(t.shape)} dtype={t.dtype} "
            f"mean={t.float().mean().item():+.6e} "
            f"std={t.float().std().item():.6e} "
            f"absmax={t.float().abs().max().item():.6e}")


def _diff(a: torch.Tensor, b: torch.Tensor) -> str:
    if a.shape != b.shape:
        return f"shape mismatch {a.shape} vs {b.shape}"
    if a.dtype != b.dtype:
        a = a.float(); b = b.float()
    d = (a.float() - b.float()).abs()
    return f"absmax={d.max().item():.6e} mean={d.mean().item():.6e}"


def main() -> int:
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    from tools.gradient_analysis.collector import GradientCollector, build_dataloader
    from tools.gradient_analysis.compat import (
        apply_use_reentrant_false, apply_index_put_fix,
    )
    from tools.gradient_analysis.matching_freeze import FrozenMatching, per_forward_seed
    from tools.gradient_analysis.temporal_state import ModelStateSnapshot
    from tools.run_gradient_analysis import _load_plugins, set_seeds

    apply_use_reentrant_false()
    apply_index_put_fix()

    with open("configs/gradient_analysis.yaml") as f:
        cfg_ana = yaml.safe_load(f)
    set_seeds(cfg_ana["seed"], False)

    model_cfg = Config.fromfile(cfg_ana["model_config"])
    _load_plugins(model_cfg)
    model = build_detector(model_cfg.model,
                           train_cfg=model_cfg.get("train_cfg"),
                           test_cfg=model_cfg.get("test_cfg"))
    model.init_weights()
    ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"]["1ep"]
    load_checkpoint(model, str(ckpt_path), map_location="cpu")
    device = cfg_ana["device"]
    device_id = int(device.split(":")[1]) if ":" in device else 0
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[device_id])

    collector = GradientCollector(
        model=model, tasks=cfg_ana["tasks"],
        shared_layer_names=cfg_ana["shared_param_groups"], device=device,
    )

    dataloader = build_dataloader(
        model_cfg, batch_size=cfg_ana["primary"]["batch_size"],
        shuffle=True, seed=cfg_ana["seed"],
    )
    data = next(iter(dataloader))

    raw = collector.model.module if hasattr(collector.model, "module") else collector.model
    handles = []
    capture: dict = {}
    name_of: dict = {}

    def make_hook(name):
        def hook(_mod, _inp, out):
            if name not in capture:
                capture[name] = []
            t = None
            if torch.is_tensor(out):
                t = out
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if torch.is_tensor(x):
                        t = x
                        break
            if t is not None:
                capture[name].append(t.detach().float().cpu().clone())
        return hook

    for name, mod in raw.named_modules():
        if len(list(mod.children())) > 0:
            continue  # only leaf modules
        h = mod.register_forward_hook(make_hook(name))
        handles.append(h)
        name_of[id(mod)] = name

    print(f"hooked {len(handles)} leaf modules")

    fm = FrozenMatching()
    state_snap = ModelStateSnapshot(collector.model)
    with fm:
        # F0: record
        state_snap.restore(); fm.next_forward()
        capture.clear()
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                f0_losses = collector.forward_losses(data)
        f0_capture = {k: list(v) for k, v in capture.items()}
        f0_det = sum(v.item() for k, v in f0_losses.items() if k.startswith("det_loss"))

        # F1: replay
        state_snap.restore(); fm.next_forward()
        capture.clear()
        with per_forward_seed(cfg_ana["seed"]):
            with torch.no_grad():
                f1_losses = collector.forward_losses(data)
        f1_capture = {k: list(v) for k, v in capture.items()}
        f1_det = sum(v.item() for k, v in f1_losses.items() if k.startswith("det_loss"))

    for h in handles:
        h.remove()

    print(f"\nF0 det_loss = {f0_det:.6f}")
    print(f"F1 det_loss = {f1_det:.6f}  Δ = {f1_det - f0_det:+.6f}")

    # Order modules in named_modules order so first divergent comes first
    ordered_names = [n for n, _m in raw.named_modules() if n in f0_capture]

    print("\nFirst 30 divergent leaf modules (sorted by named_modules order):")
    print("-" * 110)
    n_diverge = 0
    n_match = 0
    for name in ordered_names:
        f0_outs = f0_capture.get(name, [])
        f1_outs = f1_capture.get(name, [])
        if len(f0_outs) != len(f1_outs):
            print(f"  [DIFF] {name:80s} call-count {len(f0_outs)} vs {len(f1_outs)}")
            n_diverge += 1
            continue
        max_diff = 0.0
        for f0_t, f1_t in zip(f0_outs, f1_outs):
            if f0_t.shape != f1_t.shape:
                max_diff = float("inf")
                break
            d = (f0_t - f1_t).abs().max().item()
            if d > max_diff:
                max_diff = d
        if max_diff > 1e-7:
            if n_diverge < 30:
                print(f"  [DIFF] {name:80s} absmax={max_diff:.3e} (calls={len(f0_outs)})")
            n_diverge += 1
        else:
            n_match += 1
    print("-" * 110)
    print(f"Total: {n_diverge} divergent, {n_match} identical leaf modules.")
    print("First divergent module is the source of nondeterminism — that's what we need to pin/replace.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
