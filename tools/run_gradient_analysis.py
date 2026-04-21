#!/usr/bin/env python
"""Orchestrate the gradient analysis pipeline across checkpoints and modules.

Usage:
    python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --all
    python tools/run_gradient_analysis.py --config ... --modules M2,M3,M4
    python tools/run_gradient_analysis.py --config ... --checkpoints 1ep,18ep
    python tools/run_gradient_analysis.py --config ... --no-supplementary
    python tools/run_gradient_analysis.py --config ... --smoke   # 3 batches only
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so analyze_gradient_conflict is importable


def _import_runtime():
    """Import HiP-AD runtime modules. Deferred so --help works without env."""
    from mmcv import Config  # type: ignore
    from mmcv.parallel import MMDataParallel  # type: ignore
    from mmcv.runner import load_checkpoint, wrap_fp16_model  # type: ignore
    from mmdet.models import build_detector  # type: ignore  # HiP-AD convention; mmdet3d not installed

    from tools.gradient_analysis.collector import GradientCollector, build_dataloader
    from tools.gradient_analysis.conflict import run_m2, analyze_pair_batches
    from tools.gradient_analysis.probe import run_m3
    from tools.gradient_analysis.correlation import run_m4
    from tools.gradient_analysis.gradnorm import run_m5
    from tools.gradient_analysis.asymmetry import run_m7
    from tools.gradient_analysis.dynamics import run_m6

    return {
        "Config": Config, "MMDataParallel": MMDataParallel,
        "load_checkpoint": load_checkpoint, "wrap_fp16_model": wrap_fp16_model,
        "build_detector": build_detector,
        "GradientCollector": GradientCollector, "build_dataloader": build_dataloader,
        "run_m2": run_m2, "analyze_pair_batches": analyze_pair_batches,
        "run_m3": run_m3, "run_m4": run_m4, "run_m5": run_m5,
        "run_m7": run_m7, "run_m6": run_m6,
    }


def set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _load_plugins(cfg) -> None:
    """Honor cfg.custom_imports and cfg.plugin/plugin_dir so HiP-AD's custom modules
    (SparseDetector, custom heads, datasets, etc.) register with mmcv DETECTORS."""
    import importlib
    import os
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings  # type: ignore
        import_modules_from_strings(**cfg["custom_imports"])
    if getattr(cfg, "plugin", False):
        if hasattr(cfg, "plugin_dir"):
            plugin_dir = cfg.plugin_dir
        else:
            plugin_dir = os.path.dirname(cfg.filename) + "/"
        module_path = ".".join(p for p in os.path.dirname(plugin_dir).split("/") if p)
        importlib.import_module(module_path)


def load_model(rt, cfg, ckpt_path: Path, device: str, fp16: bool):
    _load_plugins(cfg)
    model = rt["build_detector"](cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    if fp16:
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            rt["wrap_fp16_model"](model)
    rt["load_checkpoint"](model, str(ckpt_path), map_location="cpu")
    device_id = int(device.split(":")[1]) if ":" in device else 0
    model = model.to(device)
    model = rt["MMDataParallel"](model, device_ids=[device_id])
    return model


def run_primary_for_checkpoint(
    rt, cfg_ana, ckpt_tag: str, ckpt_path: Path, out_dir: Path,
    modules: List[str], smoke: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = rt["Config"].fromfile(cfg_ana["model_config"])
    model = load_model(rt, model_cfg, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = rt["GradientCollector"](
        model=model,
        tasks=cfg_ana["tasks"],
        shared_layer_names=cfg_ana["shared_param_groups"],
        device=cfg_ana["device"],
    )

    num_batches = 3 if smoke else cfg_ana["primary"]["num_batches"]
    dataloader = rt["build_dataloader"](
        model_cfg,
        batch_size=cfg_ana["primary"]["batch_size"],
        shuffle=True,
        seed=cfg_ana["seed"],
    )

    # Collect gradients (needed for M2 + M5)
    cached = []
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break
        bg, _ = collector.collect_batch(i, data)
        cached.append({
            "batch_idx": bg.batch_idx,
            "shared": bg.shared,
            "full_norm": bg.full_norm,
            "shared_norm": bg.shared_norm,
            "loss_values": bg.loss_values,
        })

    groups = list(collector.shared_param_groups.keys())

    # M2 conflict
    if "M2" in modules:
        rt["run_m2"](cached, cfg_ana["tasks"], groups, out_dir / "conflict")

    # M3 probe — rebuild a fresh dataloader (iterator state reset)
    probe_df = None
    if "M3" in modules:
        dl2 = rt["build_dataloader"](model_cfg, cfg_ana["primary"]["batch_size"], True, cfg_ana["seed"])
        probe_df = rt["run_m3"](
            collector, dl2, num_batches,
            alpha=cfg_ana["probe"]["alpha"],
            steps_list=cfg_ana["probe"]["steps"],
            variants=cfg_ana["probe"]["variants"],
            out_dir=out_dir / "probe",
        )

    # M4 correlation (needs M2 + M3)
    if "M4" in modules and probe_df is not None:
        cos_dfs = {}
        for a_idx, a in enumerate(cfg_ana["tasks"]):
            for b in cfg_ana["tasks"][a_idx + 1:]:
                batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached]
                df = rt["analyze_pair_batches"](batches, a, b, groups)
                if not df.empty:
                    cos_dfs[(a, b)] = df
        if cos_dfs:
            rt["run_m4"](
                cos_dfs, probe_df,
                steps_list=cfg_ana["probe"]["steps"],
                variants=cfg_ana["probe"]["variants"],
                cosine_bins=cfg_ana["binning"]["cosine_bins"],
                out_dir=out_dir / "correlation",
            )

    # M5 gradnorm
    if "M5" in modules and probe_df is not None:
        rt["run_m5"](cached, cfg_ana["tasks"], groups, probe_df,
               steps_list=cfg_ana["probe"]["steps"], out_dir=out_dir / "gradnorm")

    # M7 asymmetry (depends on M3)
    if "M7" in modules and probe_df is not None:
        rt["run_m7"](probe_df, cfg_ana["tasks"],
               steps=1, variant="raw", out_dir=out_dir / "asymmetry")


def run_supplementary(rt, cfg_ana, out_dir: Path) -> None:
    sup = cfg_ana["supplementary"]
    if not sup["enabled"]:
        return
    ckpt_tag = sup["checkpoint"]
    ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"][ckpt_tag]
    model_cfg = rt["Config"].fromfile(cfg_ana["model_config"])
    model = load_model(rt, model_cfg, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = rt["GradientCollector"](
        model=model,
        tasks=cfg_ana["tasks"],
        shared_layer_names=cfg_ana["shared_param_groups"],
        device=cfg_ana["device"],
    )
    dataloader = rt["build_dataloader"](model_cfg, sup["batch_size"], True, cfg_ana["seed"])
    probe_df = rt["run_m3"](
        collector, dataloader, sup["num_samples"],
        alpha=cfg_ana["probe"]["alpha"],
        steps_list=cfg_ana["probe"]["steps"],
        variants=cfg_ana["probe"]["variants"],
        out_dir=out_dir,
    )
    # Collect shared grads on same batches for cos joining
    dl2 = rt["build_dataloader"](model_cfg, sup["batch_size"], True, cfg_ana["seed"])
    cached = []
    for i, data in enumerate(dl2):
        if i >= sup["num_samples"]:
            break
        bg, _ = collector.collect_batch(i, data)
        cached.append({"batch_idx": bg.batch_idx, "shared": bg.shared})
    groups = list(collector.shared_param_groups.keys())
    cos_dfs = {}
    for a_idx, a in enumerate(cfg_ana["tasks"]):
        for b in cfg_ana["tasks"][a_idx + 1:]:
            batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached]
            df = rt["analyze_pair_batches"](batches, a, b, groups)
            if not df.empty:
                cos_dfs[(a, b)] = df
    rt["run_m4"](cos_dfs, probe_df,
           steps_list=cfg_ana["probe"]["steps"],
           variants=cfg_ana["probe"]["variants"],
           cosine_bins=cfg_ana["binning"]["cosine_bins"],
           out_dir=out_dir / "correlation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--modules", default="M2,M3,M4,M5,M6,M7",
                   help="Comma-separated module codes")
    p.add_argument("--checkpoints", default=None,
                   help="Comma-separated tags (default: all in config)")
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Use only 3 batches per checkpoint (quick end-to-end test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config) as f:
        cfg_ana = yaml.safe_load(f)

    set_seeds(cfg_ana["seed"], cfg_ana.get("deterministic", True))
    if cfg_ana.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Now import HiP-AD runtime (deferred so --help works without env)
    rt = _import_runtime()

    modules = args.modules.split(",")
    out_root = Path(cfg_ana["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    tags = list(cfg_ana["checkpoints"].keys())
    if args.checkpoints:
        tags = args.checkpoints.split(",")

    per_ckpt_dirs: Dict[str, Path] = {}
    for tag in tags:
        ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"][tag]
        out_dir = out_root / f"ckpt_{tag}"
        per_ckpt_dirs[tag] = out_dir
        print(f"[{tag}] running modules {modules} → {out_dir}")
        run_primary_for_checkpoint(rt, cfg_ana, tag, ckpt_path, out_dir, modules, smoke=args.smoke)

    if "M6" in modules and len(per_ckpt_dirs) >= 2:
        print(f"[M6] aggregating dynamics across {list(per_ckpt_dirs.keys())}")
        rt["run_m6"](per_ckpt_dirs, cfg_ana["tasks"], out_root / "dynamics")

    if not args.no_supplementary and not args.smoke:
        sup_dir = out_root / "supplementary"
        print(f"[supplementary] batch_size=1 probe on {cfg_ana['supplementary']['checkpoint']}")
        run_supplementary(rt, cfg_ana, sup_dir)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
