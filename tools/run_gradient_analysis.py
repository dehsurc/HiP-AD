#!/usr/bin/env python
"""Orchestrate the gradient analysis pipeline across checkpoints and modules.
cd /home/yongjae/e2e/HiP-AD && CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. /home/yongjae/miniconda3/envs/hipad/bin/python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --checkpoints 1ep --modules M3,M4 2>&1 | tee gradient_analysis_results/supplementary.log
Usage:
    python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --all
    python tools/run_gradient_analysis.py --config ... --modules M2,M3,M4
    python tools/run_gradient_analysis.py --config ... --checkpoints 1ep,18ep
    python tools/run_gradient_analysis.py --config ... --no-supplementary
    python tools/run_gradient_analysis.py --config ... --smoke   # 3 batches only
"""
from __future__ import annotations

import argparse
import gc
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so analyze_gradient_conflict is importable

ALL_MODULES = "M2,M3,M4,M5,M6,M7,M_N1,M_N2,M_N10,M_AS,M_Q"


def _import_runtime():
    """Import runtime modules. Deferred so --help works without model env."""
    from tools.gradient_analysis.adapters.hipad import HipadAdapter
    from tools.gradient_analysis.adapters.vad import VadAdapter
    from tools.gradient_analysis.collector import GradientCollector, build_dataloader
    from tools.gradient_analysis.conflict import run_m2, analyze_pair_batches
    from tools.gradient_analysis.probe import run_m3, run_alpha_sweep
    from tools.gradient_analysis.correlation import run_m4, run_layer_m4
    from tools.gradient_analysis.gradnorm import run_m5
    from tools.gradient_analysis.asymmetry import run_m7
    from tools.gradient_analysis.dynamics import run_m6
    from tools.gradient_analysis.landscape import run_m8
    from tools.gradient_analysis.null_baseline import run_null_baseline
    from tools.gradient_analysis.distribution import run_distribution
    from tools.gradient_analysis.bootstrap import augment_summary_with_ci
    from tools.gradient_analysis.magnitude_dynamics import run_magnitude_dynamics
    from tools.gradient_analysis.query_sensitivity import run_query_sensitivity

    return {
        "HipadAdapter": HipadAdapter, "VadAdapter": VadAdapter,
        "GradientCollector": GradientCollector, "build_dataloader": build_dataloader,
        "run_m2": run_m2, "analyze_pair_batches": analyze_pair_batches,
        "run_m3": run_m3, "run_m4": run_m4, "run_layer_m4": run_layer_m4,
        "run_m5": run_m5,
        "run_m7": run_m7, "run_m6": run_m6, "run_m8": run_m8,
        "run_alpha_sweep": run_alpha_sweep,
        "run_null_baseline": run_null_baseline,
        "run_distribution": run_distribution,
        "augment_summary_with_ci": augment_summary_with_ci,
        "run_magnitude_dynamics": run_magnitude_dynamics,
        "run_query_sensitivity": run_query_sensitivity,
    }


def set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark=True lets cuDNN search for a workable algorithm; without
    # it, deterministic mode often fails with "Unable to find a valid cuDNN
    # algorithm". Reproducibility for our probe comes from RNG seeding +
    # the adapter's per-forward seed pinning, not from
    # torch.use_deterministic_algorithms.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = bool(deterministic)


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


def build_adapter(
    rt,
    cfg_ana,
    adapter_name: Optional[str] = None,
    vad_repo_root: Optional[str] = None,
    adapter_config: Optional[str] = None,
):
    """Construct the active model adapter from CLI/YAML settings."""
    name = (
        adapter_name
        or cfg_ana.get("adapter")
        or cfg_ana.get("model_adapter")
        or cfg_ana.get("model")
        or "hipad"
    )
    name = str(name).lower().replace("_", "-")
    model_config = (
        adapter_config
        or cfg_ana.get("adapter_config")
        or cfg_ana.get("model_config")
    )
    if name in {"hipad", "hi-pad"}:
        kwargs = {}
        if model_config:
            kwargs["config_path"] = Path(model_config)
        if cfg_ana.get("tasks"):
            kwargs["task_names"] = list(cfg_ana["tasks"])
        return rt["HipadAdapter"](**kwargs)
    if name == "vad":
        kwargs = {}
        repo_root = (
            vad_repo_root
            or cfg_ana.get("vad_repo_root")
            or cfg_ana.get("repo_root")
            or cfg_ana.get("vad_repo")
        )
        if repo_root:
            kwargs["repo_root"] = Path(repo_root)
        if model_config:
            kwargs["config_path"] = Path(model_config)
        return rt["VadAdapter"](**kwargs)
    raise ValueError(f"unknown gradient-analysis adapter: {name}")


def _resolve_ckpt_path(cfg_ana, ckpt_tag: str) -> Path:
    raw = Path(cfg_ana["checkpoints"][ckpt_tag])
    if raw.is_absolute():
        return raw
    return Path(cfg_ana["ckpt_root"]) / raw


def load_model(rt, adapter, ckpt_path: Path, device: str, fp16: bool):
    del rt, fp16
    return adapter.build_model(ckpt=ckpt_path, device=device)


def _unique(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def run_primary_for_checkpoint(
    rt, cfg_ana, adapter, ckpt_tag: str, ckpt_path: Path, out_dir: Path,
    modules: List[str], smoke: bool,
    target_layers: Optional[List[str]] = None,
    freeze_matching: bool = False,
    reset_temporal_state: bool = False,
    ckpt_order: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    layer_cfg = cfg_ana.get("layer_conflict", {})
    layer_enabled = bool(layer_cfg.get("enabled", True))
    layer_names = list(layer_cfg.get("layers") or target_layers or [])
    requested_groups = _unique(list(cfg_ana["shared_param_groups"]) + layer_names)
    model = load_model(rt, adapter, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = rt["GradientCollector"](
        model=model,
        adapter=adapter,
        shared_layer_names=requested_groups,
        device=cfg_ana["device"],
    )
    tasks = list(collector.tasks)

    num_batches = 3 if smoke else cfg_ana["primary"]["num_batches"]
    dataloader = rt["build_dataloader"](
        adapter,
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

    available_groups = list(collector.shared_param_groups.keys())
    groups = [g for g in cfg_ana["shared_param_groups"] if g in collector.shared_param_groups]
    if not groups:
        groups = available_groups
    layer_groups = [g for g in layer_names if g in collector.shared_param_groups]

    # M2 conflict
    if "M2" in modules:
        rt["run_m2"](cached, tasks, groups, out_dir / "conflict")
        if layer_enabled and layer_groups:
            rt["run_m2"](cached, tasks, layer_groups, out_dir / "layer_conflict")

    # M_N1 — random-baseline / permutation test (Phase 1 #1)
    if "M_N1" in modules or "null_baseline" in modules:
        nb_dir = out_dir / "null_baseline"
        nb_dir.mkdir(parents=True, exist_ok=True)
        rt["run_null_baseline"](
            cached_batches=cached,
            tasks=tasks,
            group_keys=groups,
            n_repeats=cfg_ana.get("null_baseline", {}).get("n_repeats", 1000),
            seed=cfg_ana.get("seed", 42),
            out_path=nb_dir / "null_baseline.csv",
        )

    # M_N2 — distribution diagnostics (Phase 1 #2)
    if "M_N2" in modules or "distribution" in modules:
        dist_dir = out_dir / "distribution"
        dist_dir.mkdir(parents=True, exist_ok=True)
        rt["run_distribution"](
            cached_batches=cached,
            tasks=tasks,
            group_keys=groups,
            out_path=dist_dir / "distribution_report.csv",
            emit_kde_figures=True,
            figures_dir=dist_dir / "kde",
        )

    # Bootstrap CI augmentation on the M2 conflict summary CSVs (Phase 1 #3)
    if "M2" in modules and cfg_ana.get("bootstrap", {}).get("enabled", True):
        n_resamples = cfg_ana.get("bootstrap", {}).get("n_resamples", 2000)
        for f in (out_dir / "conflict").glob("conflict_*_summary.csv"):
            per_batch_path = f.with_name(f.name.replace("_summary.csv", "_per_batch.csv"))
            if not per_batch_path.exists():
                continue
            try:
                summary_df = pd.read_csv(f)
                per_batch_df = pd.read_csv(per_batch_path)
                augmented = rt["augment_summary_with_ci"](
                    summary_df, per_batch_df,
                    group_cols=["group"],
                    value_cols=["mean_cos", "median_cos", "mean_coop_mag", "mean_conf_mag"],
                    rate_value_cols=["conflict_ratio"],
                    rate_per_batch_col="cos",
                    n_resamples=n_resamples,
                    seed=cfg_ana.get("seed", 42),
                )
                augmented.to_csv(f, index=False)
            except Exception as e:
                print(f"[bootstrap] failed on {f.name}: {e}")

    # M3 probe — rebuild a fresh dataloader (iterator state reset). Free any
    # residual autograd graphs / cached tensors from the M2 collect loop so
    # cuDNN has room to pick a workspace-hungry conv algorithm.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    probe_df = None
    if "M3" in modules:
        dl2 = rt["build_dataloader"](adapter, cfg_ana["primary"]["batch_size"], True, cfg_ana["seed"])
        forward_seed = cfg_ana.get("probe", {}).get("forward_seed", cfg_ana.get("seed"))
        probe_df = rt["run_m3"](
            collector, dl2, num_batches,
            alpha=cfg_ana["probe"]["alpha"],
            steps_list=cfg_ana["probe"]["steps"],
            variants=cfg_ana["probe"]["variants"],
            out_dir=out_dir / "probe",
            target_layers=target_layers,
            freeze_matching=freeze_matching,
            forward_seed=forward_seed,
            reset_temporal_state=reset_temporal_state,
        )

    # M_Q — query sensitivity (Part J). HiP-AD only; VadAdapter.get_task_queries
    # raises NotImplementedError. Writes ckpt_<tag>/query_sensitivity/qs.csv ;
    # the per-ckpt rows are merged into <out_root>/plan_centric/query_sensitivity.csv
    # at the end of main() so the notebook Part J cell can read a single file.
    if "M_Q" in modules or "query_sensitivity" in modules:
        if isinstance(adapter, rt["VadAdapter"]):
            print(f"[M_Q] adapter=VAD — query sensitivity unsupported, skipping {ckpt_tag}")
        else:
            dl_q = rt["build_dataloader"](adapter, cfg_ana["primary"]["batch_size"], True, cfg_ana["seed"])
            forward_seed = cfg_ana.get("probe", {}).get("forward_seed", cfg_ana.get("seed"))
            rt["run_query_sensitivity"](
                collector=collector,
                dataloader=dl_q,
                num_batches=num_batches,
                out_dir=out_dir / "query_sensitivity",
                forward_seed=forward_seed,
                freeze_matching=freeze_matching,
                model_name="HiP-AD",
                checkpoint=ckpt_tag,
                checkpoint_order=ckpt_order,
            )

    # M4/M5/M7 assume probe_df aggregates over the full param set (one ΔL per
    # (source, target) pair). Per-layer probes produce one row per layer; full
    # M4/M5/M7 are skipped, and the layer-aware M4 below handles the join.
    per_layer = bool(target_layers)
    if per_layer and any(m in modules for m in ("M4", "M5", "M7")):
        print(f"[{ckpt_tag}] per-layer probe active (layers={target_layers}); "
              f"skipping full M4/M5/M7; running layer-level M4 when available.")

    # M4 correlation (needs M2 + M3)
    if "M4" in modules and probe_df is not None and not per_layer:
        cos_dfs = {}
        for a_idx, a in enumerate(tasks):
            for b in tasks[a_idx + 1:]:
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

    # Layer-aware M4: join per-layer conflict cosine with directional
    # per-layer Δloss rows from M3. This answers: "when tasks conflict on
    # this layer, does a source-task update hurt/help the other task?"
    if "M4" in modules and probe_df is not None and per_layer and layer_groups:
        cos_dfs = {}
        for a_idx, a in enumerate(tasks):
            for b in tasks[a_idx + 1:]:
                batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached]
                df = rt["analyze_pair_batches"](batches, a, b, layer_groups)
                if not df.empty:
                    cos_dfs[(a, b)] = df
        if cos_dfs:
            rt["run_layer_m4"](
                cos_dfs, probe_df,
                steps_list=cfg_ana["probe"]["steps"],
                variants=cfg_ana["probe"]["variants"],
                cosine_bins=cfg_ana["binning"]["cosine_bins"],
                out_dir=out_dir / "layer_correlation",
            )

    # M5 gradnorm
    if "M5" in modules and probe_df is not None and not per_layer:
        rt["run_m5"](cached, tasks, groups, probe_df,
               steps_list=cfg_ana["probe"]["steps"], out_dir=out_dir / "gradnorm")

    # M7 asymmetry (depends on M3)
    if "M7" in modules and probe_df is not None and not per_layer:
        rt["run_m7"](probe_df, tasks,
               steps=1, variant="raw", out_dir=out_dir / "asymmetry")

    # M_AS — α-sensitivity sweep (Phase 1 #9). Runs only on the configured
    # checkpoint to keep cost bounded. The checkpoint match is on the tag,
    # not the path, so a YAML override applies cleanly. Uses a factory
    # closure so each α gets a fresh deterministic iterator without
    # materializing the entire dataset.
    if "M_AS" in modules or "alpha_sweep" in modules:
        sweep_cfg = cfg_ana.get("alpha_sweep", {})
        if sweep_cfg.get("enabled", False) and ckpt_tag == sweep_cfg.get("checkpoint"):
            sweep_dir = out_dir / "probe" / "alpha_sweep"
            def _sweep_dl_factory():
                return rt["build_dataloader"](
                    adapter, cfg_ana["primary"]["batch_size"], True, cfg_ana["seed"],
                )
            rt["run_alpha_sweep"](
                collector=collector,
                dl_factory=_sweep_dl_factory,
                num_batches=sweep_cfg.get("num_batches", 100),
                alphas=sweep_cfg["alphas"],
                sources=sweep_cfg["sources"],
                out_dir=sweep_dir,
                steps_list=[1],
                variants=cfg_ana["probe"]["variants"],
                target_layers=target_layers,
                freeze_matching=freeze_matching,
                forward_seed=cfg_ana.get("probe", {}).get("forward_seed",
                                                          cfg_ana.get("seed")),
                reset_temporal_state=reset_temporal_state,
            )

    # M8 gradient surface (1D line probe; independent of M2/M3)
    if "M8" in modules:
        ls_cfg = cfg_ana.get("landscape", {})
        if not ls_cfg.get("enabled", True):
            print(f"[{ckpt_tag}] M8 disabled in config (landscape.enabled=false); skipping")
        else:
            num_ls = 3 if smoke else ls_cfg.get("num_batches", 5)
            t_grid = ls_cfg.get("t_grid") or [
                -3.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0
            ]
            ls_bs = ls_cfg.get("batch_size") or cfg_ana["primary"]["batch_size"]
            dl3 = rt["build_dataloader"](adapter, ls_bs, True, cfg_ana["seed"])
            rt["run_m8"](
                collector, dl3, num_ls,
                alpha=cfg_ana["probe"]["alpha"],
                t_grid=t_grid,
                out_dir=out_dir / "landscape",
                target_layers=target_layers,
                freeze_matching=freeze_matching,
                forward_seed=cfg_ana.get("probe", {}).get("forward_seed", cfg_ana.get("seed")),
                reset_temporal_state=reset_temporal_state,
            )

def _release_gpu_resources(label: str = "") -> None:
    """Force a Python GC pass + CUDA cache release. Call this from the
    OUTER caller after a function that built a model/collector returns —
    by that point the function-frame locals are unreferenced and gc.collect
    can actually reclaim them, freeing the underlying GPU tensors."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"[cleanup{':' + label if label else ''}] "
                  f"gpu free={free/2**30:.2f}GiB / {total/2**30:.2f}GiB")
        except Exception:
            pass


def run_supplementary(
    rt, cfg_ana, adapter, out_dir: Path,
    target_layers: Optional[List[str]] = None,
    freeze_matching: bool = False,
    reset_temporal_state: bool = False,
) -> None:
    sup = cfg_ana["supplementary"]
    if not sup["enabled"]:
        return
    ckpt_tag = sup["checkpoint"]
    ckpt_path = _resolve_ckpt_path(cfg_ana, ckpt_tag)
    model = load_model(rt, adapter, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = rt["GradientCollector"](
        model=model,
        adapter=adapter,
        shared_layer_names=cfg_ana["shared_param_groups"],
        device=cfg_ana["device"],
    )
    tasks = list(collector.tasks)
    dataloader = rt["build_dataloader"](adapter, sup["batch_size"], True, cfg_ana["seed"])
    forward_seed = cfg_ana.get("probe", {}).get("forward_seed", cfg_ana.get("seed"))
    probe_df = rt["run_m3"](
        collector, dataloader, sup["num_samples"],
        alpha=cfg_ana["probe"]["alpha"],
        steps_list=cfg_ana["probe"]["steps"],
        variants=cfg_ana["probe"]["variants"],
        out_dir=out_dir,
        target_layers=target_layers,
        freeze_matching=freeze_matching,
        forward_seed=forward_seed,
        reset_temporal_state=reset_temporal_state,
    )
    # M4 correlation assumes one ΔL per (source, target) pair (full-update
    # probe). Per-layer probe_df has multiple ΔL rows per pair (one per
    # layer), which makes the cosine ↔ ΔL join multi-valued and the
    # pearsonr/spearmanr calls hit NaN/inf. Skip in per-layer mode, mirror
    # the guard in run_primary_for_checkpoint.
    if target_layers:
        print(f"[supplementary] per-layer probe active (layers={list(target_layers)}); "
              f"skipping correlation/M4 step (full-update probe required).")
        return
    # Collect shared grads on same batches for cos joining
    dl2 = rt["build_dataloader"](adapter, sup["batch_size"], True, cfg_ana["seed"])
    cached = []
    for i, data in enumerate(dl2):
        if i >= sup["num_samples"]:
            break
        bg, _ = collector.collect_batch(i, data)
        cached.append({"batch_idx": bg.batch_idx, "shared": bg.shared})
    groups = list(collector.shared_param_groups.keys())
    cos_dfs = {}
    for a_idx, a in enumerate(tasks):
        for b in tasks[a_idx + 1:]:
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
    p.add_argument("--modules", default="M2,M3,M4,M5,M6,M7,M_N1,M_N2",
                   help="Comma-separated module codes (M_N1=null_baseline, "
                        "M_N2=distribution, M_N10=magnitude_dynamics, M_AS=alpha_sweep)")
    p.add_argument("--adapter", choices=["hipad", "vad"], default=None,
                   help="Override YAML `adapter` (default: hipad).")
    p.add_argument("--vad-repo-root", default=None,
                   help="Only used when --adapter=vad. Overrides YAML `vad_repo_root`.")
    p.add_argument("--adapter-config", default=None,
                   help="Override the model config path the adapter uses.")
    p.add_argument("--checkpoints", default=None,
                   help="Comma-separated tags (default: all in config)")
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Use only 3 batches per checkpoint (quick end-to-end test)")
    p.add_argument("--probe-layers", default=None,
                   help="Comma-separated shared_param group keys (e.g. 'dec3_ffn_0,fc_after'). "
                        "When set, M3 applies the virtual update to ONLY those layers and "
                        "produces per-layer ΔL matrices. Overrides probe.target_layers in YAML.")
    p.add_argument("--freeze-matching", dest="freeze_matching", action="store_true",
                   default=None,
                   help="Pin Hungarian matching (det/map) and mode-selection argmin "
                        "(motion/plan) across baseline/grad/stepped forwards. Overrides "
                        "probe.freeze_matching in YAML.")
    p.add_argument("--no-freeze-matching", dest="freeze_matching", action="store_false",
                   help="Disable matching freeze even if YAML enables it.")
    p.add_argument("--reset-temporal-state", dest="reset_temporal_state",
                   action="store_true", default=None,
                   help="Restore InstanceBank cache + run_step before each forward "
                        "in the probe cycle (default: follow probe.reset_temporal_state). "
                        "Eliminates cache drift between baseline / grad / stepped forwards.")
    p.add_argument("--no-reset-temporal-state", dest="reset_temporal_state",
                   action="store_false",
                   help="Disable temporal-state reset even if YAML enables it.")
    p.add_argument("--output-root", default=None,
                   help="Override `output_root` in YAML — useful for parking each "
                        "experiment in its own folder (e.g. exp_perlayer_fc). "
                        "Created if missing.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config) as f:
        cfg_ana = yaml.safe_load(f)

    set_seeds(cfg_ana["seed"], cfg_ana.get("deterministic", True))
    # `torch.use_deterministic_algorithms(True)` forces every CUDA op into a
    # deterministic kernel. In practice cuDNN conv often has no such kernel
    # at probe input shapes ("Unable to find a valid cuDNN algorithm") and
    # several map/motion ops also lack deterministic implementations. Skip
    # it; same-seed reproducibility is already covered by `set_seeds` above
    # and the adapter's per-forward seed pinning inside the probe cycle.
    if cfg_ana.get("strict_deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    # Now import runtime (deferred so --help works without a model env)
    rt = _import_runtime()
    adapter = build_adapter(
        rt,
        cfg_ana,
        args.adapter,
        vad_repo_root=args.vad_repo_root,
        adapter_config=args.adapter_config,
    )
    analysis_tasks = list(adapter.tasks)

    modules = ALL_MODULES.split(",") if args.all else args.modules.split(",")
    out_root = Path(args.output_root or cfg_ana["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[output] writing results under {out_root.resolve()}")

    # Resolve probe variance-control options (CLI overrides YAML).
    probe_cfg = cfg_ana.get("probe", {})
    if args.probe_layers is not None:
        target_layers = [s for s in args.probe_layers.split(",") if s]
    else:
        target_layers = probe_cfg.get("target_layers") or None
    layer_cfg = cfg_ana.get("layer_conflict", {})
    if target_layers is None and layer_cfg.get("enabled", False):
        target_layers = layer_cfg.get("layers") or None
    if args.freeze_matching is None:
        freeze_matching = bool(probe_cfg.get("freeze_matching", False))
    else:
        freeze_matching = bool(args.freeze_matching)
    if args.reset_temporal_state is None:
        yaml_reset = probe_cfg.get("reset_temporal_state", None)
        # null/missing in YAML → mirror freeze_matching; explicit true/false → use it.
        reset_temporal_state = freeze_matching if yaml_reset is None else bool(yaml_reset)
    else:
        reset_temporal_state = bool(args.reset_temporal_state)

    tags = list(cfg_ana["checkpoints"].keys())
    if args.checkpoints:
        tags = args.checkpoints.split(",")

    per_ckpt_dirs: Dict[str, Path] = {}
    for idx, tag in enumerate(tags, start=1):
        ckpt_path = _resolve_ckpt_path(cfg_ana, tag)
        out_dir = out_root / f"ckpt_{tag}"
        per_ckpt_dirs[tag] = out_dir
        print(f"[{tag}] running modules {modules} → {out_dir} "
              f"(layers={target_layers or 'ALL'}, freeze_matching={freeze_matching}, "
              f"reset_temporal_state={reset_temporal_state})")
        run_primary_for_checkpoint(
            rt, cfg_ana, adapter, tag, ckpt_path, out_dir, modules, smoke=args.smoke,
            target_layers=target_layers, freeze_matching=freeze_matching,
            reset_temporal_state=reset_temporal_state,
            ckpt_order=idx,
        )
        # The function frame just exited, so its local model/collector/cached
        # are now unreferenced. Force a gc + cuda cache release before the
        # next ckpt builds a fresh model.
        _release_gpu_resources(label=f"after_{tag}")

    # M_Q post-processing — merge per-ckpt qs.csv files into a single root CSV
    # consumed by the notebook's Part J cell (gradient_analysis_results/plan_centric/query_sensitivity.csv).
    # Incremental: 이번 실행의 ckpt 행만 replace, 이전에 쌓인 다른 ckpt 행은 보존
    # → 분할 실행 (예: 1ep,3ep 한번 / 6ep,18ep 한번) 시에도 root csv 누적.
    if "M_Q" in modules or "query_sensitivity" in modules:
        parts = []
        for tag, d in per_ckpt_dirs.items():
            qs_csv = Path(d) / "query_sensitivity" / "qs.csv"
            if qs_csv.exists():
                parts.append(pd.read_csv(qs_csv))
        if parts:
            merged_this_run = pd.concat(parts, ignore_index=True)
            root_qs = out_root / "plan_centric" / "query_sensitivity.csv"
            root_qs.parent.mkdir(parents=True, exist_ok=True)
            if root_qs.exists():
                existing = pd.read_csv(root_qs)
                ckpts_this_run = set(merged_this_run["checkpoint"].unique())
                keep = existing[~existing["checkpoint"].isin(ckpts_this_run)]
                merged = pd.concat([keep, merged_this_run], ignore_index=True)
            else:
                merged = merged_this_run
            merged.to_csv(root_qs, index=False)
            n_ckpts = merged["checkpoint"].nunique() if "checkpoint" in merged.columns else 0
            print(f"[M_Q] root csv now has {len(merged)} rows across {n_ckpts} ckpt(s) -> {root_qs}")
        else:
            print("[M_Q] no qs.csv files found to merge")

    if "M6" in modules and len(per_ckpt_dirs) >= 2:
        print(f"[M6] aggregating dynamics across {list(per_ckpt_dirs.keys())}")
        rt["run_m6"](per_ckpt_dirs, analysis_tasks, out_root / "dynamics")

    # M_N10 — magnitude non-stationarity (Phase 1 #10). Aggregates across
    # checkpoints that already have M5 outputs.
    if "M_N10" in modules or "magnitude_dynamics" in modules:
        rows: List[Dict] = []
        for tag, d in per_ckpt_dirs.items():
            try:
                ep = float(tag.rstrip("epoch").rstrip("ep"))
            except ValueError:
                print(f"[M_N10] skipping checkpoint tag {tag} — cannot parse epoch")
                continue
            nf = Path(d) / "gradnorm" / "per_task_norm.csv"
            if not nf.exists():
                continue
            ndf = pd.read_csv(nf)
            means = ndf.groupby("task")["norm"].mean()
            for t in analysis_tasks:
                if t in means.index:
                    rows.append({"epoch": ep, "task": t, "mean_norm": float(means[t])})
        per_task_norm = pd.DataFrame(rows)
        if not per_task_norm.empty:
            print(f"[M_N10] computing magnitude dynamics across "
                  f"{sorted(per_task_norm['epoch'].unique())} epochs")
            rt["run_magnitude_dynamics"](
                per_task_norm,
                tasks=analysis_tasks,
                out_dir=out_root / "magnitude_dynamics",
            )

    if not args.no_supplementary and not args.smoke:
        sup_dir = out_root / "supplementary"
        print(f"[supplementary] batch_size=1 probe on {cfg_ana['supplementary']['checkpoint']} "
              f"(layers={target_layers or 'ALL'}, freeze_matching={freeze_matching}, "
              f"reset_temporal_state={reset_temporal_state})")
        run_supplementary(
            rt, cfg_ana, adapter, sup_dir,
            target_layers=target_layers, freeze_matching=freeze_matching,
            reset_temporal_state=reset_temporal_state,
        )
        _release_gpu_resources(label="after_supplementary")

    # Generate markdown summary
    from tools.gradient_analysis.summary import generate_summary
    generate_summary(out_root, tags)
    print(f"summary: {out_root / 'summary_report.md'}")

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
