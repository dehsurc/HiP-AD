"""HiP-AD adapter for the gradient-analysis pipeline.

Phase 2 migration: subsumes the Phase-1 inline dependencies
(`_import_hipad_utils`, `_selective_eval`, `FrozenMatching`,
`ModelStateSnapshot`, `compat` patches) behind the
`GradientAnalysisAdapter` Protocol so M1-M8 modules become model-agnostic.

Built up op-by-op alongside `vad.py` (T3-T12). HiP-AD-specific imports are
deferred (lazy) so the module is importable in environments without the
HiP-AD repo on `sys.path`.
"""
from __future__ import annotations

import contextlib
import sys
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base import TemporalSnapshot


HIPAD_DEFAULT_CONFIG = Path("ckpts/HiP-AD-Stage2_code.py")


def _ensure_hipad_paths() -> Path:
    """Make sure both the repo root and `tools/` are on sys.path so the
    HiP-AD plugin packages are importable. Returns the repo root."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # .../HiP-AD/
    tools_dir = repo_root / "tools"
    for p in (str(repo_root), str(tools_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return repo_root


class HipadAdapter:
    """Concrete adapter for HiP-AD-Stage2."""

    def __init__(self, config_path: Path = HIPAD_DEFAULT_CONFIG):
        self._repo_root = _ensure_hipad_paths()
        self._config_path = (self._repo_root / config_path
                             if not Path(config_path).is_absolute()
                             else Path(config_path))
        # Lazy: only loaded once a method that needs the config runs.
        self._cfg = None
        self._utils = None

    # -------------------------------------------------------------- ops

    @property
    def tasks(self) -> List[str]:
        utils = self._load_utils()
        return list(utils["TASK_GROUPS"].keys())

    def build_model(self, ckpt: Path, device: str) -> nn.Module:
        from mmcv import Config  # type: ignore
        from mmcv.runner import load_checkpoint  # type: ignore
        from mmdet.models import build_detector  # type: ignore

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        cfg = self._cfg
        model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"),
                                test_cfg=cfg.get("test_cfg"))
        load_checkpoint(model, str(ckpt), map_location="cpu")
        model.to(device)
        return model

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader:
        from mmcv import Config  # type: ignore
        from mmcv.parallel import collate  # type: ignore
        from projects.mmdet3d_plugin.datasets.builder import (  # type: ignore
            custom_build_dataset,
        )

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        ds = custom_build_dataset(self._cfg.data.train)
        g = torch.Generator()
        g.manual_seed(seed)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=min(4, batch_size),
            collate_fn=partial(collate, samples_per_gpu=batch_size),
            drop_last=True,
            generator=g,
        )

    # ------------------------------------------------------ HiP-AD utils

    def _load_utils(self) -> Dict[str, Any]:
        """Lazy-load HiP-AD's gradient-analysis utility surface.

        Mirrors the old `_import_hipad_utils` in collector.py — kept here
        instead of in `tools/` because every method below depends on it.
        """
        if self._utils is not None:
            return self._utils
        _ensure_hipad_paths()
        from analyze_gradient_conflict import (  # type: ignore
            TASK_GROUPS,
            _sum_task_loss,
            compute_task_gradient_grouped,
        )
        from projects.mmdet3d_plugin.core.hooks.pcgrad_optimizer_hook import (  # type: ignore
            get_shared_parameters_grouped,
        )
        self._utils = {
            "TASK_GROUPS": TASK_GROUPS,
            "_sum_task_loss": _sum_task_loss,
            "compute_task_gradient_grouped": compute_task_gradient_grouped,
            "get_shared_parameters_grouped": get_shared_parameters_grouped,
        }
        return self._utils
