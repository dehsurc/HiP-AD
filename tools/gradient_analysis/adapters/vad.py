"""VAD adapter for the gradient-analysis pipeline.

VAD-specific concerns vs HiP-AD:
  * Tasks: det / map / motion / plan (motion is "trajectory" in VAD's
    head — `loss_traj`/`loss_traj_cls`).
  * Loss-key conventions:
      det:    loss_cls, loss_bbox, d{i}.loss_cls, d{i}.loss_bbox
      map:    loss_map_cls, loss_map_bbox, loss_map_iou, loss_map_pts,
              loss_map_dir, d{i}.loss_map_*
      motion: loss_traj, loss_traj_cls, d{i}.loss_traj, d{i}.loss_traj_cls
      plan:   loss_plan_reg, loss_plan_bound, loss_plan_col, loss_plan_dir
  * Shared parameter groups: BEVFormer encoder layers (TemporalSelfAttention,
    SpatialCrossAttention, FFN, LayerNorm) and decoder layers — translated
    from a canonical group-name set into the actual nested attribute paths.
  * Stochastic-state freeze: VAD uses HungarianAssigner3D + PseudoSampler
    for det / map; `freeze_stochastic_state` patches their assign methods.
    motion / plan in VAD are mode-argmin only (no Hungarian); the freeze
    surface there is the assign-time argmin in the head.
  * Temporal state: VAD's training-mode `prev_bev` is recomputed per batch
    by `obtain_history_bev` and not cached across forwards, so the
    snapshot is largely defensive (run_step counter + the `prev_frame_info`
    dict on the detector if `video_test_mode` is on).
  * Model construction: VAD's bundled `init_detector` is inference-oriented;
    training-mode gradient analysis builds with `mmdet3d.models.build_model`
    and then loads the checkpoint.
"""
from __future__ import annotations

import contextlib
import importlib
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


VAD_DEFAULT_REPO = Path("/home/yongjae/e2e/VAD")
VAD_DEFAULT_CONFIG = VAD_DEFAULT_REPO / "data/ckpts/VAD_tiny_e2e.py"


def _ensure_vad_paths(repo_root: Path) -> None:
    """Push the VAD repo on sys.path and import its plugin so the
    `projects.mmdet3d_plugin.*` registrations fire."""
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    importlib.import_module("projects.mmdet3d_plugin")


def _repo_abs(repo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _absolutize_data_paths(node: Any, repo_root: Path) -> None:
    if isinstance(node, list):
        for item in node:
            _absolutize_data_paths(item, repo_root)
        return
    if not hasattr(node, "items"):
        return
    for key, value in list(node.items()):
        if key in {"data_root", "ann_file"} and isinstance(value, str):
            node[key] = _repo_abs(repo_root, value)
        else:
            _absolutize_data_paths(value, repo_root)


_TASKS = ["det", "map", "motion", "plan"]

# Loss-key prefixes per task. A key counts for a task if any prefix is a
# substring of `key` (handling decoder-layer prefixes like `d3.loss_cls`).
_VAD_TASK_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "det": ("loss_cls", "loss_bbox"),
    "map": (
        "loss_map_cls", "loss_map_bbox", "loss_map_iou",
        "loss_map_pts", "loss_map_dir",
    ),
    "motion": ("loss_traj",),
    "plan": (
        "loss_plan_reg", "loss_plan_bound",
        "loss_plan_col", "loss_plan_dir",
    ),
}


_VAD_DETECTOR_ATTRS: Tuple[str, ...] = ("prev_frame_info",)
_VAD_HEAD_ATTRS: Tuple[str, ...] = ("epoch",)


def _vad_deepclone(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _vad_deepclone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_vad_deepclone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_vad_deepclone(v) for v in x)
    return x


def _key_belongs_to_task(key: str, task: str) -> bool:
    """True iff the loss-dict key belongs to `task`.

    The check is on the suffix after any `dN.` prefix so decoder-layer
    losses like `d3.loss_map_cls` route to map, not det. det's
    `loss_cls`/`loss_bbox` must not match `loss_map_cls`/`loss_map_bbox`,
    so we test exact-prefix on the suffix.
    """
    suffix = key.split(".", 1)[-1] if key.startswith("d") and "." in key else key
    prefixes = _VAD_TASK_PREFIXES[task]
    if task == "det":
        return suffix.startswith("loss_cls") or suffix.startswith("loss_bbox")
    return any(suffix.startswith(p) for p in prefixes)


def _vad_per_forward_seed(seed: int = 0) -> None:
    import random as _random
    import numpy as _np

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _np.random.seed(seed)
    _random.seed(seed)


_ENC_PREFIX = "transformer.encoder.layers"
_DEC_PREFIX = "transformer.decoder.layers"
_MAP_DEC_PREFIX = "transformer.map_decoder.layers"


def _gather_params(
    head: nn.Module,
    layer_prefix: str,
    layer_idx: int,
    keep,
) -> List[nn.Parameter]:
    """Walk `head.<layer_prefix>.<layer_idx>` and return matching params."""
    base = head
    for part in layer_prefix.split("."):
        base = getattr(base, part)
    layer = base[layer_idx]
    out: List[nn.Parameter] = []
    seen_ids = set()
    for name, sub in layer.named_modules():
        if not keep(name, sub):
            continue
        for p in sub.parameters(recurse=True):
            if p.requires_grad and id(p) not in seen_ids:
                out.append(p)
                seen_ids.add(id(p))
    return out


def _module_params(module: nn.Module) -> List[nn.Parameter]:
    return [p for p in module.parameters(recurse=True) if p.requires_grad]


def _parse_idx_role(name: str, prefix: str) -> Tuple[int, str]:
    """`enc3_temporal_self_attention` -> (3, 'temporal_self_attention')."""
    rest = name[len(prefix):]
    n_str, _, role = rest.partition("_")
    return int(n_str), role


def _resolve_block(
    head: nn.Module, prefix: str, layer_idx: int, role: str,
) -> List[nn.Parameter]:
    if role == "temporal_self_attention":
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: n.startswith("attentions.0"),
        )
    if role == "spatial_cross_attention":
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: n.startswith("attentions.1"),
        )
    if role == "self_attn":
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: n.startswith("attentions.0"),
        )
    if role == "cross_attn":
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: n.startswith("attentions.1"),
        )
    if role == "ffn":
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: n.startswith("ffns."),
        )
    if role.startswith("norm"):
        idx = role.split("_", 1)[1] if "_" in role else ""
        return _gather_params(
            head, prefix, layer_idx,
            lambda n, m: isinstance(m, nn.LayerNorm)
            and (idx == "" or n == f"norms.{idx}"),
        )
    raise KeyError(f"unknown role for VAD block: {role}")


def _resolve_vad_group(head: nn.Module, name: str) -> List[nn.Parameter]:
    """Resolve a canonical VAD group name to its parameter list."""
    if name == "bev_embedding":
        return _module_params(head.bev_embedding)
    if name == "query_embedding":
        return _module_params(head.query_embedding)
    if name == "reference_points":
        return _module_params(head.transformer.reference_points)
    if name == "map_reference_points":
        return _module_params(head.transformer.map_reference_points)
    if name == "det_head_last":
        return _module_params(head.cls_branches[-1]) + _module_params(head.reg_branches[-1])
    if name == "map_head_last":
        return (
            _module_params(head.map_cls_branches[-1])
            + _module_params(head.map_reg_branches[-1])
        )
    if name == "motion_head_last":
        return _module_params(head.traj_branches[-1]) + _module_params(head.traj_cls_branches[-1])
    if name == "plan_head_last":
        return _module_params(head.ego_fut_decoder)
    if name.startswith("map_dec"):
        n_layer, role = _parse_idx_role(name, "map_dec")
        return _resolve_block(head, _MAP_DEC_PREFIX, n_layer, role)
    if name.startswith("enc"):
        n_layer, role = _parse_idx_role(name, "enc")
        return _resolve_block(head, _ENC_PREFIX, n_layer, role)
    if name.startswith("dec"):
        n_layer, role = _parse_idx_role(name, "dec")
        return _resolve_block(head, _DEC_PREFIX, n_layer, role)
    raise KeyError(f"unknown VAD group name: {name}")


VAD_DEFAULT_GROUP_NAMES: List[str] = [
    "bev_embedding",
    *[f"enc{i}_temporal_self_attention" for i in range(3)],
    *[f"enc{i}_spatial_cross_attention" for i in range(3)],
    *[f"enc{i}_ffn" for i in range(3)],
    *[f"dec{i}_self_attn" for i in range(6)],
    *[f"dec{i}_cross_attn" for i in range(6)],
    *[f"dec{i}_ffn" for i in range(6)],
    *[f"map_dec{i}_self_attn" for i in range(6)],
    *[f"map_dec{i}_cross_attn" for i in range(6)],
    *[f"map_dec{i}_ffn" for i in range(6)],
    "det_head_last",
    "map_head_last",
    "motion_head_last",
    "plan_head_last",
]


class VadAdapter:
    """Concrete adapter for VAD-tiny / VAD-base."""

    def __init__(
        self,
        repo_root: Path = VAD_DEFAULT_REPO,
        config_path: Path = VAD_DEFAULT_CONFIG,
    ):
        self._repo_root = Path(repo_root)
        self._config_path = (
            self._repo_root / config_path
            if not Path(config_path).is_absolute()
            else Path(config_path)
        )
        self._cfg = None
        self._freeze_session: Optional[Any] = None
        self._vad_paths_loaded = False

    def _ensure_vad(self) -> None:
        """Lazily push the VAD repo onto sys.path and import its plugin."""
        if not self._vad_paths_loaded:
            _ensure_vad_paths(self._repo_root)
            self._vad_paths_loaded = True

    @property
    def tasks(self) -> List[str]:
        return list(_TASKS)

    def build_model(self, ckpt: Path, device: str) -> nn.Module:
        self._ensure_vad()
        from mmcv import Config  # type: ignore
        from mmcv.runner import load_checkpoint  # type: ignore
        from mmdet3d.models import build_model  # type: ignore

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
            _absolutize_data_paths(self._cfg.data, self._repo_root)
        cfg = self._cfg
        model = build_model(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
        load_checkpoint(model, str(ckpt), map_location="cpu")
        model.to(device)
        return model

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader:
        self._ensure_vad()
        from mmcv import Config  # type: ignore
        from mmcv.parallel import collate  # type: ignore
        from mmdet3d.datasets import build_dataset  # type: ignore

        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
            _absolutize_data_paths(self._cfg.data, self._repo_root)
        ds = build_dataset(self._cfg.data.train)
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

    @staticmethod
    def _stochastic_base_types() -> Tuple[Type[nn.Module], ...]:
        return (
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]:
        types: Tuple[Type[nn.Module], ...] = self._stochastic_base_types()
        # GridMask is the only VAD-specific stochastic source (random mask
        # placement on input images during training); it lives on the
        # detector as `self.grid_mask`. It is NOT an nn.Module subclass
        # whose state we can flip via .eval(); the safer freeze surface is
        # `model.use_grid_mask = False` — handled explicitly inside
        # `_selective_eval` below.
        return types

    @contextlib.contextmanager
    def _selective_eval(self, model: nn.Module) -> Iterator[None]:
        target_types = self.selective_eval_types()
        switched: List[nn.Module] = []
        raw = model.module if hasattr(model, "module") else model
        prior_grid = getattr(raw, "use_grid_mask", None)
        original_obtain_history_bev = getattr(raw, "obtain_history_bev", None)

        def apply_eval() -> None:
            for m in model.modules():
                if isinstance(m, target_types) and m.training:
                    m.eval()
                    if m not in switched:
                        switched.append(m)
            if prior_grid is True:
                raw.use_grid_mask = False

        def patched_obtain_history_bev(*args, **kwargs):
            out = original_obtain_history_bev(*args, **kwargs)
            apply_eval()
            return out

        apply_eval()
        if original_obtain_history_bev is not None:
            setattr(raw, "obtain_history_bev", patched_obtain_history_bev)
        try:
            yield
        finally:
            if original_obtain_history_bev is not None:
                setattr(raw, "obtain_history_bev", original_obtain_history_bev)
            for m in switched:
                m.train()
            if prior_grid is True:
                raw.use_grid_mask = True

    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        from mmcv.parallel import scatter  # type: ignore

        raw = model.module if hasattr(model, "module") else model
        device = next(raw.parameters()).device
        device_id = device.index if device.type == "cuda" and device.index is not None else 0
        scattered = scatter(data, [device_id])[0]
        model.train()
        if self._freeze_session is not None:
            self._freeze_session.next_forward()
            _vad_per_forward_seed(0)
        with self._selective_eval(raw):
            losses = model(**scattered)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]
        return losses

    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]:
        if task not in _VAD_TASK_PREFIXES:
            return None
        parts: List[torch.Tensor] = []
        for k, v in loss_dict.items():
            if not torch.is_tensor(v):
                continue
            if _key_belongs_to_task(k, task):
                parts.append(v if v.ndim == 0 else v.sum())
        if not parts:
            return None
        return torch.stack(parts).sum()

    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        """Freeze VAD assignment and RNG state within a diagnostic scope."""
        import random as _random
        import numpy as _np

        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        np_state = _np.random.get_state()
        py_state = _random.getstate()

        _vad_per_forward_seed(0)
        try:
            with _VadFrozenMatchingSession() as session:
                self._freeze_session = session
                yield
        finally:
            self._freeze_session = None
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            _np.random.set_state(np_state)
            _random.setstate(py_state)

    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot:
        raw = model.module if hasattr(model, "module") else model
        payload: Dict[Any, Any] = {}
        for attr in _VAD_DETECTOR_ATTRS:
            if hasattr(raw, attr):
                payload[("detector", attr)] = _vad_deepclone(getattr(raw, attr))
        head = getattr(raw, "pts_bbox_head", None)
        if head is not None:
            for attr in _VAD_HEAD_ATTRS:
                if hasattr(head, attr):
                    payload[("head", attr)] = _vad_deepclone(getattr(head, attr))
        return TemporalSnapshot(
            model_state_keys=_VAD_DETECTOR_ATTRS + _VAD_HEAD_ATTRS,
            payload=payload,
        )

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None:
        raw = model.module if hasattr(model, "module") else model
        head = getattr(raw, "pts_bbox_head", None)
        for key, val in snapshot.payload.items():
            if not isinstance(key, tuple):
                continue
            scope, attr = key
            target = raw if scope == "detector" else head
            if target is not None and hasattr(target, attr):
                setattr(target, attr, _vad_deepclone(val))

    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]:
        raw = model.module if hasattr(model, "module") else model
        head = raw.pts_bbox_head
        out: Dict[str, List[nn.Parameter]] = {}
        for gk in group_names:
            out[gk] = _resolve_vad_group(head, gk)
        return out

    def get_task_queries(self, model, fwd_artifacts) -> "dict":
        raise NotImplementedError(
            "VadAdapter.get_task_queries is not implemented; "
            "query sensitivity analysis runs HiP-AD only."
        )


class _VadFrozenMatchingSession:
    """Patch VAD Hungarian assigners to replay first-forward assignments."""

    def __init__(self) -> None:
        self._queues: Dict[int, List[Any]] = {}
        self._play_idx: Dict[int, int] = {}
        self._patches: List[Tuple[Any, str, Any]] = []

    def _take(self, sid: int) -> Optional[Any]:
        idx = self._play_idx.get(sid, 0)
        queue = self._queues.get(sid, [])
        if idx < len(queue):
            self._play_idx[sid] = idx + 1
            return queue[idx]
        return None

    def _record(self, sid: int, value: Any) -> None:
        self._queues.setdefault(sid, []).append(value)
        self._play_idx[sid] = len(self._queues[sid])

    def next_forward(self) -> None:
        self._play_idx = {sid: 0 for sid in self._queues}

    def __enter__(self) -> "_VadFrozenMatchingSession":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()
        self._queues.clear()
        self._play_idx.clear()

    def _install_patches(self) -> None:
        from projects.mmdet3d_plugin.core.bbox.assigners.hungarian_assigner_3d import (  # type: ignore
            HungarianAssigner3D,
        )
        from projects.mmdet3d_plugin.core.bbox.assigners.map_hungarian_assigner_3d import (  # type: ignore
            MapHungarianAssigner3D,
        )

        session = self

        def _make(cls):
            orig = cls.assign

            def patched(assigner, *args, **kwargs):
                sid = id(assigner)
                cached = session._take(sid)
                if cached is not None:
                    return cached
                out = orig(assigner, *args, **kwargs)
                session._record(sid, out)
                return out

            return orig, patched

        for cls in (HungarianAssigner3D, MapHungarianAssigner3D):
            orig, patched = _make(cls)
            self._patches.append((cls, "assign", orig))
            setattr(cls, "assign", patched)

    def _uninstall_patches(self) -> None:
        for cls, attr, orig in self._patches:
            setattr(cls, attr, orig)
        self._patches = []
