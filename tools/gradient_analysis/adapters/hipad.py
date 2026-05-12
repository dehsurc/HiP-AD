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
import importlib
import sys
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import torch
import torch.utils.checkpoint as cp
from torch import nn
from torch.utils.data import DataLoader

from .base import TemporalSnapshot


HIPAD_DEFAULT_CONFIG = Path("ckpts/E2_E1_stage2_18ep/E2_E1_stage2_18ep.py")
_INDEX_PUT_PATCHED = False
_CHECKPOINT_PATCHED = False


def _apply_use_reentrant_false() -> None:
    """Keep activation checkpointing compatible with autograd.grad(inputs=...)."""
    global _CHECKPOINT_PATCHED
    if _CHECKPOINT_PATCHED:
        return
    orig = cp.checkpoint

    def patched(function, *args, use_reentrant=False, **kwargs):
        return orig(function, *args, use_reentrant=False, **kwargs)

    cp.checkpoint = patched
    torch.utils.checkpoint.checkpoint = patched
    _CHECKPOINT_PATCHED = True


def _apply_index_put_fix() -> None:
    """Patch HiP-AD map target sampling for scalar index_put CUDA builds."""
    global _INDEX_PUT_PATCHED
    if _INDEX_PUT_PATCHED:
        return

    from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget  # type: ignore

    def fixed_sample(self, cls_preds, pts_preds, cls_targets, pts_targets):
        pts_targets = [
            x.flatten(2, 3) if len(x.shape) == 4 else x for x in pts_targets
        ]
        indices = []
        for cls_pred, pts_pred, cls_target, pts_target in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            pts_pred = self.normalize_line(pts_pred)
            pts_target = self.normalize_line(pts_target)
            preds = dict(lines=pts_pred, scores=cls_pred)
            gts = dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)

        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = (
            cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        trailing = output_reg_weights.shape[2:]
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            n = pred_idx.shape[0] if torch.is_tensor(pred_idx) else len(pred_idx)
            output_reg_weights[i, pred_idx] = output_reg_weights.new_ones(
                (n,) + tuple(trailing)
            )

        return output_cls_target, output_box_target, output_reg_weights

    SparsePoint3DTarget.sample = fixed_sample
    _INDEX_PUT_PATCHED = True


def _ensure_hipad_paths() -> Path:
    """Make sure both the repo root and `tools/` are on sys.path so the
    HiP-AD plugin packages are importable. Returns the repo root."""
    here = Path(__file__).resolve()
    repo_root = here.parents[3]  # .../HiP-AD/
    tools_dir = repo_root / "tools"
    for p in (str(repo_root), str(tools_dir)):
        if p not in sys.path:
            sys.path.insert(0, p)
    importlib.import_module("projects.mmdet3d_plugin")
    _apply_use_reentrant_false()
    _apply_index_put_fix()
    return repo_root


class HipadAdapter:
    """Concrete adapter for HiP-AD-Stage2."""

    def __init__(
        self,
        config_path: Path = HIPAD_DEFAULT_CONFIG,
        task_names: Optional[List[str]] = None,
    ):
        self._repo_root = _ensure_hipad_paths()
        self._config_path = (self._repo_root / config_path
                             if not Path(config_path).is_absolute()
                             else Path(config_path))
        self._task_names = list(task_names) if task_names is not None else None
        # Lazy: only loaded once a method that needs the config runs.
        self._cfg = None
        self._utils = None
        self._freeze_session = None
        self._freeze_snapshot = None

    # -------------------------------------------------------------- ops

    @property
    def tasks(self) -> List[str]:
        from mmcv import Config  # type: ignore

        utils = self._load_utils()
        if self._task_names is not None:
            return [t for t in self._task_names if t in utils["TASK_GROUPS"]]
        if self._cfg is None:
            self._cfg = Config.fromfile(str(self._config_path))
        configured = [
            t for t in getattr(self._cfg, "task_select", [])
            if t in utils["TASK_GROUPS"]
        ]
        return configured or list(utils["TASK_GROUPS"].keys())

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

    # Stochastic / running-stat layers that MUST be in eval mode during the
    # probe. Matches the old `_STOCHASTIC_TYPES` from collector.py and adds
    # HiP-AD's `DeformableFeatureAggregation` lazily.
    @staticmethod
    def _stochastic_base_types() -> Tuple[Type[nn.Module], ...]:
        return (
            nn.Dropout, nn.Dropout2d, nn.Dropout3d,
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            nn.SyncBatchNorm,
        )

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]:
        types: Tuple[Type[nn.Module], ...] = self._stochastic_base_types()
        try:
            from projects.mmdet3d_plugin.models.blocks import (  # type: ignore
                DeformableFeatureAggregation,
            )
            types = types + (DeformableFeatureAggregation,)
        except Exception:
            pass
        return types

    @contextlib.contextmanager
    def _selective_eval(self, model: nn.Module) -> Iterator[None]:
        target_types = self.selective_eval_types()
        switched: List[nn.Module] = []
        for m in model.modules():
            if isinstance(m, target_types) and m.training:
                m.eval()
                switched.append(m)
        try:
            yield
        finally:
            for m in switched:
                m.train()

    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """Run a single training-mode forward and return the loss dict.

        Wraps the call in `_selective_eval` so BN / Dropout /
        DeformableFeatureAggregation are in eval for this forward — without
        this, the probe's baseline / grad / stepped forwards see a drifting
        model (the bug Phase 1's `_STOCHASTIC_TYPES` first fixed).
        """
        raw = model.module if hasattr(model, "module") else model
        model.train()
        if hasattr(model, "module"):
            batch = data
        else:
            from mmcv.parallel import scatter  # type: ignore

            device = next(raw.parameters()).device
            device_id = device.index if device.type == "cuda" else -1
            batch = scatter(data, [device_id])[0] if device_id >= 0 else data
        freeze_session = self._freeze_session
        seed_ctx = _per_forward_seed(42) if freeze_session is not None else contextlib.nullcontext()
        if freeze_session is not None:
            if self._freeze_snapshot is None:
                self._freeze_snapshot = _ModelStateSnapshot(model)
            self._freeze_snapshot.restore()
            freeze_session.next_forward()
        with seed_ctx:
            with self._selective_eval(raw):
                losses = model(**batch)
        if isinstance(losses, (list, tuple)):
            losses = losses[0]  # MMDataParallel returns list
        return losses

    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]:
        """Sum every loss key whose name starts with one of the prefixes
        registered in `TASK_GROUPS[task]`. Returns None if no key matched
        (task absent in this forward — caller treats as NaN)."""
        utils = self._load_utils()
        prefixes = utils["TASK_GROUPS"].get(task, [])
        task_loss = None
        for key, val in loss_dict.items():
            if (
                any(key.startswith(prefix) for prefix in prefixes)
                and isinstance(val, torch.Tensor)
            ):
                task_loss = val if task_loss is None else task_loss + val
        return task_loss

    @staticmethod
    def _shared_request_primitives(group_names: List[str]) -> List[str]:
        primitives = set()
        known = {
            "backbone", "neck", "norm", "ffn", "gnn", "inter_gnn",
            "temp_gnn", "deformable", "fc_before", "fc_after",
        }
        for name in group_names:
            if name in known:
                primitives.add(name)
            elif name.startswith("backbone_"):
                primitives.add("backbone")
            elif name.startswith("dec"):
                rest = name.split("_", 1)[1] if "_" in name else ""
                if rest.startswith("inter_gnn"):
                    primitives.add("inter_gnn")
                elif rest.startswith("temp_gnn"):
                    primitives.add("temp_gnn")
                else:
                    primitives.add(rest.split("_", 1)[0])
        return sorted(primitives)

    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]:
        """Translate group names (e.g. `dec3_ffn_0_fc1`, `backbone_layer1`)
        into actual nn.Parameter lists on the model.

        Delegates to HiP-AD's `get_shared_parameters_grouped` (registered
        in `pcgrad_optimizer_hook`); the returned dict is the same shape
        the existing `GradientCollector` consumed before Phase 2.
        """
        utils = self._load_utils()
        requested = set(group_names)
        primitives = self._shared_request_primitives(group_names)
        groups, _ids = utils["get_shared_parameters_grouped"](model, primitives)
        exact = {k: v for k, v in groups.items() if k in requested}
        if exact:
            return exact
        return groups

    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        """Pin Hungarian matching (det/map) and mode-argmin (motion/plan)
        to the result of the FIRST forward inside the scope. Subsequent
        forwards within the same scope replay cached assignments so ΔL is
        purely the weight-update effect.

        Also swaps CUDA `cumsum` for a CPU-offloaded version. CUDA cumsum is
        non-deterministic (PyTorch known limitation) and motion/plan loss
        computation cum-sums trajectory positions, so without this patch the
        loss has a per-call noise floor that overwhelms the descent signal.
        """
        prior = self._freeze_session
        prior_snapshot = self._freeze_snapshot
        with _FrozenMatchingSession() as fm, _deterministic_cumsum():
            self._freeze_session = fm
            self._freeze_snapshot = None
            try:
                yield
            finally:
                self._freeze_session = prior
                self._freeze_snapshot = prior_snapshot

    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot:
        snap = _ModelStateSnapshot(model)
        return TemporalSnapshot(
            model_state_keys=(
                tuple(_BANK_MUTABLE_ATTRS)
                + tuple(_SAMPLER_MUTABLE_ATTRS)
                + ("run_step",)
            ),
            payload={"_snap": snap},
        )

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None:
        snap = snapshot.payload.get("_snap")
        if snap is None:
            return
        snap.restore()

    def get_task_queries(self, model, fwd_artifacts) -> "dict":
        """Return HiP-AD's per-task query tensors used inside the planner.

        The runner is responsible for stashing the graph-attached query
        tensors into ``fwd_artifacts['task_queries']`` during the forward
        pass. This adapter method just retrieves them. Concrete attribute
        paths into the HiP-AD head depend on the model layout and are
        captured by the runner, not here.
        """
        if fwd_artifacts is None or "task_queries" not in fwd_artifacts:
            raise NotImplementedError(
                "HipadAdapter.get_task_queries requires fwd_artifacts['task_queries']"
                " to be populated by the runner."
            )
        return dict(fwd_artifacts["task_queries"])

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
        task_groups = {k: list(v) for k, v in TASK_GROUPS.items()}
        task_groups.setdefault("det", []).extend(["loss_det_cls", "loss_det_reg"])
        task_groups.setdefault("map", []).extend(["loss_map_cls", "loss_map_reg"])
        task_groups.setdefault("motion", []).extend(["loss_motion_cls", "loss_motion_reg"])
        task_groups.setdefault("ego", []).extend([
            "ego_loss_cls", "ego_loss_reg", "ego_loss_status", "loss_ego_status",
        ])
        task_groups.setdefault("plan", []).extend(["loss_plan_cls", "loss_plan_reg"])
        self._utils = {
            "TASK_GROUPS": task_groups,
            "_sum_task_loss": _sum_task_loss,
            "compute_task_gradient_grouped": compute_task_gradient_grouped,
            "get_shared_parameters_grouped": get_shared_parameters_grouped,
        }
        return self._utils


# ===========================================================================
# Temporal-state machinery (formerly tools/gradient_analysis/temporal_state.py)
# ===========================================================================


_BANK_MUTABLE_ATTRS: Tuple[str, ...] = (
    "cached_feature",
    "cached_anchor",
    "metas",
    "mask",
    "confidence",
    "temp_confidence",
    "instance_id",
    "prev_id",
)

_BANK_NAMES: Tuple[str, ...] = ("det", "map", "ego", "plan", "scenes")

_SAMPLER_MUTABLE_ATTRS: Tuple[str, ...] = (
    "dn_metas",
    "indices",
)


def _deepclone(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _deepclone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deepclone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_deepclone(v) for v in x)
    return x


def _capture_rng_state() -> Dict[str, Any]:
    import random as _random

    import numpy as _np

    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": _np.random.get_state(),
        "python": _random.getstate(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import random as _random

    import numpy as _np

    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    _np.random.set_state(state["numpy"])
    _random.setstate(state["python"])


def _raw_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _bank_holders(model: nn.Module) -> List[nn.Module]:
    try:
        from projects.mmdet3d_plugin.models.instance_bank import (  # type: ignore
            InstanceBank,
        )
    except Exception:
        InstanceBank = None  # type: ignore

    raw = _raw_model(model)
    holders: List[nn.Module] = []
    seen: List[int] = []

    def _try_add(obj: Any) -> None:
        if obj is None:
            return
        is_bank = (InstanceBank is not None and isinstance(obj, InstanceBank))
        if not is_bank and "InstanceBank" not in type(obj).__name__:
            return
        if id(obj) in seen:
            return
        holders.append(obj)
        seen.append(id(obj))

    for _name, module in raw.named_modules():
        for short in _BANK_NAMES:
            _try_add(getattr(module, f"{short}_instance_bank", None))
            lst = getattr(module, f"{short}_instance_bank_list", None)
            if isinstance(lst, (list, tuple)):
                for b in lst:
                    _try_add(b)
        _try_add(getattr(module, "instance_bank", None))

    if InstanceBank is not None:
        for _name, module in raw.named_modules():
            if isinstance(module, InstanceBank):
                _try_add(module)
    return holders


def _sampler_holders(model: nn.Module) -> List[Any]:
    try:
        from projects.mmdet3d_plugin.models.base_target import (  # type: ignore
            BaseTargetWithDenoising,
        )
    except Exception:
        return []

    raw = _raw_model(model)
    holders: List[Any] = []
    seen: List[int] = []

    def _try_add(obj: Any) -> None:
        if isinstance(obj, BaseTargetWithDenoising) and id(obj) not in seen:
            holders.append(obj)
            seen.append(id(obj))

    for module in raw.modules():
        for attr in module.__dict__:
            if "sampler" not in attr:
                continue
            try:
                val = getattr(module, attr)
            except AttributeError:
                continue
            _try_add(val)
            if isinstance(val, (list, tuple)):
                for item in val:
                    _try_add(item)
    return holders


class _ModelStateSnapshot:
    """Captures run_step, InstanceBank caches, and sampler DN state."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._initial: Dict[Any, Any] = {}
        self._rng_state: Dict[str, Any] = {}
        self._capture()

    def _capture(self) -> None:
        raw = _raw_model(self.model)
        self._initial.clear()
        self._rng_state = _capture_rng_state()
        if hasattr(raw, "run_step"):
            self._initial["run_step"] = raw.run_step
        for bank in _bank_holders(self.model):
            for attr in _BANK_MUTABLE_ATTRS:
                if hasattr(bank, attr):
                    self._initial[(id(bank), attr)] = _deepclone(getattr(bank, attr))
        for sampler in _sampler_holders(self.model):
            for attr in _SAMPLER_MUTABLE_ATTRS:
                if hasattr(sampler, attr):
                    self._initial[(id(sampler), attr)] = _deepclone(getattr(sampler, attr))

    def restore(self) -> None:
        raw = _raw_model(self.model)
        if "run_step" in self._initial:
            raw.run_step = self._initial["run_step"]
        for bank in _bank_holders(self.model):
            for attr in _BANK_MUTABLE_ATTRS:
                key = (id(bank), attr)
                if key in self._initial:
                    setattr(bank, attr, _deepclone(self._initial[key]))
        for sampler in _sampler_holders(self.model):
            for attr in _SAMPLER_MUTABLE_ATTRS:
                key = (id(sampler), attr)
                if key in self._initial:
                    setattr(sampler, attr, _deepclone(self._initial[key]))
        _restore_rng_state(self._rng_state)


# ===========================================================================
# Frozen-matching machinery (formerly tools/gradient_analysis/matching_freeze.py)
# Migrated under the adapter so the M3 probe never has to know about HiP-AD's
# Hungarian samplers directly.
# ===========================================================================


class _FrozenMatchingSession:
    """Stateful object holding the per-instance call queues + play indices."""

    def __init__(self) -> None:
        self._queues: Dict[int, List[Any]] = defaultdict(list)
        self._play_idx: Dict[int, int] = defaultdict(int)
        self._patches: List = []

    def _take(self, sid: int) -> Optional[Any]:
        idx = self._play_idx[sid]
        q = self._queues[sid]
        if idx < len(q):
            self._play_idx[sid] = idx + 1
            return q[idx]
        return None

    def _record(self, sid: int, value: Any) -> None:
        self._queues[sid].append(value)
        self._play_idx[sid] = len(self._queues[sid])

    def next_forward(self) -> None:
        self._play_idx = defaultdict(int)

    def reset(self) -> None:
        self._queues.clear()
        self._play_idx.clear()

    def __enter__(self) -> "_FrozenMatchingSession":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()
        self.reset()

    def _install_patches(self) -> None:
        from projects.mmdet3d_plugin.models.det.target import SparseBox3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.motion.target import SparseMotionTarget  # type: ignore
        from projects.mmdet3d_plugin.models.plan.target import (  # type: ignore
            SparsePlanTarget,
            AlignPlanTarget,
            PlanningTarget,
        )

        # SparseMotionTarget is intentionally NOT patched. Motion uses WTA
        # mode-argmin INSIDE the loss (envelope theorem), so freezing the mode
        # breaks the natural noise-canceling property: a small param step that
        # shifts reg_pred can leave the cached mode no longer optimal, and
        # cumsum amplifies the resulting per-timestep error into a large
        # positive bias that overwhelms the descent signal. Det matching
        # (motion_loss_cache['indices']) is still frozen via the det patch, so
        # reg_target / reg_weight remain deterministic; only mode selection
        # adapts.
        self._patches = [
            (SparseBox3DTarget, "sample", SparseBox3DTarget.sample,
             _make_det_patch(self, SparseBox3DTarget.sample)),
            (SparsePoint3DTarget, "sample", SparsePoint3DTarget.sample,
             _make_map_patch(self, SparsePoint3DTarget.sample)),
            (SparsePlanTarget, "sample", SparsePlanTarget.sample,
             _make_plan_patch(self, SparsePlanTarget.sample)),
            (AlignPlanTarget, "sample", AlignPlanTarget.sample,
             _make_align_plan_patch(self, AlignPlanTarget.sample)),
            (PlanningTarget, "sample", PlanningTarget.sample,
             _make_plan_patch(self, PlanningTarget.sample)),
        ]
        for cls, attr, _orig, patched in self._patches:
            setattr(cls, attr, patched)

    def _uninstall_patches(self) -> None:
        for cls, attr, orig, _patched in self._patches:
            setattr(cls, attr, orig)
        self._patches = []


def _make_det_patch(session: _FrozenMatchingSession, orig: Callable) -> Callable:
    """det: outputs are GT-derived and safe to replay verbatim."""

    def patched(self, cls_pred, box_pred, cls_target, box_target):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, cls_pred, box_pred, cls_target, box_target)
            cached_out = tuple(t.detach() if torch.is_tensor(t) else t for t in out)
            cached_indices = self.indices
            session._record(sid, ("det", cached_out, cached_indices))
            return out
        tag, cached_out, cached_indices = cached
        assert tag == "det", f"queue tag mismatch: expected det, got {tag}"
        self.indices = cached_indices
        return cached_out

    return patched


def _make_map_patch(session: _FrozenMatchingSession, orig: Callable) -> Callable:
    """map: outputs are GT-derived and safe to replay verbatim."""

    def patched(self, cls_preds, pts_preds, cls_targets, pts_targets):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, cls_preds, pts_preds, cls_targets, pts_targets)
            cached_out = tuple(t.detach() if torch.is_tensor(t) else t for t in out)
            session._record(sid, ("map", cached_out))
            return out
        tag, cached_out = cached
        assert tag == "map", f"queue tag mismatch: expected map, got {tag}"
        return cached_out

    return patched


def _make_motion_patch(session: _FrozenMatchingSession, orig: Callable) -> Callable:
    """motion: replay frozen mode indices and re-gather current predictions."""

    def patched(self, reg_pred, gt_reg_target, gt_reg_mask, motion_loss_cache):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, reg_pred, gt_reg_target, gt_reg_mask, motion_loss_cache)
            cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos = out
            session._record(sid, (
                "motion",
                cls_target.detach(),
                cls_weight.detach(),
                reg_target.detach(),
                reg_weight.detach(),
                num_pos.detach() if torch.is_tensor(num_pos) else num_pos,
            ))
            return out
        tag, cls_target, cls_weight, reg_target, reg_weight, num_pos = cached
        assert tag == "motion", f"queue tag mismatch: expected motion, got {tag}"
        bs, num_anchor, mode, ts, d = reg_pred.shape
        gather_idx = cls_target[..., None, None, None].expand(bs, num_anchor, 1, ts, d)
        best_reg = torch.gather(reg_pred, 2, gather_idx).squeeze(2)
        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos

    return patched


def _make_plan_patch(session: _FrozenMatchingSession, orig: Callable) -> Callable:
    """Plan targets use WTA; replay frozen mode indices and re-gather preds."""

    def patched(self, cls_pred, reg_pred, gt_reg_target, gt_reg_mask, data):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, cls_pred, reg_pred, gt_reg_target, gt_reg_mask, data)
            sliced_cls_pred, cls_target, cls_weight, best_reg, gt_tgt, gt_mask = out
            session._record(sid, (
                "plan",
                cls_target.detach(),
                cls_weight.detach(),
                gt_tgt.detach(),
                gt_mask.detach(),
            ))
            return out
        tag, cls_target, cls_weight, gt_tgt, gt_mask = cached
        assert tag == "plan", f"queue tag mismatch: expected plan, got {tag}"
        sliced_cls_pred, best_reg = _replan_gather(self, cls_pred, reg_pred, data, cls_target)
        return sliced_cls_pred, cls_target, cls_weight, best_reg, gt_tgt, gt_mask

    return patched


def _make_align_plan_patch(session: _FrozenMatchingSession, orig: Callable) -> Callable:
    """AlignPlanTarget has deterministic ref targets; re-gather current preds."""

    def patched(self, cls_pred, reg_pred, gt_reg_target, gt_reg_mask, data, ref_target):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, cls_pred, reg_pred, gt_reg_target, gt_reg_mask, data, ref_target)
            sliced_cls_pred, cls_target, cls_weight, best_reg, gt_tgt, gt_mask = out
            session._record(sid, (
                "align_plan",
                cls_target.detach(),
                cls_weight.detach(),
                gt_tgt.detach(),
                gt_mask.detach(),
            ))
            return out
        tag, cls_target, cls_weight, gt_tgt, gt_mask = cached
        assert tag == "align_plan", f"queue tag mismatch: expected align_plan, got {tag}"
        sliced_cls_pred, best_reg = _replan_gather(self, cls_pred, reg_pred, data, cls_target)
        return sliced_cls_pred, cls_target, cls_weight, best_reg, gt_tgt, gt_mask

    return patched


def _replan_gather(self, cls_pred, reg_pred, data, mode_idx):
    """Reproduce plan cls/best-reg gather using current predictions."""
    bs = reg_pred.shape[0]
    bs_indices = torch.arange(bs, device=reg_pred.device)
    cmd = data["gt_ego_fut_cmd"].argmax(dim=-1) if self.ego_fut_cmd > 1 else 0

    cls_pred = cls_pred.reshape(bs, self.ego_fut_cmd, 1, -1)
    reg_pred = reg_pred.reshape(bs, self.ego_fut_cmd, 1, -1, self.ego_fut_ts, 2)
    cls_pred = cls_pred[bs_indices, cmd]
    reg_pred = reg_pred[bs_indices, cmd]

    ts, d = self.ego_fut_ts, 2
    gather_idx = mode_idx[..., None, None, None].repeat(1, 1, 1, ts, d)
    best_reg = torch.gather(reg_pred, 2, gather_idx).squeeze(2)
    return cls_pred, best_reg


@contextlib.contextmanager
def _deterministic_cumsum():
    """Monkey-patch torch.cumsum / Tensor.cumsum to run on CPU. CUDA cumsum
    has no deterministic implementation; this guarantees bit-exact results
    inside a freeze scope at the cost of one D2H/H2D copy per call."""
    _orig_t = torch.Tensor.cumsum
    _orig_f = torch.cumsum

    def patched_t(self, *args, **kwargs):
        if self.is_cuda:
            return _orig_t(self.cpu(), *args, **kwargs).to(self.device)
        return _orig_t(self, *args, **kwargs)

    def patched_f(input, *args, **kwargs):
        if isinstance(input, torch.Tensor) and input.is_cuda:
            return _orig_f(input.cpu(), *args, **kwargs).to(input.device)
        return _orig_f(input, *args, **kwargs)

    torch.Tensor.cumsum = patched_t
    torch.cumsum = patched_f
    try:
        yield
    finally:
        torch.Tensor.cumsum = _orig_t
        torch.cumsum = _orig_f


@contextlib.contextmanager
def _per_forward_seed(seed: int):
    """Pin torch, numpy, and Python random streams for one forward."""
    import random as _random

    import numpy as _np

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = _np.random.get_state()
    py_state = _random.getstate()

    torch.manual_seed(seed)
    _np.random.seed(seed)
    _random.seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        _np.random.set_state(np_state)
        _random.setstate(py_state)
