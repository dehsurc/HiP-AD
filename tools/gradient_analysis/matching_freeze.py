"""Matching freeze for the gradient probe.

Context manager that pins Hungarian matching (det/map) and mode-selection
argmin (motion/plan) to the result of the FIRST forward inside its scope.
All later forwards within the same `freeze_step` reuse those cached
assignments so that ΔL between baseline and stepped forwards isolates the
weight-update effect from matching reshuffling.

Usage::

    with FrozenMatching() as fm:
        # forward 0 — captures matching; subsequent calls in this step replay
        fm.next_forward()
        baseline = collector.forward_losses(data)
        # forward 1 — replays cached matching
        fm.next_forward()
        grad_fwd = collector.forward_losses(data)
        ...
        # next batch — must drop caches
        fm.reset()

The patch is applied at the class level for the duration of the context, so
all live instances pick it up. Caches are keyed by `id(sampler_instance)` so
multi-decoder-layer calls within a single forward each get their own slot in
the queue.
"""
from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

import torch


class FrozenMatching:
    """Stateful object holding the per-instance call queues + play indices.

    Two operating modes — driven implicitly by queue length:
      - When `play_idx[sid] >= len(queue[sid])`: in **record** mode for this
        call (run original `sample`, append outputs to queue).
      - Otherwise: in **playback** mode (replay queue[sid][play_idx], advance).

    `next_forward()` resets play indices to zero so the next forward replays
    from the start of each queue. `reset()` drops the queues entirely (call
    between batches when matching legitimately differs).
    """

    def __init__(self) -> None:
        self._queues: Dict[int, List[Any]] = defaultdict(list)
        self._play_idx: Dict[int, int] = defaultdict(int)
        self._patches: List = []

    # ------------- queue plumbing -------------
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

    # ------------- public API -------------
    def next_forward(self) -> None:
        """Rewind playback indices. Call before each forward in the freeze cycle."""
        self._play_idx = defaultdict(int)

    def reset(self) -> None:
        """Drop all caches. Call between probe batches."""
        self._queues.clear()
        self._play_idx.clear()

    # ------------- context manager -------------
    def __enter__(self) -> "FrozenMatching":
        self._install_patches()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._uninstall_patches()
        self.reset()

    # ------------- patch install -------------
    def _install_patches(self) -> None:
        from projects.mmdet3d_plugin.models.det.target import SparseBox3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget  # type: ignore
        from projects.mmdet3d_plugin.models.motion.target import SparseMotionTarget  # type: ignore
        from projects.mmdet3d_plugin.models.plan.target import (  # type: ignore
            SparsePlanTarget,
            AlignPlanTarget,
            PlanningTarget,
        )

        self._patches = [
            (SparseBox3DTarget, "sample", SparseBox3DTarget.sample,
             _make_det_patch(self, SparseBox3DTarget.sample)),
            (SparsePoint3DTarget, "sample", SparsePoint3DTarget.sample,
             _make_map_patch(self, SparsePoint3DTarget.sample)),
            (SparseMotionTarget, "sample", SparseMotionTarget.sample,
             _make_motion_patch(self, SparseMotionTarget.sample)),
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


# ---------------------------------------------------------------------------
# Per-sampler patch factories.
# Each returns a function that — within a freeze session — runs the original
# `sample` on the FIRST call (per-instance) and replays cached outputs on
# subsequent calls. Outputs that depend on current predictions (gather over
# `reg_pred`) are recomputed from cached indices so gradients remain attached.
# ---------------------------------------------------------------------------

def _make_det_patch(session: FrozenMatching, orig: Callable) -> Callable:
    """det: outputs are GT-derived (cls_target / box_target / reg_weights are
    indexed from the GT lists, not the predictions). Safe to cache outputs as
    detached tensors and replay verbatim."""

    def patched(self, cls_pred, box_pred, cls_target, box_target):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, cls_pred, box_pred, cls_target, box_target)
            # Detach to avoid holding the autograd graph from the recording forward.
            cached_out = tuple(t.detach() if torch.is_tensor(t) else t for t in out)
            cached_indices = self.indices
            session._record(sid, ("det", cached_out, cached_indices))
            return out
        tag, cached_out, cached_indices = cached
        assert tag == "det", f"queue tag mismatch: expected det, got {tag}"
        # Restore self.indices so downstream consumers (motion_loss_cache) see frozen state.
        self.indices = cached_indices
        return cached_out

    return patched


def _make_map_patch(session: FrozenMatching, orig: Callable) -> Callable:
    """map: same property as det — outputs are GT-derived."""

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


def _make_motion_patch(session: FrozenMatching, orig: Callable) -> Callable:
    """motion: indices come from `motion_loss_cache` (already frozen via det).
    But `cls_target` (mode argmin) and `best_reg` (gather over `reg_pred`)
    change with predictions. Cache cls_target as the frozen mode_idx and
    re-gather best_reg from the CURRENT reg_pred to keep gradient flow."""

    def patched(self, reg_pred, gt_reg_target, gt_reg_mask, motion_loss_cache):
        sid = id(self)
        cached = session._take(sid)
        if cached is None:
            out = orig(self, reg_pred, gt_reg_target, gt_reg_mask, motion_loss_cache)
            cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos = out
            # Cache mode_idx (cls_target), GT-derived tensors, and num_pos.
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
        # Re-gather best_reg from CURRENT reg_pred using cached mode_idx so grad flows.
        bs, num_anchor, mode, ts, d = reg_pred.shape
        gather_idx = cls_target[..., None, None, None].expand(bs, num_anchor, 1, ts, d)
        best_reg = torch.gather(reg_pred, 2, gather_idx).squeeze(2)
        return cls_target, cls_weight, best_reg, reg_target, reg_weight, num_pos

    return patched


def _make_plan_patch(session: FrozenMatching, orig: Callable) -> Callable:
    """SparsePlanTarget / PlanningTarget: same WTA pattern as motion. Returns
    `(cls_pred, cls_target, cls_weight, best_reg, gt_reg_target, gt_reg_mask)`.
    `cls_pred` and `best_reg` are gather/index outputs of the CURRENT reg_pred,
    so we recompute them from cached mode_idx to keep gradients."""

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
        # Re-derive cls_pred / best_reg from CURRENT preds + cached mode_idx.
        sliced_cls_pred, best_reg = _replan_gather(self, cls_pred, reg_pred, data, cls_target)
        return sliced_cls_pred, cls_target, cls_weight, best_reg, gt_tgt, gt_mask

    return patched


def _make_align_plan_patch(session: FrozenMatching, orig: Callable) -> Callable:
    """AlignPlanTarget.sample takes an extra `ref_target` arg and uses it as
    cls_target directly (no argmin), so matching is already deterministic
    given ref_target. We still need to cache cls_weight/gt_tgt/gt_mask for
    consistency, and recompute cls_pred / best_reg from CURRENT preds."""

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
    """Reproduce the cls_pred slice + best_reg gather from SparsePlanTarget /
    PlanningTarget / AlignPlanTarget against the CURRENT predictions, using
    a frozen `mode_idx`. Mirrors the slicing logic in plan/target.py."""
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
def per_forward_seed(seed: int):
    """Pin RNG for the next forward (DN noise / temporal_dn_groups randperm).

    Captures the pre-call RNG state and restores it on exit, so this can be
    nested or interleaved without polluting the outer RNG stream."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
