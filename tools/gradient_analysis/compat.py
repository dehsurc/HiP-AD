"""Runtime compatibility patches for HiP-AD samplers.

These are NOT analysis features — they're workarounds for upstream PyTorch
behavior that interacts badly with the existing HiP-AD code on some
CUDA/PyTorch builds. They're applied unconditionally at probe startup so
the analysis pipeline can run without editing the model source tree.

Currently fixes:
  - SparsePoint3DTarget.sample line 60: ``output_reg_weights[i, pred_idx] = 1``
    triggers an INTERNAL ASSERT
    ``linearIndex.numel()*sliceSize*nElemBefore == expandedValue.numel()``
    in ``aten/.../Indexing.cu`` on some builds because the scalar ``1`` is
    not auto-expanded to the full slice shape inside the CUDA index_put
    kernel. Replacing with a properly-shaped ones tensor sidesteps the
    assert deterministically.
  - ``torch.utils.checkpoint.checkpoint`` defaults to ``use_reentrant=True``,
    which is incompatible with ``torch.autograd.grad(inputs=...)``. Force
    ``use_reentrant=False`` so backbone gradient checkpointing (with_cp=True)
    can co-exist with the per-task gradient extraction the probe needs.
    This restores the ~70% activation-memory saving that ``with_cp`` was
    designed to give.
"""
from __future__ import annotations

import torch
import torch.utils.checkpoint as cp


_APPLIED = False
_CHECKPOINT_PATCHED = False


def apply_use_reentrant_false() -> None:
    """Make ``torch.utils.checkpoint.checkpoint`` always pass
    ``use_reentrant=False`` so it plays nicely with ``autograd.grad(inputs=...)``.
    Idempotent. Affects every ``cp.checkpoint(...)`` call in the process,
    including those compiled into mmdet backbones (resnet, swin, ...)."""
    global _CHECKPOINT_PATCHED
    if _CHECKPOINT_PATCHED:
        return
    orig = cp.checkpoint

    def patched(function, *args, use_reentrant=False, **kwargs):
        # Strip any caller-supplied use_reentrant=True; the whole point is to
        # force the non-reentrant variant, which is compatible with
        # autograd.grad(inputs=...).
        return orig(function, *args, use_reentrant=False, **kwargs)

    cp.checkpoint = patched
    # Some files do `from torch.utils.checkpoint import checkpoint as cp` —
    # patch the public attr too in case of late binding.
    torch.utils.checkpoint.checkpoint = patched
    _CHECKPOINT_PATCHED = True


def apply_index_put_fix() -> None:
    """Monkey-patch SparsePoint3DTarget.sample to use tensor-valued fills
    for ``output_reg_weights[i, pred_idx]``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    from projects.mmdet3d_plugin.models.map.target import SparsePoint3DTarget  # type: ignore

    def fixed_sample(self, cls_preds, pts_preds, cls_targets, pts_targets):
        # Mirror of the original method, with the scalar ``= 1`` assignment
        # rewritten to use ``output_reg_weights.new_ones`` so the CUDA kernel
        # gets a value tensor of the exact slice shape.
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
        trailing = output_reg_weights.shape[2:]  # e.g. (40,) for line points
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
    _APPLIED = True
