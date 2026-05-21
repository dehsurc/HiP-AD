"""Oracle perception injection: replace det outputs with GT.

Scenario A — det-oracle. Map kept normal.

Given the model's det "slot" (the 4 tensors consumed by motion_query build and
inter_gnn), this module produces those tensors from ground-truth boxes so the
downstream motion/plan modules see a perfect detector.

Slot tensors:
    det_anchor:           (B, N, 11)   [X, Y, Z, log W, log L, log H, sinY, cosY, VX, VY, VZ]
    det_cls:              (B, N, C)    one-hot-like logits (high at GT class, low elsewhere)
    det_instance_feature: (B, N, D)    zeros (oracle has no learned feature)
    det_anchor_embed:     (B, N, D)    det_anchor_encoder(det_anchor)

Slots beyond GT count are filled with a "no-object" row (zero box, very
negative cls logits) so attention modules over a fixed N still work and
det_sampler's Hungarian match maps GT 1-to-1 to slots 0..M-1.
"""
import torch

from ..core.box3d import X, Y, Z, W, L, H, SIN_YAW, COS_YAW, VX, VY, VZ


_BOX_DIM = 11
_HIGH_LOGIT = 10.0
_LOW_LOGIT = -10.0


def build_oracle_det_anchor(metas, num_anchor, device, dtype):
    """Convert list[Tensor[M_b, 9]] GT boxes into a padded (B, N, 11) anchor.

    GT format (per Adaptor):  [x, y, z, w, l, h, yaw, vx, vy]
    Anchor format:            [x, y, z, log w, log l, log h, sin yaw, cos yaw, vx, vy, vz=0]
    """
    gt_list = metas["gt_bboxes_3d"]
    B = len(gt_list)
    det_anchor = torch.zeros((B, num_anchor, _BOX_DIM), device=device, dtype=dtype)
    valid_count = torch.zeros((B,), device=device, dtype=torch.long)

    for b in range(B):
        gt = gt_list[b]
        if not torch.is_tensor(gt):
            gt = torch.as_tensor(gt, device=device, dtype=dtype)
        else:
            gt = gt.to(device=device, dtype=dtype)
        M = gt.shape[0]
        n = min(M, num_anchor)
        if n == 0:
            continue

        det_anchor[b, :n, X] = gt[:n, 0]
        det_anchor[b, :n, Y] = gt[:n, 1]
        det_anchor[b, :n, Z] = gt[:n, 2]
        det_anchor[b, :n, W] = gt[:n, 3].clamp_min(1e-3).log()
        det_anchor[b, :n, L] = gt[:n, 4].clamp_min(1e-3).log()
        det_anchor[b, :n, H] = gt[:n, 5].clamp_min(1e-3).log()
        yaw = gt[:n, 6]
        det_anchor[b, :n, SIN_YAW] = torch.sin(yaw)
        det_anchor[b, :n, COS_YAW] = torch.cos(yaw)
        if gt.shape[1] >= 8:
            det_anchor[b, :n, VX] = gt[:n, 7]
        if gt.shape[1] >= 9:
            det_anchor[b, :n, VY] = gt[:n, 8]
        # VZ stays 0
        valid_count[b] = n

    return det_anchor, valid_count


def build_oracle_det_cls(metas, valid_count, num_anchor, num_cls, device, dtype):
    """Build (B, N, C) class logits — high at GT class for valid slots, low everywhere else."""
    gt_labels_list = metas["gt_labels_3d"]
    B = len(gt_labels_list)
    cls = torch.full((B, num_anchor, num_cls), _LOW_LOGIT, device=device, dtype=dtype)

    for b in range(B):
        n = int(valid_count[b].item())
        if n == 0:
            continue
        lb = gt_labels_list[b]
        if not torch.is_tensor(lb):
            lb = torch.as_tensor(lb, device=device, dtype=torch.long)
        else:
            lb = lb.to(device=device, dtype=torch.long)
        idx = lb[:n].clamp(min=0, max=num_cls - 1)
        cls[b, torch.arange(n, device=device), idx] = _HIGH_LOGIT
    return cls


def build_oracle_det_quality(num_anchor, batch_size, device, dtype):
    """Stub quality logits (centerness, yawness) for the oracle path."""
    return torch.zeros((batch_size, num_anchor, 2), device=device, dtype=dtype)


def build_oracle_det_outputs(
    metas,
    num_anchor,
    num_cls,
    embed_dims,
    device,
    dtype,
    anchor_encoder,
    instance_feature_template=None,
):
    """Produce the 4 slot tensors + det_quality.

    instance_feature_template: optional (B, N, D) tensor whose values are
        overwritten by zeros; used only to preserve dtype/device when caller
        already has one in hand. If None, a new zero tensor is allocated.
    """
    det_anchor, valid_count = build_oracle_det_anchor(metas, num_anchor, device, dtype)
    det_cls = build_oracle_det_cls(metas, valid_count, num_anchor, num_cls, device, dtype)
    det_quality = build_oracle_det_quality(num_anchor, len(metas["gt_bboxes_3d"]), device, dtype)

    if instance_feature_template is not None:
        det_instance_feature = torch.zeros_like(instance_feature_template)
    else:
        det_instance_feature = torch.zeros(
            (len(metas["gt_bboxes_3d"]), num_anchor, embed_dims), device=device, dtype=dtype
        )

    det_anchor_embed = anchor_encoder(det_anchor)
    return det_anchor, det_cls, det_quality, det_instance_feature, det_anchor_embed, valid_count
