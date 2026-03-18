import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Combined distillation losses for BEVFusion -> HiP-AD knowledge transfer.

    Loss 1 (Strategy 1): GT-Matched Prediction Distillation
        - For each GT object, find the matched student query and teacher proposal.
        - KL-divergence on classification soft logits (with temperature).
        - Smooth-L1 on regression outputs.

    Loss 2 (Strategy 2): Auxiliary BEV Heatmap Distillation
        - MSE or Focal loss between student's auxiliary BEV heatmap and
          teacher's dense_heatmap.
    """

    def __init__(
        self,
        cls_kd_loss_weight=1.0,
        reg_kd_loss_weight=0.5,
        heatmap_kd_loss_weight=0.5,
        temperature=2.0,
        num_classes=9,
    ):
        super().__init__()
        self.cls_kd_loss_weight = cls_kd_loss_weight
        self.reg_kd_loss_weight = reg_kd_loss_weight
        self.heatmap_kd_loss_weight = heatmap_kd_loss_weight
        self.temperature = temperature
        self.num_classes = num_classes

    def forward(
        self,
        student_cls_list,
        student_reg_list,
        student_heatmap,
        teacher_outputs,
        gt_labels_3d,
        gt_bboxes_3d,
        student_sampler,
    ):
        """
        Args:
            student_cls_list: list of [B, N_s, C] - per-decoder-layer cls logits
            student_reg_list: list of [B, N_s, D] - per-decoder-layer reg preds
            student_heatmap: [B, C, H, W] - from AuxBEVHeatmapHead (or None)
            teacher_outputs: dict with keys:
                - dense_heatmap: [B, C, H, W] (logits, before sigmoid)
                - heatmap: [B, C, N_t] - per-proposal cls scores (after sigmoid)
                - center: [B, 2, N_t]
                - height: [B, 1, N_t]
                - dim: [B, 3, N_t]
                - rot: [B, 2, N_t]
                - vel: [B, 2, N_t]
            gt_labels_3d: list of [num_gt] per batch
            gt_bboxes_3d: list of [num_gt, D] per batch
            student_sampler: SparseBox3DTarget instance (has .indices after sample())

        Returns:
            losses: dict
        """
        losses = {}

        # ---- Strategy 1: GT-matched prediction distillation ----
        # Use the last decoder layer's outputs for distillation
        student_cls = student_cls_list[-1]  # [B, N_s, C]
        student_reg = student_reg_list[-1]  # [B, N_s, D]

        # Teacher predictions: reshape from [B, dim, N_t] to [B, N_t, dim]
        teacher_cls = teacher_outputs["heatmap"].permute(0, 2, 1)  # [B, N_t, C]
        teacher_reg = torch.cat([
            teacher_outputs["center"],   # [B, 2, N_t]
            teacher_outputs["height"],   # [B, 1, N_t]
            teacher_outputs["dim"],      # [B, 3, N_t]
            teacher_outputs["rot"],      # [B, 2, N_t]
            teacher_outputs["vel"],      # [B, 2, N_t]
        ], dim=1).permute(0, 2, 1)       # [B, N_t, 10]

        cls_kd_loss, reg_kd_loss = self._pred_distill_loss(
            student_cls, student_reg,
            teacher_cls, teacher_reg,
            student_sampler,
            gt_labels_3d, gt_bboxes_3d,
        )
        losses["loss_cls_kd"] = cls_kd_loss * self.cls_kd_loss_weight
        losses["loss_reg_kd"] = reg_kd_loss * self.reg_kd_loss_weight

        # ---- Strategy 2: Auxiliary heatmap distillation ----
        if student_heatmap is not None and "dense_heatmap" in teacher_outputs:
            heatmap_loss = self._heatmap_distill_loss(
                student_heatmap, teacher_outputs["dense_heatmap"]
            )
            losses["loss_heatmap_kd"] = heatmap_loss * self.heatmap_kd_loss_weight

        return losses

    def _pred_distill_loss(
        self,
        student_cls, student_reg,
        teacher_cls, teacher_reg,
        student_sampler,
        gt_labels_3d, gt_bboxes_3d,
    ):
        """GT-matched prediction distillation.

        For each GT object matched by the student (via Hungarian matching),
        find the teacher's most confident prediction for the same GT class
        at a nearby location, and distill between them.

        Since the teacher uses a different matching strategy (heatmap-based
        top-k rather than Hungarian), we match teacher proposals to GT by
        finding the closest proposal to each GT center.
        """
        B = student_cls.shape[0]
        total_cls_loss = student_cls.new_tensor(0.0)
        total_reg_loss = student_cls.new_tensor(0.0)
        num_matched = 0

        # Student's Hungarian matching indices (set by sampler.sample())
        student_indices = student_sampler.indices  # list of (pred_idx, gt_idx) per batch

        for b in range(B):
            s_pred_idx, s_gt_idx = student_indices[b]
            if s_pred_idx is None or len(s_pred_idx) == 0:
                continue

            num_gt = len(gt_labels_3d[b])
            if num_gt == 0:
                continue

            # Teacher: match proposals to GT by nearest center distance
            # GT centers in BEV (x, y)
            gt_centers = gt_bboxes_3d[b][:, :2]  # [num_gt, 2]

            # Teacher proposal centers: teacher_reg[b, :, :2] = (center_x, center_y)
            t_centers = teacher_reg[b, :, :2]  # [N_t, 2]

            # Pairwise distance: [num_gt, N_t]
            dist = torch.cdist(gt_centers.float(), t_centers.float())
            # For each GT, find closest teacher proposal
            t_matched_idx = dist.argmin(dim=1)  # [num_gt]

            # Now pair up: for each student-matched GT,
            # get the corresponding teacher proposal
            for i in range(len(s_pred_idx)):
                gt_i = s_gt_idx[i]  # which GT this student query matched
                s_i = s_pred_idx[i]  # which student query
                t_i = t_matched_idx[gt_i]  # which teacher proposal

                # Classification KD: KL-div with temperature
                s_logits = student_cls[b, s_i] / self.temperature  # [C]
                t_logits = teacher_cls[b, t_i]  # [C] (already sigmoid)
                # Convert teacher sigmoid to logits for KL-div
                t_logits_raw = torch.log(t_logits.clamp(1e-6) / (1 - t_logits.clamp(1e-6)))
                t_logits_raw = t_logits_raw / self.temperature

                s_log_prob = F.log_sigmoid(s_logits)
                t_prob = torch.sigmoid(t_logits_raw)
                # Binary KL-div per class (since it's multi-label sigmoid)
                cls_loss = F.binary_cross_entropy_with_logits(
                    s_logits, t_prob, reduction="sum"
                ) * (self.temperature ** 2)
                total_cls_loss = total_cls_loss + cls_loss

                # Regression KD: Smooth-L1
                # Only on common dimensions (min of student and teacher reg dims)
                s_reg = student_reg[b, s_i]  # [D_s]
                t_reg = teacher_reg[b, t_i]  # [10]
                min_dim = min(s_reg.shape[0], t_reg.shape[0])
                reg_loss = F.smooth_l1_loss(
                    s_reg[:min_dim], t_reg[:min_dim], reduction="sum"
                )
                total_reg_loss = total_reg_loss + reg_loss
                num_matched += 1

        if num_matched > 0:
            total_cls_loss = total_cls_loss / num_matched
            total_reg_loss = total_reg_loss / num_matched

        return total_cls_loss, total_reg_loss

    def _heatmap_distill_loss(self, student_heatmap, teacher_heatmap):
        """Distillation loss between student and teacher BEV heatmaps.

        Uses MSE loss on sigmoid-activated heatmaps. The teacher's
        dense_heatmap is in logits (before sigmoid), same as student's.

        Args:
            student_heatmap: [B, C, H_s, W_s] - from AuxBEVHeatmapHead
            teacher_heatmap: [B, C, H_t, W_t] - from BEVFusion dense_heatmap
        """
        # Resize student heatmap to match teacher if sizes differ
        if student_heatmap.shape[-2:] != teacher_heatmap.shape[-2:]:
            student_heatmap = F.interpolate(
                student_heatmap,
                size=teacher_heatmap.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Apply sigmoid to both (convert logits to probabilities)
        s_prob = student_heatmap.sigmoid()
        t_prob = teacher_heatmap.sigmoid().detach()

        # MSE loss on probability maps
        loss = F.mse_loss(s_prob, t_prob, reduction="mean")
        return loss
