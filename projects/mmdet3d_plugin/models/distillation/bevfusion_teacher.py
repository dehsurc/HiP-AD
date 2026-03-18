import sys
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint


class BEVFusionTeacher(nn.Module):
    """Frozen BEVFusion wrapper for online distillation.

    Builds and loads a pre-trained BEVFusion model, runs inference with
    torch.no_grad(), and returns dense_heatmap + per-proposal predictions.

    Because BEVFusion uses a different image size (256x704) and normalisation
    (ImageNet) compared to HiP-AD (640x352, [123.675,...]), the teacher
    performs its own image preprocessing internally when given raw images or
    re-normalises the student images.

    For simplicity this implementation assumes **cached teacher outputs** are
    loaded via the data pipeline (see ``tools/cache_teacher_outputs.py``).
    An online inference path is provided but requires BEVFusion's code on
    ``sys.path`` and sufficient GPU memory.
    """

    def __init__(
        self,
        bevfusion_root="/home/yongjae/e2e/bevfusion",
        config_path=None,
        checkpoint_path=None,
        use_cache=True,
    ):
        super().__init__()
        self.use_cache = use_cache
        self.bevfusion_root = bevfusion_root

        if not use_cache:
            self.model = self._build_model(config_path, checkpoint_path)
        else:
            # When using cache, no model is needed on GPU
            self.model = None

    # ------------------------------------------------------------------
    # Online mode helpers
    # ------------------------------------------------------------------
    def _build_model(self, config_path, checkpoint_path):
        """Build BEVFusion model from its config and load checkpoint."""
        # Add bevfusion to path for imports
        if self.bevfusion_root not in sys.path:
            sys.path.insert(0, self.bevfusion_root)

        from torchpack.utils.config import configs as tp_configs

        tp_configs.load(config_path, recursive=True)
        cfg = tp_configs

        # Build model using bevfusion's builder
        from mmdet3d.models import build_model
        model = build_model(cfg.model)
        load_checkpoint(model, checkpoint_path, map_location="cpu")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    @torch.no_grad()
    def forward_online(self, img, points, data):
        """Run BEVFusion inference online.

        NOTE: This requires that ``img`` and ``points`` are preprocessed
        in a format compatible with BEVFusion (different from HiP-AD).
        Use ``tools/cache_teacher_outputs.py`` for the recommended workflow.
        """
        raise NotImplementedError(
            "Online inference requires matching BEVFusion preprocessing. "
            "Use cached mode (use_cache=True) with tools/cache_teacher_outputs.py."
        )

    # ------------------------------------------------------------------
    # Cache mode
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, data):
        """Return teacher outputs from cache stored in data dict.

        Expected keys in ``data``:
            - teacher_dense_heatmap: [B, num_classes, H, W]
            - teacher_heatmap: [B, num_classes, num_proposals]
            - teacher_center: [B, 2, num_proposals]
            - teacher_height: [B, 1, num_proposals]
            - teacher_dim: [B, 3, num_proposals]
            - teacher_rot: [B, 2, num_proposals]
            - teacher_vel: [B, 2, num_proposals]
        """
        if self.use_cache:
            return {
                "dense_heatmap": data["teacher_dense_heatmap"],
                "heatmap": data["teacher_heatmap"],
                "center": data["teacher_center"],
                "height": data["teacher_height"],
                "dim": data["teacher_dim"],
                "rot": data["teacher_rot"],
                "vel": data["teacher_vel"],
            }
        else:
            return self.forward_online(None, None, data)
