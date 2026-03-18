import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxBEVHeatmapHead(nn.Module):
    """Auxiliary head that converts sparse query features into a dense BEV
    heatmap for distillation with BEVFusion's dense_heatmap.

    The process:
    1. Each detection query has an anchor position (x, y) in LiDAR frame.
    2. Scatter (bilinear splat) query features onto a BEV grid.
    3. Apply convolutions to produce a class-specific heatmap [B, C, H, W].

    This head is only used during training for distillation and can be removed
    at inference time.
    """

    def __init__(
        self,
        embed_dims=256,
        num_classes=9,
        bev_h=128,
        bev_w=128,
        pc_range=(-51.2, -51.2, 51.2, 51.2),  # (x_min, y_min, x_max, y_max)
        mid_channels=128,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.pc_range = pc_range  # (x_min, y_min, x_max, y_max)

        # Project query features before scattering
        self.query_proj = nn.Sequential(
            nn.Linear(embed_dims, mid_channels),
            nn.ReLU(inplace=True),
        )

        # Conv layers to refine scattered features into heatmap
        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, 3, padding=1),
        )

    def forward(self, query_features, anchors):
        """
        Args:
            query_features: [B, N, embed_dims] - detection query features
            anchors: [B, N, D] - detection anchors, where D >= 2
                     anchors[..., 0] = x, anchors[..., 1] = y in LiDAR frame

        Returns:
            heatmap: [B, num_classes, bev_h, bev_w] - predicted BEV heatmap
        """
        B, N, _ = query_features.shape
        device = query_features.device

        # Project query features
        feat = self.query_proj(query_features)  # [B, N, mid_channels]

        # Convert anchor (x, y) to BEV grid coordinates
        x_min, y_min, x_max, y_max = self.pc_range
        x = anchors[..., 0]  # [B, N]
        y = anchors[..., 1]  # [B, N]

        # Normalize to [0, bev_h-1] and [0, bev_w-1]
        gx = (x - x_min) / (x_max - x_min) * (self.bev_w - 1)  # [B, N]
        gy = (y - y_min) / (y_max - y_min) * (self.bev_h - 1)  # [B, N]

        # Bilinear splatting: scatter query features to BEV grid
        bev_feat = self._bilinear_splat(feat, gx, gy, B, device)  # [B, mid, H, W]

        # Conv to produce heatmap
        heatmap = self.conv(bev_feat)  # [B, num_classes, H, W]
        return heatmap

    def _bilinear_splat(self, feat, gx, gy, B, device):
        """Bilinear splatting of sparse features onto dense BEV grid.

        For each query at fractional position (gx, gy), distribute its
        feature to the 4 nearest grid cells with bilinear weights.

        Args:
            feat: [B, N, C] - projected query features
            gx: [B, N] - x coordinates in grid space
            gy: [B, N] - y coordinates in grid space

        Returns:
            bev: [B, C, bev_h, bev_w]
        """
        C = feat.shape[-1]
        H, W = self.bev_h, self.bev_w

        bev = feat.new_zeros(B, C, H, W)
        weight = feat.new_zeros(B, 1, H, W)

        # Clamp to valid range
        gx = gx.clamp(0, W - 1)
        gy = gy.clamp(0, H - 1)

        # Four corner indices
        gx0 = gx.long()
        gy0 = gy.long()
        gx1 = (gx0 + 1).clamp(max=W - 1)
        gy1 = (gy0 + 1).clamp(max=H - 1)

        # Bilinear weights
        wx = gx - gx0.float()  # [B, N]
        wy = gy - gy0.float()  # [B, N]

        feat_t = feat.permute(0, 2, 1)  # [B, C, N]

        for (ix, iy, w) in [
            (gx0, gy0, (1 - wx) * (1 - wy)),
            (gx1, gy0, wx * (1 - wy)),
            (gx0, gy1, (1 - wx) * wy),
            (gx1, gy1, wx * wy),
        ]:
            # w: [B, N]
            w_feat = feat_t * w.unsqueeze(1)  # [B, C, N]
            idx = iy * W + ix  # [B, N] - flat index
            idx = idx.unsqueeze(1).expand(-1, C, -1)  # [B, C, N]
            bev.view(B, C, -1).scatter_add_(2, idx, w_feat)

            w_idx = idx[:, :1, :]  # [B, 1, N]
            weight.view(B, 1, -1).scatter_add_(2, w_idx, w.unsqueeze(1))

        # Normalize by accumulated weights (avoid division by zero)
        bev = bev / (weight + 1e-6)
        return bev
