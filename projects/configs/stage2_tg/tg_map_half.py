# TG variant: map loss weight halved (0.5x)
# Tests whether reducing map loss (instead of completely removing) improves planning.
# Based on TG analysis finding: map task has most negative TG (-0.033)
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_map_half"

# All tasks active, but map loss weights reduced to 0.5x
model = dict(
    ablate_tasks=[],
    head=dict(
        onedecoder_head=dict(
            # Original: loss_weight=1.0 -> 0.5
            loss_map_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,  # 1.0 * 0.5
            ),
            # Original: loss_weight=10.0 -> 5.0
            loss_map_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type="LinesL1Loss", loss_weight=5.0, beta=0.01),  # 10.0 * 0.5
                num_sample=20,  # map_num_pts
                roi_size=(30, 60),  # map_roi_size
            ),
        ),
    ),
)
