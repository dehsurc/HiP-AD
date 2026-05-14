# Ablation: smooth loss weight warmup at stage2 onset.
# Full stage2 loss set + baseline LR (1e-4 with 500-iter LR warmup) unchanged,
# but additionally ramp the 5 newly-activated task losses linearly from 0 to
# their target weights over the first 500 iters (matches LR warmup length).
# Tests hypothesis (A): "early gradient burst through random-init new heads
# damages shared det/map features at training onset".
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/loss_warmup_500iter"
wandb_name = "stage2_ablation_loss_warmup_500iter"

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(entity="e2ekd", project="hipad", name=wandb_name),
            by_epoch=False,
        ),
    ],
)

# Must re-list all custom_hooks (mmcv list-override replaces, not merges)
custom_hooks = [
    dict(
        type="WandbValVisHook",
        vis_dir="val_vis/visual",
        max_images=8,
        interval=1,
        priority="LOWEST",
    ),
    dict(
        type="LossWeightWarmupHook",
        warmup_iters=500,
        loss_names=[
            "loss_motion_cls",
            "loss_motion_reg",
            "loss_plan_cls",
            "loss_plan_reg",
            "loss_ego_status",
        ],
        head_path="head.onedecoder_head",
    ),
]
