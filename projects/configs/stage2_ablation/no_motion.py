# Ablation: zero ONLY motion losses; keep planning (plan_cls/reg + ego_status) on.
# Purpose: isolate whether motion supervision alone causes the det/map drop.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/no_motion"
wandb_name = "stage2_ablation_no_motion"

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

model = dict(
    head=dict(
        onedecoder_head=dict(
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.0),
            # planning losses unchanged from E2 baseline
        ),
    ),
)
