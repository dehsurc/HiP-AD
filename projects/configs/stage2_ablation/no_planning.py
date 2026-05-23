# Ablation: zero planning losses (plan_cls + plan_reg + ego_status); motion stays on.
# Purpose: isolate whether planning supervision is the dominant source of interference.
# Note: this is effectively what previous E8 run did — existing E8 result can be
# reused as evidence for this ablation (motion was 0.2 in E8).
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/no_planning"
wandb_name = "stage2_ablation_no_planning"

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
            loss_ego_status=dict(type="L1Loss", loss_weight=0.0),
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
            loss_plan_reg=dict(type="L1Loss", loss_weight=0.0),
            # motion losses unchanged from E2 baseline (0.2)
        ),
    ),
)
