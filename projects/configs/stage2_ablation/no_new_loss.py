# Ablation: zero ALL stage2-new losses (motion + planning + ego_status).
# Purpose: truly reproduce stage1's supervision signal on the stage2 model.
# Note: previous E8 run kept motion=0.2 on, which does NOT match stage1
# (stage1 gates motion out via task_select). This config is the strict version.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/no_new_loss"
wandb_name = "stage2_ablation_no_new_loss"

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
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.0),
        ),
    ),
)
