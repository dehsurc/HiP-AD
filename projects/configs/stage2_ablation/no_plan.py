# Ablation: zero ONLY plan_cls + plan_reg; keep ego_status and motion at baseline.
# Purpose: isolate planning (trajectory) supervision's contribution, separate from ego.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/no_plan"
wandb_name = "stage2_ablation_no_plan"

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
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.0),
            loss_plan_reg=dict(type="L1Loss", loss_weight=0.0),
            # ego_status, motion unchanged from E2 baseline
        ),
    ),
)
