# Ablation: zero ONLY ego_status loss; keep plan_cls/reg and motion at baseline.
# Purpose: isolate ego_status's contribution to det/map drop, separate from plan.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/no_ego"
wandb_name = "stage2_ablation_no_ego"

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
            # plan_cls, plan_reg, motion unchanged from E2 baseline
        ),
    ),
)
