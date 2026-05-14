# Ablation: smooth loss weight warmup at stage2 onset, 6 epochs.
# Full stage2 loss set + baseline LR (1e-4 with 500-iter LR warmup) unchanged,
# but additionally ramp the 5 newly-activated task losses linearly from 0 to
# their target weights over the first 500 iters (matches LR warmup length).
#
# 1-epoch run (loss_warmup_500iter) showed: det drop almost fully prevented
# (−0.008), but map still has residual drop (−0.039). This 6-epoch run asks:
#   (i) do motion/plan/ego catch up to baseline (E3) by ep5-6?
#   (ii) do det/map reach baseline late-epoch performance (stage1 exceeded),
#        or plateau near stage1 like low_lr?
# Together these answer whether warmup is a sufficient stand-alone solution
# (A) or whether ongoing interference remains for map (B).
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/loss_warmup_500iter_6ep"
wandb_name = "stage2_ablation_loss_warmup_500iter_6ep"

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 6

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
evaluation = dict(
    interval=num_iters_per_epoch,
    eval_mode=dict(
        with_det=True,
        with_tracking=False,
        with_map=True,
        with_motion=True,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    ),
)

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
