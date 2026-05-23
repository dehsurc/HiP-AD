# Ablation: loss weight warmup extended to 1 full epoch (2344 iters), 6 epochs total.
# Motivation: loss_warmup_500iter_6ep showed no improvement over baseline because
# the 500-iter weight warmup overlapped exactly with the existing 500-iter LR
# warmup — once LR reached peak (iter 500), the loss was already at target too,
# so the rest of ep1 (iter 500~2344, where most damage accumulates) was
# identical to baseline. Extending warmup to 1 full epoch keeps the loss weight
# ramping through the entire damage window while LR is at peak.
#
# Schedule: iter 0~500 LR ramp (1/3 -> 1) + weight ramp (0 -> 0.213 target).
#           iter 500~2344 LR peak + weight ramp (0.213 -> 1.0 target).
#           iter 2344~ full LR (cosine) + full weight.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/loss_warmup_2344iter_6ep"
wandb_name = "stage2_ablation_loss_warmup_2344iter_6ep"

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
        warmup_iters=2344,
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
