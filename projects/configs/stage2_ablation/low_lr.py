# Ablation: keep full stage2 losses ON, but reduce peak LR by 10x (1e-4 -> 1e-5).
# Purpose: rule out LR-jump as a confound. Stage1 cosine ends near 1e-7, while
# stage2 restarts at 1e-4 peak — this test asks whether a gentler restart LR,
# without touching loss weights, prevents the det/map drop.
#
# Extended to 6 epochs: 1-epoch result (det=0.2040, map=0.3360) showed no drop,
# but with LR 10x lower, the model only moves ~10x less in parameter space per
# iter — so at 1 epoch we cannot distinguish "LR is fine intrinsically" from
# "not enough time to accumulate damage yet". 6 epochs matches the observation
# window used for the baseline Stage2 drop curve (ep1-ep4 were all below stage1).
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/low_lr_6ep"
wandb_name = "stage2_ablation_low_lr_6ep"

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

optimizer = dict(
    type="AdamW",
    lr=1e-5,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
