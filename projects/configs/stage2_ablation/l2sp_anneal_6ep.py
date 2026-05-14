# Ablation: L2-SP with cosine annealing schedule (RecAdam-style objective
# shifting), 6 epochs. Same anchor scope as l2sp_lambda1_6ep so this isolates
# the annealing effect vs the fixed-lambda baseline.
#
# Why annealing:
# Previous l2sp_lambda1_6ep used a fixed lambda=1.0 for the entire 6 epochs.
# This holds det/map params at stage1 values even AFTER the early "damage
# window" has passed -> the model never gets to refine det/map features
# beyond stage1's solution, capping final performance (final mAP 0.169,
# worse than no-intervention E3 baseline 0.245).
#
# RecAdam (Chen et al., EMNLP 2020) shows the missing piece is "objective
# shifting": ramp the anchor weight DOWN over time so early iters get strong
# protection during the cold-start gradient burst from random-init new heads,
# but later iters can freely co-adapt det/map with the new tasks.
#
# Schedule:
#   iter 0~2343    (epoch 1):  lambda = 1.0  (full hold through damage window)
#   iter 2344~7031 (epochs 2-3): lambda 1.0 -> 0.0 cosine decay
#   iter 7032~ (epochs 4-6): lambda = 0.0   (full freedom for late refinement)
#
# Compare against:
#   * l2sp_lambda1_6ep (fixed lambda=1.0): isolates annealing effect
#   * loss_warmup_500iter_6ep (early-iter loss damping): different mechanism,
#       same goal of protecting damage window
#   * E3 6ep baseline: reference for "is this actually better than nothing"
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/l2sp_anneal_6ep"
wandb_name = "stage2_ablation_l2sp_anneal_6ep"

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))  # 2344
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
        type="L2SPHook",
        stage1_ckpt="./work_dirs/exp/E1_stage1_12ep/latest.pth",
        lambda_l2sp=1.0,
        schedule='cosine',
        lambda_min=0.0,
        hold_iters=num_iters_per_epoch,        # 1 epoch hold
        anneal_iters=num_iters_per_epoch * 2,  # 2 epoch decay
        include_prefixes=[
            "img_backbone.",
            "img_neck.",
            "depth_branch.",
            "head.onedecoder_head.det_",
            "head.onedecoder_head.map_",
            "head.onedecoder_head.fc_before",
            "head.onedecoder_head.fc_after",
        ],
        exclude_prefixes=[],
        log_interval=50,
        priority="LOW",
    ),
]
