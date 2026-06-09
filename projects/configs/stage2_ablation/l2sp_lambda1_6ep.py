# Ablation: L2-SP regularization toward stage1 solution, 6 epochs.
# Adds lambda * ||theta - theta_stage1||^2 to the training loss on det/map/
# backbone parameters only. Goal: prevent drift of det/map features during
# stage2 training, regardless of whether the drop mechanism is gradient
# magnitude, direction conflict, or forward-level contamination.
#
# Anchored parameter groups (exist in stage1 ckpt, directly tied to det/map):
#   - img_backbone.*              (ResNet50)
#   - img_neck.*                  (FPN)
#   - depth_branch.*              (dense depth aux)
#   - head.onedecoder_head.det_*  (det anchor_encoder / deformable / refine /
#                                  instance_bank)
#   - head.onedecoder_head.map_*  (map ... same family)
#   - head.onedecoder_head.fc_before, fc_after (decoupled-attn projections)
#
# Excluded: motion_/plan_/ego_/layers.* (shared decoder operations contain
# plan/ego-specific attention modules; anchoring them would freeze plan/ego
# learning toward their stage1 random-init values, which is worse than no
# anchor at all).
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/l2sp_lambda1_6ep"
wandb_name = "stage2_ablation_l2sp_lambda1_6ep"

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
        type="L2SPHook",
        stage1_ckpt="./work_dirs/exp/E1_stage1_12ep/latest.pth",
        lambda_l2sp=1.0,
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
