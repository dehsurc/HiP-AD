# Ablation: L2-SP with cosine annealing AND broadened anchor scope to
# include the shared decoder (head.onedecoder_head.layers.*), 6 epochs.
# Compare against l2sp_anneal_6ep to isolate the scope-broadening effect
# from the annealing effect.
#
# Why broaden scope to layers.*:
# The previous l2sp_lambda1_6ep config excluded head.onedecoder_head.layers.*
# on the assumption that "shared decoder contains plan/ego-specific attention
# modules". Re-inspecting the architecture and stage1 checkpoint:
#   * layers.* is a flat ModuleList of generic ops (norm/ffn/gnn/deformable/
#     refine) keyed by operation_order in SparseOneDecoderHead. NO task-
#     specific submodules inside.
#   * Plan/ego/motion-specific modules live in sibling prefixes
#     (plan_instance_bank.*, ego_instance_bank.*, motion_*) which are NOT
#     anchored here.
#   * All 204 layers.* parameters exist in stage1 ckpt with valid shapes,
#     i.e. they WERE trained on det+map supervision in stage1.
#
# This is precisely the "shared decoder gradient domination" location
# identified by the per-group gradient norm analysis (plan grad ~3.5x det
# at this point). Anchoring it with annealing is the literal application of
# "protect what was learned, release once new tasks have settled".
#
# Scope:
#   include: backbone, neck, depth, det_*, map_*, fc_before, fc_after,
#            layers.*   (shared decoder, NEW)
#   exclude (defense in depth): plan_*, ego_*, motion_*  (their stage1
#            values are random-init since stage1 doesn't activate these
#            losses; anchoring would freeze toward noise).
#
# Schedule: same as l2sp_anneal_6ep (cosine, hold 1ep, anneal over 2ep,
# release 3 final ep) so the only difference vs that config is scope.
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/l2sp_anneal_shared_6ep"
wandb_name = "stage2_ablation_l2sp_anneal_shared_6ep"

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
        hold_iters=num_iters_per_epoch,
        anneal_iters=num_iters_per_epoch * 2,
        include_prefixes=[
            "img_backbone.",
            "img_neck.",
            "depth_branch.",
            "head.onedecoder_head.det_",
            "head.onedecoder_head.map_",
            "head.onedecoder_head.fc_before",
            "head.onedecoder_head.fc_after",
            "head.onedecoder_head.layers.",  # shared decoder (new vs l2sp_anneal_6ep)
        ],
        exclude_prefixes=[
            "head.onedecoder_head.plan_",
            "head.onedecoder_head.ego_",
            "head.onedecoder_head.motion_",
        ],
        log_interval=50,
        priority="LOW",
    ),
]
