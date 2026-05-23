# MapTRv2 teacher -> HiP-AD map module distillation, stage1 base.
#
# Inherits hipad_nusc_stage1.py and overrides:
#   - wandb name / work_dir
#   - data.train.map_teacher_cache_path
#   - train_pipeline Collect.keys (adds teacher_map_*)
#   - model.head.onedecoder_head map_distill_*
#
# Real map GT loss stays ON (map_gt_loss_weight=1.0). Direction ambiguity
# resolved by forward/reverse min L1 in _compute_map_distill_loss, so reg KD
# is enabled (alpha_reg > 0) — differs from chanyoung's setup that disabled
# reg via alpha=0 due to raw-L1 direction noise.

_base_ = ["./hipad_nusc_stage1.py"]

# ------------------------------------------------------------------------
# Run identity
# ------------------------------------------------------------------------
wandb_project = "hipad_distill_map"
wandb_name = "hipad_nusc_stage1_12ep_map_distill"
work_dir = "work_dirs/exp/hipad_nusc_stage1_12ep_map_distill"

# Override log_config so the new wandb_project/wandb_name take effect at runtime
# (base config evaluates log_config before our overrides).
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(entity="e2ekd", project=wandb_project, name=wandb_name),
            by_epoch=False,
        ),
    ],
)

# ------------------------------------------------------------------------
# MapTRv2 teacher cache (28130 train samples; class permuted to HiP-AD order;
# pts in raw meters x in [-15,15], y in [-30,30])
# ------------------------------------------------------------------------
map_teacher_cache_path = "/data4/kyungmin/nuScenes_Converted/data/cache_map/maptrv2_teacher_train.pkl"

# ------------------------------------------------------------------------
# Map KD hyperparameters
# Initial ratio alpha_cls : alpha_reg = 1:10 mirrors HiP-AD main map loss
# (loss_map_cls.loss_weight=1.0, loss_map_reg.loss_line.loss_weight=10.0).
# Magnitude to be re-tuned after dry-run.
# ------------------------------------------------------------------------
map_distill_alpha_cls = 0.1
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = True
map_gt_loss_weight = 1.0

# ------------------------------------------------------------------------
# Onedecoder_head overrides
# ------------------------------------------------------------------------
model = dict(
    head=dict(
        onedecoder_head=dict(
            map_distill_alpha_cls=map_distill_alpha_cls,
            map_distill_alpha_reg=map_distill_alpha_reg,
            map_distill_temperature=map_distill_temperature,
            map_distill_score_thr=map_distill_score_thr,
            map_distill_dist_thr=map_distill_dist_thr,
            map_distill_last_layer_only=map_distill_last_layer_only,
            map_gt_loss_weight=map_gt_loss_weight,
        ),
    ),
)

# ------------------------------------------------------------------------
# train_pipeline redefinition (inline copy of stage1 base pipeline + extra
# teacher_map_* collect keys). Keep in sync with hipad_nusc_stage1.py if base
# pipeline changes.
# ------------------------------------------------------------------------
file_client_args = dict(backend="disk")
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args,
    ),
    dict(type="ResizeCropFlipImage"),
    # strides[:num_depth_layers] = [4,8,16,32][:3] = [4,8,16]
    dict(type="MultiScaleDepthMapGenerator", downsample=[4, 8, 16]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(
        type="CircleObjectRangeFilter",
        # len(det_class_names) == 10 in stage1 base
        class_dist_thred=[55] * 10,
    ),
    dict(
        type="InstanceNameFilter",
        classes=[
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ],
    ),
    dict(
        type="VectorizeMap",
        roi_size=(30, 60),       # = map_roi_size
        simplify=False,
        normalize=False,
        sample_num=20,           # = map_num_pts
        permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "gt_depth",
            "focal",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_map_labels",
            "gt_map_pts",
            "gt_agent_fut_trajs",
            "gt_agent_fut_masks",
            "gt_ego_fut_trajs",
            "gt_ego_fut_masks",
            "gt_ego_fut_cmd",
            "gt_ego_fut_trajs_2hz",
            "gt_ego_fut_masks_2hz",
            "ego_status",
            "ego_status_mask",
            # Map distillation tensors (added vs base)
            "teacher_map_logits",
            "teacher_map_pts",
            "teacher_map_scores",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

# ------------------------------------------------------------------------
# Wire cache path + new train_pipeline into data.train.
# val/test untouched (KD only at training time).
# ------------------------------------------------------------------------
data = dict(
    train=dict(
        pipeline=train_pipeline,
        map_teacher_cache_path=map_teacher_cache_path,
    ),
)
