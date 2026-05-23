# D2: KD-only on BOTH detection and map (no real GT supervision for either task).
#
#   - Detection KD: BEVFusion teacher cache, pseudo_gt mode, det_gt_loss_weight=0.0
#                   (same as E5_stage1_12ep_distill_wo_det). D1 dropped score_thr to
#                   chase recall and underperformed E5 → D2 reuses E5's score_thr=0.3.
#   - Map KD:        MapTRv2 teacher cache, pseudo_gt mode, map_gt_loss_weight=0.0
#                   (same as our hipad_nusc_stage1_map_distill_kdonly).
#
# Both branches use pseudo_gt mode → alpha values act only as on/off flags
# (use_distill / use_map_distill). Effective KD magnitude is set by loss_det_cls
# (FocalLoss loss_weight=2.0) / loss_det_reg + loss_map_cls (1.0) / loss_map_reg
# (LinesL1 loss_weight=10.0) on the standard loss heads. teacher_tp branch is
# disabled (distill_mode=pseudo_gt for both).
#
# Real GT branches (det + map) emit zero loss because their *_gt_loss_weight=0.
# Depth and (vestigial) plan/ego losses remain on real GT, unchanged from
# stage1 base — they were already ~0 in stage1 logs.

_base_ = ["../hipad_nusc_stage1.py"]

# ------------------------------------------------------------------------
# Run identity
# ------------------------------------------------------------------------
wandb_project = "hipad_distill_dm"
wandb_name = "D2_stage1_12ep_distill_det_map_kdonly"
work_dir = "work_dirs/exp/D2_stage1_12ep_distill_det_map_kdonly"

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
# Detection KD (BEVFusion teacher) — mirrors E5 exactly
# ------------------------------------------------------------------------
teacher_cache_path = "data/cache/det/bevfusion_teacher_train.pkl"
distill_alpha_cls = 0.2
distill_alpha_reg = 0.4
distill_temperature = 4.0
distill_score_thr = 0.3
distill_last_layer_only = True
distill_mode = "pseudo_gt"
det_gt_loss_weight = 0.0

# ------------------------------------------------------------------------
# Map KD (MapTRv2 teacher) — mirrors hipad_nusc_stage1_map_distill_kdonly
# ------------------------------------------------------------------------
map_teacher_cache_path = "/data4/kyungmin/nuScenes_Converted/data/cache_map/maptrv2_teacher_train.pkl"
map_distill_alpha_cls = 0.1
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = True
map_distill_mode = "pseudo_gt"
map_gt_loss_weight = 0.0

# ------------------------------------------------------------------------
# Onedecoder_head overrides
# ------------------------------------------------------------------------
model = dict(
    head=dict(
        onedecoder_head=dict(
            # det distill
            distill_alpha_cls=distill_alpha_cls,
            distill_alpha_reg=distill_alpha_reg,
            distill_temperature=distill_temperature,
            distill_score_thr=distill_score_thr,
            distill_last_layer_only=distill_last_layer_only,
            distill_mode=distill_mode,
            det_gt_loss_weight=det_gt_loss_weight,
            # map distill
            map_distill_alpha_cls=map_distill_alpha_cls,
            map_distill_alpha_reg=map_distill_alpha_reg,
            map_distill_temperature=map_distill_temperature,
            map_distill_score_thr=map_distill_score_thr,
            map_distill_dist_thr=map_distill_dist_thr,
            map_distill_last_layer_only=map_distill_last_layer_only,
            map_distill_mode=map_distill_mode,
            map_gt_loss_weight=map_gt_loss_weight,
        ),
    ),
)

# ------------------------------------------------------------------------
# Train pipeline: stage1 base inline + Collect.keys appends both
# det teacher (teacher_logits/boxes/scores) and map teacher (teacher_map_*).
# meta_keys also adds "token" (E5 pattern) so downstream cache lookups can
# pull from img_meta if needed.
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
    dict(type="MultiScaleDepthMapGenerator", downsample=[4, 8, 16]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * 10),
    dict(
        type="InstanceNameFilter",
        classes=[
            "car", "truck", "construction_vehicle", "bus", "trailer",
            "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
        ],
    ),
    dict(
        type="VectorizeMap",
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
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
            # Det teacher cache tensors
            "teacher_logits",
            "teacher_boxes",
            "teacher_scores",
            # Map teacher cache tensors
            "teacher_map_logits",
            "teacher_map_pts",
            "teacher_map_scores",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "token"],
    ),
]

# ------------------------------------------------------------------------
# Wire both cache paths + new train_pipeline into data.train. val/test untouched.
# ------------------------------------------------------------------------
data = dict(
    train=dict(
        pipeline=train_pipeline,
        teacher_cache_path=teacher_cache_path,
        map_teacher_cache_path=map_teacher_cache_path,
    ),
)
