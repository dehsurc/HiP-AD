# D4: KD on TOP of real GT for BOTH detection and map (additive / "w_gt").
#
# Companion to D2 (KD-only). Same teachers, same KD hyperparameters; the only
# differences are det_gt_loss_weight=1.0 and map_gt_loss_weight=1.0 so that real
# GT supervision remains fully active alongside the pseudo-GT KD branch.
#
#   - Detection KD: BEVFusion teacher cache, pseudo_gt mode → adds a separate
#     det_loss_kd_* branch through loss_det_cls / loss_det_reg in addition to
#     the standard real-GT det loss path. Mirrors E4_stage1_12ep_distill_w_det.
#   - Map KD:       MapTRv2 teacher cache, pseudo_gt mode → adds a separate
#     map_loss_kd_* branch through loss_map_cls / loss_map_reg in addition to
#     the real-GT map loss path. Mirrors hipad_nusc_stage1_map_distill (but in
#     pseudo_gt mode, matching D2's framing).
#
# In pseudo_gt mode `distill_alpha_*` / `map_distill_alpha_*` are only on/off
# flags (use_distill / use_map_distill). Effective KD magnitude is set by the
# standard loss heads (loss_det_cls=2.0, loss_det_reg.loss_box=0.25,
# loss_map_cls=1.0, loss_map_reg.loss_line=10.0).

_base_ = ["../hipad_nusc_stage1.py"]

# ------------------------------------------------------------------------
# Run identity
# ------------------------------------------------------------------------
wandb_project = "hipad_distill_dm"
wandb_name = "D4_stage1_12ep_distill_det_map_w_gt"
work_dir = "work_dirs/exp/D4_stage1_12ep_distill_det_map_w_gt"

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
# Detection KD (BEVFusion teacher) — mirrors D2; gt loss stays ON
# ------------------------------------------------------------------------
teacher_cache_path = "data/cache/det/bevfusion_teacher_train.pkl"
distill_alpha_cls = 0.2
distill_alpha_reg = 0.4
distill_temperature = 4.0
distill_score_thr = 0.3
distill_last_layer_only = True
distill_mode = "pseudo_gt"
det_gt_loss_weight = 1.0  # real det GT loss kept ON (vs D2's 0.0)

# ------------------------------------------------------------------------
# Map KD (MapTRv2 teacher) — mirrors D2; gt loss stays ON
# ------------------------------------------------------------------------
map_teacher_cache_path = "/data4/kyungmin/nuScenes_Converted/data/cache_map/maptrv2_teacher_train.pkl"
map_distill_alpha_cls = 0.1
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = True
map_distill_mode = "pseudo_gt"
map_gt_loss_weight = 1.0  # real map GT loss kept ON (vs D2's 0.0)

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
# Train pipeline: stage1 base inline + Collect.keys appends both teacher caches.
# Identical to D2 stage1 (only loss weights differ between D2 and D4).
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
