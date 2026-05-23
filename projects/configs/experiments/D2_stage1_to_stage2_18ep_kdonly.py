# D2 stage1 ckpt → stage2 18ep, KD-only CONTINUED (det/map real GT still OFF).
#
# Identical KD setting to D2 stage1 (BEVFusion det + MapTRv2 map, pseudo_gt,
# det_gt_loss_weight=0, map_gt_loss_weight=0). Motion + planning + ego tasks
# become active in stage2 (use real GT — those tasks have no teacher).
#
# Earlier D2_stage1_to_stage2_18ep.py reverted to real-GT for det/map by mistake.
# This config fixes that.

_base_ = ["../hipad_nusc_stage2.py"]

# ------------------------------------------------------------------------
# Run identity
# ------------------------------------------------------------------------
wandb_project = "hipad_distill_dm"
wandb_name = "D2_stage1_to_stage2_18ep_kdonly"
work_dir = "work_dirs/exp/D2_stage1_to_stage2_18ep_kdonly"

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
# Load D2 stage1 final ckpt (KD-only trained, 12 epoch = iter 21096)
# ------------------------------------------------------------------------
load_from = "/home/kyungmin/min_ws/rideflux/HiP-AD/work_dirs/exp/D2_stage1_12ep_distill_det_map_kdonly/iter_21096.pth"

# ------------------------------------------------------------------------
# Det KD (BEVFusion teacher) — same as D2 stage1
# ------------------------------------------------------------------------
teacher_cache_path = "data/cache/det/bevfusion_teacher_train.pkl"
distill_alpha_cls = 0.2
distill_alpha_reg = 0.4
distill_temperature = 4.0
distill_score_thr = 0.3
distill_last_layer_only = True
distill_mode = "pseudo_gt"
det_gt_loss_weight = 0.0  # real det GT stays OFF

# ------------------------------------------------------------------------
# Map KD (MapTRv2 teacher) — same as D2 stage1
# ------------------------------------------------------------------------
map_teacher_cache_path = "/data4/kyungmin/nuScenes_Converted/data/cache_map/maptrv2_teacher_train.pkl"
map_distill_alpha_cls = 0.1
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = True
map_distill_mode = "pseudo_gt"
map_gt_loss_weight = 0.0  # real map GT stays OFF

# ------------------------------------------------------------------------
# Wire KD args into onedecoder_head
# ------------------------------------------------------------------------
model = dict(
    head=dict(
        onedecoder_head=dict(
            # det KD
            distill_alpha_cls=distill_alpha_cls,
            distill_alpha_reg=distill_alpha_reg,
            distill_temperature=distill_temperature,
            distill_score_thr=distill_score_thr,
            distill_last_layer_only=distill_last_layer_only,
            distill_mode=distill_mode,
            det_gt_loss_weight=det_gt_loss_weight,
            # map KD
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
# train_pipeline: stage2 base pipeline + Collect.keys appends both teacher tensors.
# Inline (matches stage2 train_pipeline). Keep in sync if base pipeline changes.
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
# Wire teacher caches + new train_pipeline into data.train.
# val/test untouched (KD only at training time).
# ------------------------------------------------------------------------
data = dict(
    train=dict(
        pipeline=train_pipeline,
        teacher_cache_path=teacher_cache_path,
        map_teacher_cache_path=map_teacher_cache_path,
    ),
)
