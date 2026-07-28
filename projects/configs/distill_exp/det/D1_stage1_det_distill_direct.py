# GT-free det distillation, stage1 from scratch.
#
# Inherits hipad_nusc_stage1.py and:
#   - disables real det GT loss (det_gt_loss_weight=0)
#   - adds direct teacher<->student Hungarian matching (distill_mode="direct")
#   - injects teacher_{logits,boxes,scores} via train_pipeline Collect
#   - wires bevfusion teacher cache into data.train
#
# Mirrors hipad_nusc_stage1_map_distill_kdonly.py inheritance pattern (det side).

_base_ = ["../../hipad_nusc_stage1.py"]

wandb_project = "nusc_det_distill"
wandb_name = "D1_stage1_det_distill_direct"
work_dir = "work_dirs/exp/D1_stage1_det_distill_direct"

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

# BEVFusion teacher cache (per-token; class permuted to HiP-AD order)
teacher_cache_path = "data/cache/det/bevfusion_teacher_train.pkl"

# Direct GT-free matching: alphas only gate use_distill, real magnitude comes
# from loss_det_cls/reg's own loss_weight. Kept at 0.2/0.4 for consistency
# with the distill_wo_det series.
distill_alpha_cls = 0.2
distill_alpha_reg = 0.4
distill_temperature = 4.0
distill_score_thr = 0.1
distill_last_layer_only = True
distill_mode = "direct"
det_gt_loss_weight = 0.0

model = dict(
    head=dict(
        onedecoder_head=dict(
            distill_alpha_cls=distill_alpha_cls,
            distill_alpha_reg=distill_alpha_reg,
            distill_temperature=distill_temperature,
            distill_score_thr=distill_score_thr,
            distill_last_layer_only=distill_last_layer_only,
            distill_mode=distill_mode,
            det_gt_loss_weight=det_gt_loss_weight,
        ),
    ),
)

# Redefine train_pipeline (mmcv replaces lists wholesale) to add teacher_* keys.
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
            # Det distillation tensors (added vs base)
            "teacher_logits",
            "teacher_boxes",
            "teacher_scores",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "token"],
    ),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        teacher_cache_path=teacher_cache_path,
    ),
)
