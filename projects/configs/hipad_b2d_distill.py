_base_ = ["hipad_b2d_stage2.py"]

# ============================================================
# BEVFusion -> HiP-AD Knowledge Distillation Config
# ============================================================
# Prerequisites:
#   1. Run tools/cache_teacher_outputs.py in bevfusion env to generate cache
#   2. Set teacher_cache_path below to the generated cache file
#   3. Set load_from to pre-trained HiP-AD checkpoint
# ============================================================

teacher_cache_path = "/home/yongjae/e2e/HiP-AD/data/teacher_cache/teacher_cache.pkl"

# ---- Training settings for distillation fine-tuning ----
num_gpus = 4
batch_size = 4
num_epochs = 6
num_iters_per_epoch = int(234769 // (num_gpus * batch_size))
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=3)

optimizer = dict(
    type="AdamW",
    lr=5e-5,
    weight_decay=0.01,
)

load_from = None  # TODO: set to pre-trained HiP-AD stage2 checkpoint

# ---- Distillation module ----
model = dict(
    distillation=dict(
        teacher=dict(
            use_cache=True,
        ),
        loss=dict(
            cls_kd_loss_weight=1.0,
            reg_kd_loss_weight=0.5,
            heatmap_kd_loss_weight=0.5,
            temperature=2.0,
            num_classes=9,
        ),
        aux_bev_head=dict(
            embed_dims=256,
            num_classes=9,
            bev_h=128,
            bev_w=128,
            pc_range=(-51.2, -51.2, 51.2, 51.2),
            mid_channels=128,
        ),
    ),
)

# ---- Data pipeline: add teacher cache loading + teacher keys ----
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

det_class_names = [
    "car", "van", "truck", "bicycle", "traffic_sign",
    "traffic_cone", "traffic_light", "pedestrian", "others",
]
strides = [4, 8, 16, 32]
num_depth_layers = 3
map_roi_size = (30, 60)
map_num_pts = 20

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="B2DLoadPointsFromFile"),
    dict(type="LoadTeacherCache", cache_path=teacher_cache_path),
    dict(type="ResizeCropFlipImage"),
    dict(type="B2DMultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizePloyLine",
         roi_size=map_roi_size,
         simplify=False,
         normalize=False,
         sample_num=map_num_pts,
         permute=True,
    ),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth", "focal",
               "gt_ego_fut_cmd", "target_point", "ego_status", "ego_status_mask",
               "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels", "gt_map_pts",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_spat_trajs_2m", "gt_ego_spat_masks_2m",
               "gt_ego_spat_trajs_5m", "gt_ego_spat_masks_5m",
               "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "gt_ego_fut_trajs_5hz", "gt_ego_fut_masks_5hz",
               # Teacher cache keys
               "teacher_dense_heatmap",
               "teacher_heatmap",
               "teacher_center",
               "teacher_height",
               "teacher_dim",
               "teacher_rot",
               "teacher_vel",
               ],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "scene_token"],
    ),
]
