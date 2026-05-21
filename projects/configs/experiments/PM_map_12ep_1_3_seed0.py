"""PM_map: map-only training.

Architecture reduced to map modules only (task_select=["map"], query_select=["map"]).
inter_graph_model = None. graph_model / temp_graph_model contain only the map group.

Loads PM_det checkpoint to inherit img_backbone / img_neck / depth_branch (other
keys like det_* will be unexpected and skipped). Backbone+neck+depth_branch are
frozen via paramwise_cfg lr_mult=0 + norm_eval=True so the shared encoder stays
consistent with PM_det.
"""
log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
work_dir = "work_dirs/exp/PM_map_12ep_1_3_seed0"

version = 'trainval'
length = {'trainval': 9360, 'mini': 323}

num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 18
checkpoint_epoch_interval = 3

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
wandb_project = "hipad"
wandb_name = "PM_map_12ep_1_3_seed0"
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
# Inherit shared encoder (backbone/neck/depth) from PM_det. Other keys are skipped.
load_from = "./work_dirs/exp/PM_det_12ep_1_3_seed0/latest.pth"
import os as _os
_resume = "./work_dirs/exp/PM_map_12ep_1_3_seed0/latest.pth"
resume_from = _resume if _os.path.exists(_resume) else None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
num_cams = 6

# det class names (kept only for dataset compatibility; det task not trained)
det_class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
num_det_classes = len(det_class_names)

map_class_names = ["ped_crossing", "divider", "boundary"]
num_map_classes = len(map_class_names)
map_roi_size = (30, 60)
map_num_pts = 20

# kept for dataset compatibility
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_cmd = 3
ego_fut_mode = 6
ego_status_dims = 6

embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
decouple_attn = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# temporal
temporal = True
temporal_map = True

# ====== PM_map: map only ======
task_config = dict(with_onedecoder=True)
task_select = ["map"]
query_select = ["map"]

single_frame_layer = ["concat", "gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

anchor_paths = {
    "map": "data/kmeans/kmeans_map_100.npy",
}


model = dict(
    type="SparseDetector",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=True,  # freeze BN running stats (backbone loaded from PM_det)
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=False),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=False),
        no_norm_on_lateral=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseHead",
        task_config=task_config,
        evaluate_bench2dive=False,
        onedecoder_head=dict(
            type="SparseOneDecoder",
            task_select=task_select,
            query_select=query_select,
            operation_order=operation_order,
            num_single_frame_decoder=num_single_frame_decoder,
            cls_threshold_to_reg=0.05,
            # ----- instance_bank (map only) -----
            map_instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor=anchor_paths["map"],
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            # ----- anchor encoder -----
            map_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                return_points_embed=True,
            ),
            # ----- operation (map-only, no decouple) -----
            custom_op=dict(type="CustomOperation"),
            temp_graph_model=dict(
                type="TemporalSeparateAttention",
                query_select=query_select,
                query_list=[["map"]],
                key_list=[["map"]],
                decouple_list=[False],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ) if temporal else None,
            graph_model=dict(
                type="SeparateAttention",
                query_select=query_select,
                separate_list=[["map"]],
                decouple_list=[False],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ),
            inter_graph_model=None,
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            # ----- deformable (map only) -----
            map_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=use_deformable_func,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=map_num_pts,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            # ----- refine (map only) -----
            map_refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                num_cls=num_map_classes,
            ),
            # ----- sampler / loss / decoder (map only) -----
            map_sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type="HungarianLinesAssigner",
                    cost=dict(
                        type="MapQueriesCost",
                        cls_cost=dict(type="FocalLossCost", weight=1.0),
                        reg_cost=dict(type="LinesL1Cost", weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes,
                num_sample=map_num_pts,
                roi_size=map_roi_size,
            ),
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(
                type="SparseLineLoss",
                loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                num_sample=map_num_pts,
                roi_size=map_roi_size,
            ),
            map_reg_weights=[1.0] * 40,
            map_decoder=dict(type="SparsePoint3DDecoder"),
        ),
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"
data_root = "data/nuscenes/"
eval_data_root = "data/infos/nuscenes/"
anno_root = "data/infos/" if version == 'trainval' else "data/infos/mini/"
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
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(
        type="VectorizeMap",
        roi_size=map_roi_size,
        simplify=False,
        normalize=False,
        sample_num=map_num_pts,
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
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type="Collect",
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            "ego_status",
            "gt_ego_fut_cmd",
        ],
        meta_keys=["T_global", "T_global_inv", "timestamp"],
    ),
]

eval_pipeline = [
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(
        type="VectorizeMap",
        roi_size=map_roi_size,
        simplify=True,
        normalize=False,
    ),
    dict(
        type="Collect",
        keys=[
            "vectors",
            "gt_bboxes_3d",
            "gt_labels_3d",
            "gt_agent_fut_trajs",
            "gt_agent_fut_masks",
            "gt_ego_fut_trajs",
            "gt_ego_fut_masks",
            "gt_ego_fut_cmd",
            "fut_boxes",
        ],
        meta_keys=["token", "timestamp"],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

nusc_version = "v1.0-trainval" if version == "trainval" else "v1.0-mini"
data_basic_config = dict(
    type=dataset_type,
    data_root=data_root,
    classes=det_class_names,
    map_classes=map_class_names,
    ego_status_dims=ego_status_dims,
    modality=input_modality,
    version=nusc_version,
    work_dir=work_dir,
)

eval_config = dict(
    **data_basic_config,
    eval_data_root=eval_data_root,
    ann_file=anno_root + "nuscenes_infos_val.pkl",
    pipeline=eval_pipeline,
    test_mode=True,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
    "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train_1_3_seed0.pkl",
        pipeline=train_pipeline,
        test_mode=False,
        data_aug_conf=data_aug_conf,
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        data_aug_conf=data_aug_conf,
        test_mode=True,
        eval_config=eval_config,
    ),
)

# ================== training ========================
# Freeze shared encoder (loaded from PM_det) via lr_mult=0.
# norm_eval=True on img_backbone above also keeps BN running stats frozen.
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.0),
            "img_neck": dict(lr_mult=0.0),
            "depth_branch": dict(lr_mult=0.0),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(
    type="IterBasedRunner",
    max_iters=num_iters_per_epoch * num_epochs,
)

# ================== eval ========================
eval_mode = dict(
    with_det=False,
    with_tracking=False,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch * 3,
    jsonfile_prefix="val/",
    eval_mode=eval_mode,
    out_dir="val_vis",
)

custom_hooks = [
    dict(
        type="WandbValVisHook",
        vis_dir="val_vis/visual",
        max_images=8,
        interval=1,
        priority="LOWEST",
    )
]
