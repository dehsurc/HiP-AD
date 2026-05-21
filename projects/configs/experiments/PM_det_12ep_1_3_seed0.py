"""PM_det: detection-only training.

Architecture reduced to det modules only (task_select=["det"], query_select=["det"]).
inter_graph_model = None (no plan/ego cross-attention).
graph_model / temp_graph_model contain only the det group.

Starts from ImageNet ResNet50 (no load_from). All trainable.
"""
log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
work_dir = "work_dirs/exp/PM_det_12ep_1_3_seed0"

version = 'trainval'
# 1/3 stratified-by-scene subset (seed 0).
length = {'trainval': 9360, 'mini': 323}

num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 12
checkpoint_epoch_interval = 3

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
wandb_project = "hipad"
wandb_name = "PM_det_12ep_1_3_seed0"
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
load_from = None
import os as _os
_resume = "./work_dirs/exp/PM_det_12ep_1_3_seed0/latest.pth"
resume_from = _resume if _os.path.exists(_resume) else None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
num_cams = 6

# det class names
det_class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer",
    "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
num_det_classes = len(det_class_names)

# kept only because data pipeline still vectorizes them; not used by model
map_class_names = ["ped_crossing", "divider", "boundary"]
num_map_classes = len(map_class_names)
map_roi_size = (30, 60)
map_num_pts = 20

# kept for data pipeline compatibility (gt collected but unused by model)
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_cmd = 3
ego_fut_mode = 6
ego_status_dims = 6

# model
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
temporal_det = True

# ====== PM_det: detection only ======
task_config = dict(with_onedecoder=True)
task_select = ["det"]
query_select = ["det"]

# inter_gnn removed from op order (no plan/ego queries)
single_frame_layer = ["concat", "gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "norm", "split", "deformable", "concat", "ffn", "norm", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

anchor_paths = {
    "det": "data/kmeans/kmeans_det_900.npy",
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
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=True),
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
            # ----- instance_bank (det only) -----
            det_instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=anchor_paths["det"],
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal_det else -1,
                confidence_decay=0.6,
                feat_grad=False,
                class_names=det_class_names,
                zero_velocity_classes=["barrier", "traffic_cone"],
            ),
            # ----- anchor encoder -----
            det_anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            # ----- operation (det-only) -----
            custom_op=dict(type="CustomOperation"),
            temp_graph_model=dict(
                type="TemporalSeparateAttention",
                query_select=query_select,
                query_list=[["det"]],
                key_list=[["det"]],
                decouple_list=[True],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims * 2,
                        num_heads=num_groups,
                        batch_first=True,
                        dropout=drop_out,
                    ),
                ],
            ) if temporal else None,
            graph_model=dict(
                type="SeparateAttention",
                query_select=query_select,
                separate_list=[["det"]],
                decouple_list=[True],
                attn=[
                    dict(
                        type="MultiheadFlashAttention",
                        embed_dims=embed_dims * 2,
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
            # ----- deformable (det only) -----
            det_deformable=dict(
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
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[
                        [0, 0, 0],
                        [0.45, 0, 0],
                        [-0.45, 0, 0],
                        [0, 0.45, 0],
                        [0, -0.45, 0],
                        [0, 0, 0.45],
                        [0, 0, -0.45],
                    ],
                ),
            ),
            # ----- refine (det only) -----
            det_refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims,
                num_cls=num_det_classes,
                refine_yaw=True,
                with_quality_estimation=True,
            ),
            # ----- sampler / loss / decoder (det only) -----
            det_sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    det_class_names.index("traffic_cone"): [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
            ),
            loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_det_reg=dict(
                type="SparseBox3DLoss",
                loss_box=dict(type="L1Loss", loss_weight=0.25),
                loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                loss_yawness=dict(type="GaussianFocalLoss"),
            ),
            det_reg_weights=[2.0] * 3 + [1.0] * 7,
            det_decoder=dict(type="SparseBox3DDecoder"),
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
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
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
    with_det=True,
    with_tracking=False,
    with_map=False,
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
