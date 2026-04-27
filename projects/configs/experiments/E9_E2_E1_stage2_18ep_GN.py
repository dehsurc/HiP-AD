# ──────────────────────────────────────────────────────────────
# E9: Stage2 training with GradNorm (ICML 2018) on E1 ckpt.
# Base: E2_E1_stage2_18ep.py
# Delta: `model.gradnorm` added; custom_hooks adds GradNormMetadataHook.
# Tasks (5): det, map, motion, plan, ego   (depth kept as fixed auxiliary)
# Shared W: last Linear.weight of every AsymmetricFFN in SparseOneDecoder
#           (6 tensors: 1 single-frame + 5 temporal)
# α=1.5, lr_w=2.5e-2, warmup=500 iter, pivot N=50 iter, update_every=1
# Data: nuscenes_infos_train_1_3_seed0.pkl (1/3 seed0 split)
# Load: ckpts/nusc_stage1_e1.pth
# See: docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md
# ──────────────────────────────────────────────────────────────

log_level = "INFO"
dist_params = dict(backend="nccl")

# DDP flags:
#   - find_unused_parameters=True: the last decoder layer's inter_gnn
#     FlashMHA has an attn.in_proj_bias that reaches the loss only on some
#     samples (sample-dependent Gain/gating path). This makes the DDP graph
#     dynamic, so static_graph=True cannot be used.
# `mmdet_train.py` monkey-patches `torch.utils.checkpoint.checkpoint` to
# run non-reentrant, which is compatible with FUP=True. This lets us keep
# `with_cp=True` on the backbone (below) for activation memory savings.
find_unused_parameters = True

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
work_dir = "work_dirs/exp/E9_E2_E1_stage2_18ep_GN"

version = 'trainval'
# 1/3 stratified-by-scene subset (seed 0). Full trainval = 28130 infos; subset = 9360.
# Matched to E2_E1_stage2_18ep_1_3_seed0 baseline so iter/LR schedule align.
length = {'trainval': 9360, 'mini': 323}

num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 18
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
wandb_project = "hipad"
wandb_name = "E9_E2_E1_stage2_18ep_GN"
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
load_from = "/home/yongjae/e2e/HiP-AD/ckpts/nusc_stage1_e1.pth"
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
num_cams = 6

# det & map
det_class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
map_class_names = ["ped_crossing", "divider", "boundary"]

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
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
temporal_map = True
temporal_ego = True
temporal_plan = True

# tasks
task_config = dict(with_onedecoder=True)

task_select = ["det", "map", "plan", "ego", "motion"]
query_select = ["det", "map", "plan", "ego"]

single_frame_layer = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable",
                      "concat", "ffn", "norm", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split",
                        "deformable", "concat", "ffn", "norm", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# anchors
anchor_paths = {
    "det": "data/kmeans/kmeans_det_900.npy",
    "map": "data/kmeans/kmeans_map_100.npy",
    "motion": f"data/kmeans/kmeans_motion_{fut_mode}.npy",
}

plan_anchor_paths = f"data/kmeans/kmeans_plan_{ego_fut_mode}.npy"
plan_speed_refer = None
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


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
            plan_speed_refer=plan_speed_refer,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=True,
            with_target_point_embed=False,
            with_supervise_ego_status=True,
            num_command=ego_fut_cmd,
            with_ego_instance_feature=True,
            with_incremental_plan_refine=True,
            motion_anchor=anchor_paths["motion"],
            cls_threshold_to_reg=0.05,
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
            ego_instance_bank=dict(
                type="EgoInstanceBank",
                anchor_type="nus",
                embed_dims=embed_dims,
                num_temp_instances=1 if temporal_ego else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
                plan_anchor=plan_anchor_paths,
            ),
            plan_instance_bank=dict(
                type="PlanningInstanceBank",
                embed_dims=embed_dims,
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode,
                num_temp_mode=ego_fut_mode if temporal_plan else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
                anchor_paths=plan_anchor_paths,
                anchor_types=plan_anchor_types,
            ),
            det_anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            map_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                return_points_embed=True,
            ),
            plan_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=ego_fut_ts,
                return_points_embed=True,
            ),
            custom_op=dict(type="CustomOperation"),
            temp_graph_model=dict(
                type="TemporalSeparateAttention",
                query_select=query_select,
                query_list=[["det"], ["map"], ["plan", "ego"]],
                key_list=[["det"], ["map"], ["det", "map"]],
                decouple_list=[True, False, False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ) if temporal else None,
            graph_model=dict(
                type="SeparateAttention",
                query_select=query_select,
                separate_list=[["det"], ["map"]],
                decouple_list=[True, False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ),
            inter_graph_model=dict(
                type="InteractiveAttention",
                query_select=query_select,
                query_list=[["plan", "ego"]],
                key_list=[["det", "map"]],
                decouple_list=[False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ),
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
            det_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0],
                               [0, 0, 0.45], [0, 0, -0.45]],
                ),
            ),
            map_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=map_num_pts,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            ego_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=12, fix_scale=[[0.45, 0, 0]],
                ),
            ),
            plan_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=ego_fut_ts,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            det_refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims, num_cls=num_det_classes,
                refine_yaw=True, with_quality_estimation=True,
            ),
            map_refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims, num_sample=map_num_pts, num_cls=num_map_classes,
            ),
            ego_refine_layer=dict(
                type="EgoStatusRefinementModule",
                embed_dims=embed_dims, status_dims=ego_status_dims,
            ),
            plan_refine_layer=dict(
                type="SparsePlanAlignRefinementModule",
                embed_dims=embed_dims, ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
                anchor_types=plan_anchor_types,
            ),
            motion_refine_layer=dict(
                type="SparseMotionRefinementModule",
                embed_dims=embed_dims, fut_ts=fut_ts, fut_mode=fut_mode,
            ),
            det_sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0, num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32, add_neg_dn=True,
                cls_weight=2.0, box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    det_class_names.index("traffic_cone"):
                        [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
            ),
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
                num_cls=num_map_classes, num_sample=map_num_pts, roi_size=map_roi_size,
            ),
            plan_sampler=dict(
                type="SparsePlanTarget",
                ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
            ),
            align_sampler=dict(
                type="AlignPlanTarget",
                ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(type="SparseMotionTarget"),
            loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_det_reg=dict(type="SparseBox3DLoss",
                              loss_box=dict(type="L1Loss", loss_weight=0.25),
                              loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                              loss_yawness=dict(type="GaussianFocalLoss")),
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts, roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=1.0),
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
            loss_plan_reg=dict(type="L1Loss", loss_weight=1.0),
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.2),
            det_reg_weights=[2.0] * 3 + [1.0] * 7,
            map_reg_weights=[1.0] * 40,
            det_decoder=dict(type="SparseBox3DDecoder"),
            map_decoder=dict(type="SparsePoint3DDecoder"),
            plan_decoder=dict(type="SparsePlanDecoder",
                              ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd,
                              ego_fut_mode=ego_fut_mode, ego_vehicle="nus",
                              anchor_types=plan_anchor_types, anchor_refer=plan_anchor_refer,
                              speed_refer=plan_speed_refer, with_rescore=True),
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),

    # ── GradNorm (NEW) ──
    gradnorm=dict(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[4.25, 11.0, 0.4, 1.5, 1.0],
        alpha=1.5,
        lr=2.5e-2,
        update_after_step=500,
        pivot_warmup_steps=50,
        update_every=1,
        clamp_min=1e-4,
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
    dict(type="LoadPointsFromFile", coord_type="LIDAR",
         load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False,
         normalize=False, sample_num=map_num_pts, permute=True),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth",
               "focal", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels",
               "gt_map_pts", "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd",
               "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "ego_status", "ego_status_mask"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"]),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh",
               "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp"]),
]

eval_pipeline = [
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=True, normalize=False),
    dict(type="Collect",
         keys=["vectors", "gt_bboxes_3d", "gt_labels_3d",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_fut_trajs", "gt_ego_fut_masks",
               "gt_ego_fut_cmd", "fut_boxes"],
         meta_keys=["token", "timestamp"]),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=False,
)

nusc_version = "v1.0-trainval" if version == "trainval" else "v1.0-mini"
data_basic_config = dict(
    type=dataset_type, data_root=data_root, classes=det_class_names,
    map_classes=map_class_names, ego_status_dims=ego_status_dims,
    modality=input_modality, version=nusc_version, work_dir=work_dir,
)

eval_config = dict(
    **data_basic_config, eval_data_root=eval_data_root,
    ann_file=anno_root + "nuscenes_infos_val.pkl",
    pipeline=eval_pipeline, test_mode=True,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600, "rand_flip": True, "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train_1_3_seed0.pkl",
        pipeline=train_pipeline, test_mode=False,
        data_aug_conf=data_aug_conf, with_seq_flag=True,
        sequences_split_num=2, keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline, data_aug_conf=data_aug_conf,
        test_mode=True, eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline, data_aug_conf=data_aug_conf,
        test_mode=True, eval_config=eval_config,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW", lr=1e-4, weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing", warmup="linear",
    warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3,
)
runner = dict(type="IterBasedRunner",
              max_iters=num_iters_per_epoch * num_epochs)

# ================== eval ========================
eval_mode = dict(
    with_det=True, with_tracking=False, with_map=True,
    with_motion=True, with_planning=True,
    tracking_threshold=0.2, motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval * 2,
    jsonfile_prefix="val/", eval_mode=eval_mode, out_dir="val_vis",
)

custom_hooks = [
    dict(type="WandbValVisHook",
         vis_dir="val_vis/visual", max_images=8, interval=1, priority="LOWEST"),
    dict(type="GradNormMetadataHook", priority="LOWEST"),
]
