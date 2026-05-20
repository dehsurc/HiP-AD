log_level = 'INFO'
dist_params = dict(backend='nccl')
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
work_dir = 'work_dirs/exp/Mask2map_110ep_map_distill_stage1'
version = 'trainval'
length = dict(trainval=28130, mini=323)
num_gpus = 2
batch_size = 8
num_iters_per_epoch = 1758
num_epochs = 12
checkpoint_epoch_interval = 3
checkpoint_config = dict(interval=1758, max_keep_ckpts=-1)
wandb_project = 'hipad'
wandb_name = 'E9_stage1_12ep_distill_mask2map'
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='e2ekd',
                project='hipad',
                name='E9_stage1_12ep_distill_mask2map'),
            by_epoch=False)
    ])
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
num_cams = 6
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
map_class_names = ['ped_crossing', 'divider', 'boundary']
num_det_classes = 10
num_map_classes = 3
map_roi_size = (30, 60)
map_num_pts = 20
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
num_levels = 4
num_depth_layers = 3
drop_out = 0.1
decouple_attn = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True
task_config = dict(with_onedecoder=True)
task_select = ['det', 'map', 'plan', 'ego']
query_select = ['det', 'map', 'plan', 'ego']
single_frame_layer = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat',
    'ffn', 'norm', 'split', 'refine'
]
temporal_frame_layer = [
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine'
]
operation_order = [
    'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable', 'concat',
    'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn', 'gnn', 'inter_gnn',
    'norm', 'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine', 'concat', 'temp_gnn', 'gnn',
    'inter_gnn', 'norm', 'split', 'deformable', 'concat', 'ffn', 'norm',
    'split', 'refine', 'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm',
    'split', 'deformable', 'concat', 'ffn', 'norm', 'split', 'refine',
    'concat', 'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
    'concat', 'ffn', 'norm', 'split', 'refine'
]
anchor_paths = dict(
    det='data/kmeans/kmeans_det_900.npy',
    map='data/kmeans/kmeans_map_100.npy',
    motion='data/kmeans/kmeans_motion_6.npy')
plan_anchor_paths = 'data/kmeans/kmeans_plan_6.npy'
plan_speed_refer = None
plan_anchor_refer = ('temp', '2hz')
plan_anchor_types = [('temp', '2hz')]
map_teacher_cache_path = 'data/cache/map/mask2map_110ep_phase2_train.pkl'
map_distill_alpha_cls = 0.4
map_distill_alpha_reg = 0.2
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_last_layer_only = True
# Real-GT + teacher pseudo-GT map distillation. The pseudo-GT map path
# converts teacher logits to hard labels and provides both original/reversed
# polyline directions before Hungarian assignment.
map_gt_loss_weight = 0.0
map_distill_mode = 'pseudo_gt'
model = dict(
    type='SparseDetector',
    use_grid_mask=True,
    use_deformable_func=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style='pytorch',
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type='BN', requires_grad=True),
        pretrained='ckpts/resnet50-19c8e357.pth'),
    img_neck=dict(
        type='FPN',
        num_outs=4,
        start_level=0,
        out_channels=256,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        no_norm_on_lateral=True,
        in_channels=[256, 512, 1024, 2048]),
    depth_branch=dict(
        type='DenseDepthNet',
        embed_dims=256,
        num_depth_layers=3,
        loss_weight=0.2),
    head=dict(
        type='SparseHead',
        task_config=dict(with_onedecoder=True),
        evaluate_bench2dive=False,
        onedecoder_head=dict(
            type='SparseOneDecoder',
            task_select=['det', 'map', 'plan', 'ego'],
            query_select=['det', 'map', 'plan', 'ego'],
            operation_order=[
                'concat', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine', 'concat',
                'temp_gnn', 'gnn', 'inter_gnn', 'norm', 'split', 'deformable',
                'concat', 'ffn', 'norm', 'split', 'refine'
            ],
            num_single_frame_decoder=1,
            plan_speed_refer=None,
            plan_anchor_refer=('temp', '2hz'),
            with_command_embed=True,
            with_target_point_embed=False,
            with_supervise_ego_status=True,
            num_command=3,
            with_ego_instance_feature=True,
            with_incremental_plan_refine=True,
            motion_anchor='data/kmeans/kmeans_motion_6.npy',
            cls_threshold_to_reg=0.05,
            map_distill_alpha_cls=0.4,
            map_distill_alpha_reg=0.2,
            map_distill_temperature=4.0,
            map_distill_score_thr=0.3,
            map_distill_last_layer_only=True,
            map_gt_loss_weight=map_gt_loss_weight,
            map_distill_mode=map_distill_mode,
            det_instance_bank=dict(
                type='InstanceBank',
                num_anchor=900,
                embed_dims=256,
                anchor='data/kmeans/kmeans_det_900.npy',
                anchor_handler=dict(type='SparseBox3DKeyPointsGenerator'),
                num_temp_instances=600,
                confidence_decay=0.6,
                feat_grad=False,
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                zero_velocity_classes=['barrier', 'traffic_cone']),
            map_instance_bank=dict(
                type='InstanceBank',
                num_anchor=100,
                embed_dims=256,
                anchor='data/kmeans/kmeans_map_100.npy',
                anchor_handler=dict(type='SparsePoint3DKeyPointsGenerator'),
                num_temp_instances=0,
                confidence_decay=0.6,
                feat_grad=True),
            ego_instance_bank=dict(
                type='EgoInstanceBank',
                anchor_type='nus',
                embed_dims=256,
                num_temp_instances=1,
                feature_map_scale=(8.0, 22.0),
                plan_anchor='data/kmeans/kmeans_plan_6.npy'),
            plan_instance_bank=dict(
                type='PlanningInstanceBank',
                embed_dims=256,
                ego_fut_ts=6,
                ego_fut_cmd=3,
                ego_fut_mode=6,
                num_temp_mode=6,
                feature_map_scale=(8.0, 22.0),
                anchor_paths='data/kmeans/kmeans_plan_6.npy',
                anchor_types=[('temp', '2hz')]),
            det_anchor_encoder=dict(
                type='SparseBox3DEncoder',
                vel_dims=3,
                embed_dims=[128, 32, 32, 64],
                mode='cat',
                output_fc=False,
                in_loops=1,
                out_loops=4),
            map_anchor_encoder=dict(
                type='SparsePoint3DEncoder',
                embed_dims=256,
                num_sample=20,
                return_points_embed=True),
            plan_anchor_encoder=dict(
                type='SparsePoint3DEncoder',
                embed_dims=256,
                num_sample=6,
                return_points_embed=True),
            custom_op=dict(type='CustomOperation'),
            temp_graph_model=dict(
                type='TemporalSeparateAttention',
                query_select=['det', 'map', 'plan', 'ego'],
                query_list=[['det'], ['map'], ['plan', 'ego']],
                key_list=[['det'], ['map'], ['det', 'map']],
                decouple_list=[True, False, False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=512,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            graph_model=dict(
                type='SeparateAttention',
                query_select=['det', 'map', 'plan', 'ego'],
                separate_list=[['det'], ['map']],
                decouple_list=[True, False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=512,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1),
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            inter_graph_model=dict(
                type='InteractiveAttention',
                query_select=['det', 'map', 'plan', 'ego'],
                query_list=[['plan', 'ego']],
                key_list=[['det', 'map']],
                decouple_list=[False],
                attn=[
                    dict(
                        type='MultiheadFlashAttention',
                        embed_dims=256,
                        num_heads=8,
                        batch_first=True,
                        dropout=0.1)
                ]),
            norm_layer=dict(type='LN', normalized_shape=256),
            ffn=dict(
                type='AsymmetricFFN',
                in_channels=512,
                pre_norm=dict(type='LN'),
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            det_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0], [0, 0, 0.45],
                               [0, 0, -0.45]])),
            map_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=20,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            ego_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparseBox3DKeyPointsGenerator',
                    num_learnable_pts=12,
                    fix_scale=[[0.45, 0, 0]])),
            plan_deformable=dict(
                type='DeformableFeatureAggregation',
                embed_dims=256,
                num_groups=8,
                num_levels=4,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode='cat',
                kps_generator=dict(
                    type='SparsePoint3DKeyPointsGenerator',
                    embed_dims=256,
                    num_sample=6,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023)),
            det_refine_layer=dict(
                type='SparseBox3DRefinementModule',
                embed_dims=256,
                num_cls=10,
                refine_yaw=True,
                with_quality_estimation=True),
            map_refine_layer=dict(
                type='SparsePoint3DRefinementModule',
                embed_dims=256,
                num_sample=20,
                num_cls=3),
            ego_refine_layer=dict(
                type='EgoStatusRefinementModule',
                embed_dims=256,
                status_dims=6),
            plan_refine_layer=dict(
                type='SparsePlanAlignRefinementModule',
                embed_dims=256,
                ego_fut_ts=6,
                ego_fut_cmd=3,
                ego_fut_mode=6,
                anchor_types=[('temp', '2hz')]),
            motion_refine_layer=dict(
                type='SparseMotionRefinementModule',
                embed_dims=256,
                fut_ts=12,
                fut_mode=6),
            det_sampler=dict(
                type='SparseBox3DTarget',
                num_dn_groups=0,
                num_temp_dn_groups=0,
                dn_noise_scale=[
                    2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
                ],
                max_dn_gt=32,
                add_neg_dn=True,
                cls_weight=2.0,
                box_weight=0.25,
                reg_weights=[2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
                cls_wise_reg_weights=dict(
                    {9: [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]})),
            map_sampler=dict(
                type='SparsePoint3DTarget',
                assigner=dict(
                    type='HungarianLinesAssigner',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=1.0),
                        reg_cost=dict(
                            type='LinesL1Cost',
                            weight=10.0,
                            beta=0.01,
                            permute=True))),
                num_cls=3,
                num_sample=20,
                roi_size=(30, 60)),
            plan_sampler=dict(
                type='SparsePlanTarget',
                ego_fut_ts=6,
                ego_fut_cmd=3,
                ego_fut_mode=6),
            align_sampler=dict(
                type='AlignPlanTarget',
                ego_fut_ts=6,
                ego_fut_cmd=3,
                ego_fut_mode=6),
            motion_sampler=dict(type='SparseMotionTarget'),
            loss_det_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            loss_det_reg=dict(
                type='SparseBox3DLoss',
                loss_box=dict(type='L1Loss', loss_weight=0.25),
                loss_centerness=dict(
                    type='CrossEntropyLoss', use_sigmoid=True),
                loss_yawness=dict(type='GaussianFocalLoss')),
            loss_map_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_map_reg=dict(
                type='SparseLineLoss',
                loss_line=dict(
                    type='LinesL1Loss', loss_weight=10.0, beta=0.01),
                num_sample=20,
                roi_size=(30, 60)),
            loss_ego_status=dict(type='L1Loss', loss_weight=0.0),
            loss_plan_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.0),
            loss_plan_reg=dict(type='L1Loss', loss_weight=0.0),
            loss_motion_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.2),
            loss_motion_reg=dict(type='L1Loss', loss_weight=0.2),
            det_reg_weights=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            map_reg_weights=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0
            ],
            det_decoder=dict(type='SparseBox3DDecoder'),
            map_decoder=dict(type='SparsePoint3DDecoder'),
            plan_decoder=dict(
                type='SparsePlanDecoder',
                ego_fut_ts=6,
                ego_fut_cmd=3,
                ego_fut_mode=6,
                ego_vehicle='nus',
                anchor_types=[('temp', '2hz')],
                anchor_refer=('temp', '2hz'),
                speed_refer=None,
                with_rescore=True),
            motion_decoder=dict(type='SparseMotionDecoder'))))
dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
eval_data_root = 'data/infos/nuscenes/'
anno_root = 'data/infos/'
file_client_args = dict(backend='disk')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(type='ResizeCropFlipImage'),
    dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
    dict(type='BBoxRotation'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='VectorizeMap',
        roi_size=(30, 60),
        simplify=False,
        normalize=False,
        sample_num=20,
        permute=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth',
            'focal', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels',
            'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks',
            'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd',
            'gt_ego_fut_trajs_2hz', 'gt_ego_fut_masks_2hz', 'ego_status',
            'ego_status_mask', 'teacher_map_logits', 'teacher_map_pts',
            'teacher_map_scores'
        ],
        meta_keys=[
            'T_global', 'T_global_inv', 'timestamp', 'instance_id', 'token'
        ])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipImage'),
    dict(
        type='NormalizeMultiviewImage',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='NuScenesSparse4DAdaptor'),
    dict(
        type='Collect',
        keys=[
            'img', 'timestamp', 'projection_mat', 'image_wh', 'ego_status',
            'gt_ego_fut_cmd'
        ],
        meta_keys=['T_global', 'T_global_inv', 'timestamp'])
]
eval_pipeline = [
    dict(
        type='CircleObjectRangeFilter',
        class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
    dict(
        type='InstanceNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='VectorizeMap', roi_size=(30, 60), simplify=True,
        normalize=False),
    dict(
        type='Collect',
        keys=[
            'vectors', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_agent_fut_trajs',
            'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks',
            'gt_ego_fut_cmd', 'fut_boxes'
        ],
        meta_keys=['token', 'timestamp'])
]
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
nusc_version = 'v1.0-trainval'
data_basic_config = dict(
    type='NuScenes3DDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    ego_status_dims=6,
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval',
    work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1')
eval_config = dict(
    type='NuScenes3DDataset',
    data_root='data/nuscenes/',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    map_classes=['ped_crossing', 'divider', 'boundary'],
    ego_status_dims=6,
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    version='v1.0-trainval',
    work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
    eval_data_root='data/infos/nuscenes/',
    ann_file='data/infos/nuscenes_infos_val.pkl',
    pipeline=[
        dict(
            type='CircleObjectRangeFilter',
            class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
        dict(
            type='InstanceNameFilter',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ]),
        dict(
            type='VectorizeMap',
            roi_size=(30, 60),
            simplify=True,
            normalize=False),
        dict(
            type='Collect',
            keys=[
                'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'fut_boxes'
            ],
            meta_keys=['token', 'timestamp'])
    ],
    test_mode=True)
data_aug_conf = dict(
    resize_lim=(0.4, 0.47),
    final_dim=(256, 704),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(-5.4, 5.4),
    H=900,
    W=1600,
    rand_flip=True,
    rot3d_range=[0, 0])
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        ego_status_dims=6,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
        ann_file='data/infos/nuscenes_infos_train.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(type='ResizeCropFlipImage'),
            dict(type='MultiScaleDepthMapGenerator', downsample=[4, 8, 16]),
            dict(type='BBoxRotation'),
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(
                type='CircleObjectRangeFilter',
                class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
            dict(
                type='InstanceNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='VectorizeMap',
                roi_size=(30, 60),
                simplify=False,
                normalize=False,
                sample_num=20,
                permute=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs',
                    'gt_agent_fut_masks', 'gt_ego_fut_trajs',
                    'gt_ego_fut_masks', 'gt_ego_fut_cmd',
                    'gt_ego_fut_trajs_2hz', 'gt_ego_fut_masks_2hz',
                    'ego_status', 'ego_status_mask', 'teacher_map_logits',
                    'teacher_map_pts', 'teacher_map_scores'
                ],
                meta_keys=[
                    'T_global', 'T_global_inv', 'timestamp', 'instance_id',
                    'token'
                ])
        ],
        test_mode=False,
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        with_seq_flag=True,
        sequences_split_num=2,
        keep_consistent_seq_aug=True,
        map_teacher_cache_path='data/cache/map/mask2map_110ep_phase2_train.pkl'
    ),
    val=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        ego_status_dims=6,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
        ann_file='data/infos/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_fut_cmd'
                ],
                meta_keys=['T_global', 'T_global_inv', 'timestamp'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            ego_status_dims=6,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
            eval_data_root='data/infos/nuscenes/',
            ann_file='data/infos/nuscenes_infos_val.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)),
    test=dict(
        type='NuScenes3DDataset',
        data_root='data/nuscenes/',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        map_classes=['ped_crossing', 'divider', 'boundary'],
        ego_status_dims=6,
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        version='v1.0-trainval',
        work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
        ann_file='data/infos/nuscenes_infos_val.pkl',
        pipeline=[
            dict(type='LoadMultiViewImageFromFiles', to_float32=True),
            dict(type='ResizeCropFlipImage'),
            dict(
                type='NormalizeMultiviewImage',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='NuScenesSparse4DAdaptor'),
            dict(
                type='Collect',
                keys=[
                    'img', 'timestamp', 'projection_mat', 'image_wh',
                    'ego_status', 'gt_ego_fut_cmd'
                ],
                meta_keys=['T_global', 'T_global_inv', 'timestamp'])
        ],
        data_aug_conf=dict(
            resize_lim=(0.4, 0.47),
            final_dim=(256, 704),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(-5.4, 5.4),
            H=900,
            W=1600,
            rand_flip=True,
            rot3d_range=[0, 0]),
        test_mode=True,
        eval_config=dict(
            type='NuScenes3DDataset',
            data_root='data/nuscenes/',
            classes=[
                'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                'traffic_cone'
            ],
            map_classes=['ped_crossing', 'divider', 'boundary'],
            ego_status_dims=6,
            modality=dict(
                use_lidar=False,
                use_camera=True,
                use_radar=False,
                use_map=False,
                use_external=False),
            version='v1.0-trainval',
            work_dir='work_dirs/exp/Mask2map_110ep_map_distill_stage1',
            eval_data_root='data/infos/nuscenes/',
            ann_file='data/infos/nuscenes_infos_val.pkl',
            pipeline=[
                dict(
                    type='CircleObjectRangeFilter',
                    class_dist_thred=[55, 55, 55, 55, 55, 55, 55, 55, 55, 55]),
                dict(
                    type='InstanceNameFilter',
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ]),
                dict(
                    type='VectorizeMap',
                    roi_size=(30, 60),
                    simplify=True,
                    normalize=False),
                dict(
                    type='Collect',
                    keys=[
                        'vectors', 'gt_bboxes_3d', 'gt_labels_3d',
                        'gt_agent_fut_trajs', 'gt_agent_fut_masks',
                        'gt_ego_fut_trajs', 'gt_ego_fut_masks',
                        'gt_ego_fut_cmd', 'fut_boxes'
                    ],
                    meta_keys=['token', 'timestamp'])
            ],
            test_mode=True)))
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.001,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.5))))
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
runner = dict(type='IterBasedRunner', max_iters=21096)
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_map=True,
    with_motion=False,
    with_planning=False,
    tracking_threshold=0.2,
    motion_threshhold=0.2)
evaluation = dict(
    interval=5274,
    jsonfile_prefix='val/',
    eval_mode=dict(
        with_det=True,
        with_tracking=False,
        with_map=True,
        with_motion=False,
        with_planning=False,
        tracking_threshold=0.2,
        motion_threshhold=0.2))
gpu_ids = range(0, 2)
