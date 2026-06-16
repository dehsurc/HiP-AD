_base_ = ['./E2_E1_stage2_18ep.py']

work_dir = 'work_dirs/exp/E11_E10_stage2_18ep_gt_pgt_feat'
wandb_name = 'E11_E10_stage2_18ep_gt_pgt_feat'
load_from = '/home/chanyoung/RideFlux/HiP-AD/data/cache/map/iter_149430.pth'

log_config = dict(
    interval=50,
    hooks=[
        dict(
            type='ScientificTextLoggerHook',
            by_epoch=False,
            sci_keys=[
                'map_loss_kd_cls',
                'map_loss_kd_line',
                'map_loss_kd_feat',
            ],
            sci_prefixes=['map_loss_kd_feat_']),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                entity='e2ekd',
                project='hipad',
                name=wandb_name),
            by_epoch=False)
    ])

# E11 keeps the full stage2 task set from E2, but applies the current E10 map
# KD recipe: real map GT remains enabled, teacher map outputs are used as
# pseudo GT, and point-token feature distillation is kept as an auxiliary
# regularizer.
map_teacher_class_names = ['ped_crossing', 'divider', 'boundary']
map_teacher_to_student_class_perm = None
map_teacher_cache_path = 'data/cache/map/maptrv2_teacher_train_feat_top20_l01.pkl'

map_distill_mode = 'pseudo_gt'
map_distill_alpha_cls = 0.25
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = False
map_gt_loss_weight = 1.0

map_feature_distill_alpha = 1.0
map_feature_distill_layers = (0, 1)
map_feature_distill_weights = (0.5, 0.5)
map_feature_distill_cached_layers = (0, 1)
map_feature_distill_teacher_dim = 256
map_feature_distill_kd_dim = 256
map_feature_distill_num_classes = 3
map_feature_distill_cls_cost_weight = 0.0
map_feature_distill_line_cost_weight = 1.0
map_feature_distill_student_proj_depth = 1
map_feature_distill_detach_point_embed = True
map_feature_distill_use_point_tokens = True
map_feature_distill_point_tokens_as_main = True
map_feature_distill_point_aux_weight = 0.0
map_feature_distill_freeze_student_proj = False
map_feature_distill_teacher_identity = True
map_feature_distill_same_class_only = True
map_feature_match_mode = 'gt_anchor'

model = dict(
    head=dict(
        onedecoder_head=dict(
            map_distill_alpha_cls=map_distill_alpha_cls,
            map_distill_alpha_reg=map_distill_alpha_reg,
            map_distill_temperature=map_distill_temperature,
            map_distill_score_thr=map_distill_score_thr,
            map_distill_dist_thr=map_distill_dist_thr,
            map_distill_last_layer_only=map_distill_last_layer_only,
            map_distill_mode=map_distill_mode,
            map_gt_loss_weight=map_gt_loss_weight,
            map_feature_distill_alpha=map_feature_distill_alpha,
            map_feature_distill_layers=map_feature_distill_layers,
            map_feature_distill_weights=map_feature_distill_weights,
            map_feature_distill_cached_layers=map_feature_distill_cached_layers,
            map_feature_distill_teacher_dim=map_feature_distill_teacher_dim,
            map_feature_distill_kd_dim=map_feature_distill_kd_dim,
            map_feature_distill_num_classes=map_feature_distill_num_classes,
            map_feature_distill_cls_cost_weight=map_feature_distill_cls_cost_weight,
            map_feature_distill_line_cost_weight=map_feature_distill_line_cost_weight,
            map_feature_distill_student_proj_depth=map_feature_distill_student_proj_depth,
            map_feature_distill_detach_point_embed=map_feature_distill_detach_point_embed,
            map_feature_distill_use_point_tokens=map_feature_distill_use_point_tokens,
            map_feature_distill_point_tokens_as_main=map_feature_distill_point_tokens_as_main,
            map_feature_distill_point_aux_weight=map_feature_distill_point_aux_weight,
            map_feature_distill_freeze_student_proj=map_feature_distill_freeze_student_proj,
            map_feature_distill_teacher_identity=map_feature_distill_teacher_identity,
            map_feature_distill_same_class_only=map_feature_distill_same_class_only,
            map_feature_match_mode=map_feature_match_mode,
            map_teacher_to_student_class_perm=map_teacher_to_student_class_perm,
        )))

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
            'img',
            'timestamp',
            'projection_mat',
            'image_wh',
            'gt_depth',
            'focal',
            'gt_bboxes_3d',
            'gt_labels_3d',
            'gt_map_labels',
            'gt_map_pts',
            'gt_agent_fut_trajs',
            'gt_agent_fut_masks',
            'gt_ego_fut_trajs',
            'gt_ego_fut_masks',
            'gt_ego_fut_cmd',
            'gt_ego_fut_trajs_2hz',
            'gt_ego_fut_masks_2hz',
            'ego_status',
            'ego_status_mask',
            'teacher_map_logits',
            'teacher_map_pts',
            'teacher_map_scores',
            'teacher_map_features',
        ],
        meta_keys=['T_global', 'T_global_inv', 'timestamp', 'instance_id']),
]

eval_config = dict(work_dir=work_dir)

data = dict(
    train=dict(
        work_dir=work_dir,
        pipeline=train_pipeline,
        map_teacher_cache_path=map_teacher_cache_path,
        map_teacher_num_layers=2,
        map_teacher_num_queries=20,
        map_teacher_num_pts=20,
        map_teacher_feature_dim=256,
        map_teacher_require_features=True),
    val=dict(
        work_dir=work_dir,
        eval_config=dict(work_dir=work_dir)),
    test=dict(
        work_dir=work_dir,
        eval_config=dict(work_dir=work_dir)))
