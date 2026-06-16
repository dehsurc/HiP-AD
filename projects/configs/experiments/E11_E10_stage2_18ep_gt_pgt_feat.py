_base_ = ['./E2_E1_stage2_18ep.py']

work_dir = 'work_dirs/exp/E11_E10_stage2_18ep_gt_pgt_feat'
wandb_name = 'E11_E10_stage2_18ep_gt_pgt_feat'
load_from = './work_dirs/exp/E10_output_kd_pgt_l5_feat_aux/latest.pth'

num_gpus = 4
batch_size = 4
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 18
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
runner = dict(type='IterBasedRunner', max_iters=num_iters_per_epoch * num_epochs)
evaluation = dict(interval=num_iters_per_epoch * 3)

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
            # Override the base (E2) temporal attention: split plan/ego temporal
        # attention into a self block (plan/ego <- plan/ego) and a cross block
        # (plan/ego <- det,map), and enable use_updated_query. The 4th attn /
        # decouple entry is a copy of the 3rd (plan/ego self) block.
        # base vars aren't visible here, so values are inlined:
        # embed_dims=256, num_groups=8, drop_out=0.1.
        temp_graph_model=dict(
            type="TemporalSeparateAttention",
            query_select=["det", "map", "plan", "ego"],
            query_list=[["det"], ["map"], ["plan", "ego"], ["plan", "ego"]],
            key_list=[["det"], ["map"], ["plan", "ego"], ["det", "map"]],
            decouple_list=[True, False, False, False],
            use_updated_query=True,
            attn=[
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256 * 2,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
            ],
        ),
        # Match b2d/revision: graph_model adds plan/ego as a third separate
        # group (was det/map only in E2). List keys fully replace the base.
        graph_model=dict(
            type="SeparateAttention",
            query_select=["det", "map", "plan", "ego"],
            separate_list=[["det"], ["map"], ["plan", "ego"]],
            decouple_list=[True, False, False],
            with_distance_attn_mask=False,
            attn=[
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256 * 2,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
            ],
        ),
        # Match b2d/revision: inter_graph_model switches from InteractiveAttention
        # to SeparateAttention over all 4 modalities with distance + structured
        # masks. _delete_=True is REQUIRED so the base InteractiveAttention's
        # query_list/key_list keys don't leak into SeparateAttention.__init__.
        inter_graph_model=dict(
            _delete_=True,
            type="SeparateAttention",
            query_select=["det", "map", "plan", "ego"],
            separate_list=[["det", "map", "plan", "ego"]],
            decouple_list=[False],
            with_distance_attn_mask=True,
            with_structured_mask=True,
            attn=[
                dict(
                    type="MultiheadFlashAttention",
                    embed_dims=256,
                    num_heads=8,
                    batch_first=True,
                    dropout=0.1,
                ),
            ],
        ))))

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
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
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
