_base_ = ['./E9_feature_distill.py']

work_dir = 'work_dirs/exp/E10_output_kd_pgt_l5_feat_aux'
wandb_name = 'E10_output_kd_pgt_l5_feat_aux'

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

# Output-level map distillation is the primary signal in E10. Keep real GT map
# supervision on, and add teacher polylines/logits as pseudo GT through the
# regular map target/loss path.
map_teacher_cache_path = 'data/cache/map/maptrv2_teacher_train_feat_top20_l345.pkl'
map_distill_mode = 'pseudo_gt'
map_distill_alpha_cls = 0.25
map_distill_alpha_reg = 1.0
map_distill_temperature = 4.0
map_distill_score_thr = 0.3
map_distill_dist_thr = 4.0
map_distill_last_layer_only = False
map_gt_loss_weight = 1.0

# Feature KD stays as a small auxiliary regularizer, using the l3/l4/l5 cache
# and matching setup that was already validated in E9.
map_feature_distill_alpha = 16.0
map_feature_distill_layers = (5,)
map_feature_distill_weights = (1.0,)
map_feature_distill_cached_layers = (3, 4, 5)
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
map_teacher_to_student_class_perm = None

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

eval_config = dict(work_dir=work_dir)

data = dict(
    train=dict(
        work_dir=work_dir,
        map_teacher_cache_path=map_teacher_cache_path,
        map_teacher_num_layers=3,
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
