"""Generate all experiment configs from base templates.

Run once to create E2~E9 configs, then this script can be deleted.
"""
import os

DIR = os.path.dirname(os.path.abspath(__file__))


def read(path):
    with open(path) as f:
        return f.read()


def write(name, content):
    path = os.path.join(DIR, name)
    with open(path, "w") as f:
        f.write(content)
    print(f"Created {path}")


# ── Base templates ──
stage1_base = read(os.path.join(DIR, "E1_stage1_12ep.py"))
stage2_base = read(os.path.join(DIR, "..", "hipad_nusc_stage2.py"))
stage2_distill_base = read(os.path.join(DIR, "..", "distill", "hipad_nusc_stage2_distill.py"))
stage2_distill_only_base = read(os.path.join(DIR, "..", "distill", "hipad_nusc_stage2_distill_only.py"))


# ════════════════════════════════════════════════════════════
# E4: stage1(12ep) distill w/ det_loss  (det_gt_loss_weight=1.0)
# E5: stage1(12ep) distill w/o det_loss (det_gt_loss_weight=0.0)
# ════════════════════════════════════════════════════════════

def make_stage1_distill(exp_id, wandb_name, work_dir, det_gt_loss_weight):
    """Create stage1 config with distillation."""
    cfg = stage1_base

    # work_dir & wandb
    cfg = cfg.replace(
        'work_dir = "work_dirs/exp/E1_stage1_12ep"',
        f'work_dir = "{work_dir}"',
    )
    cfg = cfg.replace(
        'wandb_name = "E1_stage1_12ep"',
        f'wandb_name = "{wandb_name}"',
    )

    # Add distillation config section before model definition
    distill_section = f"""
# ================== distillation config ========================
teacher_cache_path = "data/cache/det/bevfusion_teacher_train.pkl"
distill_alpha_cls = 0.05
distill_alpha_reg = 0.1
distill_temperature = 4.0
distill_score_thr = 0.3
distill_last_layer_only = True

"""
    cfg = cfg.replace(
        "\nmodel = dict(",
        distill_section + "model = dict(",
    )

    # Add distillation params to onedecoder_head (after cls_threshold_to_reg)
    distill_head_params = f"""            # distillation
            distill_alpha_cls=distill_alpha_cls,
            distill_alpha_reg=distill_alpha_reg,
            distill_temperature=distill_temperature,
            distill_score_thr=distill_score_thr,
            distill_last_layer_only=distill_last_layer_only,
            distill_mode="teacher_tp",
            det_gt_loss_weight={det_gt_loss_weight},"""
    cfg = cfg.replace(
        '            cls_threshold_to_reg=0.05,\n            # instance_bank',
        f'            cls_threshold_to_reg=0.05,\n{distill_head_params}\n            # instance_bank',
    )

    # Add teacher keys to train_pipeline Collect
    cfg = cfg.replace(
        '            "ego_status_mask",\n        ],\n        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"],',
        '            "ego_status_mask",\n'
        '            "teacher_logits",\n'
        '            "teacher_boxes",\n'
        '            "teacher_scores",\n'
        '        ],\n'
        '        meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id", "token"],',
    )

    # Add teacher_cache_path to data train
    cfg = cfg.replace(
        "        keep_consistent_seq_aug=True,\n    ),\n    val=dict(",
        "        keep_consistent_seq_aug=True,\n"
        "        teacher_cache_path=teacher_cache_path,\n"
        "    ),\n    val=dict(",
    )

    return cfg


write("E4_stage1_12ep_distill_w_det.py",
      make_stage1_distill("E4", "E4_stage1_distill_w_det",
                          "work_dirs/exp/E4_stage1_12ep_distill_w_det", 1.0))

write("E5_stage1_12ep_distill_wo_det.py",
      make_stage1_distill("E5", "E5_stage1_distill_wo_det",
                          "work_dirs/exp/E5_stage1_12ep_distill_wo_det", 0.0))


# ════════════════════════════════════════════════════════════
# E2: E1+stage2(18ep)  — no distillation
# E3: E1+stage2(6ep)   — no distillation
# ════════════════════════════════════════════════════════════

def make_stage2_no_distill(wandb_name, work_dir, num_epochs, load_from):
    cfg = stage2_base

    cfg = cfg.replace(
        'work_dir = "work_dirs/hipad_nusc_stage2"',
        f'work_dir = "{work_dir}"',
    )
    cfg = cfg.replace(
        'wandb_name = "hipad_nusc_stage2_18ep"',
        f'wandb_name = "{wandb_name}"',
    )
    cfg = cfg.replace(
        'num_epochs = 18',
        f'num_epochs = {num_epochs}',
    )
    # Fix load_from (appears twice in stage2 config)
    cfg = cfg.replace(
        'load_from = "./work_dirs/hipad_nusc_stage1/latest.pth"',
        f'load_from = "{load_from}"',
    )
    return cfg


write("E2_E1_stage2_18ep.py",
      make_stage2_no_distill("E2_E1_stage2_18ep",
                             "work_dirs/exp/E2_E1_stage2_18ep", 18,
                             "./work_dirs/hipad_nusc_stage1/latest.pth"))

write("E3_E1_stage2_6ep.py",
      make_stage2_no_distill("E3_E1_stage2_6ep",
                             "work_dirs/exp/E3_E1_stage2_6ep", 6,
                             "./work_dirs/hipad_nusc_stage1/latest.pth"))


# ════════════════════════════════════════════════════════════
# E6: E4+stage2(6ep) distill w/ det_loss
# E7: E5+stage2(6ep) distill w/o det_loss
# E8: E1+stage2(6ep) distill w/ det_loss
# E9: E1+stage2(6ep) distill w/o det_loss
# ════════════════════════════════════════════════════════════

def make_stage2_distill(wandb_name, work_dir, load_from, det_gt_loss_weight):
    """Use the appropriate base depending on det_gt_loss_weight."""
    if det_gt_loss_weight > 0:
        cfg = stage2_distill_base
        # In the distill base, det_gt_loss_weight=1.0 is set inline
        cfg = cfg.replace(
            'work_dir = "work_dirs/hipad_nusc_stage2_distill_v2"',
            f'work_dir = "{work_dir}"',
        )
        # wandb_name uses datetime in distill base, replace the whole line
        cfg = cfg.replace(
            'import datetime\nwandb_project = "nusc_det_distill"\nwandb_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")',
            f'wandb_project = "hipad"\nwandb_name = "{wandb_name}"',
        )
    else:
        cfg = stage2_distill_only_base
        cfg = cfg.replace(
            'work_dir = "work_dirs/hipad_nusc_stage2_distill_only"',
            f'work_dir = "{work_dir}"',
        )
        cfg = cfg.replace(
            'wandb_project = "nusc_det_distill"\nwandb_name = "stage2_distill_only_tp"',
            f'wandb_project = "hipad"\nwandb_name = "{wandb_name}"',
        )

    # Fix load_from
    cfg = cfg.replace(
        'load_from = "./work_dirs/hipad_nusc_stage1/latest.pth"',
        f'load_from = "{load_from}"',
    )
    return cfg


write("E6_E4_stage2_6ep_distill_w_det.py",
      make_stage2_distill("E6_E4_stage2_distill_w_det",
                          "work_dirs/exp/E6_E4_stage2_6ep_distill_w_det",
                          "./work_dirs/exp/E4_stage1_12ep_distill_w_det/latest.pth", 1.0))

write("E7_E5_stage2_6ep_distill_wo_det.py",
      make_stage2_distill("E7_E5_stage2_distill_wo_det",
                          "work_dirs/exp/E7_E5_stage2_6ep_distill_wo_det",
                          "./work_dirs/exp/E5_stage1_12ep_distill_wo_det/latest.pth", 0.0))

write("E8_E1_stage2_6ep_distill_w_det.py",
      make_stage2_distill("E8_E1_stage2_distill_w_det",
                          "work_dirs/exp/E8_E1_stage2_6ep_distill_w_det",
                          "./work_dirs/hipad_nusc_stage1/latest.pth", 1.0))

write("E9_E1_stage2_6ep_distill_wo_det.py",
      make_stage2_distill("E9_E1_stage2_distill_wo_det",
                          "work_dirs/exp/E9_E1_stage2_6ep_distill_wo_det",
                          "./work_dirs/hipad_nusc_stage1/latest.pth", 0.0))

print("\nDone! All experiment configs generated.")
