# Transfer-Gain (leave-one-aux-out) base — 2026-06-12
#
# ForkMerge-style TG for the gradient-dynamics research question
# "which task helps/hurts planning":
#   1) short fine-tune from the CONVERGED 18ep checkpoint (iter_31644),
#      once per variant, with one task's loss gradient cut via
#      SparseDetector.ablate_tasks (loss values still logged, grads detached);
#   2) evaluate planning on nuScenes val (PlanningMetric: L2 / obj_col /
#      obj_box_col) with tools/test.py;
#   3) TG(task) = P(no_task) - P(tg_full).   P = L2 avg (collision guardrail).
#
# Geometry matches iter_31644's training config (grad_analysis: 4 GPUs x
# batch 4 -> 1758 iters/epoch), so 1 epoch == 1758 iters.
# Driver: tools/anal_tg.sh
_base_ = ["../experiments/E2_E1_stage2_18ep_grad_analysis.py"]

version = "trainval"
length = {"trainval": 28130, "mini": 323}
num_gpus = 4
batch_size = 4
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 1

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=1)

# NOTE: leave-one-out ablation is done via SparseDetector.ablate_tasks, which
# zeros the gradient with `L*0 + L.detach()` (keeps every param in the graph),
# so DDP works with the DEFAULT find_unused_parameters=False — same setting as
# tg_full. We deliberately do NOT set find_unused_parameters=True: it clashes
# with the backbone's gradient checkpointing (with_cp=True) and raises
# "Expected to mark a variable ready only once".

# Weights only (fresh iter counter / LR schedule). NOT resume_from — that
# would restore _iter=31644 (>= max_iters) and skip training entirely.
load_from = "/home/yongjae/e2e/HiP-AD/work_dirs/E2_rev/iter_31644.pth"
resume_from = None

# Fine-tune LR: 0.2x of the original 1e-4. The original run ended at the
# cosine floor (1e-7); full 1e-4 would perturb the converged model too
# violently, while the deep tail would freeze it. min_lr_ratio=0.1 keeps the
# whole epoch inside a meaningful [2e-6, 2e-5] band. Identical across
# variants, so it cancels in the TG difference.
optimizer = dict(
    type="AdamW",
    lr=2e-5,
    weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5)}),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=0.1,
)

# Text logging only — no wandb dependency for autonomous TG runs.
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook", by_epoch=False)])
custom_hooks = []

# In-train eval stays defined but the driver trains with --no-validate and
# measures every variant uniformly via tools/test.py afterwards (including
# the un-finetuned iter_31644 reference).
#
# PLANNING-FOCUSED eval: the TG deliverable is planning L2/collision, and
# `planning_eval()` is fully self-contained — predicted ego trajectory comes
# from the model outputs (`res['img_bbox']['final_planning']`) and GT
# trajectory + collision boxes (`fut_boxes`) from its own eval dataset, with
# NO dependency on the det/map/motion metric blocks. So we drop:
#   with_map=False    — the map Chamfer-AP devkit is the ~40min/eval bottleneck
#   with_motion=False — not needed for the planning verdict
# and keep with_det=True (only ~75s) as a cheap ablation sanity check
# (tg_no_det should show det mAP collapse, confirming the gradient cut bit).
eval_mode = dict(
    with_det=True,
    with_tracking=False,
    with_map=False,
    with_motion=False,
    with_planning=True,
    tracking_threshold=0.2,
    motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch,
    jsonfile_prefix="val/",
    eval_mode=eval_mode,
    out_dir="val_vis",
)
