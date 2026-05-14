# Stage2 continuation of D2 (stage1 12ep, det+map KD-only, both teachers).
#
# D2 stage1 result (12ep):
#   det mAP 0.1977 / NDS 0.2471
#   map mAP_normal 0.3235 (ped 0.242, div 0.324, bound 0.405)
#
# Stage2 picks up the D2 ckpt and trains 18 epochs further with:
#   - real GT supervision for all tasks (det, map, plan, ego, motion)
#   - motion + planning tasks newly enabled (task_select expanded vs stage1)
#   - NO distillation (KD branches off — det_gt_loss_weight defaults back to 1.0,
#     map_gt_loss_weight defaults back to 1.0, distill_alpha_* default 0.0)
#
# Hypothesis: D2 backbone learned strong det+map representation from teachers;
# stage2 fine-tunes with real GT and adds motion/planning heads on top of that.

_base_ = ["../hipad_nusc_stage2.py"]

# ------------------------------------------------------------------------
# Run identity
# ------------------------------------------------------------------------
wandb_project = "hipad_distill_dm"
wandb_name = "D2_stage1_to_stage2_18ep"
work_dir = "work_dirs/exp/D2_stage1_to_stage2_18ep"

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

# ------------------------------------------------------------------------
# Load D2 stage1 final ckpt (KD-only trained, 12 epoch = iter 21096)
# ------------------------------------------------------------------------
load_from = "/home/kyungmin/min_ws/rideflux/HiP-AD/work_dirs/exp/D2_stage1_12ep_distill_det_map_kdonly/iter_21096.pth"
