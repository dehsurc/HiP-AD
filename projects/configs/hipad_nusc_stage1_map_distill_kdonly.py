# Ablation: map distillation WITHOUT real map GT loss (KD-only).
# Inherits hipad_nusc_stage1_map_distill.py and zeros out real GT loss weight.
#
# Real map cls/reg losses are still *computed* (loss_map runs the matcher and
# loss heads as usual), but multiplied by map_gt_loss_weight=0 so they
# contribute zero gradient. KD branch is unaffected.
#
# Detection / motion / planning losses stay ON (only the map GT branch is
# silenced) — change other gt loss weights here if you want a stricter
# distill-only setup.

_base_ = ["./hipad_nusc_stage1_map_distill.py"]

# Run identity (separate wandb run + work_dir)
wandb_project = "hipad_distill_map"
wandb_name = "hipad_nusc_stage1_12ep_map_distill_kdonly"
work_dir = "work_dirs/exp/hipad_nusc_stage1_12ep_map_distill_kdonly"

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

# Turn off real map GT loss; KD remains active.
# Switch to pseudo_gt mode (mirror of det E5 pseudo_gt): teacher predictions are
# converted to pseudo GT and fed through the standard map_sampler +
# loss_map_cls / loss_map_reg pipeline. In this mode map_distill_alpha_* values
# are just on/off flags (they only need to be > 0 to keep use_map_distill True);
# the real KD magnitude comes from loss_map_cls/reg's own loss_weight (1.0 / 10.0).
map_distill_mode = "pseudo_gt"

model = dict(
    head=dict(
        onedecoder_head=dict(
            map_gt_loss_weight=0.0,
            map_distill_mode=map_distill_mode,
        ),
    ),
)
