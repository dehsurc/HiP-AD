# CONTROL for E10_attittud_b1_ft3ep: identical batch-1 finetune recipe
# (same ckpt, lr, fp32, frozen BN, epochs) but NO gradient surgery.
# Any E10_attittud vs this-run difference isolates the ATTITTUD surgery
# effect from the batch-1 + hyperparameter changes.
_base_ = ["./E2_E1_stage2_18ep_new.py"]

work_dir = "work_dirs/exp/E10_ctrl_plain_b1_ft3ep"

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 1
batch_size = 1
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 3

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
evaluation = dict(interval=num_iters_per_epoch)

load_from = "ckpts/rev_nusc/70+stage2_3ep.pth"
resume_from = None

fp16 = None

# fp32 has no Fp16OptimizerHook overflow-skip; SafeOptimizerHook skips
# non-finite steps and resets temporal instance banks (see nan_guard.py).
# accum_steps=8 matches the ATTITTUD run's effective batch (grad averaged
# over 8 batch-1 micro-samples) so the comparison isolates the surgery.
optimizer_config = dict(
    type="SafeOptimizerHook",
    accum_steps=8,
    grad_clip=dict(max_norm=25, norm_type=2),
)

model = dict(
    img_backbone=dict(
        norm_eval=True,
        with_cp=False,
    ),
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
)

optimizer = dict(
    type="AdamW",
    lr=7e-5,  # eff batch 8 (accum_steps): 1e-4 * sqrt(8/16)
    weight_decay=0.001,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.5),
        }
    ),
)
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

wandb_project = "hipad_rev"
wandb_name = "E10_ctrl_plain_b1_ft3ep"
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

custom_hooks = [
    dict(
        type="WandbValVisHook",
        vis_dir="val_vis/visual",
        max_images=8,
        interval=1,
        priority="LOWEST",
    ),
    dict(type="FreezeBNHook", priority="HIGH"),
]
