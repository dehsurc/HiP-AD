# Ablation: FAMO (Fast Adaptive Multitask Optimization, NeurIPS 2023) at
# stage2 onset, 6 epochs. Same data / LR schedule / batch size as
# loss_warmup_500iter_6ep, l2sp_lambda1_6ep, low_lr_6ep so the comparison
# isolates the gradient-balancing method.
#
# Method: dynamic per-task loss weights w = softmax(z) updated each iter
# from the previous iter's per-task log-loss decrease. Tasks improving
# slowest get more weight; tasks improving fastest get less.
#
# Why this matters for our problem (catastrophic-forgetting / stability-gap-
# like behavior at stage1->stage2 transition):
#   - Newly-activated motion/plan/ego losses start large (random-init heads)
#     -> FAMO automatically downweights them initially -> gradient burst on
#     shared decoder is naturally damped.
#   - Det/map losses are already low (inherited from stage1 ckpt) -> FAMO
#     may slightly upweight them, opposing forgetting.
#   - Single backward pass: avoids the DDP+grad-checkpoint reentrancy that
#     broke coupled L2-SP. AdamW state stays clean (unlike PCGrad).
_base_ = ["./_base_1ep.py"]

work_dir = "work_dirs/exp/stage2_ablation/famo_6ep"
wandb_name = "stage2_ablation_famo_6ep"

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 6

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
evaluation = dict(
    interval=num_iters_per_epoch,
    eval_mode=dict(
        with_det=True,
        with_tracking=False,
        with_map=True,
        with_motion=True,
        with_planning=True,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    ),
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(entity="e2ekd", project="hipad", name=wandb_name),
            by_epoch=False,
        ),
    ],
)

# FAMOHook runs at priority HIGH (=30), before the (Fp16)OptimizerHook
# (ABOVE_NORMAL=40). It rewrites runner.outputs['loss'] in place, so the
# optimizer backprops the FAMO-weighted loss on the standard single-pass
# pipeline. Compatible with with_cp=True and FP16 already in _base_.
custom_hooks = [
    dict(
        type="WandbValVisHook",
        vis_dir="val_vis/visual",
        max_images=8,
        interval=1,
        priority="LOWEST",
    ),
    dict(
        type="FAMOHook",
        weight_lr=0.025,
        gamma=1e-3,
        warmup_iters=500,   # match LR warmup; uniform weights during burn-in
        log_interval=50,
        # task_order=None  -> autodetect from model's task_losses dict
        priority="HIGH",
    ),
]
