# ATTITTUD batch-1 planning-aware finetune, 3 epochs.
#
# Hypothesis: per-sample task-gradient relationships differ, so run batch=1
# and modify auxiliary (det/map/motion/ego) gradients per sample against a
# low-rank subspace of recent planning gradients (Dery et al., ICLR 2021),
# flipping the conflicting ("bad") component.
#
# Protocol: finetune from the baseline stage2 3-epoch checkpoint
# (ckpts/rev_nusc/70+stage2_3ep.pth) so the plan head is already meaningful
# and the planning-gradient subspace is well-defined from step 1. Paired
# control: E10_ctrl_plain_b1_ft3ep.py (identical, no surgery).
#
# Batch-1 adjustments vs base config:
#   - fp32 (fp16 off): multiple retain_graph backwards + prior fp16 NaN issues
#   - with_cp off: activation checkpointing conflicts with multi-backward
#   - BN frozen via FreezeBNHook (stats from 6 views of 1 scene are degenerate)
#   - lr sqrt-scaled 1e-4 * sqrt(1/16) = 2.5e-5, warmup 1000 iters
_base_ = ["./E2_E1_stage2_18ep_new.py"]

work_dir = "work_dirs/exp/E10_attittud_b1_ft3ep"

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

# fp32: disable the base fp16 wrapper entirely.
fp16 = None

model = dict(
    img_backbone=dict(
        norm_eval=True,   # freeze BN stats (batch-1)
        with_cp=False,    # no activation ckpt: safe multi-backward
    ),
)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
)

# Effective batch = accum_steps (8) via gradient accumulation inside the
# ATTITTUD hook: per-sample forward/backward/surgery, averaged over 8 samples
# before each optimizer step. Restores the batch-averaging buffer that
# batch-1 removes (the real fix for the ego-status divergence NaN) without
# collapsing the per-sample gradient surgery. LR sqrt-scaled to eff batch 8:
# 1e-4 * sqrt(8/16) = 7.07e-5.
optimizer = dict(
    type="AdamW",
    lr=7e-5,
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

# Planning-aware auxiliary gradient decomposition (replaces optimizer hook).
attittud = dict(
    # Decoder shared ops only: inter_gnn (unified cross-task attention where
    # task queries actually interact), plus the shared ffn / norm. Excludes
    # backbone/neck so the single flattened plan subspace is not dominated by
    # millions of backbone params where the planning gradient is negligible.
    shared_layers=["inter_gnn", "ffn", "norm"],
    mode="svd_buffer",      # ring buffer of plan grads; 'rank1' = current only
    buffer_size=16,
    svd_energy=0.90,        # adaptive rank: smallest k with >=90% energy
    k_max=8,
    alpha_good=1.0,
    alpha_bad=-1.0,         # flip conflicting component (user choice)
    alpha_neutral=1.0,
    primary_task="plan",
    warmup_iters=50,        # surgery starts early (finetune ckpt: plan grad
                            # meaningful from step 1; buffer fills by ~16)
    accum_steps=8,          # gradient accumulation window = effective batch 8
    log_interval=50,
    log_verbose=True,       # task-pair conflict matrix, subspace spectrum/drift
    dump_per_sample=True,   # append per-sample gradient records to work_dir jsonl
    track_direction=True,   # per-task grad-direction convergence (EMA coherence)
    ema_beta=0.99,          # EMA horizon ~100 samples for direction stats
)

wandb_project = "hipad_rev"
wandb_name = "E10_attittud_b1_ft3ep"
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
