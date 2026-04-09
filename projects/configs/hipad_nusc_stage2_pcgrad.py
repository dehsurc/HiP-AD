_base_ = ["./hipad_nusc_stage2.py"]

# Override work_dir and wandb name
work_dir = "work_dirs/hipad_nusc_stage2_pcgrad"
wandb_name = "hipad_nusc_stage2_pcgrad"

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

# Disable fp16 — PCGrad requires FP32 for accurate gradient projection
fp16 = None

# PCGrad configuration
pcgrad = dict(
    shared_layers=['backbone', 'neck', 'norm', 'ffn', 'fc_before', 'fc_after'],
    normalize_grads=True,
    pcgrad_interval=5,
    warmup_iters=500,
    log_interval=50,
    pcgrad_groups=[
        # Backbone (high conflict hotspot)
        'backbone_stem', 'backbone_layer1',
        # Decoder norm layers
        'dec0_norm_0', 'dec0_norm_1',
        'dec1_norm_0', 'dec1_norm_1',
        'dec2_norm_0', 'dec2_norm_1',
        'dec3_norm_0', 'dec3_norm_1',
        'dec4_norm_0', 'dec4_norm_1',
        'dec5_norm_0', 'dec5_norm_1',
        # Decoder FFN layers (prenorm included)
        'dec0_ffn_0', 'dec1_ffn_0', 'dec2_ffn_0',
        'dec3_ffn_0', 'dec4_ffn_0', 'dec5_ffn_0',
    ],
)
