_base_ = ["./experiments/E2_E1_stage2_18ep.py"]

# 2 GPU, batch_size=6 → effective batch=12 (same as baseline)
# Note: base config의 length/version은 child 파싱 시점에 안 보여서 여기서 재정의
version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 18
checkpoint_epoch_interval = 1

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)  # 14064
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)

# Override work_dir and wandb name
work_dir = "/home/yongjae/e2e/HiP-AD/data_nusc/nuscenes/yong/hipad_nusc_pcgrad-e1"
wandb_name = "hipad_nusc_stage2_pcgrad-e1"

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

# Match baseline fp16 setting for fair comparison
fp16 = dict(loss_scale=32.0)

# Match dataloader batch size to config batch_size
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
)

# Disable gradient checkpointing — incompatible with multiple .backward()
# calls needed for PCGrad (DDP "marked ready twice" error)
# Disable depth_branch — its loss doesn't fit any PCGrad task prefix and would
# silently lose supervision during PCGrad iters.
model = dict(
    img_backbone=dict(with_cp=False),
)

# PCGrad configuration
pcgrad = dict(
    shared_layers=['backbone', 'neck', 'norm', 'ffn', 'fc_before', 'fc_after'],
    normalize_grads=True,
    pcgrad_interval=1,
    warmup_iters=0,
    log_interval=50,
    primary_task='plan',
    pcgrad_groups=[
        # Backbone (per-stage)
        'backbone_stem',
        'backbone_layer1', 'backbone_layer2',
        'backbone_layer3', 'backbone_layer4',
        # Neck
        'neck',
        # Decoder norm layers
        'dec0_norm_0', 'dec0_norm_1',
        'dec1_norm_0', 'dec1_norm_1',
        'dec2_norm_0', 'dec2_norm_1',
        'dec3_norm_0', 'dec3_norm_1',
        'dec4_norm_0', 'dec4_norm_1',
        'dec5_norm_0', 'dec5_norm_1',
        # Decoder FFN sub-modules (pre_norm / fc1 / fc2 / identity_fc)
        'dec0_ffn_0_pre_norm', 'dec0_ffn_0_fc1', 'dec0_ffn_0_fc2', 'dec0_ffn_0_identity_fc',
        'dec1_ffn_0_pre_norm', 'dec1_ffn_0_fc1', 'dec1_ffn_0_fc2', 'dec1_ffn_0_identity_fc',
        'dec2_ffn_0_pre_norm', 'dec2_ffn_0_fc1', 'dec2_ffn_0_fc2', 'dec2_ffn_0_identity_fc',
        'dec3_ffn_0_pre_norm', 'dec3_ffn_0_fc1', 'dec3_ffn_0_fc2', 'dec3_ffn_0_identity_fc',
        'dec4_ffn_0_pre_norm', 'dec4_ffn_0_fc1', 'dec4_ffn_0_fc2', 'dec4_ffn_0_identity_fc',
        'dec5_ffn_0_pre_norm', 'dec5_ffn_0_fc1', 'dec5_ffn_0_fc2', 'dec5_ffn_0_identity_fc',
        # Decoder fc_before / fc_after
        'fc_before', 'fc_after',
    ],
)
