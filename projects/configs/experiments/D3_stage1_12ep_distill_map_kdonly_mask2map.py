# D3: Map KD-only with Mask2Map 110ep teacher (vs D2/our kdonly which used MapTRv2 24ep).
#
# Mask2Map is a stronger map specialist (110-epoch trained on nuScenes-map).
# Cache pkl format is identical to our MapTRv2 cache:
#   - {sample_token: {logits[100,3] fp32, pts[100,20,2] fp32, scores[100] fp32}}
#   - class order already permuted to HiP-AD [ped_crossing, divider, boundary]
#   - raw meters, x in [-15,15], y in [-30,30]
# So no code changes needed — only swap the cache path.
#
# Hypothesis: a stronger teacher should lift map mAP above stage1 baseline 0.347
# (our MapTRv2 kdonly hit 0.32, below baseline; chanyoung's Mask2Map run hit 0.357,
# above baseline with the same config family).

_base_ = ["../hipad_nusc_stage1_map_distill_kdonly.py"]

# Run identity
wandb_project = "hipad_distill_map"
wandb_name = "D3_stage1_12ep_distill_map_kdonly_mask2map"
work_dir = "work_dirs/exp/D3_stage1_12ep_distill_map_kdonly_mask2map"

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

# Swap teacher cache: MapTRv2 → Mask2Map
map_teacher_cache_path = "/data4/kyungmin/nuScenes_Converted/data/cache_map/mask2map_110ep_train.pkl"

data = dict(
    train=dict(
        map_teacher_cache_path=map_teacher_cache_path,
    ),
)
