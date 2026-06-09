# Common base for 1-epoch stage2 ablations diagnosing which stage2-new loss
# drives the det/map drop at stage1→stage2 transition.
# Inherits from the full stage2 baseline and only flips to 1 epoch.
_base_ = ["../experiments/E2_E1_stage2_18ep.py"]

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}
num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 1
checkpoint_epoch_interval = 1

runner = dict(type="IterBasedRunner", max_iters=num_iters_per_epoch * num_epochs)
checkpoint_config = dict(interval=num_iters_per_epoch, max_keep_ckpts=-1)
evaluation = dict(
    interval=num_iters_per_epoch,
    jsonfile_prefix="val/",
    eval_mode=dict(
        with_det=True,
        with_tracking=False,
        with_map=True,
        with_motion=False,
        with_planning=False,
        tracking_threshold=0.2,
        motion_threshhold=0.2,
    ),
    out_dir="val_vis",
)

load_from = "./work_dirs/exp/E1_stage1_12ep/latest.pth"
