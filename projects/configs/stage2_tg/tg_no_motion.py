# TG variant: motion loss gradients cut (motion_loss_* detached).
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_no_motion"
model = dict(ablate_tasks=["motion"])
