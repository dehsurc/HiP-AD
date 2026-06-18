# TG variant: map loss gradients cut (map_loss_* detached).
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_no_map"
model = dict(ablate_tasks=["map"])
