# TG control: all task losses active, same 1-epoch fine-tune schedule.
# P(tg_full) is the baseline every leave-one-out variant is compared against.
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_full"
model = dict(ablate_tasks=[])
