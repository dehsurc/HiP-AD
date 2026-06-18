# TG variant: detection loss gradients cut (det_loss_* detached).
# TG(det) = P(tg_no_det) - P(tg_full); positive L2 delta => det was helping plan.
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_no_det"
model = dict(ablate_tasks=["det"])
