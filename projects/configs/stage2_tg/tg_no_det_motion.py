# TG joint ablation: det AND motion loss gradients cut together.
# Rationale: motion_query is built from det_instance_feature
# (sparse_onedecoder.py:986), so motion's loss keeps supervising the shared
# agent features even when det is ablated alone -> no_det understates det.
# This joint variant removes the whole agent-perception supervision stack;
# compare TG(det&motion) against TG(det)+TG(motion) for redundancy/synergy.
_base_ = ["./_base_tg_1ep.py"]

work_dir = "work_dirs/exp/stage2_tg/tg_no_det_motion"
model = dict(ablate_tasks=["det", "motion"])
