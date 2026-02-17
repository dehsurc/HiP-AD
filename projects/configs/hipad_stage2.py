# Unified stage2 config switcher for HiP-AD
# Set `dataset` to choose between Bench2Drive and nuScenes.

dataset = "b2d"  # options: "b2d", "nuscenes"

if dataset == "b2d":
    _base_ = ["hipad_b2d_stage2.py"]
elif dataset == "nuscenes":
    _base_ = ["hipad_nusc_stage2.py"]
else:
    raise ValueError(f"Unsupported dataset: {dataset}")
