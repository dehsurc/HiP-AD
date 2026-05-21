"""Merge PM_det / PM_map / PM_motion checkpoints into a single init checkpoint
for PM_full (the final stage that trains plan/ego on top of the learned
det/map/motion query weights).

Strategy:
  - Start from PM_motion checkpoint (it contains backbone + det_* + motion_*).
  - Pull map_* parameters from PM_map checkpoint.
  - Drop attention/operation layers (head.onedecoder_head.layers.*,
    fc_before/after) because PM_det and PM_map were trained with task-only
    attention groups whose shapes do NOT match PM_full's full 3-group
    attention. Those layers will be freshly initialized in PM_full.
  - plan_*, ego_*, command_embed_encoder, etc. are absent in all three PM_
    checkpoints and will be freshly initialized in PM_full.

Usage:
  python tools/merge_pm_ckpts.py \
    --det work_dirs/exp/PM_det_12ep_1_3_seed0/latest.pth \
    --map work_dirs/exp/PM_map_12ep_1_3_seed0/latest.pth \
    --motion work_dirs/exp/PM_motion_12ep_1_3_seed0/latest.pth \
    --out work_dirs/exp/PM_merged_init.pth
"""
import argparse
import os
import torch


EXCLUDE_PREFIXES = (
    "head.onedecoder_head.layers.",      # attention / op layers (task-only shape)
    "head.onedecoder_head.fc_before.",   # decouple_attn wrappers (shape mismatch risk)
    "head.onedecoder_head.fc_after.",
)

MAP_KEY_MARKERS = (
    "map_instance_bank",
    "map_anchor_encoder",
    "map_deformable",
    "map_refine",
    "suqueeze_map_instance",   # only present if with_concat_map_points
)


def _state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if "state_dict" in ckpt:
        return ckpt, ckpt["state_dict"]
    return ckpt, ckpt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det", required=True)
    parser.add_argument("--map", dest="map_path", required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    for p in [args.det, args.map_path, args.motion]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    det_ckpt, det_sd = _state_dict(args.det)
    map_ckpt, map_sd = _state_dict(args.map_path)
    motion_ckpt, motion_sd = _state_dict(args.motion)

    print(f"[PM_det]    {len(det_sd):4d} keys from {args.det}")
    print(f"[PM_map]    {len(map_sd):4d} keys from {args.map_path}")
    print(f"[PM_motion] {len(motion_sd):4d} keys from {args.motion}")

    merged = {}
    dropped_attn = 0
    for k, v in motion_sd.items():
        if any(k.startswith(p) for p in EXCLUDE_PREFIXES):
            dropped_attn += 1
            continue
        merged[k] = v
    print(f"\nFrom PM_motion: kept {len(merged)} keys "
          f"(dropped {dropped_attn} attention/op layer keys)")

    added_map = 0
    overwritten = 0
    for k, v in map_sd.items():
        if any(k.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if not any(m in k for m in MAP_KEY_MARKERS):
            continue
        if k in merged:
            overwritten += 1
        merged[k] = v
        added_map += 1
    print(f"From PM_map:    added {added_map} map_* keys "
          f"(overwrote {overwritten} existing keys)")

    # Use motion ckpt's meta as base; record provenance
    meta = dict(motion_ckpt.get("meta", {})) if isinstance(motion_ckpt, dict) else {}
    meta["pm_merge_sources"] = {
        "det": os.path.abspath(args.det),
        "map": os.path.abspath(args.map_path),
        "motion": os.path.abspath(args.motion),
    }
    meta["pm_merge_dropped_attn_keys"] = dropped_attn

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state_dict": merged, "meta": meta}, args.out)

    print(f"\n✓ Saved merged checkpoint to {args.out}")
    print(f"  Total keys: {len(merged)}")
    print(f"  Hint: PM_full will fresh-init attention/op layers, plan_*, ego_*, "
          "command_embed_encoder, etc.")


if __name__ == "__main__":
    main()
