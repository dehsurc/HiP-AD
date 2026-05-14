"""Scene-level stratified subsampling for NuScenes HiP-AD training pkl.

Stratifies scenes by (map_location, day/night, rain) and samples `ratio`
of scenes from each stratum. Filters the existing pkl by kept scene_tokens
so all pre-computed GT (future ego/agent trajs, sweeps, etc.) is preserved.

Usage:
    python tools/subsample_scenes.py \
        --in-pkl data/infos/nuscenes_infos_train.pkl \
        --out-pkl data/infos/nuscenes_infos_train_1_3.pkl \
        --dataroot data/nuscenes \
        --ratio 0.3333 --seed 0
"""
import argparse
import math
import pickle
import random
from collections import Counter, defaultdict


def build_scene_strata(scene_tokens, dataroot, version="v1.0-trainval",
                       scene2loc=None):
    """Return dict scene_token -> (location, is_night, is_rain)."""
    try:
        from nuscenes.nuscenes import NuScenes
        nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    except Exception as e:
        print(f"[warn] NuScenes devkit unavailable ({e}); "
              f"stratifying by location only.")
        return {st: (scene2loc.get(st, "unknown"), False, False)
                for st in scene_tokens}

    log2loc = {log["token"]: log["location"] for log in nusc.log}
    strata = {}
    for scene in nusc.scene:
        if scene["token"] not in scene_tokens:
            continue
        desc = scene["description"].lower()
        loc = log2loc.get(scene["log_token"], "unknown")
        strata[scene["token"]] = (loc, "night" in desc, "rain" in desc)
    # Safety: any scene present in pkl but not devkit
    for st in scene_tokens:
        strata.setdefault(st, (scene2loc.get(st, "unknown"), False, False))
    return strata


def stratified_sample(strata, ratio, seed):
    """Pick ratio of scenes from each stratum. Rounds up per stratum to
    ensure every stratum keeps at least one scene (if it had any)."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for scene_token, key in strata.items():
        buckets[key].append(scene_token)

    kept = []
    per_stratum_report = []
    for key, scenes in sorted(buckets.items()):
        scenes_sorted = sorted(scenes)  # deterministic given seed
        n_keep = max(1, round(len(scenes_sorted) * ratio))
        rng.shuffle(scenes_sorted)
        picked = scenes_sorted[:n_keep]
        kept.extend(picked)
        per_stratum_report.append((key, len(scenes), n_keep))
    return set(kept), per_stratum_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-pkl", required=True)
    parser.add_argument("--out-pkl", required=True)
    parser.add_argument("--dataroot", default="data/nuscenes")
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--ratio", type=float, default=1.0 / 3.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"[load] {args.in_pkl}")
    with open(args.in_pkl, "rb") as f:
        data = pickle.load(f)

    infos = data["infos"]
    print(f"  infos: {len(infos)}")

    scene2loc = {}
    for info in infos:
        scene2loc.setdefault(info["scene_token"], info.get("map_location", "?"))
    all_scenes = set(scene2loc.keys())
    print(f"  scenes: {len(all_scenes)}")
    print(f"  location dist (by scene): "
          f"{dict(Counter(scene2loc.values()))}")

    strata = build_scene_strata(all_scenes, args.dataroot, args.version,
                                scene2loc=scene2loc)

    kept_scenes, report = stratified_sample(strata, args.ratio, args.seed)
    print(f"\n[strata] (location, night, rain) -> total / kept")
    for key, n_total, n_keep in report:
        print(f"  {key}: {n_total} -> {n_keep}")
    print(f"[kept scenes] {len(kept_scenes)} / {len(all_scenes)} "
          f"({len(kept_scenes)/len(all_scenes):.3f})")

    kept_infos = [i for i in infos if i["scene_token"] in kept_scenes]
    print(f"[kept infos] {len(kept_infos)} / {len(infos)} "
          f"({len(kept_infos)/len(infos):.3f})")

    out_data = {"infos": kept_infos, "metadata": data.get("metadata", {})}
    out_data["metadata"] = dict(out_data["metadata"])
    out_data["metadata"].update({
        "subsample_ratio": args.ratio,
        "subsample_seed": args.seed,
        "subsample_kept_scenes": len(kept_scenes),
        "subsample_total_scenes": len(all_scenes),
    })

    print(f"[save] {args.out_pkl}")
    with open(args.out_pkl, "wb") as f:
        pickle.dump(out_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[done]")


if __name__ == "__main__":
    main()
