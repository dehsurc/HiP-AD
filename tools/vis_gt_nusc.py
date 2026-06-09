"""Visualize nuScenes ground truth (map polylines + det boxes + ego future) in BEV.

Usage:
    python tools/vis_gt_nusc.py \
        --info data/infos/nuscenes_infos_val.pkl \
        --out work_dirs/vis_gt \
        --indices 0 300 600 900 1200 1500
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mmcv
import numpy as np


MAP_CLASSES = ("ped_crossing", "divider", "boundary")
MAP_COLORS = {"ped_crossing": "tab:blue", "divider": "tab:green", "boundary": "tab:red"}

DET_COLORS = {
    "car": "tab:orange",
    "truck": "tab:olive",
    "construction_vehicle": "tab:brown",
    "bus": "tab:cyan",
    "trailer": "tab:pink",
    "barrier": "tab:purple",
    "motorcycle": "magenta",
    "bicycle": "lime",
    "pedestrian": "black",
    "traffic_cone": "gold",
}


def draw_box(ax, box, color):
    """box = [x, y, z, l, w, h, yaw, vx, vy]. Drawn in BEV ego/lidar frame."""
    x, y, _, l, w, _, yaw = box[:7]
    corners = np.array([[ l / 2,  w / 2],
                        [ l / 2, -w / 2],
                        [-l / 2, -w / 2],
                        [-l / 2,  w / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = corners @ R.T + np.array([x, y])
    poly = plt.Polygon(corners, fill=False, edgecolor=color, linewidth=1.2)
    ax.add_patch(poly)
    # heading marker
    head = (np.array([l / 2, 0]) @ R.T + np.array([x, y]))
    ax.plot([x, head[0]], [y, head[1]], color=color, linewidth=0.8)


def visualize_sample(info, out_path, roi_size=(60, 60)):
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))

    # roi_size = (x_range, y_range) in metres around ego (lidar/ego origin = (0,0))
    half_x, half_y = roi_size[0] / 2, roi_size[1] / 2

    # ----- Map polylines (lidar frame, already centered on ego at converter time) -----
    map_annos = info.get("map_annos", {})
    map_handles = []
    for label, anno_list in map_annos.items():
        cls_name = MAP_CLASSES[label] if isinstance(label, int) else label
        color = MAP_COLORS.get(cls_name, "gray")
        for line in anno_list:
            line = np.array(line)
            if line.ndim == 2 and line.shape[1] >= 2:
                ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.4, alpha=0.9)
                ax.scatter(line[:, 0], line[:, 1], color=color, s=4, alpha=0.6)
        map_handles.append(mpatches.Patch(color=color, label=cls_name))

    # ----- Det boxes -----
    boxes = info.get("gt_boxes", np.zeros((0, 9)))
    names = info.get("gt_names", [])
    mask = info.get("valid_flag", None)
    if mask is not None:
        boxes = boxes[mask]
        names = np.array(names)[mask] if len(names) else names
    box_classes = set()
    for box, name in zip(boxes, names):
        if name not in DET_COLORS:
            continue
        draw_box(ax, box, DET_COLORS[name])
        box_classes.add(name)
    det_handles = [mpatches.Patch(color=DET_COLORS[n], label=n) for n in sorted(box_classes)]

    # ----- Ego future trajectory -----
    ego_fut = info.get("gt_ego_fut_trajs", None)
    if ego_fut is not None:
        ego_fut = np.array(ego_fut)
        # gt_ego_fut_trajs is per-step delta in some converters; check the shape
        if ego_fut.ndim == 2 and ego_fut.shape[1] >= 2:
            cum = np.cumsum(ego_fut[:, :2], axis=0)
            ax.plot([0] + list(cum[:, 0]), [0] + list(cum[:, 1]),
                    color="red", linewidth=2, linestyle="--", label="ego_future")
            ax.scatter(cum[:, 0], cum[:, 1], color="red", s=15, zorder=5)

    # ----- Ego origin -----
    ax.plot(0, 0, marker="o", color="black", markersize=8, zorder=6)
    ax.arrow(0, 0, 3, 0, head_width=0.6, head_length=0.6,
             fc="black", ec="black", zorder=6)

    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m, lidar frame)")
    ax.set_ylabel("y (m, lidar frame)")
    token = info.get("token", "?")
    map_loc = info.get("map_location", "?")
    ax.set_title(f"token={token[:8]}  map={map_loc}  "
                 f"#det={len(boxes)}  #ped={len(map_annos.get(0, []))}  "
                 f"#div={len(map_annos.get(1, []))}  #bnd={len(map_annos.get(2, []))}")
    ax.legend(handles=map_handles + det_handles, loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--info", required=True, help="path to nuScenes infos pkl")
    parser.add_argument("--out", required=True, help="output dir for png")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 300, 600, 900, 1200, 1500])
    parser.add_argument("--roi", nargs=2, type=float, default=[60.0, 60.0],
                        help="BEV view size (x_range, y_range) in metres")
    args = parser.parse_args()

    print(f"Loading {args.info}...")
    infos = mmcv.load(args.info)
    if isinstance(infos, dict) and "infos" in infos:
        infos = infos["infos"]
    print(f"  num samples: {len(infos)}")

    os.makedirs(args.out, exist_ok=True)
    for idx in args.indices:
        if idx >= len(infos):
            print(f"  skip {idx} (>= {len(infos)})")
            continue
        info = infos[idx]
        out_path = Path(args.out) / f"sample_{idx:05d}.png"
        visualize_sample(info, str(out_path), roi_size=tuple(args.roi))
        print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
