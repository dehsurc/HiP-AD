"""
Re-visualize cosine similarity heatmaps and per-operation active overlap heatmaps
with fixed range [-0.5, +0.5], blue(+) / white(0) / red(-) colormap.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

BASE = "/home/yongjae/e2e/HiP-AD/gradient_analysis_results"
TASKS = ["plan", "det", "map", "motion", "ego"]
TASK_PAIRS = [
    "plan_vs_det", "plan_vs_map", "plan_vs_motion", "plan_vs_ego",
    "ego_vs_det", "ego_vs_map", "ego_vs_motion",
    "det_vs_map", "det_vs_motion",
    "map_vs_motion",
]

# Custom colormap: red(-) -> white(0) -> blue(+)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "rwb", [(0.8, 0.1, 0.1), (1, 1, 1), (0.1, 0.3, 0.8)]
)


def build_symmetric_matrix(pairwise_dict):
    """Build 5x5 symmetric matrix from pairwise cosine means."""
    n = len(TASKS)
    mat = np.eye(n)
    for key, val in pairwise_dict.items():
        parts = key.split("_vs_")
        if len(parts) != 2:
            continue
        t1, t2 = parts
        if t1 in TASKS and t2 in TASKS:
            i, j = TASKS.index(t1), TASKS.index(t2)
            mat[i, j] = val["mean"]
            mat[j, i] = val["mean"]
    return mat


def plot_cosine_heatmap(mat, save_path, title):
    """Plot 5x5 task cosine similarity heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap=cmap, vmin=-0.5, vmax=0.5, aspect="equal")

    ax.set_xticks(range(len(TASKS)))
    ax.set_yticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_yticklabels(TASKS, fontsize=11)

    # Annotate
    for i in range(len(TASKS)):
        for j in range(len(TASKS)):
            color = "white" if abs(mat[i, j]) > 0.35 else "black"
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_title(title, fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_operation_active_overlap(stats, save_path, title):
    """Plot per-operation (group) active overlap heatmap: 3 panels."""
    groups = list(stats["per_group_cosine"].keys())
    pair_keys = TASK_PAIRS

    # Build matrices: raw cosine, active overlap cosine, overlap ratio
    def build_matrix(section, field):
        mat = np.full((len(groups), len(pair_keys)), np.nan)
        for i, g in enumerate(groups):
            if g not in section:
                continue
            for j, p in enumerate(pair_keys):
                if p in section[g]:
                    val = section[g][p]
                    if isinstance(val, dict) and field in val:
                        mat[i, j] = val[field]["mean"] if isinstance(val[field], dict) else val[field]
                    elif isinstance(val, dict) and "mean" in val:
                        mat[i, j] = val["mean"]
        return mat

    raw_cos = build_matrix(stats["per_group_cosine"], "mean")
    # per_group_cosine has structure: group -> pair -> {mean, std, ...}
    # Rebuild correctly
    raw_cos = np.full((len(groups), len(pair_keys)), np.nan)
    for i, g in enumerate(groups):
        if g not in stats["per_group_cosine"]:
            continue
        for j, p in enumerate(pair_keys):
            if p in stats["per_group_cosine"][g]:
                raw_cos[i, j] = stats["per_group_cosine"][g][p]["mean"]

    active_cos = np.full((len(groups), len(pair_keys)), np.nan)
    for i, g in enumerate(groups):
        if g not in stats["per_group_active_overlap"]:
            continue
        for j, p in enumerate(pair_keys):
            if p in stats["per_group_active_overlap"][g]:
                active_cos[i, j] = stats["per_group_active_overlap"][g][p]["overlap_cosine"]["mean"]

    overlap_ratio = np.full((len(groups), len(pair_keys)), np.nan)
    for i, g in enumerate(groups):
        if g not in stats["per_group_active_overlap"]:
            continue
        for j, p in enumerate(pair_keys):
            if p in stats["per_group_active_overlap"][g]:
                overlap_ratio[i, j] = stats["per_group_active_overlap"][g][p]["overlap_ratio"]["mean"]

    # Shorten pair labels
    pair_labels = [p.replace("_vs_", " v ") for p in pair_keys]

    fig, axes = plt.subplots(1, 3, figsize=(28, max(14, len(groups) * 0.35)))

    datasets = [
        (raw_cos, "Raw Cosine Similarity", -0.5, 0.5),
        (active_cos, "Active Overlap Cosine", -0.5, 0.5),
        (overlap_ratio, "Overlap Ratio (low = disjoint)", 0.0, 1.0),
    ]

    # For overlap ratio, use a different colormap
    cmap_ratio = plt.cm.RdYlGn

    for idx, (data, subtitle, vmin, vmax) in enumerate(datasets):
        ax = axes[idx]
        if idx == 2:
            use_cmap = cmap_ratio
        else:
            use_cmap = cmap
        im = ax.imshow(data, cmap=use_cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(pair_labels)))
        ax.set_xticklabels(pair_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=6)
        ax.set_title(subtitle, fontsize=11)

        # Annotate with values
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    val = data[i, j]
                    if idx == 2:
                        color = "black" if 0.3 < val < 0.7 else "white"
                    else:
                        color = "white" if abs(val) > 0.35 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5, color=color)

        fig.colorbar(im, ax=ax, shrink=0.5)

    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    epochs = sorted(os.listdir(BASE))
    for epoch in epochs:
        epoch_dir = os.path.join(BASE, epoch)
        if not os.path.isdir(epoch_dir):
            continue
        for split in sorted(os.listdir(epoch_dir)):
            split_dir = os.path.join(epoch_dir, split)
            stats_path = os.path.join(split_dir, "gradient_conflict_stats.json")
            if not os.path.exists(stats_path):
                continue

            print(f"\n=== {epoch}/{split} ===")
            with open(stats_path) as f:
                stats = json.load(f)

            # 1) Cosine similarity heatmap
            mat = build_symmetric_matrix(stats["pairwise_cosine"])
            out1 = os.path.join(split_dir, "cosine_similarity_heatmap.png")
            plot_cosine_heatmap(mat, out1,
                f"Mean Cosine Similarity between Task Gradients\n(Negative = Conflict) [{epoch}/{split}]")

            # 2) Per-operation active overlap
            if "per_group_active_overlap" in stats:
                os.makedirs(os.path.join(split_dir, "active_overlap"), exist_ok=True)
                out2 = os.path.join(split_dir, "active_overlap", "per_operation_active_overlap.png")
                plot_per_operation_active_overlap(stats, out2,
                    f"Per-Operation: Raw vs Overlap Cosine & Overlap Ratio [{epoch}/{split}]")


if __name__ == "__main__":
    main()
