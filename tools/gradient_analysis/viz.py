"""Shared matplotlib plotting helpers for gradient analysis (headless, Agg)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scatter_with_bins(
    x: np.ndarray,
    y: np.ndarray,
    bin_summary: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    out_path: str | Path,
    y_stat_col: str,
) -> None:
    """Scatter of (x, y) overlaid with binned mean ± std bars."""
    out_path = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.25, s=8, label="per-batch")
    centers = (bin_summary["bin_lo"] + bin_summary["bin_hi"]) / 2
    ax.errorbar(
        centers,
        bin_summary[f"{y_stat_col}_mean"],
        yerr=bin_summary[f"{y_stat_col}_std"],
        fmt="o-",
        color="crimson",
        capsize=3,
        label="bin mean ± std",
    )
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: str | Path,
    cmap: str = "RdBu_r",
    center: Optional[float] = 0.0,
    fmt: str = "{:.3f}",
) -> None:
    """Annotated heatmap for square or rectangular matrices."""
    out_path = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(col_labels), 1.0 + 0.9 * len(row_labels)))
    vmax = float(np.nanmax(np.abs(matrix)))
    if center is not None:
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def violin_by_pair(
    data: dict[str, np.ndarray],
    y_label: str,
    title: str,
    out_path: str | Path,
) -> None:
    """Violin plot: one violin per task pair."""
    out_path = _ensure_dir(Path(out_path))
    labels = list(data.keys())
    arrays = [data[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4.5))
    ax.violinplot(arrays, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def timeseries(
    rows: Iterable[dict],
    x_key: str,
    y_key: str,
    series_key: str,
    title: str,
    out_path: str | Path,
) -> None:
    """Generic time-series plot: one line per series."""
    out_path = _ensure_dir(Path(out_path))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, sub in df.groupby(series_key):
        sub = sub.sort_values(x_key)
        ax.plot(sub[x_key], sub[y_key], marker="o", label=str(label))
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
