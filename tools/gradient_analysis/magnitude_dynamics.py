"""M-N10 — Magnitude non-stationarity (Phase 1 #10).

Reads per-checkpoint per-task gradient norm CSVs and produces:

  * compute_per_task_slopes — linear regression of mean_norm vs epoch.
  * compute_ratio_evolution — (task × task × epoch) ratio cube.
  * non_stationarity_index — coefficient of variation per task pair across
    epochs.

Plus a `run_magnitude_dynamics` orchestrator that writes csv + heatmap
evolution figures.

Memory rationale: prior analysis surfaced motion/map ≈ 7.5× as the dominant
imbalance signal but only at a single checkpoint. This module turns that
snapshot into a trajectory so the gating decision (Phase 1 #1 vs #10) can be
made against time-resolved evidence.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_per_task_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task with `slope`, `intercept`, `r2`. Input columns:
    `epoch`, `task`, `mean_norm`."""
    out_rows = []
    for task, sub in df.groupby("task"):
        x = sub["epoch"].to_numpy(dtype=np.float64)
        y = sub["mean_norm"].to_numpy(dtype=np.float64)
        if x.size < 2:
            slope, intercept, r2 = float("nan"), float("nan"), float("nan")
        else:
            xm, ym = x.mean(), y.mean()
            denom = float(((x - xm) ** 2).sum())
            slope = float(((x - xm) * (y - ym)).sum() / denom) if denom > 0 else 0.0
            intercept = float(ym - slope * xm)
            ss_res = float(((y - (slope * x + intercept)) ** 2).sum())
            ss_tot = float(((y - ym) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out_rows.append({"task": task, "slope": slope, "intercept": intercept, "r2": r2})
    return pd.DataFrame(out_rows)


def compute_ratio_evolution(df: pd.DataFrame, tasks: List[str]) -> np.ndarray:
    """Return a (n_tasks, n_tasks, n_epochs) array of pairwise ratios sorted
    by epoch ascending. Diagonals are 1 (self-ratio); cells with a missing
    norm are NaN."""
    epochs = sorted(df["epoch"].unique())
    cube = np.full((len(tasks), len(tasks), len(epochs)), np.nan, dtype=np.float64)
    for k, ep in enumerate(epochs):
        sub = df[df["epoch"] == ep]
        norms = {row["task"]: float(row["mean_norm"]) for _, row in sub.iterrows()}
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                ni = norms.get(ti)
                nj = norms.get(tj)
                if ni is None or nj is None or nj == 0:
                    continue
                cube[i, j, k] = ni / nj
    return cube


def non_stationarity_index(df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    """Coefficient of variation of each pairwise ratio across epochs.

    Returns a DataFrame indexed by `(task_a, task_b)` with column `cv`.
    Diagonal pairs are excluded.
    """
    cube = compute_ratio_evolution(df, tasks)
    rows = []
    for i, j in combinations(range(len(tasks)), 2):
        seq = cube[i, j, :]
        seq = seq[np.isfinite(seq)]
        if seq.size < 2:
            cv = float("nan")
        else:
            mean = seq.mean()
            cv = float(seq.std(ddof=1) / mean) if mean != 0 else float("nan")
        rows.append({"task_a": tasks[i], "task_b": tasks[j], "cv": cv})
    return pd.DataFrame(rows).set_index(["task_a", "task_b"])


def _emit_ratio_heatmap(cube: np.ndarray, tasks: List[str],
                        epochs: List, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(epochs)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    for k, ep in enumerate(epochs):
        ax = axes[0, k]
        im = ax.imshow(cube[:, :, k], cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45)
        ax.set_yticklabels(tasks)
        ax.set_title(f"epoch {ep}")
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                v = cube[i, j, k]
                if np.isfinite(v):
                    panel_mean = np.nanmean(cube[:, :, k])
                    color = "white" if (np.isfinite(panel_mean) and v > panel_mean) else "black"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=8, color=color)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Pairwise gradient norm ratio (row / col) over epochs")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_magnitude_dynamics(
    per_task_norm: pd.DataFrame,
    tasks: List[str],
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Top-level orchestrator. `per_task_norm` columns: `epoch, task, mean_norm`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slopes = compute_per_task_slopes(per_task_norm)
    slopes.to_csv(out_dir / "per_task_slopes.csv", index=False)
    nsi = non_stationarity_index(per_task_norm, tasks).reset_index()
    nsi.to_csv(out_dir / "non_stationarity_index.csv", index=False)
    epochs = sorted(per_task_norm["epoch"].unique())
    cube = compute_ratio_evolution(per_task_norm, tasks)
    np.save(out_dir / "ratio_evolution.npy", cube)
    if epochs:
        _emit_ratio_heatmap(cube, tasks, epochs, out_dir / "ratio_evolution.png")
    return {"slopes": slopes, "nsi": nsi}
