"""M5 — GradNorm Analysis.

Per-task gradient norm distributions, norm ratio matrix, and a raw-vs-normalized
probe comparison using M3 outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .viz import heatmap


def per_task_norms_from_cached(cached, tasks: List[str], groups: List[str]) -> pd.DataFrame:
    rows = []
    for cb in cached:
        for t in tasks:
            tg = cb["shared"].get(t, {})
            for g in groups:
                v = tg.get(g)
                if v is None:
                    continue
                rows.append({
                    "batch_idx": cb["batch_idx"],
                    "task": t,
                    "group": g,
                    "norm": float(v.norm()) if hasattr(v, "norm") else float(np.linalg.norm(v)),
                })
    return pd.DataFrame(rows)


def antisymmetric_frobenius(M: np.ndarray) -> float:
    """Frobenius norm of (M - M^T) / 2."""
    A = (M - M.T) / 2.0
    return float(np.sqrt((A * A).sum()))


def run_m5(
    cached,
    tasks: List[str],
    groups: List[str],
    probe_df: pd.DataFrame,
    steps_list: List[int],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norms = per_task_norms_from_cached(cached, tasks, groups)
    norms.to_csv(out_dir / "per_task_norm.csv", index=False)

    # Mean norm ratio matrix (tasks x tasks)
    means = norms.groupby("task")["norm"].mean().reindex(tasks)
    ratio = np.outer(means.values, 1.0 / means.values)
    heatmap(
        ratio,
        row_labels=tasks,
        col_labels=tasks,
        title="Task gradient norm ratio (row / col)",
        out_path=out_dir / "norm_ratio_heatmap.png",
        cmap="viridis",
        center=None,
        fmt="{:.2f}",
    )

    # Raw-vs-normalized symmetry comparison (uses M3 probe df)
    comp_rows = []
    for s in steps_list:
        for variant in ("raw", "normalized"):
            sub = probe_df[(probe_df["steps"] == s) & (probe_df["variant"] == variant)]
            if sub.empty:
                continue
            mat = sub.pivot_table(index="source_task", columns="target_task",
                                  values="delta", aggfunc="mean").reindex(index=tasks, columns=tasks)
            fro = antisymmetric_frobenius(mat.values)
            comp_rows.append({"steps": s, "variant": variant, "antisymmetric_fro": fro})
            heatmap(
                mat.values,
                row_labels=tasks, col_labels=tasks,
                title=f"Δloss matrix ({s}-step, {variant})",
                out_path=out_dir / f"affinity_matrix_{s}step_{variant}.png",
            )
    pd.DataFrame(comp_rows).to_csv(out_dir / "raw_vs_norm_symmetry.csv", index=False)
