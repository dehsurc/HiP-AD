"""M7 — Task Affinity Asymmetry."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .viz import heatmap


def decompose_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (symmetric, antisymmetric) parts of M."""
    S = (M + M.T) / 2.0
    A = (M - M.T) / 2.0
    return S, A


def top_asymmetric_pairs(M: np.ndarray, labels: Sequence[str], k: int = 5):
    """Rank task pairs by |antisymmetric component|, NaN-aware.

    Pairs whose (i, j) or (j, i) cell is NaN are still emitted (with NaN
    antisym values) so callers see them in the table, but they sort last so
    the top-k ranking is dominated by finite values.
    """
    _, A = decompose_matrix(M)
    candidates = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            v = float(A[i, j])
            abs_v = float(abs(v)) if np.isfinite(v) else float("nan")
            candidates.append({
                "pair": (labels[i], labels[j]),
                "antisym": v,
                "abs_antisym": abs_v,
            })
    # NaN-aware sort: finite values first by descending abs, NaN trailing.
    candidates.sort(
        key=lambda d: (
            0 if np.isfinite(d["abs_antisym"]) else 1,
            -d["abs_antisym"] if np.isfinite(d["abs_antisym"]) else 0.0,
        )
    )
    return candidates[:k]


def run_m7(
    probe_df: pd.DataFrame,
    tasks: List[str],
    steps: int,
    variant: str,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = probe_df[(probe_df["steps"] == steps) & (probe_df["variant"] == variant)]
    mat = sub.pivot_table(index="source_task", columns="target_task",
                          values="delta", aggfunc="mean").reindex(index=tasks, columns=tasks)
    M = mat.values
    S, A = decompose_matrix(M)

    heatmap(M, tasks, tasks, f"Original Δloss ({steps}-step {variant})",
            out_path=out_dir / "original.png")
    heatmap(S, tasks, tasks, "Symmetric part", out_path=out_dir / "symmetric.png")
    heatmap(A, tasks, tasks, "Antisymmetric part", out_path=out_dir / "antisymmetric.png")

    top = top_asymmetric_pairs(M, tasks, k=len(tasks) * (len(tasks) - 1) // 2)
    pd.DataFrame(top).to_csv(out_dir / "top_asymmetric_pairs.csv", index=False)
