"""Bin 1-D scalar data and summarize per-bin statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bin_by_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return 0-based bin index for each x, inclusive on both ends of the last bin.

    Values outside [edges[0], edges[-1]] are clipped to the nearest edge.
    """
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a 1-D array with at least 2 elements")
    # np.digitize with right=False puts x == edge[i] in bin i; we want last-bin inclusion.
    idx = np.digitize(x, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(int)


def summarize_bins(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    stat_label: str = "y",
) -> pd.DataFrame:
    """For each bin defined by `edges`, compute count, mean/std/median of y,
    and helpful_ratio = fraction of y with y < 0.

    Rows are returned for every bin, including empty bins (count = 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    bins = bin_by_edges(x, edges)
    n_bins = len(edges) - 1

    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_idx": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "count": 0,
                    f"{stat_label}_mean": float("nan"),
                    f"{stat_label}_std": float("nan"),
                    f"{stat_label}_median": float("nan"),
                    "helpful_ratio": float("nan"),
                }
            )
            continue
        sub = y[mask]
        rows.append(
            {
                "bin_idx": b,
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": n,
                f"{stat_label}_mean": float(sub.mean()),
                f"{stat_label}_std": float(sub.std(ddof=0)),
                f"{stat_label}_median": float(np.median(sub)),
                "helpful_ratio": float((sub < 0).mean()),
            }
        )
    return pd.DataFrame(rows)
