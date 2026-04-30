"""M4 — Correlation & Binning.

Joins batch-level cosine similarity (from M2) with batch-level Δloss (from M3),
computes Pearson/Spearman correlation, emits binned summaries, and plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .binning import summarize_bins
from .viz import scatter_with_bins


def join_cos_delta(
    cos_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    steps: int,
    variant: str,
) -> pd.DataFrame:
    """Inner-join on (batch_idx, task_a=source, task_b=target) for given steps/variant."""
    probe_sub = probe_df[(probe_df["steps"] == steps) & (probe_df["variant"] == variant)]
    cols = ["batch_idx", "source_task", "target_task", "delta"]
    if "rel_delta" in probe_sub.columns:
        cols.append("rel_delta")
    merged = cos_df.merge(
        probe_sub[cols],
        left_on=["batch_idx", "task_a", "task_b"],
        right_on=["batch_idx", "source_task", "target_task"],
        how="inner",
    )
    return merged


def pair_correlations(df: pd.DataFrame) -> Tuple[float, float]:
    """Return (pearson, spearman) between cos and delta.

    scipy's pearsonr/spearmanr asserts ``asarray_chkfinite`` and raises
    ``ValueError: array must not contain infs or NaNs`` when fed any
    non-finite value. The probe occasionally emits NaN ``delta`` rows
    (e.g. baseline loss exactly zero, or a task absent on a given batch);
    drop them defensively so the correlation step doesn't crash the run.
    """
    from scipy.stats import pearsonr, spearmanr

    finite = (
        np.isfinite(df["cos"].to_numpy(dtype=float))
        & np.isfinite(df["delta"].to_numpy(dtype=float))
    )
    df = df.loc[finite]
    if len(df) < 3:
        return float("nan"), float("nan")
    p, _ = pearsonr(df["cos"], df["delta"])
    s, _ = spearmanr(df["cos"], df["delta"])
    return float(p), float(s)


def run_m4(
    cos_dfs: dict[Tuple[str, str], pd.DataFrame],   # (a,b) -> cos per-batch df (must have batch_idx, group, cos)
    probe_df: pd.DataFrame,
    steps_list: List[int],
    variants: List[str],
    cosine_bins: List[float],
    out_dir: Path,
) -> pd.DataFrame:
    """Run M4 across all pairs/variants/steps. Emit CSVs + scatter plots.

    Returns a long-format correlation table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (a, b), cos_df in cos_dfs.items():
        cos_df = cos_df.copy()
        cos_df["task_a"] = a
        cos_df["task_b"] = b
        # Aggregate cos across groups to one scalar per batch (mean of groups).
        per_batch = cos_df.groupby("batch_idx", as_index=False)["cos"].mean()
        per_batch["task_a"] = a
        per_batch["task_b"] = b
        for s in steps_list:
            for v in variants:
                joined = join_cos_delta(per_batch, probe_df, steps=s, variant=v)
                if joined.empty:
                    continue
                pearson, spearman = pair_correlations(joined)
                bin_df = summarize_bins(
                    joined["cos"].to_numpy(),
                    joined["delta"].to_numpy(),
                    np.asarray(cosine_bins),
                    stat_label="delta",
                )
                bin_df.insert(0, "variant", v)
                bin_df.insert(0, "steps", s)
                bin_df.insert(0, "task_b", b)
                bin_df.insert(0, "task_a", a)
                bin_df.to_csv(out_dir / f"binned_{a}_{b}_{s}step_{v}.csv", index=False)
                scatter_with_bins(
                    joined["cos"].to_numpy(),
                    joined["delta"].to_numpy(),
                    bin_df,
                    x_label=f"cos(g_{a}, g_{b})",
                    y_label=f"ΔL_{b}",
                    title=f"{a}→{b} ({s}-step, {v})  Pearson={pearson:.2f}",
                    out_path=out_dir / f"scatter_{a}_{b}_{s}step_{v}.png",
                    y_stat_col="delta",
                )
                rows.append({
                    "task_a": a, "task_b": b, "steps": s, "variant": v,
                    "pearson": pearson, "spearman": spearman, "n": len(joined),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "correlation_table.csv", index=False)
    return table
