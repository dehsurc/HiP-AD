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


def _add_unordered_pair_cols(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        tuple(sorted((str(a), str(b))))
        for a, b in zip(out[left_col], out[right_col])
    ]
    out["_pair_a"] = [p[0] for p in pairs]
    out["_pair_b"] = [p[1] for p in pairs]
    return out


def join_layer_cos_delta(
    cos_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    steps: int,
    variant: str,
) -> pd.DataFrame:
    """Join layer-level cosine with directional per-layer probe Δloss.

    `cos_df` is the M2 per-batch output with `task_a`, `task_b`, `group`, and
    `cos`. `probe_df` is M3 output with `source_task`, `target_task`, `layer`,
    and `delta`.

    The cosine is symmetric for an unordered pair, while Δloss is directional.
    Therefore one `(task_a, task_b)` cosine row can join to both
    `task_a -> task_b` and `task_b -> task_a` probe rows for the same batch and
    layer.
    """
    if cos_df.empty or probe_df.empty:
        return pd.DataFrame()
    if "layer" not in probe_df.columns:
        return pd.DataFrame()

    probe_sub = probe_df[
        (probe_df["steps"] == steps)
        & (probe_df["variant"] == variant)
        & (probe_df["source_task"] != probe_df["target_task"])
    ].copy()
    if probe_sub.empty:
        return pd.DataFrame()

    cos = _add_unordered_pair_cols(cos_df, "task_a", "task_b")
    probe = _add_unordered_pair_cols(probe_sub, "source_task", "target_task")

    cols = [
        "batch_idx", "source_task", "target_task", "layer", "delta",
        "baseline_loss", "stepped_loss", "grad_norm",
    ]
    if "rel_delta" in probe.columns:
        cols.append("rel_delta")

    merged = cos.merge(
        probe[cols + ["_pair_a", "_pair_b"]],
        left_on=["batch_idx", "group", "_pair_a", "_pair_b"],
        right_on=["batch_idx", "layer", "_pair_a", "_pair_b"],
        how="inner",
    )
    merged = merged.drop(columns=["_pair_a", "_pair_b"])
    merged["steps"] = steps
    merged["variant"] = variant
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


def run_layer_m4(
    cos_dfs: dict[Tuple[str, str], pd.DataFrame],
    probe_df: pd.DataFrame,
    steps_list: List[int],
    variants: List[str],
    cosine_bins: List[float],
    out_dir: Path,
) -> pd.DataFrame:
    """Run layer-level M4 for per-layer probes.

    Emits:
      - `layer_joined_per_batch.csv`: one row per batch x layer x direction
      - `layer_correlation_table.csv`: Pearson/Spearman per layer/direction
      - `layer_binned_delta.csv`: Δloss summaries by cosine bin
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_parts = []
    table_rows = []
    binned_rows = []

    for (a, b), cos_df in cos_dfs.items():
        cos_df = cos_df.copy()
        cos_df["task_a"] = a
        cos_df["task_b"] = b
        for s in steps_list:
            for v in variants:
                joined = join_layer_cos_delta(cos_df, probe_df, steps=s, variant=v)
                if joined.empty:
                    continue
                joined_parts.append(joined)
                group_cols = ["layer", "source_task", "target_task"]
                for (layer, source, target), sub in joined.groupby(group_cols):
                    pearson, spearman = pair_correlations(sub)
                    finite = sub[np.isfinite(sub["cos"]) & np.isfinite(sub["delta"])]
                    conflict = finite[finite["cos"] < 0]
                    coop = finite[finite["cos"] >= 0]
                    table_rows.append({
                        "task_a": a,
                        "task_b": b,
                        "source_task": source,
                        "target_task": target,
                        "layer": layer,
                        "steps": s,
                        "variant": v,
                        "pearson": pearson,
                        "spearman": spearman,
                        "n": int(len(finite)),
                        "mean_cos": float(finite["cos"].mean()) if not finite.empty else float("nan"),
                        "conflict_ratio": float((finite["cos"] < 0).mean()) if not finite.empty else float("nan"),
                        "mean_delta": float(finite["delta"].mean()) if not finite.empty else float("nan"),
                        "helpful_ratio": float((finite["delta"] < 0).mean()) if not finite.empty else float("nan"),
                        "harmful_ratio": float((finite["delta"] > 0).mean()) if not finite.empty else float("nan"),
                        "mean_delta_when_conflict": (
                            float(conflict["delta"].mean()) if not conflict.empty else float("nan")
                        ),
                        "mean_delta_when_coop": (
                            float(coop["delta"].mean()) if not coop.empty else float("nan")
                        ),
                    })

                    if finite.empty:
                        continue
                    bin_df = summarize_bins(
                        finite["cos"].to_numpy(),
                        finite["delta"].to_numpy(),
                        np.asarray(cosine_bins),
                        stat_label="delta",
                    )
                    bin_df.insert(0, "variant", v)
                    bin_df.insert(0, "steps", s)
                    bin_df.insert(0, "layer", layer)
                    bin_df.insert(0, "target_task", target)
                    bin_df.insert(0, "source_task", source)
                    bin_df.insert(0, "task_b", b)
                    bin_df.insert(0, "task_a", a)
                    binned_rows.append(bin_df)

    joined_all = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()
    joined_all.to_csv(out_dir / "layer_joined_per_batch.csv", index=False)
    table = pd.DataFrame(table_rows)
    table.to_csv(out_dir / "layer_correlation_table.csv", index=False)
    binned = pd.concat(binned_rows, ignore_index=True) if binned_rows else pd.DataFrame()
    binned.to_csv(out_dir / "layer_binned_delta.csv", index=False)
    return table


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
