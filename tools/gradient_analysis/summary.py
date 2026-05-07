"""Generate a markdown summary highlighting top findings for paper drafting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _load_conflict_summaries(output_root: Path, checkpoint_tags: Iterable[str]) -> pd.DataFrame:
    """Concatenate every per-checkpoint conflict_*_summary.csv into one frame
    with checkpoint, task_a, task_b columns added."""
    frames: List[pd.DataFrame] = []
    for tag in checkpoint_tags:
        conflict_dir = output_root / f"ckpt_{tag}" / "conflict"
        if not conflict_dir.exists():
            continue
        for f in sorted(conflict_dir.glob("conflict_*_summary.csv")):
            df = pd.read_csv(f)
            # Filename pattern: conflict_<a>_<b>_summary.csv → recover the pair
            stem = f.stem.replace("conflict_", "").replace("_summary", "")
            parts = stem.split("_")
            if "task_a" not in df.columns and len(parts) >= 2:
                df["task_a"] = parts[0]
                df["task_b"] = "_".join(parts[1:])
            df["checkpoint"] = tag
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _write_conflict_rankings(lines: List[str], summary_df: pd.DataFrame) -> None:
    """Phase 1 #4 B4 — split rankings into real-shared vs pseudo-shared.

    Pseudo-shared groups (one task's gradient does not flow through the group
    on ≥ 50 % of batches) carry a meaningless cosine and were the dominant
    population of the previous "Lowest conflict_ratio" ranking — the new
    layout moves them to a clearly-labelled appendix.
    """
    if summary_df.empty:
        return
    cols = [c for c in ("checkpoint", "task_a", "task_b", "group",
                        "n_total", "n_valid", "conflict_ratio",
                        "mean_cos", "pseudo_shared_ratio")
            if c in summary_df.columns]
    if "is_pseudo_shared_group" in summary_df.columns:
        real = summary_df[~summary_df["is_pseudo_shared_group"].fillna(False)]
        pseudo = summary_df[summary_df["is_pseudo_shared_group"].fillna(False)]
    else:
        # Legacy summaries without the validity columns: no partition possible.
        real = summary_df
        pseudo = summary_df.iloc[0:0]

    if not real.empty and "conflict_ratio" in real.columns:
        lines.append("\n## Highest conflict_ratio (real shared groups)\n")
        top = real.sort_values("conflict_ratio", ascending=False).head(10)
        lines.append(top[cols].to_markdown(index=False) + "\n")
        lines.append("\n## Lowest conflict_ratio (real shared groups)\n")
        low = real.sort_values("conflict_ratio", ascending=True).head(10)
        lines.append(low[cols].to_markdown(index=False) + "\n")

    if not pseudo.empty:
        lines.append("\n## Pseudo-shared groups (excluded from main analysis)\n")
        lines.append(
            "These groups had ≥ 50 % of batches where one task's gradient was "
            "below EPS. Their cosine values are not interpretable as a measure "
            "of inter-task interaction.\n"
        )
        pcols = [c for c in ("checkpoint", "task_a", "task_b", "group",
                             "n_total", "n_pseudo_shared", "pseudo_shared_ratio")
                 if c in pseudo.columns]
        lines.append(pseudo[pcols].to_markdown(index=False) + "\n")


def generate_summary(output_root: Path, checkpoint_tags: Iterable[str]) -> None:
    output_root = Path(output_root)
    tags = list(checkpoint_tags)
    lines = ["# Gradient Analysis Summary\n"]

    # Correlation table — top/bottom pairs across checkpoints
    all_corr = []
    for tag in tags:
        f = output_root / f"ckpt_{tag}" / "correlation" / "correlation_table.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["checkpoint"] = tag
            all_corr.append(df)
    if all_corr:
        corr = pd.concat(all_corr, ignore_index=True)
        corr_1raw = corr[(corr["steps"] == 1) & (corr["variant"] == "raw")]
        lines.append("## Strongest correlation |cos ↔ Δloss| (1-step raw)\n")
        top = corr_1raw.reindex(corr_1raw["pearson"].abs().sort_values(ascending=False).index).head(10)
        lines.append(top.to_markdown(index=False) + "\n")

    # Conflict rankings — split into real-shared vs pseudo-shared (Phase 1 #4 B4)
    conflict_df = _load_conflict_summaries(output_root, tags)
    _write_conflict_rankings(lines, conflict_df)

    # Task affinity asymmetry highlights
    for tag in tags:
        f = output_root / f"ckpt_{tag}" / "asymmetry" / "top_asymmetric_pairs.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"\n## Top asymmetric pairs @ {tag}\n")
            lines.append(df.head(5).to_markdown(index=False) + "\n")

    # GradNorm raw vs normalized symmetry
    lines.append("\n## Raw vs Normalized probe symmetry (Frobenius of antisymmetric Δloss)\n")
    for tag in tags:
        f = output_root / f"ckpt_{tag}" / "gradnorm" / "raw_vs_norm_symmetry.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"### {tag}\n")
            lines.append(df.to_markdown(index=False) + "\n")

    # Distribution diagnostics summary (Phase 1 #2)
    dist_rows = []
    for tag in tags:
        f = output_root / f"ckpt_{tag}" / "distribution" / "distribution_report.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["checkpoint"] = tag
        dist_rows.append(df)
    if dist_rows:
        dist_df = pd.concat(dist_rows, ignore_index=True)
        lines.append("\n## Distribution shape census (per checkpoint)\n")
        census = dist_df.groupby(["checkpoint", "shape_label"]).size().unstack(fill_value=0)
        lines.append(census.to_markdown() + "\n")
        bimodal = dist_df[dist_df["shape_label"] == "bimodal"]
        if not bimodal.empty:
            lines.append("\n### Bimodal cells (mean is misleading — report modes instead)\n")
            bcols = [c for c in ("checkpoint", "task_a", "task_b", "group",
                                 "dip_p_value", "p5", "p50", "p95")
                     if c in bimodal.columns]
            lines.append(bimodal[bcols].to_markdown(index=False) + "\n")

    # Null-baseline pass rates (Phase 1 #1)
    nb_rows = []
    for tag in tags:
        f = output_root / f"ckpt_{tag}" / "null_baseline" / "null_baseline.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["checkpoint"] = tag
        nb_rows.append(df)
    if nb_rows:
        nb_df = pd.concat(nb_rows, ignore_index=True)
        lines.append("\n## Null-baseline pass rate (Phase 1 #1)\n")
        lines.append(
            "Fraction of (group, pair, kind) cells where observed cosine is "
            "distinguishable from the permutation null at |r| ≥ 0.1 and "
            "Bonferroni-corrected p ≤ 0.05.\n"
        )
        rate = nb_df.groupby("checkpoint")["passes_noise_threshold"].mean()
        lines.append(rate.to_frame("pass_rate").to_markdown() + "\n")

    (output_root / "summary_report.md").write_text("\n".join(lines))
