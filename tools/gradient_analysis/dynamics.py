"""M6 — Training Dynamics.

Reads per-checkpoint CSVs (conflict summaries, probe matrices, gradnorm tables)
and produces cross-checkpoint time-series plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .viz import timeseries


def _epoch_from_tag(tag: str) -> float:
    # "1ep" -> 1.0
    return float(tag.rstrip("epoch").rstrip("ep"))


def run_m6(
    per_ckpt_dirs: Dict[str, Path],   # tag -> ckpt_dir like gradient_analysis_results/ckpt_1ep
    tasks: List[str],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cosine dynamics (mean across groups per pair)
    cos_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        conflict_dir = Path(d) / "conflict"
        for f in conflict_dir.glob("conflict_*_summary.csv"):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                cos_rows.append({
                    "epoch": ep,
                    "pair": f"{row['task_a']}|{row['task_b']}",
                    "group": row["group"],
                    "mean_cos": row["mean_cos"],
                    "conflict_ratio": row["conflict_ratio"],
                })
    cos_df = pd.DataFrame(cos_rows)
    if not cos_df.empty:
        cos_df.to_csv(out_dir / "cos_dynamics.csv", index=False)
        # Aggregate groups: mean across groups per pair
        agg = cos_df.groupby(["epoch", "pair"], as_index=False)[["mean_cos", "conflict_ratio"]].mean()
        timeseries(
            agg.to_dict("records"),
            x_key="epoch", y_key="mean_cos", series_key="pair",
            title="Mean cosine similarity over epochs",
            out_path=out_dir / "timeseries_cosine.png",
        )
        timeseries(
            agg.to_dict("records"),
            x_key="epoch", y_key="conflict_ratio", series_key="pair",
            title="Conflict ratio over epochs",
            out_path=out_dir / "timeseries_conflict_ratio.png",
        )

    # Helpful ratio from probe
    help_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        probe_csv = Path(d) / "probe" / "probe_per_batch.csv"
        if not probe_csv.exists():
            continue
        pdf = pd.read_csv(probe_csv)
        pdf = pdf[(pdf["steps"] == 1) & (pdf["variant"] == "raw")]
        agg = pdf.groupby(["source_task", "target_task"]).agg(
            helpful_ratio=("delta", lambda s: float((s < 0).mean()))
        ).reset_index()
        for _, r in agg.iterrows():
            if r["source_task"] == r["target_task"]:
                continue
            help_rows.append({
                "epoch": ep,
                "pair": f"{r['source_task']}→{r['target_task']}",
                "helpful_ratio": r["helpful_ratio"],
            })
    help_df = pd.DataFrame(help_rows)
    if not help_df.empty:
        help_df.to_csv(out_dir / "helpful_dynamics.csv", index=False)
        timeseries(
            help_df.to_dict("records"),
            x_key="epoch", y_key="helpful_ratio", series_key="pair",
            title="Helpful ratio over epochs (1-step raw)",
            out_path=out_dir / "timeseries_helpful.png",
        )

    # Norm ratio dynamics
    norm_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        nf = Path(d) / "gradnorm" / "per_task_norm.csv"
        if not nf.exists():
            continue
        ndf = pd.read_csv(nf)
        means = ndf.groupby("task")["norm"].mean()
        for t in tasks:
            if t not in means.index:
                continue
            norm_rows.append({"epoch": ep, "task": t, "mean_norm": float(means[t])})
    norm_df = pd.DataFrame(norm_rows)
    if not norm_df.empty:
        norm_df.to_csv(out_dir / "norm_dynamics.csv", index=False)
        timeseries(
            norm_df.to_dict("records"),
            x_key="epoch", y_key="mean_norm", series_key="task",
            title="Per-task mean gradient norm over epochs",
            out_path=out_dir / "timeseries_norm.png",
        )
