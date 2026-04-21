"""Generate a markdown summary highlighting top findings for paper drafting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def generate_summary(output_root: Path, checkpoint_tags: Iterable[str]) -> None:
    output_root = Path(output_root)
    lines = ["# Gradient Analysis Summary\n"]

    # Correlation table — top/bottom pairs across checkpoints
    all_corr = []
    for tag in checkpoint_tags:
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

    # Task affinity asymmetry highlights
    for tag in checkpoint_tags:
        f = output_root / f"ckpt_{tag}" / "asymmetry" / "top_asymmetric_pairs.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"\n## Top asymmetric pairs @ {tag}\n")
            lines.append(df.head(5).to_markdown(index=False) + "\n")

    # GradNorm raw vs normalized symmetry
    lines.append("\n## Raw vs Normalized probe symmetry (Frobenius of antisymmetric Δloss)\n")
    for tag in checkpoint_tags:
        f = output_root / f"ckpt_{tag}" / "gradnorm" / "raw_vs_norm_symmetry.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"### {tag}\n")
            lines.append(df.to_markdown(index=False) + "\n")

    (output_root / "summary_report.md").write_text("\n".join(lines))
