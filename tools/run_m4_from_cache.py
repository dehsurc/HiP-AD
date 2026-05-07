"""Run M4 (cosine ↔ ΔL correlation) using existing Phase 1 CSVs — no GPU needed.

Reads:
  <results_root>/ckpt_<tag>/conflict/conflict_{a}_{b}_per_batch.csv  → cos_dfs
  <results_root>/ckpt_<tag>/probe/probe_per_batch.csv                → probe_df

Writes:
  <results_root>/ckpt_<tag>/correlation/correlation_table.csv
  <results_root>/ckpt_<tag>/correlation/binned_*.csv
  <results_root>/ckpt_<tag>/correlation/scatter_*.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.gradient_analysis.correlation import run_m4


TASKS = ["det", "map", "motion", "plan"]
COSINE_BINS = [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]

# Regex: conflict_{task_a}_{task_b}_per_batch.csv
_PAT = re.compile(r"conflict_([^_]+(?:_[^_]+)*)_([^_]+(?:_[^_]+)*)_per_batch\.csv$")


def _task_pair_from_name(fname: str):
    """Extract (task_a, task_b) from conflict filename using known TASKS list."""
    for a in TASKS:
        for b in TASKS:
            if fname == f"conflict_{a}_{b}_per_batch.csv":
                return a, b
    return None, None


def run_ckpt(ckpt_dir: Path, steps_list: list[int], variants: list[str]) -> None:
    tag = ckpt_dir.name
    conflict_dir = ckpt_dir / "conflict"
    probe_csv = ckpt_dir / "probe" / "probe_per_batch.csv"

    if not probe_csv.exists():
        print(f"[{tag}] probe_per_batch.csv not found — skipping")
        return
    if not conflict_dir.exists():
        print(f"[{tag}] conflict/ dir not found — skipping")
        return

    # Load probe_df (full-update rows only)
    probe_df = pd.read_csv(probe_csv)
    probe_df = probe_df[probe_df["layer"] == "_all"].copy()
    if probe_df.empty:
        print(f"[{tag}] no layer=_all rows in probe_per_batch.csv — skipping")
        return

    # Build cos_dfs from per-batch conflict CSVs
    cos_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    for csv_path in sorted(conflict_dir.glob("conflict_*_per_batch.csv")):
        a, b = _task_pair_from_name(csv_path.name)
        if a is None:
            print(f"[{tag}] skipping unrecognised file: {csv_path.name}")
            continue
        df = pd.read_csv(csv_path)
        if "batch_idx" not in df.columns or "cos" not in df.columns:
            print(f"[{tag}] missing columns in {csv_path.name} — skipping pair")
            continue
        cos_dfs[(a, b)] = df

    if not cos_dfs:
        print(f"[{tag}] no valid conflict CSVs — skipping")
        return

    out_dir = ckpt_dir / "correlation"
    print(f"[{tag}] running M4 ({len(cos_dfs)} pairs) → {out_dir}")
    run_m4(
        cos_dfs=cos_dfs,
        probe_df=probe_df,
        steps_list=steps_list,
        variants=variants,
        cosine_bins=COSINE_BINS,
        out_dir=out_dir,
    )
    print(f"[{tag}] done — {out_dir / 'correlation_table.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run M4 from cached CSVs (no GPU)")
    ap.add_argument("--results-root", default="gradient_analysis_results_phase1",
                    help="directory containing ckpt_* subdirs")
    ap.add_argument("--checkpoints", default="1ep,3ep,6ep,18ep",
                    help="comma-separated checkpoint tags")
    ap.add_argument("--steps", default="1", help="comma-separated step counts")
    ap.add_argument("--variants", default="raw,normalized")
    args = ap.parse_args()

    root = Path(args.results_root)
    tags = [t.strip() for t in args.checkpoints.split(",")]
    steps_list = [int(s) for s in args.steps.split(",")]
    variants = [v.strip() for v in args.variants.split(",")]

    for tag in tags:
        ckpt_dir = root / f"ckpt_{tag}"
        if not ckpt_dir.exists():
            print(f"[{tag}] directory {ckpt_dir} not found — skipping")
            continue
        run_ckpt(ckpt_dir, steps_list, variants)

    print("M4 complete.")


if __name__ == "__main__":
    main()
