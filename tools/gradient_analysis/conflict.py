"""M2 — Conflict Analysis (cosine + projection decomposition)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


EPS = 1e-8


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """Return cosine similarity; NaN if either norm is below EPS.

    Computed in float64 to avoid float32 round-off (so identical/opposed
    vectors return exactly 1.0 / -1.0), then clamped to [-1.0, 1.0] as a
    final safeguard.
    """
    n1 = g1.norm()
    n2 = g2.norm()
    if float(n1) < EPS or float(n2) < EPS:
        return float("nan")
    g1d = g1.to(torch.float64)
    g2d = g2.to(torch.float64)
    cos = float(torch.dot(g1d, g2d) / (g1d.norm() * g2d.norm()))
    if cos > 1.0:
        return 1.0
    if cos < -1.0:
        return -1.0
    return cos


def projection_decomposition(g_a: torch.Tensor, g_b: torch.Tensor) -> Tuple[float, float]:
    """Split ||proj_{g_b}(g_a)|| into cooperative (cos >= 0) and conflicting (cos < 0) magnitudes.

    Returns (cooperative_magnitude, conflicting_magnitude). Exactly one is nonzero.
    """
    n_b = g_b.norm()
    if float(n_b) < EPS:
        return 0.0, 0.0
    dot = float(torch.dot(g_a, g_b))
    mag = abs(dot) / float(n_b)
    if dot >= 0:
        return mag, 0.0
    return 0.0, mag


def analyze_pair_batches(
    batches: Iterable[Dict[str, Dict[str, torch.Tensor]]],
    task_a: str,
    task_b: str,
    group_keys: List[str],
) -> pd.DataFrame:
    """For each batch and each param group, compute cosine + projection decomposition
    between task_a and task_b gradients.

    `batches` is iterable of dicts: task -> {group_key -> flat tensor}.
    Returns a long-format DataFrame (one row per batch x group).
    """
    rows = []
    for i, b in enumerate(batches):
        ga_groups = b.get(task_a, {})
        gb_groups = b.get(task_b, {})
        for gk in group_keys:
            ga = ga_groups.get(gk)
            gb = gb_groups.get(gk)
            if ga is None or gb is None:
                continue
            cos = cosine_similarity(ga, gb)
            coop, conf = projection_decomposition(ga, gb)
            rows.append(
                {
                    "batch_idx": i,
                    "group": gk,
                    "cos": cos,
                    "coop_mag": coop,
                    "conf_mag": conf,
                    "norm_a": float(ga.norm()),
                    "norm_b": float(gb.norm()),
                }
            )
    return pd.DataFrame(rows)


def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(task_pair, group) across batches."""
    grouped = df.groupby("group").agg(
        mean_cos=("cos", "mean"),
        std_cos=("cos", "std"),
        median_cos=("cos", "median"),
        n=("cos", "count"),
        conflict_ratio=("cos", lambda s: float((s < 0).mean())),
        mean_coop_mag=("coop_mag", "mean"),
        mean_conf_mag=("conf_mag", "mean"),
    ).reset_index()
    return grouped


def run_m2(
    cached_batches: List[Dict],  # list of BatchGradients-like dicts with .shared
    tasks: List[str],
    group_keys: List[str],
    out_dir: Path,
) -> None:
    """Orchestrate M2 over all task pairs; write CSVs + plots per pair."""
    from .viz import violin_by_pair
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_violin: Dict[str, np.ndarray] = {}

    for a_idx, a in enumerate(tasks):
        for b in tasks[a_idx + 1:]:
            batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached_batches]
            df = analyze_pair_batches(batches, a, b, group_keys)
            if df.empty:
                continue
            df.to_csv(out_dir / f"conflict_{a}_{b}_per_batch.csv", index=False)
            summary = summarize_pair(df)
            summary.insert(0, "task_a", a)
            summary.insert(1, "task_b", b)
            summary.to_csv(out_dir / f"conflict_{a}_{b}_summary.csv", index=False)
            per_pair_violin[f"{a}|{b}"] = df["cos"].to_numpy()

    if per_pair_violin:
        violin_by_pair(
            per_pair_violin,
            y_label="cosine similarity",
            title="Per-pair cosine across batches (all groups)",
            out_path=out_dir / "cosine_violin.png",
        )
