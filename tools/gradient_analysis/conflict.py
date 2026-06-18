"""M2 — Conflict Analysis (cosine + projection decomposition)."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


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
    """Aggregate per-(task_pair, group) across batches with validity-aware
    counts (Phase 1 #4 B2/B3).

    Validity rule: a row contributes to ``n_valid`` iff cos is finite *and*
    norm_a > EPS *and* norm_b > EPS. Otherwise the row is "pseudo-shared"
    (one task's gradient does not flow through this group on this batch),
    and including it in the conflict-ratio denominator systematically biases
    the ratio toward zero. Cosine summary stats are computed over the valid
    subset; the legacy ratio (over n_total) is preserved one cycle as
    ``conflict_ratio_legacy`` for plot reproducibility.
    """
    summary_cols = [
        "group", "mean_cos", "std_cos", "median_cos", "n", "n_total",
        "n_valid", "n_pseudo_shared", "n_nan", "pseudo_shared_ratio",
        "is_pseudo_shared_group", "conflict_ratio", "conflict_ratio_legacy",
        "mean_coop_mag", "mean_conf_mag", "mean_norm_a_conflict",
        "mean_norm_b_conflict", "mean_norm_ratio_conflict",
        "median_norm_ratio_conflict", "n_conflict", "mean_norm_a_coop",
        "mean_norm_b_coop", "mean_norm_ratio_coop",
        "median_norm_ratio_coop", "n_coop",
    ]
    if df.empty:
        return pd.DataFrame(columns=summary_cols)

    def _agg(sub: pd.DataFrame) -> pd.Series:
        n_total = int(len(sub))
        finite = sub["cos"].notna()
        nonzero = (sub["norm_a"] > EPS) & (sub["norm_b"] > EPS)
        valid = finite & nonzero
        n_valid = int(valid.sum())
        # Pseudo-shared: at least one task gradient was below EPS (Phase 1 #4 B2).
        # Includes both NaN cosines from zero-norm and the rare finite-cos-but-
        # zero-norm edge case so the partition invariant holds:
        #   n_total = n_valid + n_pseudo_shared + n_nan_other.
        n_pseudo_shared = int((~nonzero).sum())
        # NaN-other: cosine is NaN despite both norms > EPS (numerical edge
        # case, e.g. ill-conditioned dot products). Should be very rare.
        n_nan = int(((~finite) & nonzero).sum())
        valid_sub = sub[valid]

        if n_valid > 0:
            mean_cos = float(valid_sub["cos"].mean())
            median_cos = float(valid_sub["cos"].median())
            std_cos = float(valid_sub["cos"].std())
            conflict_ratio = float((valid_sub["cos"] < 0).mean())
        else:
            mean_cos = float("nan"); median_cos = float("nan")
            std_cos = float("nan"); conflict_ratio = float("nan")

        # Legacy ratio: prior behaviour with n_total denominator (kept one cycle).
        if n_total > 0:
            cos_neg_count = int((sub["cos"].fillna(0.0) < 0).sum())
            conflict_ratio_legacy = float(cos_neg_count) / float(n_total)
        else:
            conflict_ratio_legacy = float("nan")
        pseudo_ratio = float(n_pseudo_shared) / n_total if n_total > 0 else float("nan")

        out = {
            "mean_cos": mean_cos,
            "std_cos": std_cos,
            "median_cos": median_cos,
            "n": n_total,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_pseudo_shared": n_pseudo_shared,
            "n_nan": n_nan,
            "pseudo_shared_ratio": pseudo_ratio,
            "is_pseudo_shared_group": bool(np.isfinite(pseudo_ratio) and pseudo_ratio >= 0.5),
            "conflict_ratio": conflict_ratio,
            "conflict_ratio_legacy": conflict_ratio_legacy,
            "mean_coop_mag": float(sub["coop_mag"].mean()) if n_total > 0 else float("nan"),
            "mean_conf_mag": float(sub["conf_mag"].mean()) if n_total > 0 else float("nan"),
        }

        # Conditional norm stats — restricted to the valid subset so NaN/zero-
        # norm rows don't leak in.
        conf = valid_sub[valid_sub["cos"] < 0]
        coop = valid_sub[valid_sub["cos"] >= 0]

        def _cond(frame: pd.DataFrame, suffix: str) -> Dict[str, float]:
            if frame.empty:
                return {
                    f"mean_norm_a_{suffix}": float("nan"),
                    f"mean_norm_b_{suffix}": float("nan"),
                    f"mean_norm_ratio_{suffix}": float("nan"),
                    f"median_norm_ratio_{suffix}": float("nan"),
                    f"n_{suffix}": 0,
                }
            ratio = frame["norm_a"] / frame["norm_b"].where(frame["norm_b"] > EPS)
            return {
                f"mean_norm_a_{suffix}": float(frame["norm_a"].mean()),
                f"mean_norm_b_{suffix}": float(frame["norm_b"].mean()),
                f"mean_norm_ratio_{suffix}": float(ratio.mean()),
                f"median_norm_ratio_{suffix}": float(ratio.median()),
                f"n_{suffix}": int(len(frame)),
            }
        out.update(_cond(conf, "conflict"))
        out.update(_cond(coop, "coop"))
        return pd.Series(out)

    grouped = df.groupby("group").apply(_agg).reset_index()
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

    pair_list = [(a, b) for a_idx, a in enumerate(tasks) for b in tasks[a_idx + 1:]]
    t_start = time.time()
    for a, b in tqdm(
        pair_list, desc=f"M2 conflict [{out_dir.name}]",
        dynamic_ncols=True, mininterval=1.0,
    ):
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
    print(
        f"[M2] {len(pair_list)} pairs × {len(cached_batches)} batches "
        f"({out_dir.name}) done in {(time.time() - t_start)/60:.2f} min"
    )

    if per_pair_violin:
        violin_by_pair(
            per_pair_violin,
            y_label="cosine similarity",
            title="Per-pair cosine across batches (all groups)",
            out_path=out_dir / "cosine_violin.png",
        )
