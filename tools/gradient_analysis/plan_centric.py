"""Plan-centric directed transfer analysis (Phase 2 refactor).

Compute helpers and summary builders for the post-conflict, plan-target
analysis. The matching notebook is
``docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/
phase2_hipad_vad_analysis_visualization.ipynb``; the design doc is in the
same directory dated 2026-05-12.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-8


def ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> bool:
    """Warn (don't raise) when required columns are absent.

    Returns True iff all required columns are present. Caller decides whether
    to SKIP or continue.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(
            f"[{df_name}] missing columns: {missing}",
            stacklevel=2,
        )
        return False
    return True


def bootstrap_ci(
    values,
    stat_fn=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat_fn`` over ``values``.

    Returns ``(nan, nan)`` when fewer than 5 finite samples are available.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = stat_fn(v[idx[i]])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def practical_threshold(
    series, ratio: float = 0.1, min_value: float = 1e-8
) -> float:
    """Return ``max(min_value, ratio * median(|series|))``.

    Noise-robust threshold used by helpful/harmful rate calculations so that
    arbitrarily small deltas are not counted as helpful.
    """
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(min_value)
    return float(max(min_value, ratio * float(np.median(np.abs(v)))))


def standardize_probe_df(probe: pd.DataFrame) -> pd.DataFrame:
    """Add ``gain``, ``gain_rel``, ``delta_rel`` and ``loss_before/after`` aliases.

    The on-disk probe schema uses ``baseline_loss`` / ``stepped_loss``; the
    spec text uses ``loss_before`` / ``loss_after``. We expose both names so
    downstream code can pick either.
    """
    df = probe.copy()

    if "loss_before" not in df.columns and "baseline_loss" in df.columns:
        df["loss_before"] = df["baseline_loss"]
    if "loss_after" not in df.columns and "stepped_loss" in df.columns:
        df["loss_after"] = df["stepped_loss"]

    df["gain"] = -df["delta"]

    if {"loss_before", "loss_after"}.issubset(df.columns):
        df["delta_rel"] = (df["loss_after"] - df["loss_before"]) / (
            df["loss_before"].abs() + EPS
        )
        df["gain_rel"] = -df["delta_rel"]
    elif "rel_delta" in df.columns:
        df["delta_rel"] = df["rel_delta"]
        df["gain_rel"] = -df["rel_delta"]
        warnings.warn(
            "standardize_probe_df: loss_before/loss_after absent; "
            "falling back to existing rel_delta as delta_rel.",
            stacklevel=2,
        )
    else:
        df["delta_rel"] = np.nan
        df["gain_rel"] = np.nan
        warnings.warn(
            "standardize_probe_df: loss_before/loss_after and rel_delta "
            "both absent; delta_rel/gain_rel set to NaN.",
            stacklevel=2,
        )
    return df
