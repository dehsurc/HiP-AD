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
