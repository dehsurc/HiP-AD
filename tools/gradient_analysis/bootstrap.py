"""M-N3 — Bootstrap CI helper (Phase 1 #3).

A single BCa bootstrap implementation that the rest of the pipeline calls
into. Adds {`<col>_ci_lo`, `<col>_ci_hi`} columns to existing summary
DataFrames so downstream consumers don't need to know how the CI was
computed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def bca_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (lo, hi) of the BCa bootstrap CI for `statistic` of `samples`.

    Constant-input edge case is handled explicitly because scipy's
    `bootstrap` raises a DegenerateDataWarning and returns NaNs there.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    if np.allclose(samples, samples[0]):
        v = float(statistic(samples))
        return v, v
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (samples,),
        statistic,
        method="BCa",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng,
        vectorized=False,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def augment_summary_with_ci(
    df: pd.DataFrame,
    per_batch_df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
    n_resamples: int = 2000,
    seed: int = 0,
    rate_value_cols: Optional[List[str]] = None,
    rate_per_batch_col: str = "cos",
    rate_predicate=lambda x: x < 0,
) -> pd.DataFrame:
    """Attach `<col>_ci_lo`, `<col>_ci_hi` to `df` for each `value_col` keyed
    on `group_cols`.

    Two code paths:

    * **Mean / median columns** (`value_cols`): summary column name is mapped
      to a per-batch column by stripping `mean_` / `median_` prefix; resamples
      that column's values with bootstrap.

    * **Rate columns** (`rate_value_cols`): summary column is treated as a
      Bernoulli rate computed by `rate_predicate(per_batch_df[rate_per_batch_col])`.
      Bootstrap resamples that boolean array and reports the CI of its mean.
      Used for `conflict_ratio` whose per-batch counterpart is `(cos < 0)`.
    """
    out = df.copy()

    # ---- mean / median path ----
    for vcol in value_cols:
        per_batch_col = vcol
        for prefix in ("mean_", "median_"):
            if vcol.startswith(prefix):
                per_batch_col = vcol[len(prefix):]
                break
        if per_batch_col not in per_batch_df.columns:
            # Skip silently if upstream summary contains a column the per-batch
            # CSV doesn't (e.g. mean_norm_a_conflict — derived in summary only).
            continue
        lo_col = f"{vcol}_ci_lo"
        hi_col = f"{vcol}_ci_hi"
        out[lo_col] = np.nan
        out[hi_col] = np.nan
        for idx, row in out.iterrows():
            mask = np.ones(len(per_batch_df), dtype=bool)
            for gc in group_cols:
                mask &= (per_batch_df[gc] == row[gc])
            samples = per_batch_df.loc[mask, per_batch_col].to_numpy()
            stat = np.mean if vcol.startswith("mean_") else np.median
            lo, hi = bca_ci(samples, statistic=stat, n_resamples=n_resamples,
                            seed=seed + int(idx))
            out.at[idx, lo_col] = lo
            out.at[idx, hi_col] = hi

    # ---- rate path ----
    if rate_value_cols and rate_per_batch_col in per_batch_df.columns:
        for vcol in rate_value_cols:
            lo_col = f"{vcol}_ci_lo"
            hi_col = f"{vcol}_ci_hi"
            out[lo_col] = np.nan
            out[hi_col] = np.nan
            for idx, row in out.iterrows():
                mask = np.ones(len(per_batch_df), dtype=bool)
                for gc in group_cols:
                    mask &= (per_batch_df[gc] == row[gc])
                values = per_batch_df.loc[mask, rate_per_batch_col].to_numpy()
                # Drop NaN rows so the rate matches summarize_pair's n_valid.
                values = values[~np.isnan(values)] if values.size else values
                if values.size == 0:
                    continue
                bools = rate_predicate(values).astype(np.float64)
                lo, hi = bca_ci(bools, statistic=np.mean, n_resamples=n_resamples,
                                seed=seed + int(idx) + 10_000)
                out.at[idx, lo_col] = lo
                out.at[idx, hi_col] = hi
    return out
