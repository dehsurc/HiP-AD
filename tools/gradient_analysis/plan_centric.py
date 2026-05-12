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


def get_probe_base(
    probe: pd.DataFrame, variant: str = "normalized", steps: int = 1
) -> pd.DataFrame:
    """Standardize, then filter to ``(variant, steps)`` subset."""
    df = standardize_probe_df(probe)
    return df[(df["variant"] == variant) & (df["steps"] == steps)].copy()


_MAIN_SOURCES = ("det", "map", "motion", "plan")
_GROUP_KEYS_FH = ["model", "checkpoint", "checkpoint_order", "layer", "source_task"]


def _ci_pair(values, seed_base: int) -> tuple[float, float]:
    return bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, seed=seed_base)


def build_plan_transfer_summary(base: pd.DataFrame) -> pd.DataFrame:
    """Part F. Plan-centric directed transfer summary.

    ``base`` must already be ``get_probe_base(..., variant="normalized",
    steps=1)`` output. Returns one row per ``(model, checkpoint, layer,
    source_task)`` restricted to ``target_task == "plan"``.
    """
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", "gain", "grad_norm"}
    if not ensure_columns(base, required, "plan_transfer_summary input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

    plan_df = base[
        (base["target_task"] == "plan")
        & (base["source_task"].isin(_MAIN_SOURCES))
    ].copy()
    if plan_df.empty:
        return pd.DataFrame(columns=_GROUP_KEYS_FH + [
            "n", "mean_delta", "median_delta", "p05_delta", "p95_delta",
            "mean_gain", "median_gain",
            "helpful_rate", "harmful_rate",
            "practical_helpful_rate", "large_harm_rate",
            "std_delta", "sem_delta",
            "ci_lo_delta", "ci_hi_delta",
            "ci_lo_gain", "ci_hi_gain",
            "mean_grad_norm", "effect_size", "tau",
        ])

    tau_per_mc = (
        plan_df.groupby(["model", "checkpoint"])["delta"]
        .apply(lambda s: practical_threshold(s.values))
        .to_dict()
    )
    plan_df["tau"] = plan_df.apply(
        lambda r: tau_per_mc[(r["model"], r["checkpoint"])], axis=1
    )

    rows = []
    seed = 0
    for keys, sub in plan_df.groupby(_GROUP_KEYS_FH, sort=False):
        d = sub["delta"].to_numpy()
        g = sub["gain"].to_numpy()
        tau = float(sub["tau"].iloc[0])
        n = int(d.size)
        ci_lo_d, ci_hi_d = _ci_pair(d, seed)
        ci_lo_g, ci_hi_g = _ci_pair(g, seed + 1)
        seed += 2
        std_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        sem_d = std_d / np.sqrt(n) if n > 1 else float("nan")
        mean_g = float(np.mean(g))
        rows.append({
            **dict(zip(_GROUP_KEYS_FH, keys)),
            "n": n,
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "p05_delta": float(np.quantile(d, 0.05)),
            "p95_delta": float(np.quantile(d, 0.95)),
            "mean_gain": mean_g,
            "median_gain": float(np.median(g)),
            "helpful_rate": float(np.mean(d < 0)),
            "harmful_rate": float(np.mean(d > 0)),
            "practical_helpful_rate": float(np.mean(d < -tau)),
            "large_harm_rate": float(np.mean(d > tau)),
            "std_delta": std_d,
            "sem_delta": sem_d,
            "ci_lo_delta": ci_lo_d,
            "ci_hi_delta": ci_hi_d,
            "ci_lo_gain": ci_lo_g,
            "ci_hi_gain": ci_hi_g,
            "mean_grad_norm": float(sub["grad_norm"].mean()),
            "effect_size": mean_g / (std_d + EPS) if np.isfinite(std_d) else float("nan"),
            "tau": tau,
        })
    return pd.DataFrame(rows)


def top_beneficial(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows ranked by ``by`` descending (most beneficial first)."""
    return summary.sort_values(by, ascending=False).head(k).reset_index(drop=True)


def top_harmful(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows.

    For ``by="mean_gain"`` sorts ascending (worst gain first). For
    ``by="large_harm_rate"`` sorts descending.
    """
    ascending = by == "mean_gain"
    return summary.sort_values(by, ascending=ascending).head(k).reset_index(drop=True)
