"""Unit tests for the bootstrap-CI helper."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.bootstrap import augment_summary_with_ci, bca_ci


def test_bca_ci_normal_data():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(200) * 0.1 + 0.5
    lo, hi = bca_ci(samples, statistic=np.mean, n_resamples=1000, seed=1)
    assert lo < 0.5 < hi
    assert hi - lo < 0.06


def test_bca_ci_handles_constant_input():
    samples = np.full(100, 0.3)
    lo, hi = bca_ci(samples, statistic=np.mean, n_resamples=200, seed=0)
    assert abs(lo - 0.3) < 1e-6
    assert abs(hi - 0.3) < 1e-6


def test_augment_conflict_summary_attaches_ci():
    rng = np.random.default_rng(0)
    per_batch = pd.DataFrame({
        "group": ["g0"] * 100,
        "cos": rng.standard_normal(100) * 0.1 + 0.2,
    })
    summary = pd.DataFrame({"group": ["g0"], "mean_cos": [per_batch["cos"].mean()]})
    out = augment_summary_with_ci(
        summary, per_batch,
        group_cols=["group"],
        value_cols=["mean_cos"],
        n_resamples=500,
        seed=0,
    )
    assert "mean_cos_ci_lo" in out.columns
    assert out.at[0, "mean_cos_ci_lo"] < out.at[0, "mean_cos"] < out.at[0, "mean_cos_ci_hi"]


def test_augment_summary_with_rate_ci_for_conflict_ratio():
    """Binary-rate code path: conflict_ratio's per-batch counterpart is (cos < 0)."""
    rng = np.random.default_rng(0)
    cos_vals = rng.standard_normal(100) * 0.5  # roughly 50% negative
    per_batch = pd.DataFrame({"group": ["g0"] * 100, "cos": cos_vals})
    summary = pd.DataFrame({
        "group": ["g0"],
        "conflict_ratio": [float((cos_vals < 0).mean())],
    })
    out = augment_summary_with_ci(
        summary, per_batch,
        group_cols=["group"],
        value_cols=[],
        rate_value_cols=["conflict_ratio"],
        rate_per_batch_col="cos",
        n_resamples=500,
        seed=0,
    )
    lo, hi = out.at[0, "conflict_ratio_ci_lo"], out.at[0, "conflict_ratio_ci_hi"]
    assert 0.0 <= lo <= summary.at[0, "conflict_ratio"] <= hi <= 1.0
