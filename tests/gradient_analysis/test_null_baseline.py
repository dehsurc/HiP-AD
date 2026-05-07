"""Unit tests for the null-baseline / permutation module."""
import numpy as np
import pandas as pd
import torch

from tools.gradient_analysis.null_baseline import (
    NullTestResult,
    batch_permutation_null,
    compare_to_null,
    run_null_baseline,
    sample_shuffle_null,
    sign_flip_null,
)


# --------- sample_shuffle_null ---------

def test_sample_shuffle_null_returns_correct_count():
    rng = np.random.default_rng(0)
    g_a = torch.tensor(rng.standard_normal(size=(8, 16)).astype("float32"))
    g_b = torch.tensor(rng.standard_normal(size=(8, 16)).astype("float32"))
    out = sample_shuffle_null(g_a, g_b, n_repeats=64, seed=1)
    assert out.shape == (64,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= -1.0) & (out <= 1.0))


def test_sample_shuffle_null_one_when_inputs_identical():
    g = torch.ones(4, 8)
    out = sample_shuffle_null(g, g, n_repeats=8, seed=0)
    assert np.allclose(out, 1.0, atol=1e-6)


# --------- sign_flip_null ---------

def test_sign_flip_null_returns_correct_count():
    rng = np.random.default_rng(2)
    g_a = torch.tensor(rng.standard_normal(size=(64,)).astype("float32"))
    g_b = torch.tensor(rng.standard_normal(size=(64,)).astype("float32"))
    out = sign_flip_null(g_a, g_b, n_repeats=128, seed=3)
    assert out.shape == (128,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= -1.0) & (out <= 1.0))


def test_sign_flip_null_centered_at_zero():
    rng = np.random.default_rng(4)
    g = torch.tensor(rng.standard_normal(size=(256,)).astype("float32"))
    out = sign_flip_null(g, g, n_repeats=2000, seed=5)
    assert abs(float(out.mean())) < 0.05


# --------- Mann-Whitney U + rank-biserial ---------

def test_mann_whitney_significant_when_observed_shifted():
    rng = np.random.default_rng(7)
    null = rng.standard_normal(1000) * 0.05
    observed = rng.standard_normal(100) * 0.05 + 0.3
    res = compare_to_null(observed, null)
    assert res.p_value < 1e-6
    assert abs(res.rank_biserial) > 0.5


def test_mann_whitney_null_when_observed_matches_null():
    rng = np.random.default_rng(8)
    null = rng.standard_normal(1000) * 0.05
    observed = rng.standard_normal(100) * 0.05
    res = compare_to_null(observed, null)
    assert res.p_value > 0.01
    assert abs(res.rank_biserial) < 0.2


def test_compare_to_null_handles_empty_inputs():
    res = compare_to_null(np.array([]), np.array([1.0, 2.0]))
    assert isinstance(res, NullTestResult)
    assert np.isnan(res.p_value)


# --------- aggregator ---------

def test_batch_permutation_null_returns_correct_count():
    rng = np.random.default_rng(11)
    grads_a = [torch.tensor(rng.standard_normal(64).astype("float32")) for _ in range(8)]
    grads_b = [torch.tensor(rng.standard_normal(64).astype("float32")) for _ in range(8)]
    out = batch_permutation_null(grads_a, grads_b, n_repeats=128, seed=12)
    assert out.shape == (128,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= -1.0) & (out <= 1.0))


def test_batch_permutation_null_centered_at_zero_for_random_inputs():
    """Random independent gradients across batches → null cosine ≈ 0 on average."""
    rng = np.random.default_rng(13)
    n_batches = 16
    grads_a = [torch.tensor(rng.standard_normal(128).astype("float32")) for _ in range(n_batches)]
    grads_b = [torch.tensor(rng.standard_normal(128).astype("float32")) for _ in range(n_batches)]
    out = batch_permutation_null(grads_a, grads_b, n_repeats=2000, seed=14)
    assert abs(float(out.mean())) < 0.05


def test_batch_permutation_null_single_batch_yields_nan():
    """With only 1 batch, no off-diagonal pair exists — the fallback NaN
    array signals 'cannot test' to the caller."""
    g = [torch.tensor([1.0, 2.0, 3.0])]
    out = batch_permutation_null(g, g, n_repeats=10, seed=0)
    assert out.shape == (1,)
    assert np.isnan(out[0])


def test_batch_permutation_null_avoids_self_pairing():
    """For batch_i with j != i, never compare a batch's gradient to itself."""
    # 3 distinct batches, each grad is (1, 0, 0), (0, 1, 0), (0, 0, 1).
    # If self-pairing happened, cos = 1.0 would dominate.
    grads = [
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ]
    out = batch_permutation_null(grads, grads, n_repeats=200, seed=42)
    # All draws must be 0 (orthogonal off-diagonals); self-pair would give 1.
    assert np.all(out == 0.0)


def test_run_null_baseline_writes_csv_with_required_columns(tmp_path):
    cached = []
    rng = np.random.default_rng(0)
    for b in range(3):
        shared = {}
        for task in ("a", "b"):
            shared[task] = {
                "g0": torch.tensor(rng.standard_normal(64).astype("float32")),
                "g1": torch.tensor(rng.standard_normal(64).astype("float32")),
            }
        cached.append({"batch_idx": b, "shared": shared})
    df = run_null_baseline(
        cached_batches=cached,
        tasks=["a", "b"],
        group_keys=["g0", "g1"],
        n_repeats=64,
        seed=42,
        bonferroni_family_size=None,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "task_a", "task_b", "group", "null_kind",
        "observed_mean", "null_mean", "null_ci_lo", "null_ci_hi",
        "u_statistic", "p_value", "p_value_bonferroni", "rank_biserial",
        "passes_noise_threshold", "n_observed", "n_null",
    }
    assert expected_cols.issubset(set(df.columns))
    assert (df["p_value_bonferroni"] >= df["p_value"]).all()
    # Two null kinds × 1 pair × 2 groups = 4 rows.
    assert len(df) == 4
    # null_kind labels must be the production set (sample_shuffle was retired
    # to prevent silent impostor results — see review notes 2026-05-03).
    assert set(df["null_kind"].unique()) == {"batch_permutation", "sign_flip"}


def test_run_null_baseline_bonferroni_excludes_n_kinds_factor():
    """Family size = n_groups × n_pairs (NOT also × n_kinds) per spec § 3.1.

    The two null kinds are robustness checks on the same hypothesis, not
    independent tests, so dividing α by n_kinds inflates the correction and
    biases the gating decision toward Pass B / Fail.
    """
    cached = []
    rng = np.random.default_rng(100)
    for b in range(3):
        shared = {
            t: {g: torch.tensor(rng.standard_normal(64).astype("float32"))
                for g in ("g0", "g1", "g2")}
            for t in ("a", "b", "c")
        }
        cached.append({"batch_idx": b, "shared": shared})
    df = run_null_baseline(
        cached_batches=cached,
        tasks=["a", "b", "c"],
        group_keys=["g0", "g1", "g2"],
        n_repeats=64,
        seed=0,
    )
    # n_groups=3, n_pairs=3, expected family size = 9 (NOT 18).
    finite = df.dropna(subset=["p_value"])
    if not finite.empty:
        ratios = (finite["p_value_bonferroni"] / finite["p_value"]).clip(upper=9.0001)
        assert (ratios <= 9.0 + 1e-9).all(), \
            "p_bonferroni / p_raw must not exceed family_size = 9 (n_groups × n_pairs)"
