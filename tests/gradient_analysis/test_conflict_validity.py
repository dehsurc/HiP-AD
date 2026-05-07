"""Phase 1 #4 B2/B3 — validity-aware aggregation in summarize_pair."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.conflict import summarize_pair


def test_summarize_pair_emits_validity_columns():
    df = pd.DataFrame({
        "batch_idx": list(range(8)),
        "group": ["g0"] * 8,
        "cos":      [0.1, -0.2, np.nan, 0.0, 0.05, -0.1, np.nan, 0.2],
        "coop_mag": [0.1,  0.0,  0.0,   0.0, 0.05,  0.0,  0.0,   0.2],
        "conf_mag": [0.0,  0.2,  0.0,   0.0, 0.0,   0.1,  0.0,   0.0],
        "norm_a":   [1.0,  1.0,  0.0,   1.0, 1.0,   1.0,  1.0,   1.0],
        "norm_b":   [1.0,  1.0,  1.0,   0.0, 1.0,   1.0,  0.0,   1.0],
    })
    summary = summarize_pair(df)
    row = summary.iloc[0]
    assert row["n_total"] == 8
    # Valid: cos finite AND both norms > 0 → rows 0, 1, 4, 5, 7 (5 rows).
    # Row 3 has cos=0.0 (finite) but norm_b=0 → excluded by nonzero.
    assert row["n_valid"] == 5
    # Pseudo-shared: at least one norm == 0 → rows 2, 3, 6 (3 rows).
    # Row 3 (cos finite, norm_b=0) is now correctly counted as pseudo-shared.
    assert row["n_pseudo_shared"] == 3
    # Partition invariant
    assert row["n_total"] == row["n_valid"] + row["n_pseudo_shared"] + row["n_nan"]
    assert 0.0 <= row["pseudo_shared_ratio"] <= 1.0
    assert row["is_pseudo_shared_group"] in (True, False)
    assert "conflict_ratio_legacy" in summary.columns
    # main conflict_ratio over n_valid: 2 of 5 valid rows have cos < 0 = 0.4
    assert abs(row["conflict_ratio"] - (2.0 / 5.0)) < 1e-6


def test_summarize_pair_partition_invariant_under_synthetic_distributions():
    """n_total = n_valid + n_pseudo_shared + n_nan must hold across many
    edge-case mixes."""
    rng = np.random.default_rng(0)
    for trial in range(10):
        n = 30
        cos = rng.standard_normal(n) * 0.1
        norm_a = rng.uniform(0, 2, size=n)
        norm_b = rng.uniform(0, 2, size=n)
        # Inject NaNs and zero norms into random rows
        nan_rows = rng.choice(n, size=5, replace=False)
        cos[nan_rows] = np.nan
        zero_rows = rng.choice(n, size=5, replace=False)
        norm_a[zero_rows] = 0.0
        df = pd.DataFrame({
            "batch_idx": list(range(n)),
            "group": ["g0"] * n,
            "cos": cos,
            "coop_mag": np.zeros(n),
            "conf_mag": np.zeros(n),
            "norm_a": norm_a,
            "norm_b": norm_b,
        })
        s = summarize_pair(df).iloc[0]
        assert s["n_total"] == s["n_valid"] + s["n_pseudo_shared"] + s["n_nan"], \
            f"partition broken on trial {trial}: " \
            f"{s['n_total']} != {s['n_valid']} + {s['n_pseudo_shared']} + {s['n_nan']}"


def test_summarize_pair_pseudo_shared_group_flag():
    """Group where most batches have one side zero norm → flagged."""
    df = pd.DataFrame({
        "batch_idx": list(range(10)),
        "group": ["g0"] * 10,
        "cos":      [np.nan] * 8 + [0.1, 0.2],
        "coop_mag": [0.0] * 8 + [0.1, 0.2],
        "conf_mag": [0.0] * 10,
        "norm_a":   [0.0] * 8 + [1.0, 1.0],
        "norm_b":   [1.0] * 10,
    })
    summary = summarize_pair(df)
    row = summary.iloc[0]
    # 8 NaN rows + 0 finite-cos-zero-norm rows; norm_a=0 for 8 rows → all 8 pseudo-shared.
    assert row["n_pseudo_shared"] == 8
    assert row["pseudo_shared_ratio"] == 0.8
    assert bool(row["is_pseudo_shared_group"]) is True


def test_summarize_pair_zero_total_no_crash():
    """Edge case: empty group should not raise."""
    df = pd.DataFrame({
        "batch_idx": [], "group": [], "cos": [],
        "coop_mag": [], "conf_mag": [], "norm_a": [], "norm_b": [],
    })
    # groupby on empty df returns empty result; just ensure no exception.
    out = summarize_pair(df)
    assert out.empty or "is_pseudo_shared_group" in out.columns
