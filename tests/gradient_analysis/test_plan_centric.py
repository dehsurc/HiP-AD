import warnings

import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis import plan_centric as pc


def test_ensure_columns_returns_true_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert pc.ensure_columns(df, ["a", "b"], "df") is True


def test_ensure_columns_returns_false_and_warns_when_missing():
    df = pd.DataFrame({"a": [1]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = pc.ensure_columns(df, ["a", "b", "c"], "myframe")
    assert ok is False
    assert any("myframe" in str(w.message) and "b" in str(w.message) for w in caught)


def test_bootstrap_ci_returns_nan_for_small_n():
    lo, hi = pc.bootstrap_ci([1.0, 2.0])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_brackets_mean_for_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=2.0, scale=0.5, size=200)
    lo, hi = pc.bootstrap_ci(sample, n_boot=500, seed=1)
    assert lo < 2.0 < hi
    assert hi - lo < 0.5


def test_bootstrap_ci_filters_nonfinite():
    sample = [1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0, 6.0]
    lo, hi = pc.bootstrap_ci(sample, n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_practical_threshold_floors_at_min_value():
    assert pc.practical_threshold([0.0, 0.0, 0.0]) == pytest.approx(1e-8)


def test_practical_threshold_scales_with_median_abs():
    series = [-10.0, -5.0, 0.0, 5.0, 10.0]
    assert pc.practical_threshold(series, ratio=0.1) == pytest.approx(0.5)


def test_practical_threshold_ignores_nonfinite():
    series = [np.nan, np.inf, 4.0, -4.0, 0.0]
    assert pc.practical_threshold(series, ratio=0.1) == pytest.approx(0.4)


def _make_probe(loss_before=True, rel_delta=True):
    cols = {
        "batch_idx": [0, 1],
        "source_task": ["det", "map"],
        "target_task": ["plan", "plan"],
        "steps": [1, 1],
        "variant": ["normalized", "normalized"],
        "layer": ["_all", "_all"],
        "grad_norm": [2.0, 3.0],
        "delta": [-0.1, 0.2],
    }
    if loss_before:
        cols["baseline_loss"] = [1.0, 2.0]
        cols["stepped_loss"] = [0.9, 2.2]
    if rel_delta:
        cols["rel_delta"] = [-0.1, 0.1]
    return pd.DataFrame(cols)


def test_standardize_probe_df_aliases_loss_columns():
    df = pc.standardize_probe_df(_make_probe())
    assert "loss_before" in df.columns
    assert "loss_after" in df.columns
    assert df.loc[0, "loss_before"] == pytest.approx(1.0)
    assert df.loc[1, "loss_after"] == pytest.approx(2.2)


def test_standardize_probe_df_computes_gain():
    df = pc.standardize_probe_df(_make_probe())
    assert df.loc[0, "gain"] == pytest.approx(0.1)
    assert df.loc[1, "gain"] == pytest.approx(-0.2)


def test_standardize_probe_df_computes_delta_rel_when_losses_present():
    df = pc.standardize_probe_df(_make_probe())
    assert df.loc[0, "delta_rel"] == pytest.approx(-0.1)
    assert df.loc[0, "gain_rel"] == pytest.approx(0.1)


def test_standardize_probe_df_falls_back_to_rel_delta_when_no_losses():
    base = _make_probe(loss_before=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pc.standardize_probe_df(base)
    assert df.loc[0, "delta_rel"] == pytest.approx(-0.1)
    assert any("loss_before" in str(w.message) or "loss_after" in str(w.message)
               for w in caught)


def test_standardize_probe_df_emits_nan_when_no_losses_and_no_rel_delta():
    base = _make_probe(loss_before=False, rel_delta=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pc.standardize_probe_df(base)
    assert df["delta_rel"].isna().all()
    assert df["gain_rel"].isna().all()
