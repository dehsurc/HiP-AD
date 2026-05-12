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
