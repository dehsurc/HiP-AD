import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.correlation import join_cos_delta, pair_correlations


def test_join_cos_delta_matches_on_batch_idx():
    cos_df = pd.DataFrame([
        {"batch_idx": 0, "group": "g0", "cos": 0.5, "task_a": "A", "task_b": "B"},
        {"batch_idx": 1, "group": "g0", "cos": -0.3, "task_a": "A", "task_b": "B"},
    ])
    probe_df = pd.DataFrame([
        {"batch_idx": 0, "source_task": "A", "target_task": "B", "steps": 1, "variant": "raw", "delta": -0.1},
        {"batch_idx": 1, "source_task": "A", "target_task": "B", "steps": 1, "variant": "raw", "delta": 0.2},
    ])
    joined = join_cos_delta(cos_df, probe_df, steps=1, variant="raw")
    assert len(joined) == 2
    row0 = joined[joined["batch_idx"] == 0].iloc[0]
    assert row0["cos"] == 0.5
    assert row0["delta"] == -0.1


def test_pair_correlations_computes_pearson_and_spearman():
    df = pd.DataFrame({"cos": [1.0, 2.0, 3.0, 4.0, 5.0], "delta": [-1, -2, -3, -4, -5]})
    pearson, spearman = pair_correlations(df)
    assert pearson == pytest.approx(-1.0)
    assert spearman == pytest.approx(-1.0)
