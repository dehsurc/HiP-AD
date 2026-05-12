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


def test_get_probe_base_filters_variant_and_steps():
    df = pd.DataFrame({
        "batch_idx": [0, 0, 0],
        "source_task": ["det", "det", "det"],
        "target_task": ["plan", "plan", "plan"],
        "steps": [1, 1, 2],
        "variant": ["normalized", "raw", "normalized"],
        "layer": ["_all", "_all", "_all"],
        "grad_norm": [1.0, 1.0, 1.0],
        "baseline_loss": [1.0, 1.0, 1.0],
        "stepped_loss": [0.9, 0.8, 0.7],
        "delta": [-0.1, -0.2, -0.3],
    })
    base = pc.get_probe_base(df, variant="normalized", steps=1)
    assert len(base) == 1
    assert "gain" in base.columns
    assert base.iloc[0]["gain"] == pytest.approx(0.1)


def _plan_target_fixture():
    rows = []
    rng = np.random.default_rng(0)
    for source in ["det", "map", "motion", "plan"]:
        if source == "det":
            deltas = rng.normal(-0.05, 0.02, size=20)
        elif source == "map":
            deltas = rng.normal(0.0, 0.02, size=20)
        elif source == "motion":
            deltas = rng.normal(0.05, 0.02, size=20)
        else:
            deltas = rng.normal(-0.1, 0.02, size=20)
        for i, d in enumerate(deltas):
            rows.append({
                "model": "HiP-AD",
                "checkpoint": "1ep",
                "checkpoint_order": 1,
                "batch_idx": i,
                "source_task": source,
                "target_task": "plan",
                "steps": 1,
                "variant": "normalized",
                "layer": "dec3_ffn_0",
                "grad_norm": 1.0,
                "baseline_loss": 1.0,
                "stepped_loss": 1.0 + d,
                "delta": d,
            })
    return pd.DataFrame(rows)


def test_build_plan_transfer_summary_groups_and_signs():
    base = pc.standardize_probe_df(_plan_target_fixture())
    summary = pc.build_plan_transfer_summary(base)
    expected_cols = {
        "model", "checkpoint", "checkpoint_order", "layer", "source_task",
        "n", "mean_delta", "median_delta", "p05_delta", "p95_delta",
        "mean_gain", "median_gain",
        "helpful_rate", "harmful_rate",
        "practical_helpful_rate", "large_harm_rate",
        "std_delta", "sem_delta",
        "ci_lo_delta", "ci_hi_delta",
        "ci_lo_gain", "ci_hi_gain",
        "mean_grad_norm", "effect_size", "tau",
    }
    assert expected_cols.issubset(set(summary.columns))
    det_row = summary[summary["source_task"] == "det"].iloc[0]
    motion_row = summary[summary["source_task"] == "motion"].iloc[0]
    assert det_row["mean_gain"] > 0
    assert motion_row["mean_gain"] < 0
    assert (summary["helpful_rate"].between(0, 1)).all()


def test_top_beneficial_sorts_descending_by_mean_gain():
    summary = pd.DataFrame({
        "model": ["HiP-AD"] * 3,
        "checkpoint": ["1ep"] * 3,
        "layer": ["a", "b", "c"],
        "source_task": ["det"] * 3,
        "mean_gain": [0.1, 0.3, 0.05],
        "practical_helpful_rate": [0.5, 0.7, 0.4],
        "large_harm_rate": [0.0, 0.0, 0.0],
    })
    top = pc.top_beneficial(summary, k=2, by="mean_gain")
    assert list(top["layer"]) == ["b", "a"]


def test_top_harmful_sorts_ascending_by_mean_gain():
    summary = pd.DataFrame({
        "model": ["HiP-AD"] * 3,
        "checkpoint": ["1ep"] * 3,
        "layer": ["a", "b", "c"],
        "source_task": ["det"] * 3,
        "mean_gain": [-0.2, 0.0, -0.5],
        "practical_helpful_rate": [0.1, 0.2, 0.0],
        "large_harm_rate": [0.5, 0.0, 0.8],
    })
    top = pc.top_harmful(summary, k=2, by="mean_gain")
    assert list(top["layer"]) == ["c", "a"]
    top_harm = pc.top_harmful(summary, k=2, by="large_harm_rate")
    assert list(top_harm["layer"]) == ["c", "a"]


def _asym_fixture(n=20):
    rng = np.random.default_rng(1)
    rows = []
    plans = {
        ("det", "plan"): rng.normal(-0.05, 0.01, n),
        ("plan", "det"): rng.normal(+0.05, 0.01, n),
        ("map", "plan"): rng.normal(-0.05, 0.01, n),
        ("plan", "map"): rng.normal(-0.05, 0.01, n),
        ("motion", "plan"): rng.normal(0.0, 0.001, n),
        ("plan", "motion"): rng.normal(0.0, 0.001, n),
        ("det", "det"): rng.normal(-0.2, 0.01, n),
        ("map", "map"): rng.normal(-0.2, 0.01, n),
        ("motion", "motion"): rng.normal(-0.05, 0.005, n),
    }
    for (s, t), deltas in plans.items():
        for i, d in enumerate(deltas):
            rows.append({
                "model": "HiP-AD", "checkpoint": "1ep", "checkpoint_order": 1,
                "batch_idx": i, "source_task": s, "target_task": t,
                "steps": 1, "variant": "normalized", "layer": "L0",
                "grad_norm": 1.0,
                "baseline_loss": 1.0, "stepped_loss": 1.0 + d, "delta": d,
            })
    return pd.DataFrame(rows)


def test_build_asymmetry_summary_signs_and_interpretation():
    base = pc.standardize_probe_df(_asym_fixture())
    summary = pc.build_asymmetry_summary(base)
    assert {"model", "checkpoint", "layer", "aux_task",
            "gain_A_to_plan", "gain_plan_to_A",
            "asymmetry", "plan_transfer_ratio",
            "helpful_A_to_plan", "helpful_plan_to_A",
            "interpretation"}.issubset(summary.columns)
    by_aux = summary.set_index("aux_task")
    assert by_aux.loc["det", "asymmetry"] > 0
    assert "보존하지 않는다" in by_aux.loc["det", "interpretation"]
    assert "상호 보완" in by_aux.loc["map", "interpretation"]
    assert "도달" in by_aux.loc["motion", "interpretation"] or \
           "미미" in by_aux.loc["motion", "interpretation"]


def test_interpret_asymmetry_row_covers_four_quadrants():
    tau = 0.01
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.05,
                                      "gain_plan_to_A": -0.05, "tau": tau})
    assert "보존하지" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.05,
                                      "gain_plan_to_A":  0.05, "tau": tau})
    assert "상호 보완" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan": -0.05,
                                      "gain_plan_to_A": -0.05, "tau": tau})
    assert "상호 간섭" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.0,
                                      "gain_plan_to_A":  0.05, "tau": tau})
    assert "도달" in msg or "미미" in msg


def test_build_effect_size_summary_columns_and_flags():
    base = pc.standardize_probe_df(_plan_target_fixture())
    summary = pc.build_effect_size_summary(base, target="plan")
    assert {"model", "checkpoint", "layer", "source_task",
            "n", "helpful_rate", "practical_helpful_rate", "large_harm_rate",
            "mean_gain", "median_gain", "std_delta", "effect_size",
            "ci_lo_gain", "ci_hi_gain", "ci_contains_zero",
            "flag_high_helpful_low_gain", "flag_positive_but_insig",
            "flag_high_practical_helpful", "flag_high_large_harm"}.issubset(summary.columns)
    assert summary["ci_contains_zero"].dtype == bool or set(summary["ci_contains_zero"].unique()).issubset({True, False})
    assert summary["source_task"].isin(["det", "map", "motion", "plan"]).all()
