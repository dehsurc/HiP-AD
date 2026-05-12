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


# ---------------------------------------------------------------------------
# Task 10 – Part I: detect_first_order_columns + build_first_order_residual_summary
# ---------------------------------------------------------------------------

def test_detect_first_order_columns_returns_none_when_missing():
    df = pd.DataFrame({"delta": [0.0]})
    assert pc.detect_first_order_columns(df) is None


def test_detect_first_order_columns_finds_canonical_names():
    df = pd.DataFrame({"delta": [0.0], "grad_dot": [0.0], "step_size": [1e-3]})
    cols = pc.detect_first_order_columns(df)
    assert cols == {"grad_dot": "grad_dot", "step_size": "step_size"}


def test_first_order_residual_summary_pearson_one_when_pred_matches():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "model": ["HiP-AD"] * n, "checkpoint": ["1ep"] * n,
        "checkpoint_order": [1] * n, "layer": ["L0"] * n,
        "source_task": ["det"] * n, "target_task": ["plan"] * n,
        "step_size": [1e-3] * n,
        "grad_dot": rng.normal(0, 1, n),
    })
    df["delta"] = -df["step_size"] * df["grad_dot"]
    base = pc.standardize_probe_df(
        df.assign(steps=1, variant="normalized", batch_idx=range(n),
                  grad_norm=1.0, baseline_loss=1.0,
                  stepped_loss=1.0 + df["delta"]))
    cols = {"grad_dot": "grad_dot", "step_size": "step_size"}
    summary = pc.build_first_order_residual_summary(base, cols)
    assert summary["pearson_actual_pred"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert summary["mean_residual"].iloc[0] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Task 11 – Part J: load_or_template_query_sensitivity + build_query_sensitivity_summary
# ---------------------------------------------------------------------------

def test_load_or_template_query_sensitivity_writes_template_when_missing(tmp_path):
    path = tmp_path / "qs.csv"
    df, status = pc.load_or_template_query_sensitivity(path)
    assert df is None
    assert status == "template"
    assert path.exists()
    template = pd.read_csv(path)
    expected = {
        "model", "checkpoint", "checkpoint_order", "batch_idx", "scene_token",
        "layer", "task_query_type", "query_index", "query_norm",
        "grad_plan_wrt_query_norm",
    }
    assert expected.issubset(template.columns)


def test_load_or_template_query_sensitivity_reads_existing(tmp_path):
    path = tmp_path / "qs.csv"
    pd.DataFrame({
        "model": ["HiP-AD"], "checkpoint": ["1ep"], "checkpoint_order": [1],
        "batch_idx": [0], "scene_token": ["s"], "layer": ["L"],
        "task_query_type": ["det"], "query_index": [0],
        "query_norm": [1.0], "grad_plan_wrt_query_norm": [0.5],
    }).to_csv(path, index=False)
    df, status = pc.load_or_template_query_sensitivity(path)
    assert status == "loaded"
    assert len(df) == 1


def test_build_query_sensitivity_summary_basic():
    df = pd.DataFrame({
        "model": ["HiP-AD"] * 6,
        "checkpoint": ["1ep"] * 6,
        "checkpoint_order": [1] * 6,
        "layer": ["L0"] * 6,
        "task_query_type": ["det", "det", "map", "map", "motion", "motion"],
        "query_norm": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        "grad_plan_wrt_query_norm": [0.2, 0.4, 0.1, 0.1, 0.0, 0.0],
    })
    out = pc.build_query_sensitivity_summary(df)
    assert {"model", "checkpoint", "layer", "task_query_type",
            "mean_sensitivity", "median_sensitivity", "p05", "p95",
            "mean_query_norm", "sensitivity_normed_mean",
            "share_task"}.issubset(out.columns)
    by_task = out.set_index("task_query_type")
    assert by_task.loc["det", "share_task"] > by_task.loc["motion", "share_task"]


# ---------------------------------------------------------------------------
# Task 12 – Part K: load_or_template_elasticity + build_elasticity_summary
#            + planning_safe_weight_range
# ---------------------------------------------------------------------------

def test_load_or_template_elasticity_writes_template(tmp_path):
    path = tmp_path / "el.csv"
    df, status = pc.load_or_template_elasticity(path)
    assert df is None and status == "template" and path.exists()
    template = pd.read_csv(path)
    assert {"model", "run_id", "lambda_det", "lambda_map", "lambda_motion",
            "lambda_plan", "plan_l2"}.issubset(template.columns)


def test_build_elasticity_summary_long_format_and_directions():
    df = pd.DataFrame({
        "model": ["A"] * 6,
        "run_id": [f"r{i}" for i in range(6)],
        "lambda_det": [1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        "lambda_map": [1.0, 1.0, 1.0, 0.5, 2.0, 1.0],
        "lambda_motion": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "lambda_plan": [1.0] * 6,
        "det_metric": [0.30, 0.28, 0.32, 0.30, 0.30, 0.30],
        "map_metric": [0.50, 0.50, 0.50, 0.45, 0.55, 0.50],
        "motion_loss": [0.20] * 6,
        "plan_l2": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
        "is_baseline": [True, False, False, False, False, False],
    })
    summary = pc.build_elasticity_summary(df)
    assert {"model", "swept_task", "lambda_value", "log_lambda",
            "task_metric", "plan_metric",
            "relative_task_metric", "relative_plan_metric"}.issubset(summary.columns)
    det2 = summary[(summary["swept_task"] == "det") & (summary["lambda_value"] == 2.0)]
    assert det2["relative_task_metric"].iloc[0] > 0


def test_planning_safe_weight_range_filters_by_tolerance():
    summary = pd.DataFrame({
        "model": ["A"] * 3,
        "swept_task": ["det"] * 3,
        "lambda_value": [0.5, 1.0, 2.0],
        "log_lambda": [np.log(0.5), 0.0, np.log(2.0)],
        "task_metric": [0.28, 0.30, 0.32],
        "plan_metric": [1.005, 1.000, 1.030],
        "relative_task_metric": [-0.067, 0.0, 0.067],
        "relative_plan_metric": [0.005, 0.0, 0.030],
    })
    safe = pc.planning_safe_weight_range(summary, tol=0.01)
    assert (safe["planning_safe"] == (safe["relative_plan_metric"] <= 0.01)).all()
