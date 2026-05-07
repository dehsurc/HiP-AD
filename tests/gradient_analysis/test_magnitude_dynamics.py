"""Phase 1 #10 — per-task norm slopes + pairwise ratio matrix evolution."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.magnitude_dynamics import (
    compute_per_task_slopes,
    compute_ratio_evolution,
    non_stationarity_index,
    run_magnitude_dynamics,
)


def test_compute_per_task_slopes_returns_one_row_per_task():
    df = pd.DataFrame({
        "epoch": [1, 3, 6, 18, 1, 3, 6, 18],
        "task":  ["det"] * 4 + ["motion"] * 4,
        "mean_norm": [10.0, 12.0, 15.0, 20.0, 5.0, 7.0, 12.0, 25.0],
    })
    out = compute_per_task_slopes(df)
    assert set(out["task"]) == {"det", "motion"}
    s_det = float(out.loc[out["task"] == "det", "slope"].iloc[0])
    s_motion = float(out.loc[out["task"] == "motion", "slope"].iloc[0])
    assert s_motion > s_det


def test_compute_ratio_evolution_shape():
    df = pd.DataFrame({
        "epoch": [1, 1, 6, 6],
        "task":  ["det", "motion", "det", "motion"],
        "mean_norm": [10.0, 5.0, 15.0, 12.0],
    })
    cube = compute_ratio_evolution(df, tasks=["det", "motion"])
    assert cube.shape == (2, 2, 2)
    assert abs(cube[0, 1, 0] - 2.0) < 1e-6
    assert abs(cube[0, 1, 1] - 1.25) < 1e-6


def test_non_stationarity_index_is_cv_of_ratios():
    df = pd.DataFrame({
        "epoch": [1, 1, 6, 6],
        "task":  ["det", "motion", "det", "motion"],
        "mean_norm": [10.0, 5.0, 15.0, 12.0],
    })
    nsi = non_stationarity_index(df, tasks=["det", "motion"])
    expected = float(np.std([2.0, 1.25], ddof=1) / np.mean([2.0, 1.25]))
    val = float(nsi.loc[("det", "motion"), "cv"])
    assert abs(val - expected) < 1e-6


def test_run_magnitude_dynamics_writes_artifacts(tmp_path):
    df = pd.DataFrame({
        "epoch": [1, 1, 6, 6],
        "task":  ["det", "motion", "det", "motion"],
        "mean_norm": [10.0, 5.0, 15.0, 12.0],
    })
    out = run_magnitude_dynamics(df, tasks=["det", "motion"], out_dir=tmp_path)
    assert (tmp_path / "per_task_slopes.csv").exists()
    assert (tmp_path / "non_stationarity_index.csv").exists()
    assert (tmp_path / "ratio_evolution.npy").exists()
    assert (tmp_path / "ratio_evolution.png").exists()
    assert "slopes" in out and "nsi" in out
