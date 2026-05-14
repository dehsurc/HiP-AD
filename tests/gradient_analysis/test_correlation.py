import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.correlation import (
    join_cos_delta,
    join_layer_cos_delta,
    pair_correlations,
    run_layer_m4,
)


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


def test_join_layer_cos_delta_matches_layer_and_both_directions():
    cos_df = pd.DataFrame([
        {"batch_idx": 0, "group": "L", "cos": -0.5, "task_a": "A", "task_b": "B"},
        {"batch_idx": 1, "group": "L", "cos": 0.3, "task_a": "A", "task_b": "B"},
    ])
    probe_df = pd.DataFrame([
        {
            "batch_idx": 0, "source_task": "A", "target_task": "B",
            "layer": "L", "steps": 1, "variant": "raw", "delta": 0.2,
            "baseline_loss": 1.0, "stepped_loss": 1.2, "grad_norm": 2.0,
        },
        {
            "batch_idx": 1, "source_task": "B", "target_task": "A",
            "layer": "L", "steps": 1, "variant": "raw", "delta": -0.1,
            "baseline_loss": 1.0, "stepped_loss": 0.9, "grad_norm": 3.0,
        },
        {
            "batch_idx": 1, "source_task": "A", "target_task": "B",
            "layer": "other", "steps": 1, "variant": "raw", "delta": 9.0,
            "baseline_loss": 1.0, "stepped_loss": 10.0, "grad_norm": 4.0,
        },
    ])

    joined = join_layer_cos_delta(cos_df, probe_df, steps=1, variant="raw")

    assert len(joined) == 2
    assert set(zip(joined["source_task"], joined["target_task"])) == {("A", "B"), ("B", "A")}
    assert set(joined["layer"]) == {"L"}


def test_run_layer_m4_writes_directional_layer_tables(tmp_path):
    cos_df = pd.DataFrame([
        {"batch_idx": 0, "group": "L", "cos": -1.0},
        {"batch_idx": 1, "group": "L", "cos": 0.0},
        {"batch_idx": 2, "group": "L", "cos": 1.0},
    ])
    probe_rows = []
    for i, delta in enumerate([1.0, 0.0, -1.0]):
        probe_rows.append({
            "batch_idx": i, "source_task": "A", "target_task": "B",
            "layer": "L", "steps": 1, "variant": "raw", "delta": delta,
            "baseline_loss": 1.0, "stepped_loss": 1.0 + delta, "grad_norm": 1.0,
        })
        probe_rows.append({
            "batch_idx": i, "source_task": "B", "target_task": "A",
            "layer": "L", "steps": 1, "variant": "raw", "delta": -delta,
            "baseline_loss": 1.0, "stepped_loss": 1.0 - delta, "grad_norm": 1.0,
        })
    probe_df = pd.DataFrame(probe_rows)

    table = run_layer_m4(
        {("A", "B"): cos_df},
        probe_df,
        steps_list=[1],
        variants=["raw"],
        cosine_bins=[-1.0, -0.1, 0.1, 1.0],
        out_dir=tmp_path,
    )

    assert {tuple(x) for x in table[["source_task", "target_task"]].to_numpy()} == {
        ("A", "B"), ("B", "A"),
    }
    assert (tmp_path / "layer_joined_per_batch.csv").exists()
    assert (tmp_path / "layer_correlation_table.csv").exists()
    assert (tmp_path / "layer_binned_delta.csv").exists()
