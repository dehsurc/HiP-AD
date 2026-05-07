"""Unit tests for the distribution-diagnostics module."""
import numpy as np
import torch

from tools.gradient_analysis.distribution import (
    classify_shape,
    compute_distribution_features,
    run_distribution,
)


def test_compute_features_unimodal_centered():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(500) * 0.05
    feats = compute_distribution_features(samples)
    assert -0.05 < feats["mean"] < 0.05
    assert feats["dip_p_value"] > 0.05
    assert feats["shape_label"] in ("unimodal-near-0", "unimodal-pos", "unimodal-neg")


def test_compute_features_bimodal_detected():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(250) * 0.05 - 0.5
    b = rng.standard_normal(250) * 0.05 + 0.5
    samples = np.concatenate([a, b])
    feats = compute_distribution_features(samples)
    assert feats["dip_p_value"] < 0.05
    assert feats["shape_label"] == "bimodal"


def test_classify_shape_returns_one_of_known_labels():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal(500) * 0.05 + 0.3
    feats = compute_distribution_features(samples)
    assert feats["shape_label"] in {
        "unimodal-near-0", "unimodal-pos", "unimodal-neg",
        "bimodal", "heavy-tail",
    }


def test_classify_shape_handles_empty_array():
    feats = compute_distribution_features(np.array([], dtype=np.float64))
    assert feats["shape_label"] == "empty"
    assert np.isnan(feats["mean"])


def test_run_distribution_writes_csv(tmp_path):
    cached = []
    rng = np.random.default_rng(0)
    for b in range(8):
        shared = {
            "a": {"g0": torch.tensor(rng.standard_normal(64).astype("float32"))},
            "b": {"g0": torch.tensor(rng.standard_normal(64).astype("float32"))},
        }
        cached.append({"batch_idx": b, "shared": shared})
    out_path = tmp_path / "distribution_report.csv"
    df = run_distribution(
        cached_batches=cached,
        tasks=["a", "b"],
        group_keys=["g0"],
        out_path=out_path,
        emit_kde_figures=False,
    )
    assert out_path.exists()
    assert {"task_a", "task_b", "group", "shape_label", "dip_p_value", "n"} \
        .issubset(set(df.columns))
    assert len(df) == 1
