import numpy as np
import torch

from tools.gradient_analysis.conflict import (
    cosine_similarity,
    projection_decomposition,
    analyze_pair_batches,
)


def test_cosine_identical_vectors_is_one():
    g = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(g, g) == 1.0


def test_cosine_opposite_vectors_is_negative_one():
    g = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(g, -g) == -1.0


def test_cosine_zero_norm_returns_nan():
    g = torch.tensor([1.0, 2.0, 3.0])
    zero = torch.zeros(3)
    assert np.isnan(cosine_similarity(zero, g))


def test_projection_decomposition_cooperative_only_when_aligned():
    g_a = torch.tensor([1.0, 0.0])
    g_b = torch.tensor([1.0, 0.0])
    coop, conf = projection_decomposition(g_a, g_b)
    assert coop == 1.0
    assert conf == 0.0


def test_projection_decomposition_conflicting_only_when_opposed():
    g_a = torch.tensor([-1.0, 0.0])
    g_b = torch.tensor([1.0, 0.0])
    coop, conf = projection_decomposition(g_a, g_b)
    assert coop == 0.0
    assert conf == 1.0


def test_analyze_pair_batches_reports_expected_shape():
    # Two batches, single group 'g0'
    batches = [
        {"A": {"g0": torch.tensor([1.0, 0.0])}, "B": {"g0": torch.tensor([0.5, 0.1])}},
        {"A": {"g0": torch.tensor([1.0, 0.0])}, "B": {"g0": torch.tensor([-0.5, 0.1])}},
    ]
    df = analyze_pair_batches(batches, task_a="A", task_b="B", group_keys=["g0"])
    assert len(df) == 2
    assert set(df.columns) >= {"batch_idx", "group", "cos", "coop_mag", "conf_mag"}
    # First batch aligned → coop > 0, conf == 0
    row = df[df["batch_idx"] == 0].iloc[0]
    assert row["cos"] > 0
    assert row["conf_mag"] == 0
