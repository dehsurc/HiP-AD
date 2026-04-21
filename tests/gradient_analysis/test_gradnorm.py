import numpy as np
import pandas as pd
import pytest
import torch

from tools.gradient_analysis.gradnorm import (
    per_task_norms_from_cached,
    antisymmetric_frobenius,
)


def test_per_task_norms_from_cached():
    cached = [
        {
            "batch_idx": 0,
            "shared": {
                "A": {"g0": torch.tensor([3.0, 4.0])},  # norm = 5
                "B": {"g0": torch.tensor([0.0, 2.0])},  # norm = 2
            },
        },
    ]
    df = per_task_norms_from_cached(cached, tasks=["A", "B"], groups=["g0"])
    row_a = df[(df["task"] == "A") & (df["group"] == "g0")].iloc[0]
    assert row_a["norm"] == pytest.approx(5.0)


def test_antisymmetric_frobenius_zero_for_symmetric():
    M = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert antisymmetric_frobenius(M) == pytest.approx(0.0)


def test_antisymmetric_frobenius_positive_for_asymmetric():
    M = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert antisymmetric_frobenius(M) > 0.0
