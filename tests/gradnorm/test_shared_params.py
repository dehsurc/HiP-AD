import pytest
import torch
from torch import nn

from projects.mmdet3d_plugin.core.gradnorm.shared_params import (
    collect_last_linear_weights,
)


class _FakeFFN(nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(in_ch, hid), nn.ReLU(), nn.Dropout(0.0)),
            nn.Linear(hid, out_ch),
            nn.Dropout(0.0),
        )


def test_collects_one_weight_per_ffn_in_order():
    ffn_a = _FakeFFN(8, 16, 4)
    ffn_b = _FakeFFN(4, 32, 4)
    layers = [nn.Identity(), ffn_a, nn.Identity(), ffn_b]
    operation_order = ["norm", "ffn", "norm", "ffn"]

    params = collect_last_linear_weights(operation_order, layers)

    assert len(params) == 2
    assert params[0] is ffn_a.layers[1].weight
    assert params[1] is ffn_b.layers[1].weight
    assert params[0].shape == (4, 16)
    assert params[1].shape == (4, 32)


def test_asserts_when_ffn_has_no_linear():
    bad_ffn = nn.Module()
    bad_ffn.layers = nn.Sequential(nn.ReLU(), nn.Dropout(0.0))
    with pytest.raises(AssertionError):
        collect_last_linear_weights(["ffn"], [bad_ffn])


def test_ignores_non_ffn_ops():
    params = collect_last_linear_weights(
        ["norm", "norm"], [nn.Identity(), nn.Identity()]
    )
    assert params == []
