"""Regression: probe_one_batch emits grad_dot and step_size on the 1-step path."""
from collections import OrderedDict

import numpy as np
import pytest
import torch
from torch import nn


class _Adapter:
    def split_losses(self, fwd, task):
        return fwd[task]
    def snapshot_temporal_state(self, model):
        return None
    def restore_temporal_state(self, model, snap):
        pass
    def freeze_stochastic_state(self):
        from contextlib import nullcontext
        return nullcontext()


class _Collector:
    """Minimal collector contract used by probe_one_batch."""

    def __init__(self):
        torch.manual_seed(0)
        self.shared = nn.Linear(4, 4)
        self.head = {"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)}
        self.tasks = ["a", "b"]
        self.adapter = _Adapter()
        self.full_params = list(self.shared.parameters())
        self.shared_param_groups = OrderedDict(L0=self.full_params)

        self._x = torch.randn(8, 4)
        self._ya = torch.randn(8, 1)
        self._yb = torch.randn(8, 1)

    @property
    def model(self):
        return self.shared

    def forward_losses(self, data):
        h = self.shared(self._x)
        return {"a": (self.head["a"](h) - self._ya).pow(2).mean(),
                "b": (self.head["b"](h) - self._yb).pow(2).mean()}


def test_probe_one_batch_emits_grad_dot_and_step_size():
    from tools.gradient_analysis.probe import probe_one_batch, rows_to_dataframe

    col = _Collector()
    rows = probe_one_batch(
        col, data=None, alpha=1e-3, steps_list=[1],
        variants=["normalized", "raw"], batch_idx=0,
        target_layers=None, freeze_matching=False, forward_seed=None,
    )
    df = rows_to_dataframe(rows)
    assert "grad_dot" in df.columns
    assert "step_size" in df.columns

    self_rows = df[(df["source_task"] == "a") & (df["target_task"] == "a")
                   & (df["variant"] == "normalized")]
    grad_norm = float(self_rows["grad_norm"].iloc[0])
    assert np.isfinite(self_rows["grad_dot"].iloc[0])
    assert self_rows["grad_dot"].iloc[0] == pytest.approx(grad_norm ** 2, rel=1e-4)
    assert self_rows["step_size"].iloc[0] == pytest.approx(1e-3 / grad_norm, rel=1e-6)
    raw_self = df[(df["source_task"] == "a") & (df["target_task"] == "a")
                  & (df["variant"] == "raw")]
    assert raw_self["step_size"].iloc[0] == pytest.approx(1e-3, rel=1e-9)
