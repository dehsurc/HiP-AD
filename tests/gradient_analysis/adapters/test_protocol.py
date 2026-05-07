"""Conformance tests for the GradientAnalysisAdapter Protocol shape."""
import inspect

from tools.gradient_analysis.adapters.base import (
    GradientAnalysisAdapter,
    TemporalSnapshot,
)


def test_temporal_snapshot_default_payload():
    snap = TemporalSnapshot(model_state_keys=("k1", "k2"))
    assert snap.model_state_keys == ("k1", "k2")
    assert snap.payload == {}


def test_temporal_snapshot_payload_set():
    snap = TemporalSnapshot(model_state_keys=(), payload={"a": 1})
    assert snap.payload == {"a": 1}


def test_protocol_has_required_methods():
    expected = {
        "tasks", "build_model", "build_dataloader",
        "forward_losses", "split_losses", "shared_param_groups",
        "selective_eval_types", "freeze_stochastic_state",
        "snapshot_temporal_state", "restore_temporal_state",
    }
    members = {n for n, _ in inspect.getmembers(GradientAnalysisAdapter)}
    missing = expected - members
    assert not missing, f"Protocol missing: {missing}"


import contextlib
from typing import Iterator, List, Mapping, Tuple, Type
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MockAdapter:
    """Minimal in-memory adapter exercising every Protocol method.

    Used so M1-M8 module tests do not depend on either real model. The
    construction is *not* an instance of GradientAnalysisAdapter at the
    typing level (we don't subclass the Protocol), but `runtime_checkable`
    on the Protocol means `isinstance(MockAdapter(), GradientAnalysisAdapter)`
    is True iff the structural shape matches — and the test below asserts
    exactly that.
    """

    @property
    def tasks(self) -> List[str]:
        return ["t0", "t1"]

    def build_model(self, ckpt, device):
        m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        return m.to(device)

    def build_dataloader(self, batch_size, seed, shuffle=False):
        torch.manual_seed(seed)
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        return DataLoader(TensorDataset(x, y), batch_size=batch_size,
                          shuffle=shuffle)

    def forward_losses(self, model, data):
        x, y = data
        logits = model(x)
        return {
            "t0": logits.pow(2).mean(),
            "t1": (logits - y[:, None].float()).pow(2).mean(),
        }

    def split_losses(self, loss_dict, task):
        return loss_dict.get(task)

    def shared_param_groups(self, model, group_names):
        out = {}
        for gk in group_names:
            if gk == "linear0":
                out[gk] = list(model[0].parameters())
            elif gk == "linear1":
                out[gk] = list(model[1].parameters())
        return out

    def selective_eval_types(self):
        return (nn.Dropout,)

    @contextlib.contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]:
        yield

    def snapshot_temporal_state(self, model):
        from tools.gradient_analysis.adapters.base import TemporalSnapshot
        return TemporalSnapshot(model_state_keys=("dummy",),
                                payload={"dummy": 0})

    def restore_temporal_state(self, model, snapshot):
        return None


def test_mockadapter_satisfies_protocol_runtime():
    """`runtime_checkable` Protocol — structural conformance check."""
    a = MockAdapter()
    assert isinstance(a, GradientAnalysisAdapter)


def test_mockadapter_full_lifecycle():
    a = MockAdapter()
    assert a.tasks == ["t0", "t1"]
    model = a.build_model(ckpt=None, device="cpu")
    dl = a.build_dataloader(batch_size=2, seed=0)
    batch = next(iter(dl))
    losses = a.forward_losses(model, batch)
    assert "t0" in losses and "t1" in losses
    assert a.split_losses(losses, "t0") is losses["t0"]
    assert a.split_losses(losses, "missing") is None
    groups = a.shared_param_groups(model, ["linear0", "linear1"])
    assert set(groups) == {"linear0", "linear1"}
    assert all(isinstance(p, nn.Parameter) for ps in groups.values() for p in ps)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    with a.freeze_stochastic_state():
        snap = a.snapshot_temporal_state(model)
    assert "dummy" in snap.model_state_keys
    a.restore_temporal_state(model, snap)
