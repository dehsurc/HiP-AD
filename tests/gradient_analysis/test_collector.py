"""Collector invariants. Uses a toy linear model to keep tests fast."""
from collections import OrderedDict

import torch
from torch import nn

from tools.gradient_analysis.collector import (
    compute_task_full_gradient,
    slice_shared_from_full,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head_a = nn.Linear(4, 2)
        self.head_b = nn.Linear(4, 2)

    def forward(self, x):
        h = self.shared(x)
        return {"a": self.head_a(h), "b": self.head_b(h)}


def _task_loss(out, task):
    return out[task].pow(2).sum()


def test_full_gradient_excludes_unreachable_params():
    torch.manual_seed(0)
    model = ToyModel()
    x = torch.randn(8, 4)
    out = model(x)
    loss_a = _task_loss(out, "a")
    # Build full-param list
    full_params = [p for p in model.parameters() if p.requires_grad]
    grads = compute_task_full_gradient(loss_a, full_params)
    # head_b params should have zero (or None-replaced) gradient
    id2idx = {id(p): i for i, p in enumerate(full_params)}
    for name, p in model.named_parameters():
        idx = id2idx[id(p)]
        if name.startswith("head_b"):
            assert torch.allclose(grads[idx], torch.zeros_like(grads[idx]))
        else:
            assert grads[idx].abs().sum() > 0


def test_shared_slice_matches_shared_only_computation():
    """V1 sanity: g^shared from full-param call must equal the gradient computed
    directly on shared params only."""
    torch.manual_seed(0)
    model = ToyModel()
    x = torch.randn(8, 4)

    shared_params = list(model.shared.parameters())
    full_params = [p for p in model.parameters() if p.requires_grad]

    # Route 1: full-param grad then slice
    out = model(x)
    loss_a = _task_loss(out, "a")
    full_grad = compute_task_full_gradient(loss_a, full_params, retain_graph=True)
    sliced = slice_shared_from_full(full_grad, full_params, shared_params)

    # Route 2: direct shared-only grad
    direct = torch.autograd.grad(loss_a, shared_params, retain_graph=False)
    direct_flat = torch.cat([g.detach().flatten() for g in direct])

    assert torch.allclose(sliced, direct_flat, atol=1e-6)
