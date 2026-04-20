"""Shared fixtures for gradient analysis tests."""
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def toy_params():
    """Two-group toy param set: group 'a' (2 params), group 'b' (1 param)."""
    from collections import OrderedDict
    groups = OrderedDict()
    groups["a"] = [torch.nn.Parameter(torch.randn(3, 2)), torch.nn.Parameter(torch.randn(4))]
    groups["b"] = [torch.nn.Parameter(torch.randn(5))]
    return groups
