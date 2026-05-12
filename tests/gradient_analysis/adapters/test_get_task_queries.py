"""Adapter contract: get_task_queries returns graph-attached query tensors."""
import pytest


def test_base_adapter_get_task_queries_default_raises_not_implemented():
    from tools.gradient_analysis.adapters.base import BaseAdapter

    class _A(BaseAdapter):
        pass

    with pytest.raises(NotImplementedError):
        _A().get_task_queries(model=None, fwd_artifacts=None)


def test_vad_adapter_get_task_queries_raises_not_implemented():
    from tools.gradient_analysis.adapters.vad import VadAdapter
    with pytest.raises(NotImplementedError):
        VadAdapter().get_task_queries(model=None, fwd_artifacts=None)


def test_hipad_adapter_get_task_queries_returns_dict_from_fwd_artifacts():
    """When the runner has stashed task_queries into fwd_artifacts, retrieve them."""
    import torch
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    queries = {"det": torch.randn(2, 4, requires_grad=True),
               "map": torch.randn(2, 4, requires_grad=True),
               "motion": torch.randn(2, 4, requires_grad=True),
               "plan": torch.randn(2, 4, requires_grad=True)}
    fwd = {"plan_loss": torch.tensor(1.0), "task_queries": queries}
    out = HipadAdapter().get_task_queries(model=None, fwd_artifacts=fwd)
    assert set(out.keys()) == {"det", "map", "motion", "plan"}
    for q in out.values():
        assert q.requires_grad


def test_hipad_adapter_get_task_queries_without_fwd_artifacts_raises():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter
    with pytest.raises(NotImplementedError):
        HipadAdapter().get_task_queries(model=None, fwd_artifacts=None)
