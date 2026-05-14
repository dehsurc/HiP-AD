"""Single-batch smoke tests for the HiP-AD adapter.

Skipped automatically if the HiP-AD repo + ckpt are not available, so this
file can live in CI without infrastructure.
"""
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HIPAD_CKPT = REPO_ROOT / "ckpts" / "E2_E1_stage2_18ep" / "iter_2344.pth"
HIPAD_CONFIG = REPO_ROOT / "ckpts" / "E2_E1_stage2_18ep" / "E2_E1_stage2_18ep.py"
IS_HIPAD_ENV = "hipad" in Path(sys.executable).parts

needs_hipad = pytest.mark.skipif(
    not (HIPAD_CKPT.exists() and HIPAD_CONFIG.exists() and IS_HIPAD_ENV),
    reason="HiP-AD ckpt/config not present or not running in hipad env",
)


@needs_hipad
def test_hipad_adapter_tasks_and_build_model():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    assert sorted(a.tasks) == ["det", "ego", "map", "motion", "plan"]

    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    assert sum(p.numel() for p in model.parameters()) > 0
    assert any(p.requires_grad for p in model.parameters())


@needs_hipad
def test_hipad_adapter_build_dataloader():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    dl = a.build_dataloader(batch_size=1, seed=0, shuffle=False)
    batch = next(iter(dl))
    assert batch is not None


@needs_hipad
def test_hipad_adapter_forward_and_split():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    losses = a.forward_losses(model, batch)
    assert isinstance(losses, dict)
    for t in a.tasks:
        sub = a.split_losses(losses, t)
        assert sub is not None, f"missing task in split: {t}"
        assert sub.requires_grad


@needs_hipad
def test_hipad_adapter_selective_eval_types():
    from torch import nn
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    assert nn.BatchNorm2d in types
    assert any("DeformableFeatureAggregation" in t.__name__ for t in types)


@needs_hipad
def test_hipad_adapter_shared_param_groups():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    groups = a.shared_param_groups(
        model,
        group_names=["dec0_gnn_0", "dec3_ffn_0_fc1", "backbone_layer1"],
    )
    assert set(groups) == {"dec0_gnn_0", "dec3_ffn_0_fc1", "backbone_layer1"}
    for gk, params in groups.items():
        assert len(params) > 0, gk
        for p in params:
            assert p.requires_grad


@needs_hipad
def test_hipad_adapter_freeze_stochastic_state_two_forwards_match():
    """Inside `freeze_stochastic_state`, two consecutive forwards over the
    same batch produce identical per-task losses (matching is frozen)."""
    import torch
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    with a.freeze_stochastic_state():
        l1 = a.forward_losses(model, batch)
        l2 = a.forward_losses(model, batch)
    for k in set(l1) & set(l2):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l2[k]):
            assert torch.allclose(l1[k], l2[k], atol=1e-3), \
                f"freeze leaked at key {k}: {l1[k].item()} != {l2[k].item()}"


@needs_hipad
def test_hipad_adapter_snapshot_restore_round_trip():
    import torch
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    model = a.build_model(ckpt=HIPAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    snap = a.snapshot_temporal_state(model)
    l1 = a.forward_losses(model, batch)
    _ = a.forward_losses(model, batch)
    a.restore_temporal_state(model, snap)
    l3 = a.forward_losses(model, batch)
    for k in set(l1) & set(l3):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l3[k]):
            assert torch.allclose(l1[k], l3[k], atol=1e-3), k
