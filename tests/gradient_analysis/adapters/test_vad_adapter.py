"""Single-batch smoke tests for the VAD adapter.

Skipped if the VAD repo + ckpt are not available. Tests run from the
HiP-AD repo root with `PYTHONPATH=.`; the adapter pushes the VAD repo on
sys.path internally.
"""
from pathlib import Path
import sys

import pytest

VAD_REPO = Path("/home/yongjae/e2e/VAD")
VAD_CKPT = VAD_REPO / "data/ckpts/epoch_1.pth"
VAD_CONFIG = VAD_REPO / "data/ckpts/VAD_tiny_e2e.py"
IS_VAD_ENV = "vad" in Path(sys.executable).parts

needs_vad = pytest.mark.skipif(
    not (VAD_CKPT.exists() and VAD_CONFIG.exists() and IS_VAD_ENV),
    reason="VAD ckpt/config not present or not running in vad env",
)


@needs_vad
def test_vad_adapter_tasks():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    assert sorted(a.tasks) == ["det", "map", "motion", "plan"]


@needs_vad
def test_vad_adapter_build_model_and_dataloader():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    assert sum(p.numel() for p in model.parameters()) > 0

    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    assert "img" in batch


@needs_vad
def test_vad_adapter_forward_and_split():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))
    losses = a.forward_losses(model, batch)
    for t in ["det", "map", "motion", "plan"]:
        sub = a.split_losses(losses, t)
        assert sub is not None, f"missing task in split: {t}"
        assert sub.requires_grad


@needs_vad
def test_vad_adapter_selective_eval_types():
    from torch import nn
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    types = a.selective_eval_types()
    assert nn.Dropout in types
    assert nn.BatchNorm2d in types


@needs_vad
def test_vad_adapter_shared_param_groups():
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    groups = a.shared_param_groups(
        model,
        group_names=[
            "enc0_temporal_self_attention",
            "enc0_spatial_cross_attention",
            "enc0_ffn",
            "enc0_norm_0",
            "dec0_self_attn",
            "dec0_cross_attn",
            "dec0_ffn",
        ],
    )
    assert set(groups).issuperset({
        "enc0_temporal_self_attention",
        "enc0_spatial_cross_attention",
        "enc0_ffn",
    })
    for gk, params in groups.items():
        assert len(params) > 0, gk


@needs_vad
def test_vad_adapter_freeze_stochastic_state_two_forwards_match():
    import torch
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    with a.freeze_stochastic_state():
        l1 = a.forward_losses(model, batch)
        l2 = a.forward_losses(model, batch)
    for k in set(l1) & set(l2):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l2[k]):
            assert torch.allclose(l1[k], l2[k], atol=1e-3), \
                f"freeze leaked at key {k}: {l1[k].item()} != {l2[k].item()}"


@needs_vad
def test_vad_adapter_snapshot_restore_round_trip():
    import torch
    from tools.gradient_analysis.adapters.vad import VadAdapter

    a = VadAdapter(repo_root=VAD_REPO, config_path=VAD_CONFIG)
    model = a.build_model(ckpt=VAD_CKPT, device="cuda:0")
    dl = a.build_dataloader(batch_size=1, seed=0)
    batch = next(iter(dl))

    snap = a.snapshot_temporal_state(model)
    with a.freeze_stochastic_state():
        l1 = a.forward_losses(model, batch)
        _ = a.forward_losses(model, batch)
        a.restore_temporal_state(model, snap)
        l3 = a.forward_losses(model, batch)
    for k in set(l1) & set(l3):
        if torch.is_tensor(l1[k]) and torch.is_tensor(l3[k]):
            assert torch.allclose(l1[k], l3[k], atol=1e-3), k
