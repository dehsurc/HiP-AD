"""Single-batch smoke tests for the HiP-AD adapter.

Skipped automatically if the HiP-AD repo + ckpt are not available, so this
file can live in CI without infrastructure.
"""
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
HIPAD_CKPT = REPO_ROOT / "ckpts" / "code_epoch1.pth"               # <- DEVIATION: actual filename
HIPAD_CONFIG = REPO_ROOT / "ckpts" / "HiP-AD-Stage2_code.py"

needs_hipad = pytest.mark.skipif(
    not (HIPAD_CKPT.exists() and HIPAD_CONFIG.exists()),
    reason="HiP-AD ckpt or config not present",
)


@needs_hipad
def test_hipad_adapter_tasks_and_build_model():
    from tools.gradient_analysis.adapters.hipad import HipadAdapter

    a = HipadAdapter(config_path=HIPAD_CONFIG)
    assert sorted(a.tasks) == ["det", "map", "motion", "plan"]

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
