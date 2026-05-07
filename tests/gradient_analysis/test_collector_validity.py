"""Phase 1 #4 B1 — per-param non-zero mask carried alongside cached gradients."""
import torch

from tools.gradient_analysis.collector import BatchGradients


def test_batch_gradients_carries_v2_masks(tmp_path):
    bg = BatchGradients(
        batch_idx=0,
        shared={"a": {"g0": torch.randn(8)}},
        full_norm={"a": 1.0},
        shared_norm={"a": 1.0},
        loss_values={"a": 0.0},
        nonzero_masks={"a": {"g0": torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.bool)}},
    )
    bg.save(tmp_path)
    loaded = BatchGradients.load(tmp_path / "batch_00000.pt")
    assert "a" in loaded.nonzero_masks
    assert loaded.nonzero_masks["a"]["g0"].sum().item() == 5


def test_batch_gradients_v1_load_backward_compat(tmp_path):
    """A v1 .pt file (no `nonzero_masks` key) must load with empty masks so
    existing 100-batch caches keep working without re-collection."""
    legacy = {
        "batch_idx": 0,
        "shared": {"a": {"g0": torch.randn(4)}},
        "full_norm": {"a": 1.0},
        "shared_norm": {"a": 1.0},
        "loss_values": {"a": 0.0},
    }
    (tmp_path / "batch_00000.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save(legacy, tmp_path / "batch_00000.pt")
    loaded = BatchGradients.load(tmp_path / "batch_00000.pt")
    assert loaded.nonzero_masks == {}


def test_batch_gradients_default_factory_isolates_instances():
    """Two default-constructed BatchGradients should not share a dict."""
    bg1 = BatchGradients(batch_idx=0, shared={}, full_norm={}, shared_norm={}, loss_values={})
    bg2 = BatchGradients(batch_idx=1, shared={}, full_norm={}, shared_norm={}, loss_values={})
    bg1.nonzero_masks["a"] = {"g0": torch.zeros(4, dtype=torch.bool)}
    assert "a" not in bg2.nonzero_masks
