import pytest
import torch

from projects.mmdet3d_plugin.models.sparse_detector import SparseDetector


def _tensor(v):
    return torch.tensor(float(v))


def test_aggregate_task_losses_basic():
    output = {
        "det_loss_cls": _tensor(1.0),
        "det_loss_box": _tensor(2.0),
        "det_loss_cns": _tensor(0.5),
        "det_loss_yns": _tensor(0.5),
        "map_loss_cls": _tensor(0.3),
        "map_loss_line": _tensor(0.7),
        "motion_loss_cls": _tensor(0.1),
        "motion_loss_reg": _tensor(0.2),
        "plan_loss_temp_cls": _tensor(0.4),
        "plan_loss_temp_reg": _tensor(0.6),
        "ego_loss_status": _tensor(0.9),
        "loss_dense_depth": _tensor(10.0),
    }
    task_names = ["det", "map", "motion", "plan", "ego"]
    agg = SparseDetector._aggregate_task_losses(output, task_names)
    assert float(agg["det"].item()) == pytest.approx(4.0)
    assert float(agg["map"].item()) == pytest.approx(1.0)
    assert float(agg["motion"].item()) == pytest.approx(0.3)
    assert float(agg["plan"].item()) == pytest.approx(1.0)
    assert float(agg["ego"].item()) == pytest.approx(0.9)
    assert "depth" not in agg


def test_aggregate_asserts_when_prefix_missing():
    output = {"det_loss_cls": _tensor(1.0)}
    with pytest.raises(AssertionError):
        SparseDetector._aggregate_task_losses(output, ["det", "map"])


def test_rename_as_monitor_detaches_and_removes_loss_substring():
    x = torch.tensor(1.0, requires_grad=True)
    output = {
        "det_loss_cls": x * 2.0,
        "map_loss_line": x * 3.0,
        "loss_dense_depth": x * 5.0,
        "some_scalar": 7,
    }
    task_prefixes = {"det_loss_", "map_loss_"}
    out = SparseDetector._rename_as_monitor(output, task_prefixes)
    assert set(out.keys()) == {
        "monitor_det_cls",
        "monitor_map_line",
        "loss_dense_depth",
        "some_scalar",
    }
    assert not out["monitor_det_cls"].requires_grad
    assert not out["monitor_map_line"].requires_grad
    assert out["loss_dense_depth"] is output["loss_dense_depth"]
    assert out["some_scalar"] == 7
    assert "loss" not in "monitor_det_cls"
    assert "loss" not in "monitor_map_line"
