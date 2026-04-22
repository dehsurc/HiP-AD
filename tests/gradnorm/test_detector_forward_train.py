"""Dry-run smoke test: exercises the post-head GradNorm branch logic used by
``SparseDetector.forward_train`` with a fake head output and a fake shared-param
provider. Heavy model init is NOT executed.
"""

import pytest
import torch

from projects.mmdet3d_plugin.models.sparse_detector import SparseDetector
from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter


class _FakeOneDecoder(torch.nn.Module):
    def __init__(self, n_shared=2, dim=4):
        super().__init__()
        self.shared = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(dim, dim, requires_grad=True))
            for _ in range(n_shared)
        ])

    def collect_ffn_last_fc_params(self):
        return list(self.shared)


def _make_output(decoder):
    dim = decoder.shared[0].shape[-1]
    x = torch.randn(dim)

    def dep(scale):
        # Depend on every shared param so autograd.grad with allow_unused=False
        # works for each one.
        acc = 0.0
        for p in decoder.shared:
            acc = acc + (p @ x).pow(2).sum() * 1e-3
        return acc + float(scale)

    return {
        "det_loss_cls": dep(0.5),
        "det_loss_box": dep(0.3),
        "det_loss_cns": dep(0.1),
        "det_loss_yns": dep(0.1),
        "map_loss_cls": dep(0.4),
        "map_loss_line": dep(0.6),
        "motion_loss_cls": dep(0.1),
        "motion_loss_reg": dep(0.2),
        "plan_loss_temp_cls": dep(0.3),
        "plan_loss_temp_reg": dep(0.4),
        "ego_loss_status": dep(0.8),
    }


def test_forward_train_gradnorm_path_produces_expected_keys():
    torch.manual_seed(0)
    weighter = GradNormLossWeighter(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        alpha=1.0,
        lr=1e-2,
        update_after_step=0,
        pivot_warmup_steps=1,
        update_every=1,
    )
    fake_decoder = _FakeOneDecoder(n_shared=2, dim=4)

    # Drive through pivot setup (1 iter) before testing gn_active logic
    output_warmup = _make_output(fake_decoder)
    task_names = weighter.task_names
    task_losses_warmup = SparseDetector._aggregate_task_losses(output_warmup, task_names)
    weighter(task_losses_warmup, fake_decoder.collect_ffn_last_fc_params())

    # Now a gn_active iter
    output = _make_output(fake_decoder)
    task_prefixes = {t + "_loss_" for t in task_names}
    task_losses = SparseDetector._aggregate_task_losses(output, task_names)
    shared_params = fake_decoder.collect_ffn_last_fc_params()
    weighted, gn_log = weighter(task_losses, shared_params)

    output = SparseDetector._rename_as_monitor(output, task_prefixes)
    output["loss_gradnorm_total"] = weighted
    for k, v in gn_log.items():
        safe_key = "gn_" + k
        if "loss" in safe_key:
            safe_key = safe_key.replace("loss", "Loss")
        output[safe_key] = torch.as_tensor(
            v, device=weighted.device, dtype=torch.float32
        )

    # Assertions
    assert "loss_gradnorm_total" in output
    for k in ["det_loss_cls", "map_loss_line", "ego_loss_status"]:
        assert k not in output
    for k in ["monitor_det_cls", "monitor_map_line", "monitor_ego_status"]:
        assert k in output
    assert any(k.startswith("gn_w_") for k in output)
    assert output["loss_gradnorm_total"].ndim == 0
    # Only "loss_gradnorm_total" has 'loss' substring (depth not included here)
    summable = [k for k in output if "loss" in k]
    assert set(summable) == {"loss_gradnorm_total"}
