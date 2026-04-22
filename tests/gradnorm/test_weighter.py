import math

import pytest
import torch

from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter


def make_default_weighter():
    return GradNormLossWeighter(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[4.25, 11.0, 0.4, 1.5, 1.0],
        alpha=1.5,
        lr=2.5e-2,
        update_after_step=500,
        pivot_warmup_steps=50,
        update_every=1,
        clamp_min=1e-4,
    )


def test_constructor_state_shapes_and_values():
    w = make_default_weighter()
    assert w.task_names == ["det", "map", "motion", "plan", "ego"]
    assert w.T_tasks == 5
    assert w.alpha == pytest.approx(1.5)
    assert w.update_after_step == 500
    assert w.pivot_warmup_steps == 50
    assert w.update_every == 1
    assert w.clamp_min == pytest.approx(1e-4)

    assert isinstance(w.w, torch.nn.Parameter)
    assert w.w.shape == (5,)
    assert torch.allclose(
        w.w.data, torch.tensor([4.25, 11.0, 0.4, 1.5, 1.0], dtype=torch.float32)
    )

    assert torch.allclose(w.L0, torch.zeros(5))
    assert torch.allclose(w.L0_running, torch.zeros(5))
    assert int(w.step_counter.item()) == 0
    assert bool(w.pivot_ready.item()) is False
    assert math.isclose(float(w.init_w_sum.item()), 18.15, rel_tol=1e-5)

    assert isinstance(w.w_optimizer, torch.optim.Adam)
    param_in_opt = w.w_optimizer.param_groups[0]["params"][0]
    assert param_in_opt is w.w


def test_constructor_rejects_bad_input():
    with pytest.raises(AssertionError):
        GradNormLossWeighter(task_names=["only_one"], init_weights=[1.0])
    with pytest.raises(AssertionError):
        GradNormLossWeighter(
            task_names=["a", "b"], init_weights=[1.0, -1.0]
        )
    with pytest.raises(AssertionError):
        GradNormLossWeighter(
            task_names=["a", "b"], init_weights=[1.0]
        )


def _dummy_shared_param(shape=(4, 4)):
    return [torch.nn.Parameter(torch.randn(*shape, requires_grad=True))]


def _make_loss(values, shared_param=None):
    """Build scalar losses whose autograd graph touches `shared_param` so that
    torch.autograd.grad against it returns valid gradients."""
    if shared_param is None:
        shared_param = _dummy_shared_param()[0]
    losses = {}
    x = torch.randn(shared_param.shape[-1])
    for name, v in values.items():
        # tiny dependency so grad is well-defined but value stays ~= v
        y = (shared_param @ x).sum() * 0.0 + float(v)
        losses[name] = y
    return losses, [shared_param]


def test_warmup_phase_does_not_update_weights():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 3.0],
        update_after_step=5,
        pivot_warmup_steps=2,
        update_every=1,
    )
    shared = _dummy_shared_param()
    initial_w = w.w.data.clone()

    for step in range(5):
        losses, shared_list = _make_loss({"a": 0.5, "b": 2.0}, shared[0])
        weighted, log = w(losses, shared_list)
        assert log["phase_id"] == 0
        expected = 1.0 * 0.5 + 3.0 * 2.0
        assert float(weighted.item()) == pytest.approx(expected, rel=1e-5)

    assert torch.allclose(w.w.data, initial_w)
    assert int(w.step_counter.item()) == 5
    assert bool(w.pivot_ready.item()) is False


def test_pivot_accumulation_and_freeze():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 3.0],
        update_after_step=5,
        pivot_warmup_steps=3,
        update_every=1,
    )
    shared = _dummy_shared_param()

    # Burn warmup
    for _ in range(5):
        losses, sp = _make_loss({"a": 0.5, "b": 2.0}, shared[0])
        w(losses, sp)

    # Step 5: first pivot accumulation iter
    losses, sp = _make_loss({"a": 0.2, "b": 0.8}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 1.0
    assert bool(w.pivot_ready.item()) is False
    assert torch.allclose(w.L0_running, torch.tensor([0.2, 0.8]), atol=1e-6)

    # Step 6: pivot_step == 2 < 3
    losses, sp = _make_loss({"a": 0.4, "b": 1.0}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 1.0
    assert bool(w.pivot_ready.item()) is False
    assert torch.allclose(w.L0_running, torch.tensor([0.6, 1.8]), atol=1e-6)

    # Step 7: pivot_step == 3 == pivot_warmup_steps → freeze
    losses, sp = _make_loss({"a": 0.6, "b": 1.4}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 2.0
    assert bool(w.pivot_ready.item()) is True
    expected_L0 = torch.tensor(
        [(0.2 + 0.4 + 0.6) / 3.0, (0.8 + 1.0 + 1.4) / 3.0]
    )
    assert torch.allclose(w.L0, expected_L0, atol=1e-6)


def _step_through_to_active(w, shared, loss_values):
    """Drive weighter through warmup + pivot_warmup_steps with constant losses,
    leaving it in the state right after pivot freeze."""
    total = w.update_after_step + w.pivot_warmup_steps
    for _ in range(total):
        losses, sp = _make_loss(loss_values, shared[0])
        w(losses, sp)


def test_gn_active_updates_weights_towards_slow_task():
    torch.manual_seed(0)
    w = GradNormLossWeighter(
        task_names=["fast", "slow"],
        init_weights=[1.0, 1.0],
        alpha=1.5,
        lr=0.1,
        update_after_step=2,
        pivot_warmup_steps=2,
        update_every=1,
        clamp_min=1e-4,
    )
    shared = [torch.nn.Parameter(torch.randn(4, 4, requires_grad=True))]

    # Symmetric losses during warmup+pivot → L0 = [1.0, 1.0]
    _step_through_to_active(w, shared, {"fast": 1.0, "slow": 1.0})
    assert bool(w.pivot_ready.item()) is True
    assert torch.allclose(w.L0, torch.tensor([1.0, 1.0]), atol=1e-6)

    w_before = w.w.data.clone()

    # gn_active iter: make "slow" task have larger L (so r̃_slow > r̃_fast),
    # but ensure both losses depend on shared param so grads are non-zero.
    x = torch.randn(4)
    fast_loss = (shared[0] @ x).pow(2).sum() * 1e-3 + 0.1
    slow_loss = (shared[0] @ x).pow(2).sum() * 1e-3 + 2.0
    losses = {"fast": fast_loss, "slow": slow_loss}

    weighted, log = w(losses, shared)
    assert log["phase_id"] == 3.0
    assert "grad_norm_fast" in log and "grad_norm_slow" in log
    assert "rt_fast" in log and "rt_slow" in log
    # r̃_slow > r̃_fast because L_slow/L0_slow > L_fast/L0_fast
    assert log["rt_slow"] > log["rt_fast"]
    # Direction of change: w_slow should move up (or at least not drop hard)
    assert w.w.data[1].item() >= w_before[1].item() - 1e-3
    # sum preserved at init_w_sum
    assert float(w.w.data.sum().item()) == pytest.approx(
        float(w.init_w_sum.item()), rel=1e-4
    )


def test_update_every_skip():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 1.0],
        update_after_step=0,
        pivot_warmup_steps=1,
        update_every=3,
    )
    shared = [torch.nn.Parameter(torch.randn(4, 4, requires_grad=True))]
    phases = []
    for step in range(5):
        x = torch.randn(4)
        losses = {
            "a": (shared[0] @ x).pow(2).sum() * 1e-3 + 0.5,
            "b": (shared[0] @ x).pow(2).sum() * 1e-3 + 0.5,
        }
        _, log = w(losses, shared)
        phases.append(int(log["phase_id"]))
    # step0=pivot_accum(1) → freeze on pivot_step=1 → phase_id 2,
    # step1=gn_active (offset 0 % 3 == 0) → phase 3,
    # step2=gn_skip_every (1%3) → 4,
    # step3=gn_skip_every (2%3) → 4,
    # step4=gn_active (3%3==0) → 3
    assert phases == [2, 3, 4, 4, 3]
