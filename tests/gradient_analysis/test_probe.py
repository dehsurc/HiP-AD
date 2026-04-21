import torch
from torch import nn

from tools.gradient_analysis.probe import apply_virtual_step, restore_params


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head_a = nn.Linear(4, 2)

    def forward(self, x):
        return self.head_a(self.shared(x))


def _loss(model, x, target):
    return (model(x) - target).pow(2).sum()


def test_apply_virtual_step_then_restore_is_identity():
    torch.manual_seed(0)
    m = ToyModel()
    snapshot = {id(p): p.data.clone() for p in m.parameters()}
    grads = [torch.randn_like(p) for p in m.parameters()]
    apply_virtual_step(list(m.parameters()), grads, alpha=0.01, normalize=False)
    # After restore, params equal snapshot
    restore_params(list(m.parameters()), snapshot)
    for p in m.parameters():
        assert torch.allclose(p.data, snapshot[id(p)])


def test_v2_source_task_own_loss_decreases_after_full_param_step():
    """Sanity: stepping along -g_A on all params reachable from L_A must decrease L_A
    for small enough alpha. This is the assertion that catches the 'shared-only' bug."""
    torch.manual_seed(0)
    m = ToyModel()
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    before = _loss(m, x, target).item()
    params = list(m.parameters())
    grads = list(torch.autograd.grad(_loss(m, x, target), params))
    snapshot = {id(p): p.data.clone() for p in params}
    apply_virtual_step(params, grads, alpha=1e-3, normalize=False)
    after = _loss(m, x, target).item()
    restore_params(params, snapshot)
    assert after < before, f"expected loss decrease but got before={before}, after={after}"
