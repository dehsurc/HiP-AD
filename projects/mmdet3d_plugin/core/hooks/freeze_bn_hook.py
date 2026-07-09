"""Freeze all BatchNorm layers during training.

Batch-1 training makes BN batch statistics degenerate (per iteration the
backbone only sees the 6 camera views of a single scene), so running-stat
updates drift. This hook forces every BN module to eval() right before each
train iteration — IterBasedRunner calls ``model.train()`` at the start of
every iteration, which would otherwise re-enable them.

Affine BN weights still receive gradients; only the batch-statistic
computation / running-stat updates are frozen.
"""

from mmcv.runner import HOOKS, Hook
from torch.nn.modules.batchnorm import _BatchNorm


@HOOKS.register_module()
class FreezeBNHook(Hook):

    def before_train_iter(self, runner):
        model = runner.model
        if hasattr(model, 'module'):
            model = model.module
        for m in model.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
