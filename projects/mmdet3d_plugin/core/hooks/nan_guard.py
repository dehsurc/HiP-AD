"""Non-finite (NaN/Inf) step protection for batch-1 fp32 training.

The stock recipe trains fp16 with ``Fp16OptimizerHook`` whose loss-scaler
silently skips steps on overflow. Running fp32 (needed for multi-backward
gradient surgery) loses that protection: HiP-AD's ego-status head can
transiently explode through the temporal instance-bank feedback (per-sample
losses in the 1e3 range on rare frames) until the fp16 flash-attention
overflows and a NaN forward permanently poisons the cached bank features.

Guard policy on any non-finite loss / gradient:
  1. zero all gradients and skip the optimizer step,
  2. reset every temporal InstanceBank (equivalent to a sequence restart)
     so cached NaN features cannot contaminate later iterations.
"""

import logging

import torch
from mmcv.runner import HOOKS, OptimizerHook

logger = logging.getLogger(__name__)


def reset_instance_banks(model) -> int:
    """Call ``reset()`` on every *InstanceBank-like module. Returns count."""
    if hasattr(model, 'module'):
        model = model.module
    n = 0
    for m in model.modules():
        if 'InstanceBank' in type(m).__name__ and hasattr(m, 'reset'):
            m.reset()
            n += 1
    return n


def skip_bad_step(runner, reason: str) -> None:
    """Zero grads, reset banks, and log one skipped step."""
    runner.optimizer.zero_grad()
    n_banks = reset_instance_banks(runner.model)
    logger.warning(
        f"[nan-guard] iter {runner.iter}: skipping step ({reason}); "
        f"reset {n_banks} instance banks")
    runner.log_buffer.update({'nan_guard/skipped': 1.0},
                             runner.outputs.get('num_samples', 1))


@HOOKS.register_module()
class SafeOptimizerHook(OptimizerHook):
    """Standard OptimizerHook + non-finite step skipping and bank reset.

    Supports gradient accumulation over ``accum_steps`` micro-samples so a
    batch-1 control run matches the ATTITTUD run's effective batch size (the
    per-sample grads accumulate in ``.grad``, then a single averaged step).
    """

    def __init__(self, accum_steps: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert accum_steps >= 1, accum_steps
        self.accum_steps = accum_steps
        self._win_count = 0

    def after_train_iter(self, runner):
        loss = runner.outputs['loss']
        if not torch.isfinite(loss):
            skip_bad_step(runner, f"non-finite loss {loss.item()}")
            self._win_count = 0
            return

        if self._win_count == 0:
            runner.optimizer.zero_grad()
        loss.backward()
        self._win_count += 1

        if self._win_count < self.accum_steps:
            return

        n = self._win_count
        for p in runner.model.parameters():
            if p.grad is not None:
                p.grad.div_(n)

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                if not torch.isfinite(grad_norm):
                    skip_bad_step(runner, "non-finite grad norm")
                    self._win_count = 0
                    return
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        runner.optimizer.step()
        self._win_count = 0
