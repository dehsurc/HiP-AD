"""FAMO (Fast Adaptive Multitask Optimization) hook for HiP-AD.

Reference: Liu et al., "Famo: Fast Adaptive Multitask Optimization",
NeurIPS 2023.

Idea
----
Replace static per-task loss weights with dynamically adapted weights
``w = softmax(z)`` chosen so that all tasks make comparable progress in log
loss space. Per iteration:

1. Read per-task scalar losses ``l_i`` (already exposed by the model when
   ``_famo_enabled=True``; same path PCGrad uses).
2. Build a single weighted loss for backward::

       L_tilde = sum_i (w_i / l_i_detached) * l_i  +  aux_loss

   Detaching ``l_i_detached`` makes the per-task gradient contribution to
   ``L_tilde`` proportional to ``w_i`` regardless of loss magnitude - i.e.
   FAMO normalizes by the current loss scale.
3. Backward + optimizer step run via the standard (Fp16)OptimizerHook on
   the modified ``runner.outputs['loss']``.
4. Update ``z`` based on the previous iter's observed log-loss decrease per
   task: tasks that improved less get higher weight next iter.

Why this is safe under DDP + gradient checkpointing
---------------------------------------------------
FAMO performs a *single* backward pass, so the "marked ready twice" failure
mode that bit PCGrad and coupled L2-SP under ``with_cp=True`` does not
arise. Compatible with FP16 because the loss scaler operates on the
already-FAMO-weighted loss.

Hook integration
----------------
We rewrite ``runner.outputs['loss']`` inside ``after_train_iter`` at
priority ``HIGH`` (=30), which runs before the OptimizerHook
(``ABOVE_NORMAL``=40). The optimizer then performs backward on the
FAMO-weighted loss exactly as it would on the original loss. The FAMO
weight update is folded into the same hook callback (operates on detached
losses, so ordering relative to the optimizer step is irrelevant).

Note on optimizer state pollution
---------------------------------
Unlike PCGrad which mutates ``param.grad`` post-backward (mixing projected
gradients into AdamW's m/v statistics), FAMO produces a single coherent
gradient via a normal backward pass. AdamW's per-parameter stats stay
clean.
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from mmcv.runner import HOOKS, Hook

logger = logging.getLogger(__name__)


@HOOKS.register_module()
class FAMOHook(Hook):
    """Dynamic per-task loss reweighting via FAMO.

    Args:
        weight_lr (float): learning rate for the weight optimizer (Adam over
            log-weights ``z``). Default 0.025 per FAMO paper.
        gamma (float): weight decay for the weight optimizer. Default 1e-3.
        warmup_iters (int): use uniform task weights for the first N iters
            (no FAMO update); typical to align with LR warmup.
        log_interval (int): push current task weights to log_buffer every
            N iters.
        task_order (list[str], optional): explicit task ordering. If None,
            taken from the first observed ``task_losses`` dict (insertion
            order).
        eps (float): numerical stability constant for log/division.
    """

    def __init__(
        self,
        weight_lr: float = 0.025,
        gamma: float = 1e-3,
        warmup_iters: int = 0,
        log_interval: int = 50,
        task_order: Optional[List[str]] = None,
        eps: float = 1e-8,
    ):
        self.weight_lr = float(weight_lr)
        self.gamma = float(gamma)
        self.warmup_iters = int(warmup_iters)
        self.log_interval = int(log_interval)
        self.task_order = list(task_order) if task_order else None
        self.eps = float(eps)

        self._tasks: Optional[List[str]] = None
        self._z: Optional[torch.Tensor] = None
        self._w_opt: Optional[torch.optim.Adam] = None
        self._prev_loss: Optional[torch.Tensor] = None
        self._enabled_set = False

    def _ensure_model_flag(self, runner):
        """Set ``_famo_enabled=True`` on the inner model so train_step
        exposes per-task losses."""
        if self._enabled_set:
            return
        model = runner.model
        inner = model.module if hasattr(model, 'module') else model
        inner._famo_enabled = True
        self._enabled_set = True
        logger.info("[FAMO] enabled per-task loss exposure on model")

    def _lazy_init(self, task_losses, device):
        if self._z is not None:
            return
        if self.task_order is not None:
            tasks = [t for t in self.task_order if t in task_losses]
            missing = [t for t in self.task_order if t not in task_losses]
            if missing:
                logger.warning(
                    f"[FAMO] task_order entries missing from observed "
                    f"task_losses: {missing}")
        else:
            tasks = list(task_losses.keys())
        self._tasks = tasks
        n = len(tasks)
        self._z = torch.zeros(n, device=device, requires_grad=True)
        self._w_opt = torch.optim.Adam(
            [self._z], lr=self.weight_lr, weight_decay=self.gamma)
        logger.info(
            f"[FAMO] init z (uniform softmax) over {n} tasks: {tasks}; "
            f"weight_lr={self.weight_lr}, gamma={self.gamma}, "
            f"warmup_iters={self.warmup_iters}")

    def before_run(self, runner):
        self._ensure_model_flag(runner)

    def before_train_iter(self, runner):
        # Defensive in case before_run was bypassed (e.g. resume_from path).
        self._ensure_model_flag(runner)

    def after_train_iter(self, runner):
        outputs = runner.outputs
        task_losses = outputs.get('task_losses', None) if outputs else None
        if not task_losses:
            # Model didn't expose per-task losses (e.g. PCGrad path conflict
            # or model not flagged yet); fall back to whatever is in 'loss'.
            return

        device = next(iter(task_losses.values())).device
        self._lazy_init(task_losses, device)

        # Stack in fixed task order. Force each task loss to a 0-d scalar
        # via .sum() since per-task losses arrive with mixed shapes (some
        # 0-d [], some 1-d [1]) depending on the underlying loss module.
        # .sum() preserves grad and yields a 0-d tensor either way.
        loss_list = [task_losses[t].sum() for t in self._tasks]
        cur = torch.stack(loss_list)            # tracks grad
        cur_det = cur.detach()                  # used for weighting / FAMO update

        # ---- 1) FAMO weight update using PREVIOUS iter's loss ----
        if (self._prev_loss is not None
                and runner.iter >= self.warmup_iters):
            # Δ_i = log(prev_l_i) - log(cur_l_i): positive if task i improved.
            delta = ((self._prev_loss + self.eps).log()
                     - (cur_det + self.eps).log()).detach()
            # FAMO objective on z: minimize sum_i w_i * Δ_i.
            # Adam descends -> mass shifts toward tasks with smaller Δ
            # (= tasks improving least), giving them more weight next iter.
            w_curr = F.softmax(self._z, dim=0)
            objective = (w_curr * delta).sum()
            self._w_opt.zero_grad()
            objective.backward()
            self._w_opt.step()

        self._prev_loss = cur_det.clone()

        # ---- 2) Build FAMO-weighted loss for backward ----
        with torch.no_grad():
            w = F.softmax(self._z, dim=0)
            coef = w / (cur_det + self.eps)         # shape [n_tasks]

        weighted_task = (coef * cur).sum()           # scalar with grad

        aux = outputs.get('aux_loss', None)
        if isinstance(aux, torch.Tensor) and aux.requires_grad:
            new_loss = weighted_task + aux
        else:
            new_loss = weighted_task

        # Replace runner.outputs['loss'] so the OptimizerHook (which runs
        # after this hook at ABOVE_NORMAL priority) backprops the FAMO loss.
        outputs['loss'] = new_loss

        # ---- 3) Logging ----
        if runner.iter % self.log_interval == 0:
            log = {}
            for i, t in enumerate(self._tasks):
                log[f'famo/w/{t}'] = float(w[i])
                log[f'famo/loss/{t}'] = float(cur_det[i])
            log['famo/weighted_loss'] = float(new_loss.detach())
            num_samples = outputs.get('num_samples', 1)
            runner.log_buffer.update(log, num_samples)
