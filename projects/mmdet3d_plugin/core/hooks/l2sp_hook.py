"""L2-SP (decoupled) regularization hook for HiP-AD.

Applies a decoupled L2-SP update analogous to decoupled weight decay (AdamW):
after each optimizer step, anchor specified parameters back toward their
stage1 values via a separate in-place update:

    theta <- theta - lr * 2 * lambda(t) * (theta - theta_stage1)

This is mathematically distinct from coupled L2-SP (the original Li et al.
2018 formulation that adds ``lambda * ||theta - theta_0||^2`` to the training
loss), but is the standard "decoupled" variant used with adaptive optimizers
- analogous to AdamW vs Adam+L2.

Why decoupled here:
The coupled formulation creates a second autograd path through anchored
parameters. Combined with gradient checkpointing (``with_cp=True`` on the
ResNet50 backbone) and DDP, this triggers
``Expected to mark a variable ready only once`` because DDP's gradient-ready
hook fires twice per parameter (once via the reentrant checkpoint backward,
once via the L2-SP loss term). Decoupling avoids autograd entirely.

Reference: Li, Grandvalet, Davoine, "Explicit Inductive Bias for Transfer
Learning with Convolutional Networks" (ICML 2018) - coupled formulation.
Chen et al., "Recall and Learn / RecAdam" (EMNLP 2020) - objective shifting
(time-varying regularization weight) on top of pretrained-weight anchoring.

Schedule support:
``lambda(t)`` can be annealed across training to implement RecAdam-style
objective shifting: hold strong anchoring through the early "damage window"
(when newly-activated random-init heads emit large gradients into shared
modules), then relax so the model can fully exploit stage2 capacity later.
"""

import logging
import math

import torch
from mmcv.runner import HOOKS, Hook

logger = logging.getLogger(__name__)


@HOOKS.register_module()
class L2SPHook(Hook):
    """Decoupled L2-SP regularization toward stage1 parameters.

    Args:
        stage1_ckpt (str): path to stage1 checkpoint .pth.
        lambda_l2sp (float): initial (and max) penalty coefficient.
            0 disables the hook.
        include_prefixes (list[str]): only parameters whose names start with
            any of these prefixes are anchored.
        exclude_prefixes (list[str]): parameters matching any of these are
            excluded even if they matched an include prefix.
        schedule (str): one of {'constant', 'linear', 'cosine'}.
            'constant' preserves the original fixed-lambda behavior.
            'linear' / 'cosine' anneal from ``lambda_l2sp`` to ``lambda_min``
            over ``anneal_iters`` starting at iter ``hold_iters``.
        lambda_min (float): final lambda after annealing completes. Set to
            0 to fully release the anchor late in training, or a small
            positive value to keep a residual pull.
        hold_iters (int): number of initial iters to hold lambda at the max
            value before annealing begins. Use this to cover the early
            damage window at full strength.
        anneal_iters (int): number of iters over which lambda decays from
            lambda_l2sp to lambda_min. After ``hold_iters + anneal_iters``
            the effective lambda is fixed at ``lambda_min``.
        log_interval (int): update log_buffer every N iters with drift norm.
    """

    def __init__(
        self,
        stage1_ckpt,
        lambda_l2sp=1.0,
        include_prefixes=None,
        exclude_prefixes=None,
        schedule='constant',
        lambda_min=0.0,
        hold_iters=0,
        anneal_iters=0,
        log_interval=50,
    ):
        self.stage1_ckpt_path = str(stage1_ckpt)
        self.lambda_l2sp = float(lambda_l2sp)
        self.include_prefixes = list(include_prefixes) if include_prefixes else []
        self.exclude_prefixes = list(exclude_prefixes) if exclude_prefixes else []
        if schedule not in ('constant', 'linear', 'cosine'):
            raise ValueError(
                f"[L2-SP] unknown schedule {schedule!r}; "
                "expected 'constant', 'linear', or 'cosine'."
            )
        self.schedule = schedule
        self.lambda_min = float(lambda_min)
        self.hold_iters = int(hold_iters)
        self.anneal_iters = int(anneal_iters)
        if self.schedule != 'constant' and self.anneal_iters <= 0:
            raise ValueError(
                f"[L2-SP] schedule={self.schedule!r} requires anneal_iters > 0."
            )
        self.log_interval = int(log_interval)

        self._ref_params = None
        self._anchored_names = None
        self._initialized = False

    def _effective_lambda(self, cur_iter):
        if self.schedule == 'constant':
            return self.lambda_l2sp
        if cur_iter < self.hold_iters:
            return self.lambda_l2sp
        t = cur_iter - self.hold_iters
        if t >= self.anneal_iters:
            return self.lambda_min
        progress = t / float(self.anneal_iters)
        if self.schedule == 'linear':
            return self.lambda_l2sp + (self.lambda_min - self.lambda_l2sp) * progress
        # cosine: 1 -> 0 over progress, mapped to [lambda_min, lambda_l2sp]
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lambda_min + (self.lambda_l2sp - self.lambda_min) * factor

    def _name_matches(self, name):
        if self.include_prefixes and not any(
            name.startswith(p) for p in self.include_prefixes
        ):
            return False
        if any(name.startswith(p) for p in self.exclude_prefixes):
            return False
        return True

    def _lazy_init(self, model):
        if self._initialized:
            return

        inner = model.module if hasattr(model, 'module') else model

        logger.info(f"[L2-SP] loading stage1 checkpoint: {self.stage1_ckpt_path}")
        ckpt = torch.load(self.stage1_ckpt_path, map_location='cpu', weights_only=False)
        stage1_sd = ckpt.get('state_dict', ckpt)

        self._ref_params = {}
        self._anchored_names = []
        skipped_not_in_ckpt = 0
        skipped_shape_mismatch = 0
        total_params = 0

        for name, param in inner.named_parameters():
            if not param.requires_grad:
                continue
            if not self._name_matches(name):
                continue
            if name not in stage1_sd:
                skipped_not_in_ckpt += 1
                continue
            ref = stage1_sd[name]
            if ref.shape != param.shape:
                skipped_shape_mismatch += 1
                continue
            ref_tensor = ref.detach().to(device=param.device, dtype=param.dtype).clone()
            ref_tensor.requires_grad_(False)
            self._ref_params[name] = ref_tensor
            self._anchored_names.append(name)
            total_params += param.numel()

        if len(self._anchored_names) == 0:
            raise RuntimeError(
                f"[L2-SP] no parameters were anchored. Check include_prefixes "
                f"({self.include_prefixes}) against actual model parameter names."
            )

        if self.schedule == 'constant':
            schedule_desc = f"lambda={self.lambda_l2sp} (constant)"
        else:
            schedule_desc = (
                f"lambda: {self.lambda_l2sp}->{self.lambda_min} "
                f"({self.schedule}, hold={self.hold_iters}, "
                f"anneal={self.anneal_iters})"
            )
        logger.info(
            f"[L2-SP] anchored {len(self._anchored_names)} tensors "
            f"({total_params:,} parameters) with {schedule_desc} "
            "(decoupled, post-step)"
        )
        if skipped_not_in_ckpt:
            logger.info(
                f"[L2-SP] {skipped_not_in_ckpt} matched-name params missing in ckpt "
                f"(expected for stage2-new modules)"
            )
        if skipped_shape_mismatch:
            logger.warning(
                f"[L2-SP] {skipped_shape_mismatch} params had shape mismatch and were skipped"
            )
        sample = self._anchored_names[:3] + (
            ['...'] + self._anchored_names[-2:]
            if len(self._anchored_names) > 5 else []
        )
        logger.info(f"[L2-SP] sample names: {sample}")

        self._initialized = True

    def before_run(self, runner):
        if self.lambda_l2sp <= 0:
            logger.info("[L2-SP] lambda=0, hook will be a no-op")
            return
        self._lazy_init(runner.model)

    def after_train_iter(self, runner):
        if self.lambda_l2sp <= 0 or self._ref_params is None:
            return

        eff_lambda = self._effective_lambda(runner.iter)
        if eff_lambda <= 0:
            if runner.iter % self.log_interval == 0:
                runner.log_buffer.update(
                    {'l2sp_lambda': 0.0},
                    runner.outputs.get('num_samples', 1),
                )
            return

        inner = runner.model.module if hasattr(runner.model, 'module') else runner.model

        cur_lr = runner.current_lr()
        if isinstance(cur_lr, list):
            lr = float(cur_lr[0])
        elif isinstance(cur_lr, dict):
            first_group = next(iter(cur_lr.values()))
            lr = float(first_group[0]) if isinstance(first_group, list) else float(first_group)
        else:
            lr = float(cur_lr)

        coef = 2.0 * eff_lambda * lr
        drift_sq_sum = 0.0
        with torch.no_grad():
            for name, param in inner.named_parameters():
                ref = self._ref_params.get(name)
                if ref is None:
                    continue
                diff = param.data - ref
                param.data.sub_(diff, alpha=coef)
                drift_sq_sum += float(diff.pow(2).sum())

        if runner.iter % self.log_interval == 0:
            runner.log_buffer.update(
                {
                    'l2sp_drift': drift_sq_sum ** 0.5,
                    'l2sp_lambda': eff_lambda,
                },
                runner.outputs.get('num_samples', 1),
            )
