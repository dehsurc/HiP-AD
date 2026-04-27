from __future__ import annotations

from typing import List

import torch
from torch import nn


class GradNormLossWeighter(nn.Module):
    """GradNorm (Chen et al., ICML 2018) loss weighter for HiP-AD stage2.

    See docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md Section 3.
    """

    def __init__(
        self,
        task_names: List[str],
        init_weights: List[float],
        alpha: float = 1.5,
        lr: float = 2.5e-2,
        update_after_step: int = 500,
        pivot_warmup_steps: int = 50,
        update_every: int = 1,
        clamp_min: float = 1e-4,
    ) -> None:
        super().__init__()

        assert len(task_names) == len(init_weights) >= 2, (
            "task_names and init_weights must have equal length >= 2"
        )
        assert alpha >= 0.0, "alpha must be non-negative"
        assert lr > 0.0, "lr must be positive"
        assert update_after_step >= 0
        assert pivot_warmup_steps >= 1
        assert update_every >= 1
        assert all(v > 0 for v in init_weights), "init_weights must be strictly positive"

        self.task_names = list(task_names)
        self.T_tasks = len(task_names)
        self.alpha = float(alpha)
        self.clamp_min = float(clamp_min)
        self.update_after_step = int(update_after_step)
        self.pivot_warmup_steps = int(pivot_warmup_steps)
        self.update_every = int(update_every)

        self.w = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        self.init_weights_cached = list(float(v) for v in init_weights)
        self.w_optimizer = torch.optim.Adam([self.w], lr=lr)

        self.register_buffer("L0", torch.zeros(self.T_tasks))
        self.register_buffer("L0_running", torch.zeros(self.T_tasks))
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        self.register_buffer("pivot_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer(
            "init_w_sum",
            torch.tensor(float(sum(init_weights)), dtype=torch.float32),
        )

    def forward(
        self,
        task_losses: "dict[str, torch.Tensor]",
        shared_params: List[torch.nn.Parameter],
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        step = int(self.step_counter.item())
        self.step_counter += 1

        L = torch.stack([task_losses[n] for n in self.task_names])
        # IMPORTANT: `.detach().clone()` — `.detach()` alone shares storage
        # with self.w, and our in-place `self.w.data.clamp_()` / `.mul_()` below
        # would bump the version of the captured tensor, triggering
        # "variable needed for gradient computation has been modified by an
        # inplace operation" on the main backward.
        w_det = self.w.detach().clone()
        weighted_loss = (w_det * L).sum()

        log: "dict[str, float]" = {}
        for i, n in enumerate(self.task_names):
            log[f"w_{n}"] = float(w_det[i].item())
        log["step"] = float(step)

        # Safety 2 gate 1: warmup
        if step < self.update_after_step:
            log["phase_id"] = 0.0
            return weighted_loss, log

        # Safety 2 gate 2: pivot accumulation / freeze
        if not bool(self.pivot_ready.item()):
            self.L0_running += L.detach().float()
            pivot_step = step - self.update_after_step + 1
            if pivot_step >= self.pivot_warmup_steps:
                self.L0.copy_(self.L0_running / float(self.pivot_warmup_steps))
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        self.L0, op=torch.distributed.ReduceOp.SUM
                    )
                    self.L0.mul_(1.0 / float(torch.distributed.get_world_size()))
                self.pivot_ready.fill_(True)
                log["phase_id"] = 2.0
            else:
                log["phase_id"] = 1.0
            return weighted_loss, log

        # update_every gate
        if (step - self.update_after_step - self.pivot_warmup_steps) % self.update_every != 0:
            log["phase_id"] = 4.0
            return weighted_loss, log

        # ── GradNorm main body ──
        # Memory-efficient formulation: ‖∇(w_i·L_i)‖ = w_i · ‖∇L_i‖ for w_i > 0
        # (clamped via clamp_min). We compute ‖∇L_i‖ with create_graph=False
        # (no second-order tape) and multiply by self.w afterward, keeping the
        # leaf connection so gn_loss.backward() still reaches self.w.
        # This eliminates the 5× second-order activation memory that the
        # original create_graph=True path kept alive, collapsing the peak
        # from ~46 GB → ~18 GB on nuScenes batch 6.
        grad_norms = []
        for i, Li in enumerate(L):
            grads = torch.autograd.grad(
                outputs=Li,
                inputs=shared_params,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            # Safety 1: fp32 promotion
            g_flat = torch.cat([g.float().flatten() for g in grads])
            grad_norms.append(g_flat.norm(p=2))
        norms = torch.stack(grad_norms).detach()   # ‖∇L_i‖, detached scalar[T]
        gw = self.w * norms                        # differentiable wrt self.w

        loss_ratio = L.detach().float() / self.L0.clamp_min(1e-8)
        rt = loss_ratio / loss_ratio.mean().clamp_min(1e-8)

        gw_avg = gw.mean().detach()
        target = (gw_avg * (rt ** self.alpha)).detach()

        gn_loss = (gw - target).abs().sum()

        self.w_optimizer.zero_grad(set_to_none=True)
        gn_loss.backward()
        self.w_optimizer.step()

        # Safety 3: clamp + sum-to-init-sum renorm + DDP sync
        with torch.no_grad():
            self.w.data.clamp_(min=self.clamp_min)
            self.w.data.mul_(
                self.init_w_sum / self.w.data.sum().clamp_min(1e-8)
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    self.w.data, op=torch.distributed.ReduceOp.SUM
                )
                self.w.data.mul_(1.0 / float(torch.distributed.get_world_size()))

        # Extended logging
        for i, n in enumerate(self.task_names):
            log[f"grad_norm_{n}"]  = float(gw[i].item())
            log[f"rt_{n}"]         = float(rt[i].item())
            log[f"L_{n}"]          = float(L[i].item())
            log[f"L0_{n}"]         = float(self.L0[i].item())
            log[f"weighted_L_{n}"] = float((w_det[i] * L[i]).item())
        log["gn_loss"]   = float(gn_loss.item())
        log["phase_id"]  = 3.0
        return weighted_loss, log
