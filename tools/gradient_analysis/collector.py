"""M1 — Gradient Collector.

Collects per-task gradients at two scopes:
  - shared: gradients on the configured shared-parameter groups (for M2/M5 conflict/norm analysis)
  - full:   gradients on all requires_grad params reachable from a task's loss
            (for M3 probe virtual updates; unreachable params get zero)

Caches per-(checkpoint, batch, task) gradient dicts to `.pt` files so downstream
modules (M2/M3/M5) can read without re-running backward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from .adapters.base import GradientAnalysisAdapter


# Phase 1 #4 B1 — non-zero threshold for per-param gradient validity masks.
# Anything below this is treated as "task gradient does not flow through this
# parameter" and pseudo-shared filtering will exclude it.
EPS_GRAD = 1e-12


# ----------------------------- public pure helpers -----------------------------

def compute_task_full_gradient(
    task_loss: torch.Tensor,
    params: List[nn.Parameter],
    retain_graph: bool = False,
) -> List[torch.Tensor]:
    """Compute gradient of `task_loss` w.r.t. each param in `params`.

    Unreachable params receive a zero tensor of the appropriate shape.
    Output tensors are detached, on the same device/dtype as inputs.
    """
    grads = torch.autograd.grad(
        outputs=task_loss,
        inputs=params,
        retain_graph=retain_graph,
        allow_unused=True,
        create_graph=False,
    )
    out: List[torch.Tensor] = []
    for p, g in zip(params, grads):
        if g is None:
            out.append(torch.zeros_like(p).detach())
        else:
            out.append(g.detach())
    return out


def slice_shared_from_full(
    full_grads: List[torch.Tensor],
    full_params: List[nn.Parameter],
    shared_params: List[nn.Parameter],
) -> torch.Tensor:
    """Return the concatenated flat gradient for `shared_params` by slicing from
    the full-param gradient list, using id()-based matching.
    """
    id2idx = {id(p): i for i, p in enumerate(full_params)}
    parts: List[torch.Tensor] = []
    for p in shared_params:
        idx = id2idx[id(p)]
        parts.append(full_grads[idx].flatten())
    return torch.cat(parts) if parts else torch.empty(0)


# ----------------------------- collection dataclasses -----------------------------

@dataclass
class BatchGradients:
    """Per-batch, per-task gradient bundle.

    v2 (Phase 1 #4 B1): adds `nonzero_masks` carrying per-(task, group, param)
    boolean masks over the flat group tensor. Pair (a, b) analysis can intersect
    these masks to operate only on parameters where both tasks' gradients
    actually flow, making the cosine semantics well-defined ("truly shared" vs
    "pseudo-shared").

    The on-disk schema is forward-compatible: a v1 file (no `nonzero_masks`
    key) loads with an empty dict so existing 100-batch caches keep working.
    """
    batch_idx: int
    shared: Dict[str, Dict[str, torch.Tensor]]  # task -> {group_key -> flat tensor}
    full_norm: Dict[str, float]                 # task -> ||g^full||
    shared_norm: Dict[str, float]               # task -> ||g^shared|| (across all groups)
    loss_values: Dict[str, float]               # task -> L_task(theta)
    nonzero_masks: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "batch_idx": self.batch_idx,
                "shared": self.shared,
                "full_norm": self.full_norm,
                "shared_norm": self.shared_norm,
                "loss_values": self.loss_values,
                "nonzero_masks": self.nonzero_masks,
            },
            out_dir / f"batch_{self.batch_idx:05d}.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "BatchGradients":
        d = torch.load(path, map_location="cpu")
        d.pop("schema_version", None)
        d.setdefault("nonzero_masks", {})
        return cls(**d)


# ----------------------------- main collector -----------------------------

class GradientCollector:
    """Run per-task backward passes over a dataloader and cache gradients.

    Full gradients are NOT cached to disk (they can be GB-sized and are only
    used transiently by M3 probe, which we run in the same session via
    `collect_and_probe`). Shared gradients (much smaller) ARE cached.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: GradientAnalysisAdapter,
        shared_layer_names: List[str],
        device: str,
    ):
        self.model = model
        self.adapter = adapter
        self.tasks = adapter.tasks
        self.device = device
        self.shared_param_groups = adapter.shared_param_groups(model, shared_layer_names)
        self.full_params: List[nn.Parameter] = [
            p for p in self._raw_model().parameters() if p.requires_grad
        ]
        self.shared_params_flat: List[nn.Parameter] = []
        for params in self.shared_param_groups.values():
            self.shared_params_flat.extend(params)

    def _raw_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def forward_losses(self, data) -> Dict[str, torch.Tensor]:
        """Run model forward through the model-specific adapter."""
        return self.adapter.forward_losses(self.model, data)

    def collect_batch(
        self,
        batch_idx: int,
        data,
    ) -> Tuple[BatchGradients, Dict[str, List[torch.Tensor]]]:
        """Compute per-task shared and full gradients for one batch.

        Returns:
          - BatchGradients (for caching)
          - full_grads: task -> list of full-param gradients (kept in memory for M3 probe;
                        caller is responsible for freeing)
        """
        losses = self.forward_losses(data)

        shared: Dict[str, Dict[str, torch.Tensor]] = {}
        nonzero_masks: Dict[str, Dict[str, torch.Tensor]] = {}
        full_grads: Dict[str, List[torch.Tensor]] = {}
        full_norm: Dict[str, float] = {}
        shared_norm: Dict[str, float] = {}
        loss_values: Dict[str, float] = {}

        n_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            task_loss = self.adapter.split_losses(losses, task)
            if task_loss is None:
                # Task absent; record NaN and continue
                shared[task] = {}
                nonzero_masks[task] = {}
                full_grads[task] = [torch.zeros_like(p) for p in self.full_params]
                full_norm[task] = float("nan")
                shared_norm[task] = float("nan")
                loss_values[task] = float("nan")
                continue

            retain = i < n_tasks - 1
            fg = compute_task_full_gradient(task_loss, self.full_params, retain_graph=retain)
            full_grads[task] = fg

            # Shared slice (per-group dict for M2 consumers) plus per-param
            # non-zero masks (Phase 1 #4 B1) so downstream pair-analysis can
            # intersect the two task masks and operate on truly-shared params.
            shared_groups: Dict[str, torch.Tensor] = {}
            mask_groups: Dict[str, torch.Tensor] = {}
            id2idx = {id(p): k for k, p in enumerate(self.full_params)}
            for gk, params in self.shared_param_groups.items():
                parts: List[torch.Tensor] = []
                masks: List[torch.Tensor] = []
                for p in params:
                    g = fg[id2idx[id(p)]].detach()
                    parts.append(g.flatten())
                    masks.append((g.abs().flatten() > EPS_GRAD))
                if parts:
                    shared_groups[gk] = torch.cat(parts).cpu()
                    mask_groups[gk] = torch.cat(masks).cpu()
                else:
                    shared_groups[gk] = torch.empty(0)
                    mask_groups[gk] = torch.empty(0, dtype=torch.bool)
            shared[task] = shared_groups
            nonzero_masks[task] = mask_groups

            # Norms
            shared_concat = torch.cat([v for v in shared_groups.values()]) if shared_groups else torch.empty(0)
            shared_norm[task] = float(shared_concat.norm().item()) if shared_concat.numel() else float("nan")
            full_concat = torch.cat([g.detach().flatten() for g in fg])
            full_norm[task] = float(full_concat.norm().item())
            loss_values[task] = float(task_loss.item())

        bg = BatchGradients(
            batch_idx=batch_idx,
            shared=shared,
            full_norm=full_norm,
            shared_norm=shared_norm,
            loss_values=loss_values,
            nonzero_masks=nonzero_masks,
        )
        return bg, full_grads


# ----------------------------- dataloader factory -----------------------------

def build_dataloader(
    adapter: GradientAnalysisAdapter,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a train dataloader through the active adapter."""
    return adapter.build_dataloader(
        batch_size=batch_size,
        seed=seed,
        shuffle=shuffle,
    )
