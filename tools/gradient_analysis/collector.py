"""M1 — Gradient Collector.

Collects per-task gradients at two scopes:
  - shared: gradients on the configured shared-parameter groups (for M2/M5 conflict/norm analysis)
  - full:   gradients on all requires_grad params reachable from a task's loss
            (for M3 probe virtual updates; unreachable params get zero)

Caches per-(checkpoint, batch, task) gradient dicts to `.pt` files so downstream
modules (M2/M3/M5) can read without re-running backward.
"""
from __future__ import annotations

import contextlib
import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


# Stochastic / running-stat layers that MUST be in eval mode during the probe.
# Without this, every forward updates BN running stats and re-samples dropout
# masks, so baseline / grad / stepped forwards see a drifting model. That
# contamination was responsible for the systematic ΔL_det blow-up at 1ep
# regardless of source task: the *upstream* features (where det is most
# sensitive) drifted, while downstream queries that buffer through additional
# transforms (map/motion/plan) showed only minor drift — exactly the pattern
# observed before this fix landed.
_STOCHASTIC_TYPES = (
    nn.Dropout, nn.Dropout2d, nn.Dropout3d,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
)


@contextlib.contextmanager
def _selective_eval(model: nn.Module):
    """Force Dropout / BN / DeformableFeatureAggregation to eval for the
    duration of the block, while keeping the rest of the model in train mode
    (so loss-computing branches still fire). Restores prior training state on
    exit."""
    # Lazy import — DeformableFeatureAggregation is HiP-AD-specific.
    try:
        from projects.mmdet3d_plugin.models.blocks import (  # type: ignore
            DeformableFeatureAggregation,
        )
        extra_types = (DeformableFeatureAggregation,)
    except Exception:
        extra_types = tuple()

    switched: List[nn.Module] = []
    target_types = _STOCHASTIC_TYPES + extra_types
    for m in model.modules():
        if isinstance(m, target_types) and m.training:
            m.eval()
            switched.append(m)
    try:
        yield
    finally:
        for m in switched:
            m.train()


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
    """Per-batch, per-task gradient bundle."""
    batch_idx: int
    shared: Dict[str, Dict[str, torch.Tensor]]  # task -> {group_key -> flat tensor}
    full_norm: Dict[str, float]                 # task -> ||g^full||
    shared_norm: Dict[str, float]               # task -> ||g^shared|| (across all groups)
    loss_values: Dict[str, float]               # task -> L_task(theta)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "batch_idx": self.batch_idx,
                "shared": self.shared,
                "full_norm": self.full_norm,
                "shared_norm": self.shared_norm,
                "loss_values": self.loss_values,
            },
            out_dir / f"batch_{self.batch_idx:05d}.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "BatchGradients":
        d = torch.load(path, map_location="cpu")
        return cls(**d)


# ----------------------------- main collector -----------------------------

# Lazy imports for HiP-AD-specific utilities — only loaded when GradientCollector is used.
# This lets the pure helpers above be importable in pure-Python test environments.
def _import_hipad_utils():
    repo_root = Path(__file__).resolve().parents[2]
    tools_dir = repo_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from analyze_gradient_conflict import (  # type: ignore
        TASK_GROUPS,
        _sum_task_loss,
        compute_task_gradient_grouped,
    )
    from projects.mmdet3d_plugin.core.hooks.pcgrad_optimizer_hook import (  # type: ignore
        get_shared_parameters_grouped,
    )
    return {
        "TASK_GROUPS": TASK_GROUPS,
        "_sum_task_loss": _sum_task_loss,
        "compute_task_gradient_grouped": compute_task_gradient_grouped,
        "get_shared_parameters_grouped": get_shared_parameters_grouped,
    }


class GradientCollector:
    """Run per-task backward passes over a dataloader and cache gradients.

    Full gradients are NOT cached to disk (they can be GB-sized and are only
    used transiently by M3 probe, which we run in the same session via
    `collect_and_probe`). Shared gradients (much smaller) ARE cached.
    """

    def __init__(
        self,
        model: nn.Module,
        tasks: List[str],
        shared_layer_names: List[str],
        device: str,
    ):
        self.model = model
        self.tasks = tasks
        self.device = device
        self._utils = _import_hipad_utils()
        self.shared_param_groups, self.shared_param_ids = self._utils["get_shared_parameters_grouped"](
            model, shared_layer_names
        )
        self.full_params: List[nn.Parameter] = [
            p for p in self._raw_model().parameters() if p.requires_grad
        ]
        self.shared_params_flat: List[nn.Parameter] = []
        for params in self.shared_param_groups.values():
            self.shared_params_flat.extend(params)

    def _raw_model(self) -> nn.Module:
        return self.model.module if hasattr(self.model, "module") else self.model

    def forward_losses(self, data) -> Dict[str, torch.Tensor]:
        """Run model forward and return the loss dict with grad.

        Uses ``_selective_eval`` so BN/Dropout/DeformableFeatureAggregation are
        in eval mode for this forward, while the rest of the model stays in
        train (so the loss-computing branches still fire). Without this, the
        probe's baseline / grad / stepped forwards see a drifting model
        because each ``model(**data)`` call updates BN running stats and
        resamples dropout masks.
        """
        self.model.train()
        with _selective_eval(self._raw_model()):
            losses = self.model(**data)
        if isinstance(losses, (list, tuple)):
            # MMDataParallel wrapping may return a list; take first (single GPU)
            losses = losses[0]
        return losses

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
        _sum_task_loss = self._utils["_sum_task_loss"]
        losses = self.forward_losses(data)

        shared: Dict[str, Dict[str, torch.Tensor]] = {}
        full_grads: Dict[str, List[torch.Tensor]] = {}
        full_norm: Dict[str, float] = {}
        shared_norm: Dict[str, float] = {}
        loss_values: Dict[str, float] = {}

        n_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            task_loss = _sum_task_loss(losses, task)
            if task_loss is None:
                # Task absent; record NaN and continue
                shared[task] = {}
                full_grads[task] = [torch.zeros_like(p) for p in self.full_params]
                full_norm[task] = float("nan")
                shared_norm[task] = float("nan")
                loss_values[task] = float("nan")
                continue

            retain = i < n_tasks - 1
            fg = compute_task_full_gradient(task_loss, self.full_params, retain_graph=retain)
            full_grads[task] = fg

            # Shared slice (per-group dict for M2 consumers)
            shared_groups: Dict[str, torch.Tensor] = {}
            id2idx = {id(p): k for k, p in enumerate(self.full_params)}
            for gk, params in self.shared_param_groups.items():
                parts = [fg[id2idx[id(p)]].flatten() for p in params]
                shared_groups[gk] = torch.cat(parts).detach().cpu()
            shared[task] = shared_groups

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
        )
        return bg, full_grads


# ----------------------------- dataloader factory -----------------------------

def build_dataloader(cfg, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Build a train dataloader using HiP-AD's custom dataset builder."""
    from projects.mmdet3d_plugin.datasets.builder import custom_build_dataset  # type: ignore
    from mmcv.parallel import collate  # type: ignore

    dataset = custom_build_dataset(cfg.data.train)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=min(4, batch_size),
        collate_fn=partial(collate, samples_per_gpu=batch_size),
        drop_last=True,
        generator=g,
    )
