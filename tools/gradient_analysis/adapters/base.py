"""Adapter Protocol shared by HiP-AD and VAD.

Ten operations capture every model-specific behaviour the M1-M8 modules
currently inline against HiP-AD. Implementations live in `hipad.py` and
`vad.py` and must satisfy `MockAdapter`-driven Protocol conformance tests
(see `tests/gradient_analysis/adapters/test_protocol.py`).

The interface is intentionally NOT a single-model abstraction: every method
exists because BOTH adapters need it, and is added during the interleaved
authoring (T3-T12 of phase2 plan).

The spec §4.1 sketched 9 operations; we add `restore_temporal_state` for
symmetry with `snapshot_temporal_state`; the Protocol makes the pair
explicit.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TemporalSnapshot:
    """Opaque payload holding the per-forward mutable state an adapter needs
    to keep stable across a probe cycle (counters, recurrent caches,
    sampler payloads, RNG state, ...).

    Adapters are free to put whatever they need in `payload`; M3 probe code
    only ever calls `adapter.restore_temporal_state(model, snap)`, never
    inspects the payload.

    `model_state_keys` is documentation-only — names of the attributes the
    snapshot tracks, used by tests / logs to verify the adapter actually
    captured something.
    """
    model_state_keys: Tuple[str, ...]
    payload: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GradientAnalysisAdapter(Protocol):
    """Model-specific surface used by every M1-M8 module.

    Method ordering matches the lifecycle of a probe batch:
      1. tasks                         — declare which tasks exist
      2. build_model / build_dataloader — construct evaluation harness
      3. forward_losses                — run forward; return raw loss dict
      4. split_losses                  — (loss_dict, task) -> single tensor
      5. shared_param_groups           — grouped parameters for M1
      6. selective_eval_types          — types to put in eval() during probe
      7. freeze_stochastic_state       — context manager for matching freeze
      8. snapshot_temporal_state       — capture mutable temporal caches
      9. restore_temporal_state        — restore from a snapshot
    """

    @property
    def tasks(self) -> List[str]: ...

    def build_model(self, ckpt: Path, device: str) -> nn.Module: ...

    def build_dataloader(
        self, batch_size: int, seed: int, shuffle: bool = False,
    ) -> DataLoader: ...

    def forward_losses(
        self, model: nn.Module, data: Mapping[str, Any],
    ) -> Dict[str, torch.Tensor]: ...

    def split_losses(
        self, loss_dict: Mapping[str, torch.Tensor], task: str,
    ) -> Optional[torch.Tensor]: ...

    def shared_param_groups(
        self, model: nn.Module, group_names: List[str],
    ) -> Dict[str, List[nn.Parameter]]: ...

    def selective_eval_types(self) -> Tuple[Type[nn.Module], ...]: ...

    @contextmanager
    def freeze_stochastic_state(self) -> Iterator[None]: ...

    def snapshot_temporal_state(self, model: nn.Module) -> TemporalSnapshot: ...

    def restore_temporal_state(
        self, model: nn.Module, snapshot: TemporalSnapshot,
    ) -> None: ...


class BaseAdapter:
    """Concrete base class providing default (raising) implementations of
    optional adapter methods. Subclasses that need them override selectively.

    Distinct from ``GradientAnalysisAdapter`` (a ``Protocol``) — this is a
    real class you can inherit from when you want the defaults.
    """

    def get_task_queries(self, model, fwd_artifacts) -> "dict":
        """Return task-specific query tensors that participate in L_plan's graph.

        Default implementation raises ``NotImplementedError``. Subclasses that
        support query-sensitivity analysis (currently HiP-AD only) override
        this and return a mapping ``{task_query_type: Tensor}`` where each
        Tensor still requires gradient (no ``.detach()``).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_task_queries is not implemented"
        )
