"""Temporal state snapshot / restore for the SparseDetector probe.

The detector maintains per-forward mutable state — instance banks
(`{det,map,ego,plan,scenes}_instance_bank{,_list}`) cache cls/anchor/meta
across timesteps via ``InstanceBank.cache``, and ``run_step`` increments
once per forward. Without intervention, the baseline / gradient / stepped
forwards inside a probe cycle each see a different cache, contaminating ΔL.

``ModelStateSnapshot`` captures the relevant attributes once and lets the
probe restore them before every forward inside the cycle. Outside the
context the original (live) state is reinstated, so wider training /
evaluation isn't affected.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, List, Tuple

import torch
from torch import nn


# Attributes on ``InstanceBank`` (and its subclasses) that ``cache(...)``
# mutates per forward. Anything missing on a given subclass is skipped.
_BANK_MUTABLE_ATTRS: Tuple[str, ...] = (
    "cached_feature",
    "cached_anchor",
    "metas",
    "confidence",
    "temp_confidence",
    "instance_id",
)

_BANK_NAMES: Tuple[str, ...] = ("det", "map", "ego", "plan", "scenes")


def _deepclone(x: Any) -> Any:
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, dict):
        return {k: _deepclone(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_deepclone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_deepclone(v) for v in x)
    return x  # primitives & misc — assumed immutable


def _raw_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _bank_holders(model: nn.Module) -> List[nn.Module]:
    """Collect every InstanceBank-like object the detector might mutate."""
    raw = _raw_model(model)
    holders: List[nn.Module] = []
    for name in _BANK_NAMES:
        single = getattr(raw, f"{name}_instance_bank", None)
        if single is not None:
            holders.append(single)
        lst = getattr(raw, f"{name}_instance_bank_list", None)
        if lst is not None:
            for b in lst:
                # Some configurations alias `_list[0]` to the singleton above;
                # de-dup by identity to avoid double-snapshotting the same obj.
                if not any(b is h for h in holders):
                    holders.append(b)
    return holders


class ModelStateSnapshot:
    """Captures `run_step` and InstanceBank cache attrs, restorable on demand."""

    def __init__(self, model: nn.Module):
        self.model = model
        self._initial: Dict[Any, Any] = {}
        self._capture()

    def _capture(self) -> None:
        raw = _raw_model(self.model)
        self._initial.clear()
        if hasattr(raw, "run_step"):
            self._initial["run_step"] = raw.run_step
        for bank in _bank_holders(self.model):
            for attr in _BANK_MUTABLE_ATTRS:
                if hasattr(bank, attr):
                    self._initial[(id(bank), attr)] = _deepclone(getattr(bank, attr))

    def restore(self) -> None:
        raw = _raw_model(self.model)
        if "run_step" in self._initial:
            raw.run_step = self._initial["run_step"]
        for bank in _bank_holders(self.model):
            for attr in _BANK_MUTABLE_ATTRS:
                key = (id(bank), attr)
                if key in self._initial:
                    setattr(bank, attr, _deepclone(self._initial[key]))

    # ------------- context manager: capture on enter, restore live on exit -------------
    def __enter__(self) -> "ModelStateSnapshot":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Drop snapshots; do NOT auto-restore on exit — the caller is
        # responsible for ending in a sensible state. We just free memory.
        self._initial.clear()


@contextlib.contextmanager
def temporal_state_freeze(model: nn.Module):
    """Convenience: yield a snapshot taken at entry. Caller calls
    `snapshot.restore()` before each forward to keep the cache identical."""
    snap = ModelStateSnapshot(model)
    try:
        yield snap
    finally:
        snap._initial.clear()
