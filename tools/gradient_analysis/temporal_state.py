"""Temporal state snapshot / restore for the SparseDetector probe.

The detector maintains per-forward mutable state — instance banks
(`{det,map,ego,plan,scenes}_instance_bank{,_list}`) cache cls/anchor/meta
across timesteps via ``InstanceBank.cache``, and ``run_step`` increments
once per forward. Without intervention, the baseline / gradient / stepped
forwards inside a probe cycle each see a different cache, contaminating ΔL.

In addition, every ``BaseTargetWithDenoising`` sampler (det/map heads)
caches `dn_metas` at the end of forward via ``cache_dn`` and splices that
payload back into the next forward via ``update_dn`` — leaking F0 DN state
into F1's predictions. The samplers are NOT ``nn.Module`` instances, so
they never appear in ``model.modules()``; we discover them via the
``*_sampler`` attributes hung off of ``nn.Module`` heads.

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


# Attributes on ``InstanceBank`` (and its subclasses) that ``cache(...)`` /
# ``get`` / ``update`` / ``get_instance_id`` mutate per forward. Anything
# missing on a given subclass is skipped.
_BANK_MUTABLE_ATTRS: Tuple[str, ...] = (
    "cached_feature",
    "cached_anchor",
    "metas",
    "mask",
    "confidence",
    "temp_confidence",
    "instance_id",
    "prev_id",
)

_BANK_NAMES: Tuple[str, ...] = ("det", "map", "ego", "plan", "scenes")

# Attributes on ``BaseTargetWithDenoising`` samplers mutated per forward.
# `dn_metas` is the leak that breaks ``freeze_matching``: it accumulates
# DN payload across forwards and is spliced back into the next forward's
# predictions via ``update_dn``.
_SAMPLER_MUTABLE_ATTRS: Tuple[str, ...] = (
    "dn_metas",
    "indices",
)


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
    """Collect every InstanceBank-like object the detector might mutate.

    Banks live nested inside the head (e.g. ``head.onedecoder_head.det_instance_bank``)
    or directly on the detector for other configs (``raw.det_instance_bank``).
    Walk the full ``raw.named_modules()`` tree and pick anything that exposes
    the ``InstanceBank`` mutable-attr surface, plus check the named *_list
    siblings on every nn.Module so deepcopied list-banks aren't missed."""
    try:
        from projects.mmdet3d_plugin.models.instance_bank import (  # type: ignore
            InstanceBank,
        )
    except Exception:
        InstanceBank = None  # type: ignore

    raw = _raw_model(model)
    holders: List[nn.Module] = []
    seen: List[int] = []

    def _try_add(obj: Any) -> None:
        if obj is None:
            return
        is_bank = (InstanceBank is not None and isinstance(obj, InstanceBank))
        if not is_bank:
            cn = type(obj).__name__
            if "InstanceBank" not in cn:
                return
        if id(obj) in seen:
            return
        holders.append(obj)
        seen.append(id(obj))

    # 1) Walk every nn.Module and check named_*_instance_bank{,_list} attrs.
    for _name, module in raw.named_modules():
        for short in _BANK_NAMES:
            single = getattr(module, f"{short}_instance_bank", None)
            _try_add(single)
            lst = getattr(module, f"{short}_instance_bank_list", None)
            if isinstance(lst, (list, tuple)):
                for b in lst:
                    _try_add(b)
        # Some heads expose just `instance_bank` (e.g. det_head.instance_bank).
        _try_add(getattr(module, "instance_bank", None))

    # 2) Also catch any registered submodule whose class is an InstanceBank.
    if InstanceBank is not None:
        for _name, module in raw.named_modules():
            if isinstance(module, InstanceBank):
                _try_add(module)
    return holders


def _sampler_holders(model: nn.Module) -> List[Any]:
    """Collect every ``BaseTargetWithDenoising`` instance reachable from the
    detector. They are plain ABCs, not ``nn.Module``s, so we scan attribute
    slots on every nn.Module and de-dup by identity."""
    try:
        from projects.mmdet3d_plugin.models.base_target import (  # type: ignore
            BaseTargetWithDenoising,
        )
    except Exception:
        return []

    raw = _raw_model(model)
    holders: List[Any] = []
    seen: List[int] = []

    def _try_add(obj: Any) -> None:
        if isinstance(obj, BaseTargetWithDenoising) and id(obj) not in seen:
            holders.append(obj)
            seen.append(id(obj))

    for module in raw.modules():
        # Direct attribute names: every head that holds a sampler does so
        # via a `*_sampler` or `sampler` attribute (det_head.sampler,
        # sparse_onedecoder.{det,map,ego,plan,align,motion}_sampler, etc.).
        for attr in module.__dict__:
            if "sampler" not in attr:
                continue
            try:
                val = getattr(module, attr)
            except AttributeError:
                continue
            _try_add(val)
            if isinstance(val, (list, tuple)):
                for item in val:
                    _try_add(item)
    return holders


class ModelStateSnapshot:
    """Captures ``run_step``, InstanceBank cache attrs, and target sampler
    ``dn_metas``/``indices``, restorable on demand."""

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
        for sampler in _sampler_holders(self.model):
            for attr in _SAMPLER_MUTABLE_ATTRS:
                if hasattr(sampler, attr):
                    self._initial[(id(sampler), attr)] = _deepclone(getattr(sampler, attr))

    def restore(self) -> None:
        raw = _raw_model(self.model)
        if "run_step" in self._initial:
            raw.run_step = self._initial["run_step"]
        for bank in _bank_holders(self.model):
            for attr in _BANK_MUTABLE_ATTRS:
                key = (id(bank), attr)
                if key in self._initial:
                    setattr(bank, attr, _deepclone(self._initial[key]))
        for sampler in _sampler_holders(self.model):
            for attr in _SAMPLER_MUTABLE_ATTRS:
                key = (id(sampler), attr)
                if key in self._initial:
                    setattr(sampler, attr, _deepclone(self._initial[key]))

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
