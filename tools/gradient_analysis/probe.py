"""M3 — One/Two-Step Probe.

Applies virtual updates and measures the target task's loss change. Supports
two scopes:

  1. **Full update** (default): updates every `requires_grad` parameter
     reachable from the source task's loss. Measures the total head+shared
     interference for that task pair.

  2. **Per-layer update**: when ``target_layers`` is provided, the virtual
     step is restricted to a single shared parameter group (e.g.
     ``dec3_ffn_0``) at a time. This isolates the contribution of one layer
     to inter-task interference, at the cost of one extra forward sequence
     per layer.

Variance control: if ``freeze_matching=True``, Hungarian matching (det/map)
and mode-selection argmin (motion/plan) are pinned to the result of the
baseline forward, so ΔL between baseline and stepped forwards cannot be
contaminated by re-shuffled assignments. ``forward_seed`` (when set) also
pins the RNG before each forward to fix DN noise sampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .matching_freeze import FrozenMatching, per_forward_seed
from .temporal_state import ModelStateSnapshot


EPS = 1e-8


# ----------------------------- primitives -----------------------------

def apply_virtual_step(
    params: List[nn.Parameter],
    grads: List[torch.Tensor],
    alpha: float,
    normalize: bool,
) -> None:
    """In-place `p <- p - alpha * g` (optionally normalized by ||g||)."""
    if normalize:
        flat = torch.cat([g.detach().flatten() for g in grads])
        n = float(flat.norm())
        if n < EPS:
            return
        scale = alpha / n
    else:
        scale = alpha
    with torch.no_grad():
        for p, g in zip(params, grads):
            p.data.add_(g, alpha=-scale)


def restore_params(params: List[nn.Parameter], snapshot: Dict[int, torch.Tensor]) -> None:
    with torch.no_grad():
        for p in params:
            p.data.copy_(snapshot[id(p)])


def snapshot_params(params: List[nn.Parameter]) -> Dict[int, torch.Tensor]:
    return {id(p): p.data.clone() for p in params}


def _flat_norm(grads: List[torch.Tensor]) -> float:
    if not grads:
        return float("nan")
    return float(torch.cat([g.detach().flatten() for g in grads]).norm().item())


# ----------------------------- probe runner -----------------------------

@dataclass
class ProbeRow:
    batch_idx: int
    source_task: str
    target_task: str
    steps: int
    variant: str
    layer: str           # "_all" for full-param update, group_key otherwise
    grad_norm: float     # ||g|| restricted to the layer's params
    baseline_loss: float
    stepped_loss: float
    delta: float
    rel_delta: float


def _resolve_param_set(
    collector,
    target_layer: Optional[str],
):
    """Pick the param list the virtual update is applied to.

    Returns ``(params, label)``. ``params`` is a flat list of ``nn.Parameter``;
    ``label`` is the layer key (or ``"_all"`` for the full-update case).

    ``shared_param_groups[key]`` may be either an ``OrderedDict`` (older
    grouping) or a plain ``list`` of ``nn.Parameter`` (pcgrad grouping); we
    accept both.
    """
    if target_layer is None:
        return collector.full_params, "_all"
    groups = collector.shared_param_groups
    if target_layer not in groups:
        raise KeyError(
            f"target_layer={target_layer!r} not in shared_param_groups. "
            f"Available: {list(groups.keys())}"
        )
    val = groups[target_layer]
    if hasattr(val, "values"):
        layer_params = list(val.values())
    else:
        layer_params = list(val)
    return layer_params, target_layer


def probe_one_batch(
    collector,
    data,
    data_next=None,
    alpha: float = 1e-3,
    steps_list: Sequence[int] = (1, 2),
    variants: Sequence[str] = ("raw", "normalized"),
    batch_idx: int = 0,
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = False,
    forward_seed: Optional[int] = None,
    reset_temporal_state: bool = False,
) -> List[ProbeRow]:
    """For each (source_task, target_layer) apply a k-step virtual update and
    record ΔL for all targets."""
    from .collector import compute_task_full_gradient
    from analyze_gradient_conflict import _sum_task_loss, TASK_GROUPS, match_loss_key  # type: ignore

    def _sum_task_loss_any(loss_dict, task_name):
        """Like _sum_task_loss but does NOT filter by requires_grad — needed
        for stepped forwards under torch.no_grad()."""
        prefixes = TASK_GROUPS.get(task_name, [])
        total = None
        for key, val in loss_dict.items():
            if match_loss_key(key, prefixes) and isinstance(val, torch.Tensor):
                total = val if total is None else total + val
        return total

    layers: List[Optional[str]] = list(target_layers) if target_layers else [None]
    rows: List[ProbeRow] = []

    fm = FrozenMatching() if freeze_matching else None
    state_snap = ModelStateSnapshot(collector.model) if reset_temporal_state else None

    def _seeded_forward_on(d):
        # Order matters: temporal cache restore first (so the model rolls back
        # to the same starting state), then matching-freeze rewind, then RNG.
        if state_snap is not None:
            state_snap.restore()
        if fm is not None:
            fm.next_forward()
        if forward_seed is not None:
            with per_forward_seed(forward_seed):
                return collector.forward_losses(d)
        return collector.forward_losses(d)

    def _seeded_forward():
        return _seeded_forward_on(data)

    ctx = fm if fm is not None else _NullCM()
    with ctx:
        # Baseline forward — no_grad. This is the call that records matchings
        # (det/map indices, motion/plan mode_idx) into the freeze queues.
        with torch.no_grad():
            losses = _seeded_forward()
            baseline: Dict[str, float] = {}
            for t in collector.tasks:
                tl = _sum_task_loss_any(losses, t)
                baseline[t] = float(tl.item()) if tl is not None else float("nan")

        for source in collector.tasks:
            for layer_key in layers:
                try:
                    update_params, layer_label = _resolve_param_set(collector, layer_key)
                except KeyError as e:
                    print(f"[probe] skip layer {layer_key}: {e}")
                    continue

                for variant in variants:
                    for steps in steps_list:
                        snap = snapshot_params(update_params)

                        # Step 1 — needs grad. Forward graph (G1) holds the
                        # entire backbone activation tape; we must release it
                        # before step 2 builds G2, otherwise both graphs sit
                        # in GPU memory simultaneously (~2× peak).
                        fwd = _seeded_forward()
                        tl = _sum_task_loss(fwd, source)
                        if tl is None:
                            restore_params(update_params, snap)
                            continue
                        g = compute_task_full_gradient(tl, update_params, retain_graph=False)
                        gn = _flat_norm(g)
                        apply_virtual_step(update_params, g, alpha=alpha,
                                           normalize=(variant == "normalized"))
                        # `tl` only frees its own graph branch under
                        # retain_graph=False; the other tasks' losses still
                        # held by `fwd` keep their branches alive. Drop the
                        # whole dict + the residual gradient list explicitly.
                        del fwd, tl, g
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if steps >= 2:
                            data2 = data if data_next is None else data_next
                            fwd2 = _seeded_forward_on(data2)
                            tl2 = _sum_task_loss(fwd2, source)
                            if tl2 is None:
                                restore_params(update_params, snap)
                                continue
                            g2 = compute_task_full_gradient(tl2, update_params, retain_graph=False)
                            apply_virtual_step(update_params, g2, alpha=alpha,
                                               normalize=(variant == "normalized"))
                            del fwd2, tl2, g2
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        # Stepped forward — no_grad
                        with torch.no_grad():
                            fwd_after = _seeded_forward()
                            for target in collector.tasks:
                                tl_after = _sum_task_loss_any(fwd_after, target)
                                if tl_after is None:
                                    continue
                                la = float(tl_after.item())
                                lb = baseline[target]
                                delta = la - lb
                                rel = delta / lb if abs(lb) > EPS else float("nan")
                                if not np.isfinite(delta):
                                    continue
                                rows.append(ProbeRow(
                                    batch_idx=batch_idx,
                                    source_task=source,
                                    target_task=target,
                                    steps=steps,
                                    variant=variant,
                                    layer=layer_label,
                                    grad_norm=gn,
                                    baseline_loss=lb,
                                    stepped_loss=la,
                                    delta=delta,
                                    rel_delta=rel,
                                ))
                        restore_params(update_params, snap)

    return rows


class _NullCM:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def rows_to_dataframe(rows: List[ProbeRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def aggregate_affinity_matrix(
    df: pd.DataFrame, steps: int, variant: str, layer: str = "_all",
) -> pd.DataFrame:
    sub = df[(df["steps"] == steps) & (df["variant"] == variant) & (df["layer"] == layer)]
    return sub.pivot_table(
        index="source_task", columns="target_task", values="delta", aggfunc="mean"
    )


def run_m3(
    collector,
    dataloader,
    num_batches: int,
    alpha: float,
    steps_list: List[int],
    variants: List[str],
    out_dir: Path,
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = False,
    forward_seed: Optional[int] = None,
    reset_temporal_state: bool = False,
) -> pd.DataFrame:
    """Execute M3 on up to ``num_batches`` batches.

    When ``target_layers`` is non-empty, the probe is repeated once per layer
    (cost ≈ N_layers× the default forward count). When empty/None, the
    probe falls back to updating all reachable parameters at once and the
    output ``layer`` column reads ``"_all"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[ProbeRow] = []
    prev_data = None
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break
        rows = probe_one_batch(
            collector, data, data_next=prev_data, alpha=alpha,
            steps_list=steps_list, variants=variants, batch_idx=i,
            target_layers=target_layers,
            freeze_matching=freeze_matching,
            forward_seed=forward_seed,
            reset_temporal_state=reset_temporal_state,
        )
        all_rows.extend(rows)
        prev_data = data

    df = rows_to_dataframe(all_rows)
    df.to_csv(out_dir / "probe_per_batch.csv", index=False)
    layers = sorted(df["layer"].unique()) if not df.empty else ["_all"]
    for layer in layers:
        suffix = "" if layer == "_all" else f"_{layer}"
        for s in steps_list:
            for v in variants:
                mat = aggregate_affinity_matrix(df, s, v, layer)
                if mat.empty:
                    continue
                mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}.csv")
    return df
