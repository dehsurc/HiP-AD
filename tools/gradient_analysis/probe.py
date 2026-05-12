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

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


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


@contextlib.contextmanager
def _seed_scope(seed: int):
    import random as _random

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    py_state = _random.getstate()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        _random.setstate(py_state)


# ----------------------------- probe runner -----------------------------

@dataclass
class ProbeRow:
    batch_idx: int
    source_task: str
    target_task: str
    steps: int
    variant: str
    layer: str           # "_all" for full-param update, group_key otherwise
    grad_norm: float     # ||g_src|| restricted to the layer's params
    grad_dot: float      # <g_src, g_tgt> at the same param scope
    step_size: float     # effective scalar applied to g_src
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

    def _split_task_loss_any(loss_dict, task_name):
        return collector.adapter.split_losses(loss_dict, task_name)

    layers: List[Optional[str]] = list(target_layers) if target_layers else [None]
    rows: List[ProbeRow] = []

    adapter = collector.adapter
    state_snap = (
        adapter.snapshot_temporal_state(collector.model)
        if reset_temporal_state else None
    )

    def _seeded_forward_on(d):
        # Order matters: temporal cache restore first (so the model rolls back
        # to the same starting state), then matching-freeze rewind, then RNG.
        if state_snap is not None:
            adapter.restore_temporal_state(collector.model, state_snap)
        if forward_seed is not None:
            with _seed_scope(forward_seed):
                return collector.forward_losses(d)
        return collector.forward_losses(d)

    def _seeded_forward():
        return _seeded_forward_on(data)

    ctx = adapter.freeze_stochastic_state() if freeze_matching else _NullCM()
    with ctx:
        # Baseline forward — no_grad.
        with torch.no_grad():
            losses = _seeded_forward()
            baseline: Dict[str, float] = {}
            for t in collector.tasks:
                tl = _split_task_loss_any(losses, t)
                baseline[t] = float(tl.item()) if tl is not None else float("nan")

        for layer_key in layers:
            try:
                update_params, layer_label = _resolve_param_set(collector, layer_key)
            except KeyError as e:
                print(f"[probe] skip layer {layer_key}: {e}")
                continue

            # Pre-compute gradient of each task's loss at this layer's params.
            # Cost: T backward passes per layer per batch; no extra forwards beyond
            # those already required by the per-source loop.
            task_grads: Dict[str, List[torch.Tensor]] = {}
            for t in collector.tasks:
                fwd_g = _seeded_forward()
                tl_g = collector.adapter.split_losses(fwd_g, t)
                if tl_g is None:
                    task_grads[t] = []
                    continue
                gt = compute_task_full_gradient(tl_g, update_params, retain_graph=False)
                task_grads[t] = [x.detach().clone() for x in gt]
                del fwd_g, tl_g, gt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def _grad_dot(gs: List[torch.Tensor], gt: List[torch.Tensor]) -> float:
                if not gs or not gt:
                    return float("nan")
                total = 0.0
                for a, b in zip(gs, gt):
                    total += float((a * b).sum().item())
                return total

            for source in collector.tasks:
                g_src = task_grads.get(source, [])
                if not g_src:
                    continue
                gn = _flat_norm(g_src)

                for variant in variants:
                    for steps in steps_list:
                        step_size = (
                            alpha / max(gn, EPS) if variant == "normalized" else alpha
                        )

                        snap = snapshot_params(update_params)
                        apply_virtual_step(update_params, g_src, alpha=alpha,
                                           normalize=(variant == "normalized"))

                        if steps >= 2:
                            data2 = data if data_next is None else data_next
                            fwd2 = _seeded_forward_on(data2)
                            tl2 = collector.adapter.split_losses(fwd2, source)
                            if tl2 is None:
                                restore_params(update_params, snap)
                                continue
                            g2 = compute_task_full_gradient(tl2, update_params,
                                                            retain_graph=False)
                            apply_virtual_step(update_params, g2, alpha=alpha,
                                               normalize=(variant == "normalized"))
                            del fwd2, tl2, g2
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        with torch.no_grad():
                            fwd_after = _seeded_forward()
                            for target in collector.tasks:
                                tl_after = _split_task_loss_any(fwd_after, target)
                                if tl_after is None:
                                    continue
                                la = float(tl_after.item())
                                lb = baseline[target]
                                delta = la - lb
                                rel = delta / lb if abs(lb) > EPS else float("nan")
                                if not np.isfinite(delta):
                                    continue
                                grad_dot = _grad_dot(g_src, task_grads.get(target, []))
                                rows.append(ProbeRow(
                                    batch_idx=batch_idx,
                                    source_task=source,
                                    target_task=target,
                                    steps=steps,
                                    variant=variant,
                                    layer=layer_label,
                                    grad_norm=gn,
                                    grad_dot=grad_dot,
                                    step_size=step_size,
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


def aggregate_affinity_matrix_with_ci(
    df: pd.DataFrame, steps: int, variant: str, layer: str = "_all",
    n_resamples: int = 2000, seed: int = 0,
):
    """Phase 1 #9 — bootstrap variance bands on the affinity matrix.

    Like ``aggregate_affinity_matrix`` but returns three matrices:
    (mean, ci_lo, ci_hi) of the per-batch ΔL distribution per (source, target).
    Cells whose CI brackets zero are not flagged here — that's downstream
    interpretation responsibility.
    """
    from .bootstrap import bca_ci

    sub = df[(df["steps"] == steps) & (df["variant"] == variant) & (df["layer"] == layer)]
    sources = sorted(sub["source_task"].unique()) if not sub.empty else []
    targets = sorted(sub["target_task"].unique()) if not sub.empty else []
    mean = pd.DataFrame(np.nan, index=sources, columns=targets)
    lo = pd.DataFrame(np.nan, index=sources, columns=targets)
    hi = pd.DataFrame(np.nan, index=sources, columns=targets)
    for i, s in enumerate(sources):
        for j, t in enumerate(targets):
            cell = sub[(sub["source_task"] == s) & (sub["target_task"] == t)]["delta"].to_numpy()
            cell = cell[np.isfinite(cell)]
            if cell.size == 0:
                continue
            mean.iloc[i, j] = float(cell.mean())
            l, h = bca_ci(cell, statistic=np.mean, n_resamples=n_resamples,
                          seed=seed + i * len(targets) + j)
            lo.iloc[i, j] = l
            hi.iloc[i, j] = h
    return mean, lo, hi


def run_alpha_sweep(
    collector,
    dataloader=None,
    num_batches: int = 100,
    alphas: Sequence[float] = (1e-4, 1e-3),
    sources: Sequence[str] = ("motion",),
    out_dir: Path = Path("alpha_sweep"),
    steps_list: Sequence[int] = (1,),
    variants: Sequence[str] = ("raw", "normalized"),
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = True,
    forward_seed: Optional[int] = 42,
    reset_temporal_state: bool = True,
    dl_factory: Optional[Callable] = None,
) -> pd.DataFrame:
    """Phase 1 #9 — α-sensitivity sweep, motion-only by default.

    Re-runs the probe on a per-α basis, restricted to the given source tasks.
    Output: one CSV per α at ``out_dir/sweep_alpha_<a>.csv`` plus a combined
    ``sweep_summary.csv`` with the diag>0 violation rate as a function of α.

    Dataloader handling:

    * Pass `dl_factory=lambda: build_dataloader(...)` (preferred). The factory
      is called once per α to produce a fresh iterator with the same seed →
      same batch order → α comparisons are deterministic without holding
      the entire dataset in memory.

    * Pass `dataloader=...` (legacy). The first `num_batches` items are
      materialized via `itertools.islice` and replayed across α values.
      Memory cost: `num_batches × per-batch tensor size`, bounded.
    """
    import itertools
    from .bootstrap import bca_ci

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []

    if dl_factory is None and dataloader is None:
        raise ValueError("run_alpha_sweep requires either `dl_factory` or `dataloader`")

    # If only a static dataloader is given, materialize the first num_batches
    # via islice (NOT list(iter(dl)) which exhausts the entire dataset).
    materialized: Optional[List] = None
    if dl_factory is None:
        materialized = list(itertools.islice(dataloader, num_batches))

    sources = list(sources)
    for a in alphas:
        # Per-α iterator: prefer factory (deterministic, low-mem) over the
        # materialized snapshot (legacy fallback).
        if dl_factory is not None:
            dl_iter = itertools.islice(iter(dl_factory()), num_batches)
        else:
            dl_iter = iter(materialized)
        rows: List[ProbeRow] = []
        prev_data = None
        for i, data in enumerate(dl_iter):
            rows.extend(probe_one_batch(
                collector, data, data_next=prev_data, alpha=a,
                steps_list=list(steps_list), variants=list(variants), batch_idx=i,
                target_layers=target_layers,
                freeze_matching=freeze_matching, forward_seed=forward_seed,
                reset_temporal_state=reset_temporal_state,
            ))
            prev_data = data
        df = rows_to_dataframe(rows)
        if not df.empty:
            df = df[df["source_task"].isin(sources)]
        df.to_csv(out_dir / f"sweep_alpha_{a:.0e}.csv", index=False)
        for variant in variants:
            sub = df[(df["steps"] == 1) & (df["variant"] == variant)] if not df.empty \
                  else df.iloc[0:0]
            for s in sources:
                target_set = sorted(sub["target_task"].unique()) if not sub.empty else []
                for t in target_set:
                    cell = sub[(sub["source_task"] == s) & (sub["target_task"] == t)]
                    if cell.empty:
                        continue
                    deltas = cell["delta"].to_numpy()
                    diag_violation = (
                        float((deltas > 0).mean()) if s == t and deltas.size > 0
                        else float("nan")
                    )
                    lo, hi = bca_ci(deltas, n_resamples=1000, seed=0)
                    summary_rows.append({
                        "alpha": a, "variant": variant,
                        "source_task": s, "target_task": t,
                        "n": int(len(cell)),
                        "diag_violation_rate": diag_violation,
                        "mean_delta": float(deltas.mean()) if deltas.size else float("nan"),
                        "mean_delta_ci_lo": lo,
                        "mean_delta_ci_hi": hi,
                    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    return summary


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
                # Phase 1 #9 — write mean + bootstrap CI bands alongside.
                mean_mat, lo_mat, hi_mat = aggregate_affinity_matrix_with_ci(
                    df, s, v, layer, n_resamples=2000, seed=42,
                )
                if mean_mat.empty:
                    continue
                mean_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}.csv")
                lo_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}_ci_lo.csv")
                hi_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}_ci_hi.csv")
    return df
