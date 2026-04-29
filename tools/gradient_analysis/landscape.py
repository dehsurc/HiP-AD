"""M8 — Gradient-direction loss surface (1D line probe).

For each (source_task, layer) the module computes ``g = ∇L_source`` w.r.t.
the chosen parameter set, normalizes it, and sweeps a multiplier ``t`` over
``t_grid`` measuring ``L_target(θ − t·α·ĝ)`` for every target task. With
this sign convention ``t > 0`` is the descent direction for ``L_source``
(loss should decrease for ``target=source`` in a locally convex region) and
``t < 0`` is the ascent direction.

This complements M3:
  - M3 reports a single ΔL at a fixed ``α`` (one point on the surface)
  - M8 reports the full ``L(t)`` curve so you can see asymmetry around ``t=0``,
    locate the convex / non-convex region, and read off curvature.

Cost: ``num_batches × num_sources × num_layers × len(t_grid)`` forwards + one
backward per (batch, source, layer). Defaults are intentionally small.

Output:
  - ``landscape_per_batch.csv`` — long format, one row per (batch, source,
    target, layer, t)
  - ``landscape_aggregate.csv`` — mean loss / delta per (source, target,
    layer, t) across batches
  - ``plots/<source>_to_<target>_<layer>.png`` — line plot per pair (when
    matplotlib is importable; otherwise skipped silently)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .matching_freeze import FrozenMatching, per_forward_seed
from .temporal_state import ModelStateSnapshot
from .probe import (
    EPS,
    apply_virtual_step,
    restore_params,
    snapshot_params,
    _resolve_param_set,
    _flat_norm,
    _NullCM,
)


@dataclass
class LandscapeRow:
    batch_idx: int
    source_task: str
    target_task: str
    layer: str
    t: float
    grad_norm: float
    loss: float
    delta: float
    rel_delta: float


def line_probe_one_batch(
    collector,
    data,
    alpha: float,
    t_grid: Sequence[float],
    batch_idx: int = 0,
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = False,
    forward_seed: Optional[int] = None,
    reset_temporal_state: bool = False,
) -> List[LandscapeRow]:
    """Sweep loss along the (normalized) gradient direction for each
    (source_task, layer), recording losses for every target task at each t."""
    from .collector import compute_task_full_gradient
    from analyze_gradient_conflict import _sum_task_loss, TASK_GROUPS, match_loss_key  # type: ignore

    def _sum_task_loss_any(loss_dict, task_name):
        prefixes = TASK_GROUPS.get(task_name, [])
        total = None
        for key, val in loss_dict.items():
            if match_loss_key(key, prefixes) and isinstance(val, torch.Tensor):
                total = val if total is None else total + val
        return total

    layers: List[Optional[str]] = list(target_layers) if target_layers else [None]
    rows: List[LandscapeRow] = []

    fm = FrozenMatching() if freeze_matching else None
    state_snap = ModelStateSnapshot(collector.model) if reset_temporal_state else None

    def fwd():
        if state_snap is not None:
            state_snap.restore()
        if fm is not None:
            fm.next_forward()
        if forward_seed is not None:
            with per_forward_seed(forward_seed):
                return collector.forward_losses(data)
        return collector.forward_losses(data)

    ctx = fm if fm is not None else _NullCM()
    with ctx:
        # Baseline (t=0) — this is also the call that fills the freeze queue.
        with torch.no_grad():
            losses = fwd()
            baseline: Dict[str, float] = {}
            for t in collector.tasks:
                tl = _sum_task_loss_any(losses, t)
                baseline[t] = float(tl.item()) if tl is not None else float("nan")

        for source in collector.tasks:
            for layer_key in layers:
                try:
                    update_params, layer_label = _resolve_param_set(collector, layer_key)
                except KeyError as e:
                    print(f"[landscape] skip layer {layer_key}: {e}")
                    continue

                snap = snapshot_params(update_params)

                # One backward to obtain the descent direction.
                fwd_with_grad = fwd()
                tl = _sum_task_loss(fwd_with_grad, source)
                if tl is None:
                    continue
                grads = compute_task_full_gradient(tl, update_params, retain_graph=False)
                gn = _flat_norm(grads)
                if not np.isfinite(gn) or gn < EPS:
                    print(f"[landscape] skip {source}/{layer_label}: grad norm ~0")
                    continue

                # We sweep t·α along the unit direction ĝ. Re-using
                # apply_virtual_step's `normalize=True` codepath gives us
                # alpha-units of motion in the direction of ĝ.
                for t in t_grid:
                    if t == 0.0:
                        # Trivial — just record baseline once per (source, layer).
                        for target in collector.tasks:
                            lb = baseline[target]
                            if not np.isfinite(lb):
                                continue
                            rows.append(LandscapeRow(
                                batch_idx=batch_idx,
                                source_task=source,
                                target_task=target,
                                layer=layer_label,
                                t=0.0,
                                grad_norm=gn,
                                loss=lb,
                                delta=0.0,
                                rel_delta=0.0,
                            ))
                        continue

                    apply_virtual_step(update_params, grads,
                                       alpha=t * alpha, normalize=True)
                    with torch.no_grad():
                        fwd_after = fwd()
                        for target in collector.tasks:
                            tl_after = _sum_task_loss_any(fwd_after, target)
                            if tl_after is None:
                                continue
                            la = float(tl_after.item())
                            lb = baseline[target]
                            delta = la - lb
                            rel = delta / lb if abs(lb) > EPS else float("nan")
                            if not np.isfinite(la):
                                continue
                            rows.append(LandscapeRow(
                                batch_idx=batch_idx,
                                source_task=source,
                                target_task=target,
                                layer=layer_label,
                                t=float(t),
                                grad_norm=gn,
                                loss=la,
                                delta=delta,
                                rel_delta=rel,
                            ))
                    restore_params(update_params, snap)

    return rows


def _plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """One PNG per (source, target, layer) — line plot of L(t)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate over batches: mean loss per (source, target, layer, t).
    agg = (
        df.groupby(["source_task", "target_task", "layer", "t"], as_index=False)
        .agg(loss_mean=("loss", "mean"),
             loss_std=("loss", "std"),
             delta_mean=("delta", "mean"))
    )
    for (source, target, layer), sub in agg.groupby(["source_task", "target_task", "layer"]):
        sub = sub.sort_values("t")
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        ax.errorbar(sub["t"], sub["loss_mean"], yerr=sub["loss_std"].fillna(0.0),
                    fmt="o-", capsize=2, lw=1.2, ms=3.5)
        ax.axvline(0.0, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel(r"$t$ (multiplier of $\alpha$;  $\theta - t\alpha\hat{g}_{\rm src}$ — $t>0$ = descent)")
        ax.set_ylabel(f"L_{target}")
        ax.set_title(f"{source} → {target}  ({layer})", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fname = f"{source}_to_{target}_{layer}.png".replace("/", "_")
        fig.savefig(plot_dir / fname, dpi=120)
        plt.close(fig)


def run_m8(
    collector,
    dataloader,
    num_batches: int,
    alpha: float,
    t_grid: Sequence[float],
    out_dir: Path,
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = False,
    forward_seed: Optional[int] = None,
    reset_temporal_state: bool = False,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[LandscapeRow] = []
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break
        rows = line_probe_one_batch(
            collector, data, alpha=alpha, t_grid=t_grid,
            batch_idx=i,
            target_layers=target_layers,
            freeze_matching=freeze_matching,
            forward_seed=forward_seed,
            reset_temporal_state=reset_temporal_state,
        )
        all_rows.extend(rows)

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    df.to_csv(out_dir / "landscape_per_batch.csv", index=False)
    if not df.empty:
        agg = (
            df.groupby(["source_task", "target_task", "layer", "t"], as_index=False)
            .agg(loss_mean=("loss", "mean"),
                 loss_std=("loss", "std"),
                 delta_mean=("delta", "mean"),
                 grad_norm_mean=("grad_norm", "mean"))
        )
        agg.to_csv(out_dir / "landscape_aggregate.csv", index=False)
        _plot_curves(df, out_dir)
    return df
