"""M3 — One/Two-Step Probe.

Applies virtual updates on the FULL set of parameters reachable from a source
task's loss, then measures the target task's loss change. This is the fixed
version of the earlier shared-param-only probe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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
    """In-place `p <- p - alpha * g` (optionally normalized by ||g^full||)."""
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


# ----------------------------- probe runner -----------------------------

@dataclass
class ProbeRow:
    batch_idx: int
    source_task: str
    target_task: str
    steps: int
    variant: str
    baseline_loss: float
    stepped_loss: float
    delta: float
    rel_delta: float


def probe_one_batch(
    collector,            # GradientCollector instance
    data,                 # first-batch data
    data_next=None,       # optional second batch for 2-step
    alpha: float = 1e-3,
    steps_list: List[int] = (1, 2),
    variants: List[str] = ("raw", "normalized"),
    batch_idx: int = 0,
) -> List[ProbeRow]:
    """For each source task, apply k-step virtual update and record ΔL for all targets."""
    from .collector import compute_task_full_gradient
    from analyze_gradient_conflict import _sum_task_loss, TASK_GROUPS, match_loss_key  # type: ignore

    def _sum_task_loss_any(loss_dict, task_name):
        """Like _sum_task_loss but does NOT filter by requires_grad.
        Needed for post-virtual-step forwards under torch.no_grad()."""
        prefixes = TASK_GROUPS.get(task_name, [])
        total = None
        for key, val in loss_dict.items():
            if match_loss_key(key, prefixes) and isinstance(val, torch.Tensor):
                total = val if total is None else total + val
        return total

    rows: List[ProbeRow] = []
    params = collector.full_params

    # Baseline losses (forward in no_grad mode just for recording)
    with torch.no_grad():
        losses = collector.forward_losses(data)
        baseline: Dict[str, float] = {}
        for t in collector.tasks:
            tl = _sum_task_loss_any(losses, t)
            baseline[t] = float(tl.item()) if tl is not None else float("nan")

    for source in collector.tasks:
        for variant in variants:
            for steps in steps_list:
                snap = snapshot_params(params)
                # Step 1 — needs grad
                fwd = collector.forward_losses(data)
                tl = _sum_task_loss(fwd, source)  # requires_grad path
                if tl is None:
                    restore_params(params, snap)
                    continue
                g = compute_task_full_gradient(tl, params, retain_graph=False)
                apply_virtual_step(params, g, alpha=alpha, normalize=(variant == "normalized"))

                if steps >= 2:
                    if data_next is None:
                        # Reuse same batch for second step
                        data2 = data
                    else:
                        data2 = data_next
                    fwd2 = collector.forward_losses(data2)
                    tl2 = _sum_task_loss(fwd2, source)
                    if tl2 is None:
                        restore_params(params, snap)
                        continue
                    g2 = compute_task_full_gradient(tl2, params, retain_graph=False)
                    apply_virtual_step(params, g2, alpha=alpha, normalize=(variant == "normalized"))

                # Stepped losses — no_grad forward, so use _sum_task_loss_any
                with torch.no_grad():
                    fwd_after = collector.forward_losses(data)
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
                            baseline_loss=lb,
                            stepped_loss=la,
                            delta=delta,
                            rel_delta=rel,
                        ))
                restore_params(params, snap)

    return rows


def rows_to_dataframe(rows: List[ProbeRow]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows])


def aggregate_affinity_matrix(df: pd.DataFrame, steps: int, variant: str) -> pd.DataFrame:
    sub = df[(df["steps"] == steps) & (df["variant"] == variant)]
    mat = sub.pivot_table(
        index="source_task", columns="target_task", values="delta", aggfunc="mean"
    )
    return mat


def run_m3(
    collector,
    dataloader,
    num_batches: int,
    alpha: float,
    steps_list: List[int],
    variants: List[str],
    out_dir: Path,
) -> pd.DataFrame:
    """Execute M3 on up to num_batches. Returns combined per-row DataFrame."""
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
        )
        all_rows.extend(rows)
        prev_data = data

    df = rows_to_dataframe(all_rows)
    df.to_csv(out_dir / "probe_per_batch.csv", index=False)
    for s in steps_list:
        for v in variants:
            mat = aggregate_affinity_matrix(df, s, v)
            mat.to_csv(out_dir / f"probe_matrix_{s}step_{v}.csv")
    return df
