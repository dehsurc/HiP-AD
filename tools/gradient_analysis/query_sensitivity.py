"""HiP-AD planning-query sensitivity dumper.

Computes ``||dL_plan/dQ_task||`` for each task-specific query tensor exposed
by ``adapter.get_task_queries``. VAD adapter raises NotImplementedError; the
runner catches it and writes an empty CSV with a SKIP message.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch


EPS = 1e-8


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


def _maybe_freeze(adapter, on: bool):
    return adapter.freeze_stochastic_state() if on else contextlib.nullcontext()


def _resolve_task_queries(adapter, model, fwd):
    """Prefer fwd['task_queries'] (set by the model wrapper) over adapter call."""
    if isinstance(fwd, dict) and "task_queries" in fwd:
        return fwd["task_queries"]
    return adapter.get_task_queries(model=model, fwd_artifacts=fwd)


def _seeded_forward(collector, data, seed):
    with _seed_scope(seed):
        return collector.forward_losses(data)


def run_query_sensitivity(
    collector,
    dataloader,
    num_batches: int,
    out_dir: Path,
    capture_vectors: bool = False,
    capture_delta: bool = False,
    forward_seed: Optional[int] = 42,
    freeze_matching: bool = True,
    model_name: str = "",
    checkpoint: str = "",
    checkpoint_order: int = 0,
) -> pd.DataFrame:
    """Per batch: compute ||dL_plan/dQ_task|| for each task query type."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "qs.csv"

    rows = []
    sidecar = []
    try:
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
            # 매 batch마다 새 contextmanager 인스턴스 — contextmanager-decorated
            # 객체는 한 번만 __enter__ 가능하기 때문.
            with _maybe_freeze(collector.adapter, freeze_matching):
                fwd = (collector.forward_losses(data) if forward_seed is None
                       else _seeded_forward(collector, data, forward_seed))
                plan_loss = collector.adapter.split_losses(fwd, "plan")
                if plan_loss is None:
                    continue
                queries = _resolve_task_queries(collector.adapter, collector.model, fwd)
                names = list(queries.keys())
                tensors = [queries[t] for t in names]
                grads = torch.autograd.grad(plan_loss, tensors,
                                            retain_graph=False, allow_unused=True)
                for task, Q, gQ in zip(names, tensors, grads):
                    q_flat = Q.detach().reshape(-1, Q.shape[-1])
                    q_norms = q_flat.norm(dim=-1).cpu().numpy()
                    if gQ is None:
                        # Query is not in plan_loss's autograd graph (e.g. parallel
                        # branch). Record zero-norm rows so the notebook can still
                        # plot bar charts per task; this is a meaningful negative
                        # finding, not a missing-data hole.
                        g_norms = np.zeros_like(q_norms)
                    else:
                        g_flat = gQ.detach().reshape(-1, gQ.shape[-1])
                        g_norms = g_flat.norm(dim=-1).cpu().numpy()
                    for idx, (qn, gn) in enumerate(zip(q_norms, g_norms)):
                        rows.append({
                            "model": model_name,
                            "checkpoint": checkpoint,
                            "checkpoint_order": checkpoint_order,
                            "batch_idx": i,
                            "scene_token": "",
                            "layer": "_all",
                            "task_query_type": task,
                            "query_index": idx,
                            "query_norm": float(qn),
                            "grad_plan_wrt_query_norm": float(gn),
                        })
                    if capture_vectors and gQ is not None:
                        sidecar.append((i, task, g_flat.cpu().clone()))
    except NotImplementedError as e:
        print(f"[query_sensitivity] SKIP: {e}")

    # Always write a CSV with header even when SKIP empties `rows`, so the
    # downstream merge in run_gradient_analysis.main() doesn't trip EmptyDataError.
    _SCHEMA = ["model", "checkpoint", "checkpoint_order", "batch_idx",
               "scene_token", "layer", "task_query_type", "query_index",
               "query_norm", "grad_plan_wrt_query_norm"]
    df = pd.DataFrame(rows, columns=None if rows else _SCHEMA)
    df.to_csv(out_csv, index=False)
    if capture_vectors:
        torch.save(sidecar, out_dir / "qs_vectors.pt")
    return df
