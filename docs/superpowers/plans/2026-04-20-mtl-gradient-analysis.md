# MTL Gradient Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable gradient analysis pipeline for HiP-AD that separates conflict analysis (shared params) from one-step probe (full reachable params), retains batch-level distributions, adds binning/correlation/projection/gradnorm/asymmetry/dynamics, and produces publication-ready outputs.

**Architecture:** A new package `tools/gradient_analysis/` that reuses existing utilities from `analyze_gradient_conflict.py` (param grouping, per-task gradient computation, task-loss summation) and `pcgrad_optimizer_hook.py` (shared param identification). Gradients are collected once per checkpoint and cached as `.pt` files; all downstream modules read from the cache. A CLI entrypoint orchestrates across 4 checkpoints (1/3/6/18 ep) + optional per-sample supplementary run.

**Tech Stack:** PyTorch, MMCV (Config, build_detector, load_checkpoint, MMDataParallel), matplotlib (headless Agg backend), numpy, pandas for CSV output, pyyaml for config, pytest for sanity-check tests.

**Spec:** [docs/superpowers/specs/2026-04-20-mtl-gradient-analysis-design.md](../specs/2026-04-20-mtl-gradient-analysis-design.md)

---

## Task 0: Repo hygiene & package scaffolding

**Files:**
- Create: `tools/gradient_analysis/__init__.py`
- Create: `tools/gradient_analysis/README.md`
- Create: `configs/gradient_analysis.yaml`
- Create: `tests/gradient_analysis/__init__.py`
- Create: `tests/gradient_analysis/conftest.py`
- Create: `requirements_gradient_analysis.txt`

- [ ] **Step 1: Create package directories**

```bash
mkdir -p tools/gradient_analysis tests/gradient_analysis gradient_analysis_results
```

- [ ] **Step 2: Write `tools/gradient_analysis/__init__.py`**

```python
"""Gradient analysis pipeline for HiP-AD multi-task learning.

See docs/superpowers/specs/2026-04-20-mtl-gradient-analysis-design.md.
"""

__all__ = [
    "collector",
    "conflict",
    "probe",
    "correlation",
    "gradnorm",
    "dynamics",
    "asymmetry",
    "binning",
    "viz",
]
```

- [ ] **Step 3: Write `configs/gradient_analysis.yaml`**

```yaml
# MTL gradient analysis configuration. All quantities are user-adjustable.

tasks: [det, map, motion, ego, plan]

ckpt_root: data/ckpts/E2_E1_stage2_18ep
model_config: projects/configs/hipad_nusc_stage2.py  # matches nusc/pcgrad training config for E2_E1_stage2_18ep

checkpoints:
  1ep:  iter_2344.pth
  3ep:  iter_7032.pth
  6ep:  iter_14064.pth
  18ep: iter_42192.pth

primary:
  batch_size: 12
  num_batches: 100

supplementary:
  enabled: true
  batch_size: 1
  num_samples: 500
  checkpoint: 1ep

probe:
  alpha: 0.001
  steps: [1, 2]
  variants: [raw, normalized]

binning:
  cosine_bins: [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]

shared_param_groups:
  - backbone
  - neck
  - norm
  - ffn
  - gnn
  - inter_gnn
  - fc_before
  - fc_after

output_root: gradient_analysis_results

seed: 42
deterministic: true
fp16: true
device: cuda:0
```

- [ ] **Step 4: Write `requirements_gradient_analysis.txt`**

```
pyyaml>=6.0
pandas>=1.5
matplotlib>=3.6
numpy>=1.23
scipy>=1.10
pytest>=7.0
```

- [ ] **Step 5: Write `tests/gradient_analysis/__init__.py`** (empty file to make it a package)

```python
```

- [ ] **Step 6: Write `tests/gradient_analysis/conftest.py`**

```python
"""Shared fixtures for gradient analysis tests."""
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)
    yield


@pytest.fixture
def toy_params():
    """Two-group toy param set: group 'a' (2 params), group 'b' (1 param)."""
    from collections import OrderedDict
    groups = OrderedDict()
    groups["a"] = [torch.nn.Parameter(torch.randn(3, 2)), torch.nn.Parameter(torch.randn(4))]
    groups["b"] = [torch.nn.Parameter(torch.randn(5))]
    return groups
```

- [ ] **Step 7: Smoke-test the scaffolding**

```bash
cd /home/yongjae/e2e/HiP-AD
python -c "import yaml; yaml.safe_load(open('configs/gradient_analysis.yaml'))"
python -c "import tools.gradient_analysis as ga; print(ga.__all__)"
```
Expected: YAML parses, package imports list.

- [ ] **Step 8: Commit**

```bash
git add tools/gradient_analysis/ tests/gradient_analysis/ configs/gradient_analysis.yaml requirements_gradient_analysis.txt
git commit -m "anal: gradient_analysis package scaffolding + config"
```

---

## Task 1: Binning utility (pure-math, TDD)

**Files:**
- Create: `tools/gradient_analysis/binning.py`
- Create: `tests/gradient_analysis/test_binning.py`

- [ ] **Step 1: Write failing tests in `tests/gradient_analysis/test_binning.py`**

```python
"""Tests for binning utility."""
import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.binning import bin_by_edges, summarize_bins


def test_bin_by_edges_assigns_integers():
    x = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    edges = np.array([-1.0, -0.3, -0.1, 0.1, 0.3, 1.0])
    bins = bin_by_edges(x, edges)
    # Expect bin indices 0..4 for values falling in each of 5 bins
    assert bins.tolist() == [0, 1, 2, 3, 4]


def test_bin_by_edges_handles_boundary_values():
    x = np.array([-1.0, 1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    bins = bin_by_edges(x, edges)
    # Left edge inclusive, right edge inclusive for last bin
    assert bins[0] == 0
    assert bins[1] == 1


def test_summarize_bins_reports_count_and_mean():
    x = np.array([-0.5, -0.4, 0.2, 0.25, 0.8])
    y = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    edges = np.array([-1.0, 0.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert list(df["bin_idx"]) == [0, 1]
    assert df.loc[df["bin_idx"] == 0, "count"].iloc[0] == 2
    assert df.loc[df["bin_idx"] == 0, "delta_mean"].iloc[0] == pytest.approx(2.0)
    assert df.loc[df["bin_idx"] == 1, "count"].iloc[0] == 3
    assert df.loc[df["bin_idx"] == 1, "delta_mean"].iloc[0] == pytest.approx((10 + 20 + 100) / 3)


def test_summarize_bins_reports_helpful_ratio():
    x = np.array([-0.5, -0.4])
    y = np.array([-1.0, 2.0])  # one helpful (negative delta), one harmful
    edges = np.array([-1.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert df["helpful_ratio"].iloc[0] == pytest.approx(0.5)


def test_summarize_bins_emits_empty_bins_as_rows_with_zero_count():
    x = np.array([0.5])
    y = np.array([1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert len(df) == 2
    empty = df[df["bin_idx"] == 0]
    assert empty["count"].iloc[0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/gradient_analysis/test_binning.py -v
```
Expected: FAIL with ModuleNotFoundError on `tools.gradient_analysis.binning`.

- [ ] **Step 3: Implement `tools/gradient_analysis/binning.py`**

```python
"""Bin 1-D scalar data and summarize per-bin statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bin_by_edges(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Return 0-based bin index for each x, inclusive on both ends of the last bin.

    Values outside [edges[0], edges[-1]] are clipped to the nearest edge.
    """
    x = np.asarray(x, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must be a 1-D array with at least 2 elements")
    # np.digitize with right=False puts x == edge[i] in bin i; we want last-bin inclusion.
    idx = np.digitize(x, edges, right=False) - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx.astype(int)


def summarize_bins(
    x: np.ndarray,
    y: np.ndarray,
    edges: np.ndarray,
    stat_label: str = "y",
) -> pd.DataFrame:
    """For each bin defined by `edges`, compute count, mean/std/median of y,
    and helpful_ratio = fraction of y with y < 0.

    Rows are returned for every bin, including empty bins (count = 0).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    bins = bin_by_edges(x, edges)
    n_bins = len(edges) - 1

    rows = []
    for b in range(n_bins):
        mask = bins == b
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bin_idx": b,
                    "bin_lo": float(edges[b]),
                    "bin_hi": float(edges[b + 1]),
                    "count": 0,
                    f"{stat_label}_mean": float("nan"),
                    f"{stat_label}_std": float("nan"),
                    f"{stat_label}_median": float("nan"),
                    "helpful_ratio": float("nan"),
                }
            )
            continue
        sub = y[mask]
        rows.append(
            {
                "bin_idx": b,
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "count": n,
                f"{stat_label}_mean": float(sub.mean()),
                f"{stat_label}_std": float(sub.std(ddof=0)),
                f"{stat_label}_median": float(np.median(sub)),
                "helpful_ratio": float((sub < 0).mean()),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/gradient_analysis/test_binning.py -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/binning.py tests/gradient_analysis/test_binning.py
git commit -m "anal: binning utility for cosine/Δloss bucketing"
```

---

## Task 2: Plotting utilities

**Files:**
- Create: `tools/gradient_analysis/viz.py`

- [ ] **Step 1: Implement `tools/gradient_analysis/viz.py`**

```python
"""Shared matplotlib plotting helpers for gradient analysis (headless, Agg)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def scatter_with_bins(
    x: np.ndarray,
    y: np.ndarray,
    bin_summary: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    out_path: str | Path,
    y_stat_col: str,
) -> None:
    """Scatter of (x, y) overlaid with binned mean ± std bars."""
    out_path = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(x, y, alpha=0.25, s=8, label="per-batch")
    centers = (bin_summary["bin_lo"] + bin_summary["bin_hi"]) / 2
    ax.errorbar(
        centers,
        bin_summary[f"{y_stat_col}_mean"],
        yerr=bin_summary[f"{y_stat_col}_std"],
        fmt="o-",
        color="crimson",
        capsize=3,
        label="bin mean ± std",
    )
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    out_path: str | Path,
    cmap: str = "RdBu_r",
    center: Optional[float] = 0.0,
    fmt: str = "{:.3f}",
) -> None:
    """Annotated heatmap for square or rectangular matrices."""
    out_path = _ensure_dir(Path(out_path))
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(col_labels), 1.0 + 0.9 * len(row_labels)))
    vmax = float(np.nanmax(np.abs(matrix)))
    if center is not None:
        im = ax.imshow(matrix, cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def violin_by_pair(
    data: dict[str, np.ndarray],
    y_label: str,
    title: str,
    out_path: str | Path,
) -> None:
    """Violin plot: one violin per task pair."""
    out_path = _ensure_dir(Path(out_path))
    labels = list(data.keys())
    arrays = [data[k] for k in labels]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(labels)), 4.5))
    ax.violinplot(arrays, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def timeseries(
    rows: Iterable[dict],
    x_key: str,
    y_key: str,
    series_key: str,
    title: str,
    out_path: str | Path,
) -> None:
    """Generic time-series plot: one line per series."""
    out_path = _ensure_dir(Path(out_path))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, sub in df.groupby(series_key):
        sub = sub.sort_values(x_key)
        ax.plot(sub[x_key], sub[y_key], marker="o", label=str(label))
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "
import numpy as np, pandas as pd
from tools.gradient_analysis.viz import scatter_with_bins, heatmap
scatter_with_bins(np.random.randn(50), np.random.randn(50),
  pd.DataFrame([{'bin_lo':-1,'bin_hi':0,'delta_mean':0.1,'delta_std':0.2},
                {'bin_lo':0,'bin_hi':1,'delta_mean':-0.1,'delta_std':0.3}]),
  'cos','delta','toy','/tmp/t1.png','delta')
heatmap(np.array([[0.1,-0.2],[0.3,0.4]]), ['a','b'], ['x','y'], 'toy', '/tmp/t2.png')
print('ok')
"
```
Expected: `ok`, files created at `/tmp/t1.png`, `/tmp/t2.png`.

- [ ] **Step 3: Commit**

```bash
git add tools/gradient_analysis/viz.py
git commit -m "anal: shared matplotlib helpers (scatter+bins, heatmap, violin, timeseries)"
```

---

## Task 3: Gradient Collector (M1)

**Files:**
- Create: `tools/gradient_analysis/collector.py`
- Create: `tests/gradient_analysis/test_collector.py`

The collector wraps existing HiP-AD utilities. Key reuse:
- `get_shared_parameters_grouped` (from `projects/mmdet3d_plugin/core/hooks/pcgrad_optimizer_hook.py`)
- `compute_task_gradient_grouped` + `TASK_GROUPS` (from `tools/analyze_gradient_conflict.py`)

It adds:
- `g_A^full`: per-task gradients over **all** requires_grad params reachable from the task's loss (params with `None` grad are excluded).
- Cache `.pt` files keyed by `(checkpoint_tag, batch_idx, task)`.

- [ ] **Step 1: Write failing test `tests/gradient_analysis/test_collector.py`**

```python
"""Collector invariants. Uses a toy linear model to keep tests fast."""
from collections import OrderedDict

import torch
from torch import nn

from tools.gradient_analysis.collector import (
    compute_task_full_gradient,
    slice_shared_from_full,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head_a = nn.Linear(4, 2)
        self.head_b = nn.Linear(4, 2)

    def forward(self, x):
        h = self.shared(x)
        return {"a": self.head_a(h), "b": self.head_b(h)}


def _task_loss(out, task):
    return out[task].pow(2).sum()


def test_full_gradient_excludes_unreachable_params():
    torch.manual_seed(0)
    model = ToyModel()
    x = torch.randn(8, 4)
    out = model(x)
    loss_a = _task_loss(out, "a")
    # Build full-param list
    full_params = [p for p in model.parameters() if p.requires_grad]
    grads = compute_task_full_gradient(loss_a, full_params)
    # head_b params should have zero (or None-replaced) gradient
    for name, p in model.named_parameters():
        idx = full_params.index(p)
        if name.startswith("head_b"):
            assert torch.allclose(grads[idx], torch.zeros_like(grads[idx]))
        else:
            assert grads[idx].abs().sum() > 0


def test_shared_slice_matches_shared_only_computation():
    """V1 sanity: g^shared from full-param call must equal the gradient computed
    directly on shared params only."""
    torch.manual_seed(0)
    model = ToyModel()
    x = torch.randn(8, 4)

    shared_params = list(model.shared.parameters())
    full_params = [p for p in model.parameters() if p.requires_grad]

    # Route 1: full-param grad then slice
    out = model(x)
    loss_a = _task_loss(out, "a")
    full_grad = compute_task_full_gradient(loss_a, full_params, retain_graph=True)
    sliced = slice_shared_from_full(full_grad, full_params, shared_params)

    # Route 2: direct shared-only grad
    direct = torch.autograd.grad(loss_a, shared_params, retain_graph=False)
    direct_flat = torch.cat([g.detach().flatten() for g in direct])

    assert torch.allclose(sliced, direct_flat, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/gradient_analysis/test_collector.py -v
```
Expected: FAIL with import error.

- [ ] **Step 3: Implement `tools/gradient_analysis/collector.py`**

```python
"""M1 — Gradient Collector.

Collects per-task gradients at two scopes:
  - shared: gradients on the configured shared-parameter groups (for M2/M5 conflict/norm analysis)
  - full:   gradients on all requires_grad params reachable from a task's loss
            (for M3 probe virtual updates; unreachable params get zero)

Caches per-(checkpoint, batch, task) gradient dicts to `.pt` files so downstream
modules (M2/M3/M5) can read without re-running backward.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

# Reuse existing HiP-AD utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `import analyze_gradient_conflict` works
from analyze_gradient_conflict import (  # type: ignore
    TASK_GROUPS,
    _sum_task_loss,
    compute_task_gradient_grouped,
)
from projects.mmdet3d_plugin.core.hooks.pcgrad_optimizer_hook import (  # type: ignore
    get_shared_parameters_grouped,
)


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
        self.shared_param_groups, self.shared_param_ids = get_shared_parameters_grouped(
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
        """Run model forward in train mode and return the loss dict with grad."""
        self.model.train()
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
```

- [ ] **Step 4: Run collector tests**

```bash
pytest tests/gradient_analysis/test_collector.py -v
```
Expected: both tests PASS. (Toy model tests don't require HiP-AD imports; they hit only the pure helpers.)

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/collector.py tests/gradient_analysis/test_collector.py
git commit -m "anal: M1 gradient collector (shared+full scopes, V1 shared==slice sanity)"
```

---

## Task 4: Conflict Analysis (M2)

**Files:**
- Create: `tools/gradient_analysis/conflict.py`
- Create: `tests/gradient_analysis/test_conflict.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/gradient_analysis/test_conflict.py
import numpy as np
import torch

from tools.gradient_analysis.conflict import (
    cosine_similarity,
    projection_decomposition,
    analyze_pair_batches,
)


def test_cosine_identical_vectors_is_one():
    g = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(g, g) == 1.0


def test_cosine_opposite_vectors_is_negative_one():
    g = torch.tensor([1.0, 2.0, 3.0])
    assert cosine_similarity(g, -g) == -1.0


def test_cosine_zero_norm_returns_nan():
    g = torch.tensor([1.0, 2.0, 3.0])
    zero = torch.zeros(3)
    assert np.isnan(cosine_similarity(zero, g))


def test_projection_decomposition_cooperative_only_when_aligned():
    g_a = torch.tensor([1.0, 0.0])
    g_b = torch.tensor([1.0, 0.0])
    coop, conf = projection_decomposition(g_a, g_b)
    assert coop == 1.0
    assert conf == 0.0


def test_projection_decomposition_conflicting_only_when_opposed():
    g_a = torch.tensor([-1.0, 0.0])
    g_b = torch.tensor([1.0, 0.0])
    coop, conf = projection_decomposition(g_a, g_b)
    assert coop == 0.0
    assert conf == 1.0


def test_analyze_pair_batches_reports_expected_shape():
    # Two batches, single group 'g0'
    batches = [
        {"A": {"g0": torch.tensor([1.0, 0.0])}, "B": {"g0": torch.tensor([0.5, 0.1])}},
        {"A": {"g0": torch.tensor([1.0, 0.0])}, "B": {"g0": torch.tensor([-0.5, 0.1])}},
    ]
    df = analyze_pair_batches(batches, task_a="A", task_b="B", group_keys=["g0"])
    assert len(df) == 2
    assert set(df.columns) >= {"batch_idx", "group", "cos", "coop_mag", "conf_mag"}
    # First batch aligned → coop > 0, conf == 0
    row = df[df["batch_idx"] == 0].iloc[0]
    assert row["cos"] > 0
    assert row["conf_mag"] == 0
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/gradient_analysis/test_conflict.py -v
```
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `tools/gradient_analysis/conflict.py`**

```python
"""M2 — Conflict Analysis (cosine + projection decomposition)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch


EPS = 1e-8


def cosine_similarity(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """Return cosine similarity; NaN if either norm is below EPS."""
    n1 = g1.norm()
    n2 = g2.norm()
    if float(n1) < EPS or float(n2) < EPS:
        return float("nan")
    return float(torch.dot(g1, g2) / (n1 * n2))


def projection_decomposition(g_a: torch.Tensor, g_b: torch.Tensor) -> Tuple[float, float]:
    """Split ||proj_{g_b}(g_a)|| into cooperative (cos ≥ 0) and conflicting (cos < 0) magnitudes.

    Returns (cooperative_magnitude, conflicting_magnitude). Exactly one is nonzero.
    """
    n_b = g_b.norm()
    if float(n_b) < EPS:
        return 0.0, 0.0
    dot = float(torch.dot(g_a, g_b))
    mag = abs(dot) / float(n_b)
    if dot >= 0:
        return mag, 0.0
    return 0.0, mag


def analyze_pair_batches(
    batches: Iterable[Dict[str, Dict[str, torch.Tensor]]],
    task_a: str,
    task_b: str,
    group_keys: List[str],
) -> pd.DataFrame:
    """For each batch and each param group, compute cosine + projection decomposition
    between task_a and task_b gradients.

    `batches` is iterable of dicts: task -> {group_key -> flat tensor}.
    Returns a long-format DataFrame (one row per batch × group).
    """
    rows = []
    for i, b in enumerate(batches):
        ga_groups = b.get(task_a, {})
        gb_groups = b.get(task_b, {})
        for gk in group_keys:
            ga = ga_groups.get(gk)
            gb = gb_groups.get(gk)
            if ga is None or gb is None:
                continue
            cos = cosine_similarity(ga, gb)
            coop, conf = projection_decomposition(ga, gb)
            rows.append(
                {
                    "batch_idx": i,
                    "group": gk,
                    "cos": cos,
                    "coop_mag": coop,
                    "conf_mag": conf,
                    "norm_a": float(ga.norm()),
                    "norm_b": float(gb.norm()),
                }
            )
    return pd.DataFrame(rows)


def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(task_pair, group) across batches."""
    grouped = df.groupby("group").agg(
        mean_cos=("cos", "mean"),
        std_cos=("cos", "std"),
        median_cos=("cos", "median"),
        n=("cos", "count"),
        conflict_ratio=("cos", lambda s: float((s < 0).mean())),
        mean_coop_mag=("coop_mag", "mean"),
        mean_conf_mag=("conf_mag", "mean"),
    ).reset_index()
    return grouped


def run_m2(
    cached_batches: List[Dict],  # list of BatchGradients-like dicts with .shared
    tasks: List[str],
    group_keys: List[str],
    out_dir: Path,
) -> None:
    """Orchestrate M2 over all task pairs; write CSVs + plots per pair."""
    from .viz import violin_by_pair
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_pair_violin: Dict[str, np.ndarray] = {}

    for a_idx, a in enumerate(tasks):
        for b in tasks[a_idx + 1:]:
            batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached_batches]
            df = analyze_pair_batches(batches, a, b, group_keys)
            if df.empty:
                continue
            df.to_csv(out_dir / f"conflict_{a}_{b}_per_batch.csv", index=False)
            summary = summarize_pair(df)
            summary.insert(0, "task_a", a)
            summary.insert(1, "task_b", b)
            summary.to_csv(out_dir / f"conflict_{a}_{b}_summary.csv", index=False)
            per_pair_violin[f"{a}|{b}"] = df["cos"].to_numpy()

    if per_pair_violin:
        violin_by_pair(
            per_pair_violin,
            y_label="cosine similarity",
            title="Per-pair cosine across batches (all groups)",
            out_path=out_dir / "cosine_violin.png",
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/gradient_analysis/test_conflict.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/conflict.py tests/gradient_analysis/test_conflict.py
git commit -m "anal: M2 conflict analysis (cosine + projection decomposition)"
```

---

## Task 5: One/Two-Step Probe (M3)

**Files:**
- Create: `tools/gradient_analysis/probe.py`
- Create: `tests/gradient_analysis/test_probe.py`

- [ ] **Step 1: Write failing test (includes V2 reverse sanity)**

```python
# tests/gradient_analysis/test_probe.py
import torch
from torch import nn

from tools.gradient_analysis.probe import apply_virtual_step, restore_params


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(4, 4)
        self.head_a = nn.Linear(4, 2)

    def forward(self, x):
        return self.head_a(self.shared(x))


def _loss(model, x, target):
    return (model(x) - target).pow(2).sum()


def test_apply_virtual_step_then_restore_is_identity():
    torch.manual_seed(0)
    m = ToyModel()
    snapshot = {id(p): p.data.clone() for p in m.parameters()}
    grads = [torch.randn_like(p) for p in m.parameters()]
    apply_virtual_step(list(m.parameters()), grads, alpha=0.01, normalize=False)
    # After restore, params equal snapshot
    restore_params(list(m.parameters()), snapshot)
    for p in m.parameters():
        assert torch.allclose(p.data, snapshot[id(p)])


def test_v2_source_task_own_loss_decreases_after_full_param_step():
    """Sanity: stepping along -g_A on all params reachable from L_A must decrease L_A
    for small enough alpha. This is the assertion that catches the 'shared-only' bug."""
    torch.manual_seed(0)
    m = ToyModel()
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    before = _loss(m, x, target).item()
    params = list(m.parameters())
    grads = list(torch.autograd.grad(_loss(m, x, target), params))
    snapshot = {id(p): p.data.clone() for p in params}
    apply_virtual_step(params, grads, alpha=1e-3, normalize=False)
    after = _loss(m, x, target).item()
    restore_params(params, snapshot)
    assert after < before, f"expected loss decrease but got before={before}, after={after}"
```

- [ ] **Step 2: Run test**

```bash
pytest tests/gradient_analysis/test_probe.py -v
```
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `tools/gradient_analysis/probe.py`**

```python
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
    """For each source task, apply k-step virtual update and record ΔL for all targets.

    Process:
      1. Forward with current θ → baseline losses for all targets.
      2. For each source task A:
         a. Snapshot params.
         b. Compute g_A^full on current params (using collector.compute_task_full_gradient
            via collector.collect_batch, but we reuse collector.forward_losses + autograd.grad).
         c. Apply virtual step.
         d. If steps==2: recompute g_A^full on data_next and step again.
         e. Forward again → stepped losses for all targets.
         f. Restore params.
    """
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

    # Baseline losses (forward in grad mode so we can use same path uniformly;
    # baseline here is just for recording — graph is discarded immediately).
    with torch.no_grad():
        losses = collector.forward_losses(data)
        baseline: Dict[str, float] = {}
        for t in collector.tasks:
            tl = _sum_task_loss_any(losses, t)
            baseline[t] = float(tl.item()) if tl is not None else float("nan")
    # Also cache loss dict for gradient computation below
    # We need fresh autograd graphs for each source task; so we recompute forward each source.

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
```

- [ ] **Step 4: Run probe tests**

```bash
pytest tests/gradient_analysis/test_probe.py -v
```
Expected: both PASS. The V2 test (`test_v2_source_task_own_loss_decreases_after_full_param_step`) is the critical sanity.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/probe.py tests/gradient_analysis/test_probe.py
git commit -m "anal: M3 one/two-step probe (full-param virtual update + V2 reverse test)"
```

---

## Task 6: Correlation & Binning Analysis (M4)

**Files:**
- Create: `tools/gradient_analysis/correlation.py`
- Create: `tests/gradient_analysis/test_correlation.py`

- [ ] **Step 1: Write tests**

```python
# tests/gradient_analysis/test_correlation.py
import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.correlation import join_cos_delta, pair_correlations


def test_join_cos_delta_matches_on_batch_idx():
    cos_df = pd.DataFrame([
        {"batch_idx": 0, "group": "g0", "cos": 0.5, "task_a": "A", "task_b": "B"},
        {"batch_idx": 1, "group": "g0", "cos": -0.3, "task_a": "A", "task_b": "B"},
    ])
    probe_df = pd.DataFrame([
        {"batch_idx": 0, "source_task": "A", "target_task": "B", "steps": 1, "variant": "raw", "delta": -0.1},
        {"batch_idx": 1, "source_task": "A", "target_task": "B", "steps": 1, "variant": "raw", "delta": 0.2},
    ])
    joined = join_cos_delta(cos_df, probe_df, steps=1, variant="raw")
    assert len(joined) == 2
    row0 = joined[joined["batch_idx"] == 0].iloc[0]
    assert row0["cos"] == 0.5
    assert row0["delta"] == -0.1


def test_pair_correlations_computes_pearson_and_spearman():
    df = pd.DataFrame({"cos": [1.0, 2.0, 3.0, 4.0, 5.0], "delta": [-1, -2, -3, -4, -5]})
    pearson, spearman = pair_correlations(df)
    assert pearson == pytest.approx(-1.0)
    assert spearman == pytest.approx(-1.0)
```

- [ ] **Step 2: Run test**

```bash
pytest tests/gradient_analysis/test_correlation.py -v
```
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `tools/gradient_analysis/correlation.py`**

```python
"""M4 — Correlation & Binning.

Joins batch-level cosine similarity (from M2) with batch-level Δloss (from M3),
computes Pearson/Spearman correlation, emits binned summaries, and plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .binning import summarize_bins
from .viz import scatter_with_bins


def join_cos_delta(
    cos_df: pd.DataFrame,
    probe_df: pd.DataFrame,
    steps: int,
    variant: str,
) -> pd.DataFrame:
    """Inner-join on (batch_idx, task_a=source, task_b=target) for given steps/variant."""
    probe_sub = probe_df[(probe_df["steps"] == steps) & (probe_df["variant"] == variant)]
    merged = cos_df.merge(
        probe_sub[["batch_idx", "source_task", "target_task", "delta", "rel_delta"]],
        left_on=["batch_idx", "task_a", "task_b"],
        right_on=["batch_idx", "source_task", "target_task"],
        how="inner",
    )
    return merged


def pair_correlations(df: pd.DataFrame) -> Tuple[float, float]:
    """Return (pearson, spearman) between cos and delta."""
    from scipy.stats import pearsonr, spearmanr

    if len(df) < 3:
        return float("nan"), float("nan")
    p, _ = pearsonr(df["cos"], df["delta"])
    s, _ = spearmanr(df["cos"], df["delta"])
    return float(p), float(s)


def run_m4(
    cos_dfs: dict[Tuple[str, str], pd.DataFrame],   # (a,b) -> cos per-batch df (must have batch_idx, group, cos)
    probe_df: pd.DataFrame,
    steps_list: List[int],
    variants: List[str],
    cosine_bins: List[float],
    out_dir: Path,
) -> pd.DataFrame:
    """Run M4 across all pairs/variants/steps. Emit CSVs + scatter plots.

    Returns a long-format correlation table.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for (a, b), cos_df in cos_dfs.items():
        cos_df = cos_df.copy()
        cos_df["task_a"] = a
        cos_df["task_b"] = b
        # Aggregate cos across groups to one scalar per batch (mean of groups).
        per_batch = cos_df.groupby("batch_idx", as_index=False)["cos"].mean()
        per_batch["task_a"] = a
        per_batch["task_b"] = b
        for s in steps_list:
            for v in variants:
                joined = join_cos_delta(per_batch, probe_df, steps=s, variant=v)
                if joined.empty:
                    continue
                pearson, spearman = pair_correlations(joined)
                bin_df = summarize_bins(
                    joined["cos"].to_numpy(),
                    joined["delta"].to_numpy(),
                    np.asarray(cosine_bins),
                    stat_label="delta",
                )
                bin_df.insert(0, "variant", v)
                bin_df.insert(0, "steps", s)
                bin_df.insert(0, "task_b", b)
                bin_df.insert(0, "task_a", a)
                bin_df.to_csv(out_dir / f"binned_{a}_{b}_{s}step_{v}.csv", index=False)
                scatter_with_bins(
                    joined["cos"].to_numpy(),
                    joined["delta"].to_numpy(),
                    bin_df,
                    x_label=f"cos(g_{a}, g_{b})",
                    y_label=f"ΔL_{b}",
                    title=f"{a}→{b} ({s}-step, {v})  Pearson={pearson:.2f}",
                    out_path=out_dir / f"scatter_{a}_{b}_{s}step_{v}.png",
                    y_stat_col="delta",
                )
                rows.append({
                    "task_a": a, "task_b": b, "steps": s, "variant": v,
                    "pearson": pearson, "spearman": spearman, "n": len(joined),
                })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "correlation_table.csv", index=False)
    return table
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/gradient_analysis/test_correlation.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/correlation.py tests/gradient_analysis/test_correlation.py
git commit -m "anal: M4 correlation + binned summaries (Pearson/Spearman + scatter)"
```

---

## Task 7: GradNorm Analysis (M5)

**Files:**
- Create: `tools/gradient_analysis/gradnorm.py`
- Create: `tests/gradient_analysis/test_gradnorm.py`

- [ ] **Step 1: Write tests**

```python
# tests/gradient_analysis/test_gradnorm.py
import numpy as np
import pandas as pd
import pytest
import torch

from tools.gradient_analysis.gradnorm import (
    per_task_norms_from_cached,
    antisymmetric_frobenius,
)


def test_per_task_norms_from_cached():
    cached = [
        {
            "batch_idx": 0,
            "shared": {
                "A": {"g0": torch.tensor([3.0, 4.0])},  # norm = 5
                "B": {"g0": torch.tensor([0.0, 2.0])},  # norm = 2
            },
        },
    ]
    df = per_task_norms_from_cached(cached, tasks=["A", "B"], groups=["g0"])
    row_a = df[(df["task"] == "A") & (df["group"] == "g0")].iloc[0]
    assert row_a["norm"] == pytest.approx(5.0)


def test_antisymmetric_frobenius_zero_for_symmetric():
    M = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert antisymmetric_frobenius(M) == pytest.approx(0.0)


def test_antisymmetric_frobenius_positive_for_asymmetric():
    M = np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert antisymmetric_frobenius(M) > 0.0
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/gradient_analysis/test_gradnorm.py -v
```
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `tools/gradient_analysis/gradnorm.py`**

```python
"""M5 — GradNorm Analysis.

Per-task gradient norm distributions, norm ratio matrix, and a raw-vs-normalized
probe comparison using M3 outputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .viz import heatmap


def per_task_norms_from_cached(cached, tasks: List[str], groups: List[str]) -> pd.DataFrame:
    rows = []
    for cb in cached:
        for t in tasks:
            tg = cb["shared"].get(t, {})
            for g in groups:
                v = tg.get(g)
                if v is None:
                    continue
                rows.append({
                    "batch_idx": cb["batch_idx"],
                    "task": t,
                    "group": g,
                    "norm": float(v.norm()) if hasattr(v, "norm") else float(np.linalg.norm(v)),
                })
    return pd.DataFrame(rows)


def antisymmetric_frobenius(M: np.ndarray) -> float:
    """Frobenius norm of (M - M^T) / 2."""
    A = (M - M.T) / 2.0
    return float(np.sqrt((A * A).sum()))


def run_m5(
    cached,
    tasks: List[str],
    groups: List[str],
    probe_df: pd.DataFrame,
    steps_list: List[int],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norms = per_task_norms_from_cached(cached, tasks, groups)
    norms.to_csv(out_dir / "per_task_norm.csv", index=False)

    # Mean norm ratio matrix (tasks x tasks)
    means = norms.groupby("task")["norm"].mean().reindex(tasks)
    ratio = np.outer(means.values, 1.0 / means.values)
    heatmap(
        ratio,
        row_labels=tasks,
        col_labels=tasks,
        title="Task gradient norm ratio (row / col)",
        out_path=out_dir / "norm_ratio_heatmap.png",
        cmap="viridis",
        center=None,
        fmt="{:.2f}",
    )

    # Raw-vs-normalized symmetry comparison (uses M3 probe df)
    comp_rows = []
    for s in steps_list:
        for variant in ("raw", "normalized"):
            sub = probe_df[(probe_df["steps"] == s) & (probe_df["variant"] == variant)]
            if sub.empty:
                continue
            mat = sub.pivot_table(index="source_task", columns="target_task",
                                  values="delta", aggfunc="mean").reindex(index=tasks, columns=tasks)
            fro = antisymmetric_frobenius(mat.values)
            comp_rows.append({"steps": s, "variant": variant, "antisymmetric_fro": fro})
            heatmap(
                mat.values,
                row_labels=tasks, col_labels=tasks,
                title=f"Δloss matrix ({s}-step, {variant})",
                out_path=out_dir / f"affinity_matrix_{s}step_{variant}.png",
            )
    pd.DataFrame(comp_rows).to_csv(out_dir / "raw_vs_norm_symmetry.csv", index=False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/gradient_analysis/test_gradnorm.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/gradnorm.py tests/gradient_analysis/test_gradnorm.py
git commit -m "anal: M5 gradnorm (per-task norm dist + raw vs normalized symmetry)"
```

---

## Task 8: Task Affinity Asymmetry (M7)

**Files:**
- Create: `tools/gradient_analysis/asymmetry.py`
- Create: `tests/gradient_analysis/test_asymmetry.py`

- [ ] **Step 1: Write tests**

```python
# tests/gradient_analysis/test_asymmetry.py
import numpy as np
import pytest

from tools.gradient_analysis.asymmetry import decompose_matrix, top_asymmetric_pairs


def test_decompose_matrix_satisfies_identity():
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    S, A = decompose_matrix(M)
    assert np.allclose(S + A, M)
    assert np.allclose(S, S.T)
    assert np.allclose(A, -A.T)


def test_top_asymmetric_pairs_ranked_by_absolute_antisym():
    M = np.array([[0.0, 1.0, 0.1], [-1.0, 0.0, 0.2], [-0.1, -0.2, 0.0]])
    labels = ["a", "b", "c"]
    ranked = top_asymmetric_pairs(M, labels, k=2)
    assert ranked[0]["pair"] == ("a", "b")
```

- [ ] **Step 2: Run test**

```bash
pytest tests/gradient_analysis/test_asymmetry.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement `tools/gradient_analysis/asymmetry.py`**

```python
"""M7 — Task Affinity Asymmetry."""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .viz import heatmap


def decompose_matrix(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (symmetric, antisymmetric) parts of M."""
    S = (M + M.T) / 2.0
    A = (M - M.T) / 2.0
    return S, A


def top_asymmetric_pairs(M: np.ndarray, labels: Sequence[str], k: int = 5):
    _, A = decompose_matrix(M)
    candidates = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            candidates.append({
                "pair": (labels[i], labels[j]),
                "antisym": float(A[i, j]),
                "abs_antisym": float(abs(A[i, j])),
            })
    candidates.sort(key=lambda d: d["abs_antisym"], reverse=True)
    return candidates[:k]


def run_m7(
    probe_df: pd.DataFrame,
    tasks: List[str],
    steps: int,
    variant: str,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = probe_df[(probe_df["steps"] == steps) & (probe_df["variant"] == variant)]
    mat = sub.pivot_table(index="source_task", columns="target_task",
                          values="delta", aggfunc="mean").reindex(index=tasks, columns=tasks)
    M = mat.values
    S, A = decompose_matrix(M)

    heatmap(M, tasks, tasks, f"Original Δloss ({steps}-step {variant})",
            out_path=out_dir / "original.png")
    heatmap(S, tasks, tasks, "Symmetric part", out_path=out_dir / "symmetric.png")
    heatmap(A, tasks, tasks, "Antisymmetric part", out_path=out_dir / "antisymmetric.png")

    top = top_asymmetric_pairs(M, tasks, k=len(tasks) * (len(tasks) - 1) // 2)
    pd.DataFrame(top).to_csv(out_dir / "top_asymmetric_pairs.csv", index=False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/gradient_analysis/test_asymmetry.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/asymmetry.py tests/gradient_analysis/test_asymmetry.py
git commit -m "anal: M7 task affinity symmetric/antisymmetric decomposition"
```

---

## Task 9: Training Dynamics (M6)

**Files:**
- Create: `tools/gradient_analysis/dynamics.py`

This module aggregates CSVs from per-checkpoint runs (M2/M3/M5) into time series plots. It's post-hoc; no gradient computation of its own.

- [ ] **Step 1: Implement `tools/gradient_analysis/dynamics.py`**

```python
"""M6 — Training Dynamics.

Reads per-checkpoint CSVs (conflict summaries, probe matrices, gradnorm tables)
and produces cross-checkpoint time-series plots.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .viz import timeseries


def _epoch_from_tag(tag: str) -> float:
    # "1ep" -> 1.0
    return float(tag.rstrip("epoch").rstrip("ep"))


def run_m6(
    per_ckpt_dirs: Dict[str, Path],   # tag -> ckpt_dir like gradient_analysis_results/ckpt_1ep
    tasks: List[str],
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cosine dynamics (mean across groups per pair)
    cos_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        conflict_dir = Path(d) / "conflict"
        for f in conflict_dir.glob("conflict_*_summary.csv"):
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                cos_rows.append({
                    "epoch": ep,
                    "pair": f"{row['task_a']}|{row['task_b']}",
                    "group": row["group"],
                    "mean_cos": row["mean_cos"],
                    "conflict_ratio": row["conflict_ratio"],
                })
    cos_df = pd.DataFrame(cos_rows)
    if not cos_df.empty:
        cos_df.to_csv(out_dir / "cos_dynamics.csv", index=False)
        # Aggregate groups: mean across groups per pair
        agg = cos_df.groupby(["epoch", "pair"], as_index=False)[["mean_cos", "conflict_ratio"]].mean()
        timeseries(
            agg.to_dict("records"),
            x_key="epoch", y_key="mean_cos", series_key="pair",
            title="Mean cosine similarity over epochs",
            out_path=out_dir / "timeseries_cosine.png",
        )
        timeseries(
            agg.to_dict("records"),
            x_key="epoch", y_key="conflict_ratio", series_key="pair",
            title="Conflict ratio over epochs",
            out_path=out_dir / "timeseries_conflict_ratio.png",
        )

    # Helpful ratio from probe
    help_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        probe_csv = Path(d) / "probe" / "probe_per_batch.csv"
        if not probe_csv.exists():
            continue
        pdf = pd.read_csv(probe_csv)
        pdf = pdf[(pdf["steps"] == 1) & (pdf["variant"] == "raw")]
        agg = pdf.groupby(["source_task", "target_task"]).agg(
            helpful_ratio=("delta", lambda s: float((s < 0).mean()))
        ).reset_index()
        for _, r in agg.iterrows():
            if r["source_task"] == r["target_task"]:
                continue
            help_rows.append({
                "epoch": ep,
                "pair": f"{r['source_task']}→{r['target_task']}",
                "helpful_ratio": r["helpful_ratio"],
            })
    help_df = pd.DataFrame(help_rows)
    if not help_df.empty:
        help_df.to_csv(out_dir / "helpful_dynamics.csv", index=False)
        timeseries(
            help_df.to_dict("records"),
            x_key="epoch", y_key="helpful_ratio", series_key="pair",
            title="Helpful ratio over epochs (1-step raw)",
            out_path=out_dir / "timeseries_helpful.png",
        )

    # Norm ratio dynamics
    norm_rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = _epoch_from_tag(tag)
        nf = Path(d) / "gradnorm" / "per_task_norm.csv"
        if not nf.exists():
            continue
        ndf = pd.read_csv(nf)
        means = ndf.groupby("task")["norm"].mean()
        for t in tasks:
            if t not in means.index:
                continue
            norm_rows.append({"epoch": ep, "task": t, "mean_norm": float(means[t])})
    norm_df = pd.DataFrame(norm_rows)
    if not norm_df.empty:
        norm_df.to_csv(out_dir / "norm_dynamics.csv", index=False)
        timeseries(
            norm_df.to_dict("records"),
            x_key="epoch", y_key="mean_norm", series_key="task",
            title="Per-task mean gradient norm over epochs",
            out_path=out_dir / "timeseries_norm.png",
        )
```

- [ ] **Step 2: Smoke-test (no per-ckpt data available yet, just import)**

```bash
python -c "from tools.gradient_analysis.dynamics import run_m6; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add tools/gradient_analysis/dynamics.py
git commit -m "anal: M6 training dynamics aggregator (cosine/helpful/norm time series)"
```

---

## Task 10: CLI entrypoint

**Files:**
- Create: `tools/run_gradient_analysis.py`

- [ ] **Step 1: Implement CLI**

```python
#!/usr/bin/env python
"""Orchestrate the gradient analysis pipeline across checkpoints and modules.

Usage:
    python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --all
    python tools/run_gradient_analysis.py --config ... --modules M2,M3,M4
    python tools/run_gradient_analysis.py --config ... --checkpoints 1ep,18ep
    python tools/run_gradient_analysis.py --config ... --no-supplementary
    python tools/run_gradient_analysis.py --config ... --smoke   # 3 batches only
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml
from mmcv import Config  # type: ignore
from mmcv.parallel import MMDataParallel  # type: ignore
from mmcv.runner import load_checkpoint, wrap_fp16_model  # type: ignore

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so analyze_gradient_conflict is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmdet3d.models import build_detector  # type: ignore

from tools.gradient_analysis.collector import GradientCollector, build_dataloader
from tools.gradient_analysis.conflict import run_m2, analyze_pair_batches
from tools.gradient_analysis.probe import run_m3
from tools.gradient_analysis.correlation import run_m4
from tools.gradient_analysis.gradnorm import run_m5
from tools.gradient_analysis.asymmetry import run_m7
from tools.gradient_analysis.dynamics import run_m6


def set_seeds(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_model(cfg, ckpt_path: Path, device: str, fp16: bool):
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    model.init_weights()
    if fp16:
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
    load_checkpoint(model, str(ckpt_path), map_location="cpu")
    device_id = int(device.split(":")[1]) if ":" in device else 0
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[device_id])
    return model


def run_primary_for_checkpoint(
    cfg_ana, ckpt_tag: str, ckpt_path: Path, out_dir: Path, modules: List[str], smoke: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = Config.fromfile(cfg_ana["model_config"])
    model = load_model(model_cfg, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = GradientCollector(
        model=model,
        tasks=cfg_ana["tasks"],
        shared_layer_names=cfg_ana["shared_param_groups"],
        device=cfg_ana["device"],
    )

    num_batches = 3 if smoke else cfg_ana["primary"]["num_batches"]
    dataloader = build_dataloader(
        model_cfg,
        batch_size=cfg_ana["primary"]["batch_size"],
        shuffle=True,
        seed=cfg_ana["seed"],
    )

    # Collect gradients (needed for M2 + M5)
    cached = []
    for i, data in enumerate(dataloader):
        if i >= num_batches:
            break
        bg, _ = collector.collect_batch(i, data)
        cached.append({
            "batch_idx": bg.batch_idx,
            "shared": bg.shared,
            "full_norm": bg.full_norm,
            "shared_norm": bg.shared_norm,
            "loss_values": bg.loss_values,
        })

    groups = list(collector.shared_param_groups.keys())

    # M2 conflict
    if "M2" in modules:
        run_m2(cached, cfg_ana["tasks"], groups, out_dir / "conflict")

    # M3 probe — rebuild a fresh dataloader (iterator state reset)
    probe_df = None
    if "M3" in modules:
        dl2 = build_dataloader(model_cfg, cfg_ana["primary"]["batch_size"], True, cfg_ana["seed"])
        probe_df = run_m3(
            collector, dl2, num_batches,
            alpha=cfg_ana["probe"]["alpha"],
            steps_list=cfg_ana["probe"]["steps"],
            variants=cfg_ana["probe"]["variants"],
            out_dir=out_dir / "probe",
        )

    # M4 correlation (needs M2 + M3)
    if "M4" in modules and probe_df is not None:
        cos_dfs = {}
        for a_idx, a in enumerate(cfg_ana["tasks"]):
            for b in cfg_ana["tasks"][a_idx + 1:]:
                batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached]
                df = analyze_pair_batches(batches, a, b, groups)
                if not df.empty:
                    cos_dfs[(a, b)] = df
        if cos_dfs:
            run_m4(
                cos_dfs, probe_df,
                steps_list=cfg_ana["probe"]["steps"],
                variants=cfg_ana["probe"]["variants"],
                cosine_bins=cfg_ana["binning"]["cosine_bins"],
                out_dir=out_dir / "correlation",
            )

    # M5 gradnorm
    if "M5" in modules and probe_df is not None:
        run_m5(cached, cfg_ana["tasks"], groups, probe_df,
               steps_list=cfg_ana["probe"]["steps"], out_dir=out_dir / "gradnorm")

    # M7 asymmetry (depends on M3)
    if "M7" in modules and probe_df is not None:
        run_m7(probe_df, cfg_ana["tasks"],
               steps=1, variant="raw", out_dir=out_dir / "asymmetry")


def run_supplementary(cfg_ana, out_dir: Path) -> None:
    sup = cfg_ana["supplementary"]
    if not sup["enabled"]:
        return
    ckpt_tag = sup["checkpoint"]
    ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"][ckpt_tag]
    model_cfg = Config.fromfile(cfg_ana["model_config"])
    model = load_model(model_cfg, ckpt_path, cfg_ana["device"], cfg_ana.get("fp16", True))
    collector = GradientCollector(
        model=model,
        tasks=cfg_ana["tasks"],
        shared_layer_names=cfg_ana["shared_param_groups"],
        device=cfg_ana["device"],
    )
    dataloader = build_dataloader(model_cfg, sup["batch_size"], True, cfg_ana["seed"])
    probe_df = run_m3(
        collector, dataloader, sup["num_samples"],
        alpha=cfg_ana["probe"]["alpha"],
        steps_list=cfg_ana["probe"]["steps"],
        variants=cfg_ana["probe"]["variants"],
        out_dir=out_dir,
    )
    # Collect shared grads on same batches for cos joining
    dl2 = build_dataloader(model_cfg, sup["batch_size"], True, cfg_ana["seed"])
    cached = []
    for i, data in enumerate(dl2):
        if i >= sup["num_samples"]:
            break
        bg, _ = collector.collect_batch(i, data)
        cached.append({"batch_idx": bg.batch_idx, "shared": bg.shared})
    groups = list(collector.shared_param_groups.keys())
    cos_dfs = {}
    for a_idx, a in enumerate(cfg_ana["tasks"]):
        for b in cfg_ana["tasks"][a_idx + 1:]:
            batches = [{a: cb["shared"].get(a, {}), b: cb["shared"].get(b, {})} for cb in cached]
            df = analyze_pair_batches(batches, a, b, groups)
            if not df.empty:
                cos_dfs[(a, b)] = df
    run_m4(cos_dfs, probe_df,
           steps_list=cfg_ana["probe"]["steps"],
           variants=cfg_ana["probe"]["variants"],
           cosine_bins=cfg_ana["binning"]["cosine_bins"],
           out_dir=out_dir / "correlation")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--all", action="store_true")
    p.add_argument("--modules", default="M2,M3,M4,M5,M6,M7",
                   help="Comma-separated module codes")
    p.add_argument("--checkpoints", default=None,
                   help="Comma-separated tags (default: all in config)")
    p.add_argument("--no-supplementary", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Use only 3 batches per checkpoint (quick end-to-end test)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config) as f:
        cfg_ana = yaml.safe_load(f)

    set_seeds(cfg_ana["seed"], cfg_ana.get("deterministic", True))
    if cfg_ana.get("deterministic"):
        torch.use_deterministic_algorithms(True, warn_only=True)

    modules = args.modules.split(",")
    out_root = Path(cfg_ana["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)

    tags = list(cfg_ana["checkpoints"].keys())
    if args.checkpoints:
        tags = args.checkpoints.split(",")

    per_ckpt_dirs: Dict[str, Path] = {}
    for tag in tags:
        ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"][tag]
        out_dir = out_root / f"ckpt_{tag}"
        per_ckpt_dirs[tag] = out_dir
        print(f"[{tag}] running modules {modules} → {out_dir}")
        run_primary_for_checkpoint(cfg_ana, tag, ckpt_path, out_dir, modules, smoke=args.smoke)

    if "M6" in modules and len(per_ckpt_dirs) >= 2:
        print(f"[M6] aggregating dynamics across {list(per_ckpt_dirs.keys())}")
        run_m6(per_ckpt_dirs, cfg_ana["tasks"], out_root / "dynamics")

    if not args.no_supplementary and not args.smoke:
        sup_dir = out_root / "supplementary"
        print(f"[supplementary] batch_size=1 probe on {cfg_ana['supplementary']['checkpoint']}")
        run_supplementary(cfg_ana, sup_dir)

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Make executable and import-check**

```bash
chmod +x tools/run_gradient_analysis.py
python -c "from tools.run_gradient_analysis import main, parse_args; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add tools/run_gradient_analysis.py
git commit -m "anal: CLI entrypoint orchestrating M2-M7 across checkpoints"
```

---

## Task 11: Smoke test on real model (end-to-end, 1 checkpoint, 3 batches)

**Files:**
- Modify: (none — this is a test run)
- Create: `gradient_analysis_results/smoke/` (output of the smoke run)

- [ ] **Step 1: Run smoke test**

```bash
cd /home/yongjae/e2e/HiP-AD
python tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --smoke \
    --no-supplementary
```
Expected: completes in ~5-15 minutes; outputs in `gradient_analysis_results/ckpt_1ep/` with at least `conflict/`, `probe/`, `correlation/`, `gradnorm/`, `asymmetry/` subdirectories.

- [ ] **Step 2: Inspect outputs**

```bash
find gradient_analysis_results/ckpt_1ep -type f | head -40
```
Expected: CSVs and PNGs in each module's subdir.

- [ ] **Step 3: Run V5 sanity check (aligned vs conflicting mean Δloss)**

```bash
python -c "
import pandas as pd
from pathlib import Path
root = Path('gradient_analysis_results/ckpt_1ep/correlation')
fails = []
for f in root.glob('binned_*_1step_raw.csv'):
    df = pd.read_csv(f)
    aligned = df[(df['bin_lo'] >= 0.3) & (df['count'] > 0)]
    conflict = df[(df['bin_hi'] <= -0.3) & (df['count'] > 0)]
    if aligned.empty or conflict.empty:
        continue
    a_mean = aligned['delta_mean'].mean()
    c_mean = conflict['delta_mean'].mean()
    if not (a_mean <= c_mean):
        fails.append((f.name, a_mean, c_mean))
print('V5 violations:', fails if fails else 'none (or insufficient data)')
"
```
Expected: no V5 violations (aligned bins have equal-or-lower Δloss than conflicting bins). If violations appear with many samples, the hypothesis or implementation needs investigation. At smoke-test scale (3 batches), may simply be under-sampled — note but don't block.

- [ ] **Step 4: Commit smoke-run output (optional, to preserve evidence)**

```bash
# Only if user wants outputs tracked — otherwise add gradient_analysis_results to .gitignore
echo "gradient_analysis_results/" >> .gitignore
git add .gitignore
git commit -m "anal: gitignore gradient_analysis_results"
```

---

## Task 12: Full primary run (4 checkpoints, 100 batches)

This is a long-running job (~1–1.5 days). Run in background, monitor.

- [ ] **Step 1: Launch full primary run**

```bash
nohup python tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --no-supplementary \
    > gradient_analysis_results/primary.log 2>&1 &
echo $! > gradient_analysis_results/primary.pid
```

- [ ] **Step 2: Monitor progress**

```bash
tail -f gradient_analysis_results/primary.log
```
Expected: per-checkpoint progress prints, no NaN/Inf warnings flood.

- [ ] **Step 3: Verify outputs after completion**

```bash
ls gradient_analysis_results/
# Expected: ckpt_1ep/ ckpt_3ep/ ckpt_6ep/ ckpt_18ep/ dynamics/
```

- [ ] **Step 4: Run V5 batch-sanity across all checkpoints**

```bash
python -c "
import pandas as pd
from pathlib import Path
for tag in ('1ep','3ep','6ep','18ep'):
    print(f'=== ckpt_{tag} ===')
    root = Path(f'gradient_analysis_results/ckpt_{tag}/correlation')
    for f in sorted(root.glob('binned_*_1step_raw.csv')):
        df = pd.read_csv(f)
        aligned = df[(df['bin_lo'] >= 0.3) & (df['count'] > 0)]
        conflict = df[(df['bin_hi'] <= -0.3) & (df['count'] > 0)]
        if aligned.empty or conflict.empty:
            continue
        a = aligned['delta_mean'].mean()
        c = conflict['delta_mean'].mean()
        ok = a <= c
        print(f'  {f.name}: aligned={a:+.4f} conflict={c:+.4f} {"OK" if ok else "VIOLATION"}')
"
```

---

## Task 13: Supplementary per-sample run (1ep, batch=1, 500 samples)

- [ ] **Step 1: Launch supplementary**

```bash
nohup python tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M3,M4 \
    > gradient_analysis_results/supplementary.log 2>&1 &
echo $! > gradient_analysis_results/supplementary.pid
```

- [ ] **Step 2: Monitor**

```bash
tail -f gradient_analysis_results/supplementary.log
```

- [ ] **Step 3: Generate comparison figure (batch=12 vs batch=1 at 1ep)**

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

bs12 = Path("gradient_analysis_results/ckpt_1ep/correlation")
bs1 = Path("gradient_analysis_results/supplementary/correlation")
out = Path("gradient_analysis_results/supplementary/bs12_vs_bs1.png")

# Example: motion→plan, 1-step raw
pair = ("motion", "plan")
f_bs12 = bs12 / f"binned_{pair[0]}_{pair[1]}_1step_raw.csv"
f_bs1 = bs1 / f"binned_{pair[0]}_{pair[1]}_1step_raw.csv"
if not (f_bs12.exists() and f_bs1.exists()):
    print("skip: missing files"); raise SystemExit(0)

d12 = pd.read_csv(f_bs12)
d1 = pd.read_csv(f_bs1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, d, label in ((axes[0], d12, "batch=12"), (axes[1], d1, "batch=1")):
    c = (d["bin_lo"] + d["bin_hi"]) / 2
    ax.errorbar(c, d["delta_mean"], yerr=d["delta_std"], marker="o")
    ax.set_title(f"{pair[0]}→{pair[1]} ({label}, n_total={int(d['count'].sum())})")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("cosine"); ax.set_ylabel(r"$\Delta L_{target}$")
fig.tight_layout()
fig.savefig(out, dpi=150)
print("wrote", out)
PY
```

---

## Task 14: Summary report generator

**Files:**
- Create: `tools/gradient_analysis/summary.py`
- Modify: `tools/run_gradient_analysis.py` — wire the summary step.

- [ ] **Step 1: Implement `tools/gradient_analysis/summary.py`**

```python
"""Generate a markdown summary highlighting top findings for paper drafting."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def generate_summary(output_root: Path, checkpoint_tags: Iterable[str]) -> None:
    output_root = Path(output_root)
    lines = ["# Gradient Analysis Summary\n"]

    # Correlation table — top/bottom pairs across checkpoints
    all_corr = []
    for tag in checkpoint_tags:
        f = output_root / f"ckpt_{tag}" / "correlation" / "correlation_table.csv"
        if f.exists():
            df = pd.read_csv(f)
            df["checkpoint"] = tag
            all_corr.append(df)
    if all_corr:
        corr = pd.concat(all_corr, ignore_index=True)
        corr_1raw = corr[(corr["steps"] == 1) & (corr["variant"] == "raw")]
        lines.append("## Strongest correlation |cos ↔ Δloss| (1-step raw)\n")
        top = corr_1raw.reindex(corr_1raw["pearson"].abs().sort_values(ascending=False).index).head(10)
        lines.append(top.to_markdown(index=False) + "\n")

    # Task affinity asymmetry highlights
    for tag in checkpoint_tags:
        f = output_root / f"ckpt_{tag}" / "asymmetry" / "top_asymmetric_pairs.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"\n## Top asymmetric pairs @ {tag}\n")
            lines.append(df.head(5).to_markdown(index=False) + "\n")

    # GradNorm raw vs normalized symmetry
    lines.append("\n## Raw vs Normalized probe symmetry (Frobenius of antisymmetric Δloss)\n")
    for tag in checkpoint_tags:
        f = output_root / f"ckpt_{tag}" / "gradnorm" / "raw_vs_norm_symmetry.csv"
        if f.exists():
            df = pd.read_csv(f)
            lines.append(f"### {tag}\n")
            lines.append(df.to_markdown(index=False) + "\n")

    (output_root / "summary_report.md").write_text("\n".join(lines))
```

- [ ] **Step 2: Wire into CLI — add to end of `main()` in `tools/run_gradient_analysis.py`**

```python
    # after M6 dynamics block:
    from tools.gradient_analysis.summary import generate_summary
    generate_summary(out_root, tags)
    print(f"summary: {out_root / 'summary_report.md'}")
```

- [ ] **Step 3: Verify summary generation**

```bash
python -c "
from pathlib import Path
from tools.gradient_analysis.summary import generate_summary
generate_summary(Path('gradient_analysis_results'), ['1ep','3ep','6ep','18ep'])
print(open('gradient_analysis_results/summary_report.md').read()[:500])
"
```

- [ ] **Step 4: Commit**

```bash
git add tools/gradient_analysis/summary.py tools/run_gradient_analysis.py
git commit -m "anal: markdown summary report generator (highlights for paper)"
```

---

## Task 15: Final repo hygiene

- [ ] **Step 1: Write `tools/gradient_analysis/README.md`**

```markdown
# Gradient Analysis Pipeline

See `docs/superpowers/specs/2026-04-20-mtl-gradient-analysis-design.md` for the full design.

## Quick start

```bash
# Smoke test (3 batches, 1 checkpoint, ~10 min)
python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --smoke --checkpoints 1ep --no-supplementary

# Full primary run (4 checkpoints × 100 batches, ~1.5 days)
python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --no-supplementary

# Supplementary per-sample (batch=1 on 1ep, ~6-8 hours)
python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --checkpoints 1ep --modules M3,M4

# Aggregate dynamics across pre-computed checkpoints (post-hoc)
python tools/run_gradient_analysis.py --config ... --modules M6
```

## Modules

- **M2** Conflict — per-layer-group cosine + projection decomposition
- **M3** Probe — 1/2-step virtual update (full param) + raw/normalized variants
- **M4** Correlation — binned cosine↔Δloss with Pearson/Spearman
- **M5** GradNorm — per-task norm, norm ratio, raw-vs-normalized symmetry
- **M6** Dynamics — cross-checkpoint time series
- **M7** Asymmetry — symmetric/antisymmetric decomposition of affinity matrix

Outputs land in `gradient_analysis_results/`. See each module's CSVs for numbers, PNGs for plots, and `summary_report.md` for highlights.
```

- [ ] **Step 2: Run full test suite**

```bash
pytest tests/gradient_analysis/ -v
```
Expected: all tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tools/gradient_analysis/README.md
git commit -m "docs: gradient analysis README"
```

---

## Appendix — Risks & escape hatches

- **OOM on full-param gradient for HiP-AD + batch=12:** if encountered, reduce `primary.batch_size` to 6 (matches training per-GPU setup), keeping `num_batches` the same. Cost doubles but pipeline unchanged.
- **V2 reverse-sanity failure on real model:** means the full-param set is incomplete (some reachable param was excluded). Re-check `collector.full_params = [p for p in raw_model.parameters() if p.requires_grad]`; this should be exhaustive.
- **V5 violations on real data:** could be real (hypothesis wrong for this model/pair/layer) or a bug. Inspect correlation scatter plots — if scatter looks noisy but bin means still monotonic across the whole range, the binned summary just lacks resolution; increase `num_batches`.
- **Non-determinism despite seed:** some cuDNN paths ignore deterministic flags. Check that `cfg_ana["deterministic"]` is honored in the model forward; if not, accept run-to-run variation up to ~1e-3 on metrics.
