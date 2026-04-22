# HiP-AD Stage2 GradNorm Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate GradNorm (Chen et al. ICML 2018) into HiP-AD stage2 training so the 5 task loss weights (det, map, motion, plan, ego) are adaptively balanced based on shared-decoder gradient magnitudes. Depth loss is kept as a fixed auxiliary. A ready-to-run config `E9_E2_E1_stage2_18ep_GN.py` plus a reusable `GradNormLossWeighter` module is produced.

**Architecture:** A HiP-AD-independent `GradNormLossWeighter(nn.Module)` holds learnable `w`, runs its own Adam, and computes per-task gradient norms against the last `nn.Linear.weight` of every `AsymmetricFFN` in the decoder. `SparseDetector.forward_train` aggregates the existing task loss dict by prefix, feeds task scalars to the weighter, and renames original loss keys to `monitor_*` so mmcv's `_parse_losses` sees only `loss_gradnorm_total` plus `loss_dense_depth`. A rank-0 CSV dump captures every iter's diagnostics.

**Tech Stack:** PyTorch, mmcv 1.x / mmdet 2.x (repo baseline), pytest (for unit tests of the isolated helper and weighter), existing nuScenes data pipeline.

**Design spec:** [docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md](../specs/2026-04-22-hipad-stage2-gradnorm-design.md)

**Commit policy (per user request):** Each task below ends with `git add` (staging) only. A single consolidated commit is made at the end of Task 14. Do NOT run `git commit` in intermediate tasks.

---

## File Structure

### Files created
| Path | Responsibility |
|---|---|
| `projects/mmdet3d_plugin/core/gradnorm/__init__.py` | Public re-export |
| `projects/mmdet3d_plugin/core/gradnorm/weighter.py` | `GradNormLossWeighter` class (algorithm, state, Adam, renorm, DDP sync, CSV-safe log) |
| `projects/mmdet3d_plugin/core/gradnorm/shared_params.py` | `collect_last_linear_weights(operation_order, layers)` — pure helper, unit-testable |
| `projects/mmdet3d_plugin/core/gradnorm/csv_dumper.py` | `GradNormCSVDumper` — rank-0 file writer |
| `tests/gradnorm/__init__.py` | (empty) |
| `tests/gradnorm/test_weighter.py` | Unit tests for `GradNormLossWeighter` |
| `tests/gradnorm/test_shared_params.py` | Unit tests for `collect_last_linear_weights` |
| `tests/gradnorm/test_csv_dumper.py` | Unit tests for `GradNormCSVDumper` |
| `tests/gradnorm/test_detector_helpers.py` | Unit tests for `SparseDetector._aggregate_task_losses` / `_rename_as_monitor` |

### Files modified
| Path | Change |
|---|---|
| `projects/mmdet3d_plugin/models/sparse_onedecoder.py` | Add `collect_ffn_last_fc_params()` method using the pure helper |
| `projects/mmdet3d_plugin/models/sparse_detector.py` | Accept `gradnorm` kwarg in `__init__`, extend `forward_train`, add 3 helpers (`_aggregate_task_losses`, `_rename_as_monitor`, `_maybe_dump_gn_csv`) |
| `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py` | REPLACE entire content with GradNorm-enabled config |

### Files NOT touched
- `projects/configs/experiments/E1_stage1_12ep.py`
- `projects/mmdet3d_plugin/apis/*.py`
- `tools/train.py`, `tools/dist_train.sh`
- All existing loss / head / backbone files

---

## Pre-flight

- [ ] **Step 0.1: Ensure pytest is available**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "import pytest; print(pytest.__version__)"
```
If it errors, run:
```bash
cd /home/yongjae/e2e/HiP-AD && /home/yongjae/miniconda3/envs/hipad/bin/pip install pytest
```
Expected: version number printed.

- [ ] **Step 0.2: Confirm working git state**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git status --short
```
Expected: only the untracked files listed in the spec (`.omc/`, `DISTILLATION_SESSION_PROMPT.md`, `GAR_epoch2_100/`, `GAR_epoch6_100/`, `data_nusc/`, `eval_monitor.log`, `evaluation/`, `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py`, `tools/gradient_conflict_anal.py`, `wait_and_eval.sh`) plus this new plan file. No other tracked-file edits pending.

---

## Task 1: `GradNormLossWeighter` constructor + buffers

**Files:**
- Create: `projects/mmdet3d_plugin/core/gradnorm/__init__.py`
- Create: `projects/mmdet3d_plugin/core/gradnorm/weighter.py`
- Test: `tests/gradnorm/__init__.py`, `tests/gradnorm/test_weighter.py`

- [ ] **Step 1.1: Create empty package init files**

Write `projects/mmdet3d_plugin/core/gradnorm/__init__.py`:
```python
from .weighter import GradNormLossWeighter

__all__ = ["GradNormLossWeighter"]
```

Write `tests/gradnorm/__init__.py`:
```python
```
(empty file)

- [ ] **Step 1.2: Write failing test for constructor state**

Write `tests/gradnorm/test_weighter.py`:
```python
import math

import pytest
import torch

from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter


def make_default_weighter():
    return GradNormLossWeighter(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[4.25, 11.0, 0.4, 1.5, 1.0],
        alpha=1.5,
        lr=2.5e-2,
        update_after_step=500,
        pivot_warmup_steps=50,
        update_every=1,
        clamp_min=1e-4,
    )


def test_constructor_state_shapes_and_values():
    w = make_default_weighter()
    assert w.task_names == ["det", "map", "motion", "plan", "ego"]
    assert w.T_tasks == 5
    assert w.alpha == pytest.approx(1.5)
    assert w.update_after_step == 500
    assert w.pivot_warmup_steps == 50
    assert w.update_every == 1
    assert w.clamp_min == pytest.approx(1e-4)

    # Parameter
    assert isinstance(w.w, torch.nn.Parameter)
    assert w.w.shape == (5,)
    assert torch.allclose(
        w.w.data, torch.tensor([4.25, 11.0, 0.4, 1.5, 1.0], dtype=torch.float32)
    )

    # Buffers
    assert torch.allclose(w.L0, torch.zeros(5))
    assert torch.allclose(w.L0_running, torch.zeros(5))
    assert int(w.step_counter.item()) == 0
    assert bool(w.pivot_ready.item()) is False
    assert math.isclose(float(w.init_w_sum.item()), 18.15, rel_tol=1e-5)

    # Internal optimizer exists and targets self.w
    assert isinstance(w.w_optimizer, torch.optim.Adam)
    param_in_opt = w.w_optimizer.param_groups[0]["params"][0]
    assert param_in_opt is w.w


def test_constructor_rejects_bad_input():
    with pytest.raises(AssertionError):
        GradNormLossWeighter(task_names=["only_one"], init_weights=[1.0])
    with pytest.raises(AssertionError):
        GradNormLossWeighter(
            task_names=["a", "b"], init_weights=[1.0, -1.0]
        )
    with pytest.raises(AssertionError):
        GradNormLossWeighter(
            task_names=["a", "b"], init_weights=[1.0]
        )
```

- [ ] **Step 1.3: Run test to verify it fails**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` for `weighter` (module not yet written).

- [ ] **Step 1.4: Implement weighter skeleton**

Write `projects/mmdet3d_plugin/core/gradnorm/weighter.py`:
```python
from __future__ import annotations

from typing import List

import torch
from torch import nn


class GradNormLossWeighter(nn.Module):
    """GradNorm (Chen et al., ICML 2018) loss weighter for HiP-AD stage2.

    See docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md Section 3.
    """

    def __init__(
        self,
        task_names: List[str],
        init_weights: List[float],
        alpha: float = 1.5,
        lr: float = 2.5e-2,
        update_after_step: int = 500,
        pivot_warmup_steps: int = 50,
        update_every: int = 1,
        clamp_min: float = 1e-4,
    ) -> None:
        super().__init__()

        assert len(task_names) == len(init_weights) >= 2, (
            "task_names and init_weights must have equal length >= 2"
        )
        assert alpha >= 0.0, "alpha must be non-negative"
        assert lr > 0.0, "lr must be positive"
        assert update_after_step >= 0
        assert pivot_warmup_steps >= 1
        assert update_every >= 1
        assert all(v > 0 for v in init_weights), "init_weights must be strictly positive"

        self.task_names = list(task_names)
        self.T_tasks = len(task_names)
        self.alpha = float(alpha)
        self.clamp_min = float(clamp_min)
        self.update_after_step = int(update_after_step)
        self.pivot_warmup_steps = int(pivot_warmup_steps)
        self.update_every = int(update_every)

        self.w = nn.Parameter(torch.tensor(init_weights, dtype=torch.float32))
        self.w_optimizer = torch.optim.Adam([self.w], lr=lr)

        self.register_buffer("L0", torch.zeros(self.T_tasks))
        self.register_buffer("L0_running", torch.zeros(self.T_tasks))
        self.register_buffer("step_counter", torch.zeros((), dtype=torch.long))
        self.register_buffer("pivot_ready", torch.zeros((), dtype=torch.bool))
        self.register_buffer(
            "init_w_sum",
            torch.tensor(float(sum(init_weights)), dtype=torch.float32),
        )
```

- [ ] **Step 1.5: Run test to verify it passes**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v
```
Expected: 2 passed.

- [ ] **Step 1.6: Stage changes (no commit yet)**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/__init__.py \
  projects/mmdet3d_plugin/core/gradnorm/weighter.py \
  tests/gradnorm/__init__.py \
  tests/gradnorm/test_weighter.py
```

---

## Task 2: Forward pass — warmup phase (no weight update)

**Files:**
- Modify: `projects/mmdet3d_plugin/core/gradnorm/weighter.py`
- Modify: `tests/gradnorm/test_weighter.py`

- [ ] **Step 2.1: Append failing test for warmup branch**

Append to `tests/gradnorm/test_weighter.py`:
```python
def _dummy_shared_param(shape=(4, 4)):
    return [torch.nn.Parameter(torch.randn(*shape, requires_grad=True))]


def _make_loss(values, shared_param=None):
    # Constructs scalar losses whose autograd graph touches `shared_param`
    # so that torch.autograd.grad against shared_param returns valid gradients.
    if shared_param is None:
        shared_param = _dummy_shared_param()[0]
    losses = {}
    x = torch.randn(4, requires_grad=False)
    for name, v in values.items():
        y = (shared_param @ x).sum() * 0.0 + float(v)
        losses[name] = y
    return losses, [shared_param]


def test_warmup_phase_does_not_update_weights():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 3.0],
        update_after_step=5,
        pivot_warmup_steps=2,
        update_every=1,
    )
    shared = _dummy_shared_param()
    initial_w = w.w.data.clone()

    for step in range(5):
        losses, shared_list = _make_loss({"a": 0.5, "b": 2.0}, shared[0])
        weighted, log = w(losses, shared_list)
        assert log["phase_id"] == 0  # warmup
        # weighted_loss == w_det * L_det + w_map * L_map using init weights
        expected = 1.0 * 0.5 + 3.0 * 2.0
        assert float(weighted.item()) == pytest.approx(expected, rel=1e-5)

    assert torch.allclose(w.w.data, initial_w)
    assert int(w.step_counter.item()) == 5
    assert bool(w.pivot_ready.item()) is False
```

- [ ] **Step 2.2: Run test to verify it fails**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py::test_warmup_phase_does_not_update_weights -v
```
Expected: `AttributeError` / `NotImplementedError` (forward missing) or `TypeError` from Module default `forward`.

- [ ] **Step 2.3: Implement warmup branch of `forward`**

Append to `projects/mmdet3d_plugin/core/gradnorm/weighter.py` inside the class:
```python
    def forward(
        self,
        task_losses: "dict[str, torch.Tensor]",
        shared_params: List[torch.nn.Parameter],
    ) -> "tuple[torch.Tensor, dict[str, float]]":
        step = int(self.step_counter.item())
        self.step_counter += 1

        L = torch.stack([task_losses[n] for n in self.task_names])
        w_det = self.w.detach()
        weighted_loss = (w_det * L).sum()

        log: "dict[str, float]" = {}
        for i, n in enumerate(self.task_names):
            log[f"w_{n}"] = float(w_det[i].item())
        log["step"] = float(step)

        # Safety 2 gate 1: warmup
        if step < self.update_after_step:
            log["phase_id"] = 0.0
            return weighted_loss, log

        raise NotImplementedError("phase >= update_after_step not implemented yet")
```

- [ ] **Step 2.4: Run test to verify it passes**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v
```
Expected: 3 passed.

- [ ] **Step 2.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/weighter.py \
  tests/gradnorm/test_weighter.py
```

---

## Task 3: Forward pass — pivot accumulation and freeze

**Files:**
- Modify: `projects/mmdet3d_plugin/core/gradnorm/weighter.py`
- Modify: `tests/gradnorm/test_weighter.py`

- [ ] **Step 3.1: Append failing test for pivot accumulation**

Append to `tests/gradnorm/test_weighter.py`:
```python
def test_pivot_accumulation_and_freeze():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 3.0],
        update_after_step=5,
        pivot_warmup_steps=3,
        update_every=1,
    )
    shared = _dummy_shared_param()

    # Burn warmup
    for _ in range(5):
        losses, sp = _make_loss({"a": 0.5, "b": 2.0}, shared[0])
        w(losses, sp)

    # Step 5: first pivot accumulation iter (step - update_after_step + 1 == 1 < 3)
    losses, sp = _make_loss({"a": 0.2, "b": 0.8}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 1.0
    assert bool(w.pivot_ready.item()) is False
    assert torch.allclose(w.L0_running, torch.tensor([0.2, 0.8]), atol=1e-6)

    # Step 6: pivot_step == 2 < 3
    losses, sp = _make_loss({"a": 0.4, "b": 1.0}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 1.0
    assert bool(w.pivot_ready.item()) is False
    assert torch.allclose(w.L0_running, torch.tensor([0.6, 1.8]), atol=1e-6)

    # Step 7: pivot_step == 3 == pivot_warmup_steps → freeze
    losses, sp = _make_loss({"a": 0.6, "b": 1.4}, shared[0])
    _, log = w(losses, sp)
    assert log["phase_id"] == 2.0
    assert bool(w.pivot_ready.item()) is True
    expected_L0 = torch.tensor([(0.2 + 0.4 + 0.6) / 3.0,
                                (0.8 + 1.0 + 1.4) / 3.0])
    assert torch.allclose(w.L0, expected_L0, atol=1e-6)
```

- [ ] **Step 3.2: Run to verify it fails**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py::test_pivot_accumulation_and_freeze -v
```
Expected: `NotImplementedError` from Task 2 placeholder.

- [ ] **Step 3.3: Replace the `NotImplementedError` with pivot branch**

Open `projects/mmdet3d_plugin/core/gradnorm/weighter.py`, find the line that currently reads:
```python
        raise NotImplementedError("phase >= update_after_step not implemented yet")
```
Replace it with:
```python
        # Safety 2 gate 2: pivot accumulation / freeze
        if not bool(self.pivot_ready.item()):
            self.L0_running += L.detach().float()
            pivot_step = step - self.update_after_step + 1
            if pivot_step >= self.pivot_warmup_steps:
                self.L0.copy_(self.L0_running / float(self.pivot_warmup_steps))
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        self.L0, op=torch.distributed.ReduceOp.SUM
                    )
                    self.L0.mul_(1.0 / float(torch.distributed.get_world_size()))
                self.pivot_ready.fill_(True)
                log["phase_id"] = 2.0
            else:
                log["phase_id"] = 1.0
            return weighted_loss, log

        raise NotImplementedError("gn_active phase not implemented yet")
```

- [ ] **Step 3.4: Run to verify pass**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v
```
Expected: 4 passed.

- [ ] **Step 3.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/weighter.py \
  tests/gradnorm/test_weighter.py
```

---

## Task 4: Forward pass — GradNorm main step

**Files:**
- Modify: `projects/mmdet3d_plugin/core/gradnorm/weighter.py`
- Modify: `tests/gradnorm/test_weighter.py`

- [ ] **Step 4.1: Append failing test that drives the main branch**

Append to `tests/gradnorm/test_weighter.py`:
```python
def _step_through_to_active(w, shared, loss_values):
    for _ in range(w.update_after_step + w.pivot_warmup_steps):
        losses, sp = _make_loss(loss_values, shared[0])
        w(losses, sp)


def test_gn_active_updates_weights_towards_slow_task():
    torch.manual_seed(0)
    task_names = ["fast", "slow"]
    w = GradNormLossWeighter(
        task_names=task_names,
        init_weights=[1.0, 1.0],
        alpha=1.5,
        lr=0.1,
        update_after_step=2,
        pivot_warmup_steps=2,
        update_every=1,
        clamp_min=1e-4,
    )
    shared = [torch.nn.Parameter(torch.randn(4, 4, requires_grad=True))]

    # phase: warmup (steps 0,1) then pivot_accum (2,3) → pivot_set on step 3
    # Use symmetric per-iter losses so pivot L0 = [1.0, 1.0]
    _step_through_to_active(w, shared, {"fast": 1.0, "slow": 1.0})

    assert bool(w.pivot_ready.item()) is True
    assert torch.allclose(w.L0, torch.tensor([1.0, 1.0]), atol=1e-6)
    w_before = w.w.data.clone()

    # Now create gn_active iter with "slow" task not progressing
    # Give shared-param-dependent losses (so gradients are non-trivial)
    x = torch.randn(4)
    fast_loss = (shared[0] @ x).sum() * 0.01                   # small → "fast"
    slow_loss = (shared[0] @ x).pow(2).sum() * 1.0             # large → "slow"
    # Normalize so pivot ratio magnitudes match intent:
    # L_fast=0.1, L_slow=2.0 → r̃_slow > r̃_fast → w_slow should rise
    fast_loss = (shared[0] @ x).sum() * 0.0 + 0.1
    # make slow_loss depend on shared so grad is non-zero, but scale to 2.0
    slow_loss = (shared[0] @ x).sum() * 1e-6 + 2.0
    # Both losses must depend on shared_param. Inject small dep for "fast":
    fast_loss = (shared[0] @ x).sum() * 1e-6 + 0.1
    losses = {"fast": fast_loss, "slow": slow_loss}

    weighted, log = w(losses, shared)
    assert log["phase_id"] == 3.0
    assert "grad_norm_fast" in log and "grad_norm_slow" in log
    assert "rt_fast" in log and "rt_slow" in log
    # r̃_slow > r̃_fast because L_slow/L0_slow > L_fast/L0_fast
    assert log["rt_slow"] > log["rt_fast"]
    # After update and renorm, w_slow should be >= w_before[slow]
    # (direction of change is what matters — magnitude depends on gradients)
    # clamp + sum-to-init-sum ensures w>0 and sum preserved
    assert w.w.data[1].item() >= w_before[1].item() - 1e-3
    # sum must stay at init_w_sum
    assert float(w.w.data.sum().item()) == pytest.approx(
        float(w.init_w_sum.item()), rel=1e-4
    )


def test_update_every_skip():
    w = GradNormLossWeighter(
        task_names=["a", "b"],
        init_weights=[1.0, 1.0],
        update_after_step=0,
        pivot_warmup_steps=1,
        update_every=3,
    )
    shared = [torch.nn.Parameter(torch.randn(4, 4, requires_grad=True))]
    # step 0 → pivot_set, step 1 → gn_active (0 % 3 == 0),
    # step 2,3 → gn_skip_every, step 4 → gn_active (3 % 3 == 0)
    phases = []
    for step in range(5):
        x = torch.randn(4)
        losses = {
            "a": (shared[0] @ x).sum() * 1e-6 + 0.5,
            "b": (shared[0] @ x).sum() * 1e-6 + 0.5,
        }
        _, log = w(losses, shared)
        phases.append(int(log["phase_id"]))
    assert phases == [2, 3, 4, 4, 3]
```

- [ ] **Step 4.2: Run to verify it fails**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v -k "gn_active or update_every"
```
Expected: `NotImplementedError` from Task 3 placeholder.

- [ ] **Step 4.3: Replace `NotImplementedError` with GradNorm main body (includes Safety 1, Safety 3, DDP sync)**

Open `projects/mmdet3d_plugin/core/gradnorm/weighter.py`. Find the line:
```python
        raise NotImplementedError("gn_active phase not implemented yet")
```
Replace it with:
```python
        # update_every gate
        if (step - self.update_after_step - self.pivot_warmup_steps) % self.update_every != 0:
            log["phase_id"] = 4.0
            return weighted_loss, log

        # ── GradNorm main body ──
        grad_norms = []
        for i, Li in enumerate(L):
            grads = torch.autograd.grad(
                outputs=self.w[i] * Li,
                inputs=shared_params,
                retain_graph=True,
                create_graph=True,
                allow_unused=False,
            )
            # Safety 1: fp32 promotion
            g_flat = torch.cat([g.float().flatten() for g in grads])
            grad_norms.append(g_flat.norm(p=2))
        gw = torch.stack(grad_norms)

        loss_ratio = L.detach().float() / self.L0.clamp_min(1e-8)
        rt = loss_ratio / loss_ratio.mean().clamp_min(1e-8)

        gw_avg = gw.mean().detach()
        target = (gw_avg * (rt ** self.alpha)).detach()

        gn_loss = (gw - target).abs().sum()

        self.w_optimizer.zero_grad(set_to_none=True)
        gn_loss.backward()
        self.w_optimizer.step()

        # Safety 3: clamp + sum-to-init-sum renorm + DDP sync
        with torch.no_grad():
            self.w.data.clamp_(min=self.clamp_min)
            self.w.data.mul_(
                self.init_w_sum / self.w.data.sum().clamp_min(1e-8)
            )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    self.w.data, op=torch.distributed.ReduceOp.SUM
                )
                self.w.data.mul_(1.0 / float(torch.distributed.get_world_size()))

        # Extended logging
        for i, n in enumerate(self.task_names):
            log[f"grad_norm_{n}"]  = float(gw[i].item())
            log[f"rt_{n}"]         = float(rt[i].item())
            log[f"L_{n}"]          = float(L[i].item())
            log[f"L0_{n}"]         = float(self.L0[i].item())
            log[f"weighted_L_{n}"] = float((w_det[i] * L[i]).item())
        log["gn_loss"]   = float(gn_loss.item())
        log["phase_id"]  = 3.0
        return weighted_loss, log
```

- [ ] **Step 4.4: Run full test file**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_weighter.py -v
```
Expected: 6 passed.

- [ ] **Step 4.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/weighter.py \
  tests/gradnorm/test_weighter.py
```

---

## Task 5: Pure helper `collect_last_linear_weights` + unit test

**Files:**
- Create: `projects/mmdet3d_plugin/core/gradnorm/shared_params.py`
- Create: `tests/gradnorm/test_shared_params.py`

- [ ] **Step 5.1: Write failing test**

Write `tests/gradnorm/test_shared_params.py`:
```python
import pytest
import torch
from torch import nn

from projects.mmdet3d_plugin.core.gradnorm.shared_params import (
    collect_last_linear_weights,
)


class _FakeFFN(nn.Module):
    def __init__(self, in_ch, hid, out_ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(in_ch, hid), nn.ReLU(), nn.Dropout(0.0)),
            nn.Linear(hid, out_ch),
            nn.Dropout(0.0),
        )


def test_collects_one_weight_per_ffn_in_order():
    ffn_a = _FakeFFN(8, 16, 4)
    ffn_b = _FakeFFN(4, 32, 4)
    layers = [nn.Identity(), ffn_a, nn.Identity(), ffn_b]
    operation_order = ["norm", "ffn", "norm", "ffn"]

    params = collect_last_linear_weights(operation_order, layers)

    assert len(params) == 2
    assert params[0] is ffn_a.layers[1].weight
    assert params[1] is ffn_b.layers[1].weight
    assert params[0].shape == (4, 16)
    assert params[1].shape == (4, 32)


def test_asserts_when_ffn_has_no_linear():
    bad_ffn = nn.Module()
    bad_ffn.layers = nn.Sequential(nn.ReLU(), nn.Dropout(0.0))
    with pytest.raises(AssertionError):
        collect_last_linear_weights(["ffn"], [bad_ffn])


def test_ignores_non_ffn_ops():
    params = collect_last_linear_weights(["norm", "norm"], [nn.Identity(), nn.Identity()])
    assert params == []
```

- [ ] **Step 5.2: Run to verify fail**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_shared_params.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 5.3: Implement helper**

Write `projects/mmdet3d_plugin/core/gradnorm/shared_params.py`:
```python
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn


def collect_last_linear_weights(
    operation_order: Sequence[str],
    layers: Sequence[nn.Module],
) -> List[torch.nn.Parameter]:
    """For every position ``i`` where ``operation_order[i] == "ffn"``, return
    ``layers[i].layers``'s last ``nn.Linear.weight`` in order.

    The caller guarantees ``layers[i]`` is an ``AsymmetricFFN``-compatible
    module exposing ``.layers`` as an ``nn.Sequential``-like container.

    Raises AssertionError if an ``"ffn"`` position has no ``nn.Linear``.
    """
    assert len(operation_order) == len(layers), (
        "operation_order and layers must have the same length"
    )
    params: List[torch.nn.Parameter] = []
    for op, module in zip(operation_order, layers):
        if op != "ffn":
            continue
        inner = getattr(module, "layers", None)
        assert inner is not None, "ffn module must expose .layers attribute"
        last_linear = None
        for sub in reversed(list(inner.children())):
            if isinstance(sub, nn.Linear):
                last_linear = sub
                break
        assert last_linear is not None, (
            "ffn module's .layers contains no nn.Linear child"
        )
        params.append(last_linear.weight)
    return params
```

- [ ] **Step 5.4: Run to verify pass**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_shared_params.py -v
```
Expected: 3 passed.

- [ ] **Step 5.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/shared_params.py \
  tests/gradnorm/test_shared_params.py
```

---

## Task 6: Add `collect_ffn_last_fc_params` method to `SparseOneDecoder`

**Files:**
- Modify: `projects/mmdet3d_plugin/models/sparse_onedecoder.py`

- [ ] **Step 6.1: Add method near other public methods**

Open `projects/mmdet3d_plugin/models/sparse_onedecoder.py`. Immediately **before** the `def loss(` line (line ~1123), add the following method. Use an import at file top if not already present.

At the top of the file (after existing imports), add if missing:
```python
from projects.mmdet3d_plugin.core.gradnorm.shared_params import (
    collect_last_linear_weights,
)
```

Inside the class, above `def loss(self, ...)`:
```python
    def collect_ffn_last_fc_params(self):
        """Return the list of nn.Parameter (weight only) taken from the last
        nn.Linear of every AsymmetricFFN that appears at an "ffn" position in
        ``self.operation_order``. Used as the shared W for GradNorm.
        """
        return collect_last_linear_weights(self.operation_order, self.layers)
```

- [ ] **Step 6.2: Smoke-check import path by compiling the file**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "import ast, pathlib; ast.parse(pathlib.Path('projects/mmdet3d_plugin/models/sparse_onedecoder.py').read_text())" && echo "OK"
```
Expected: `OK`.

- [ ] **Step 6.3: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add projects/mmdet3d_plugin/models/sparse_onedecoder.py
```

---

## Task 7: `SparseDetector` helpers — aggregate & rename (pure, unit-testable)

**Files:**
- Create: `tests/gradnorm/test_detector_helpers.py`
- Modify: `projects/mmdet3d_plugin/models/sparse_detector.py`

- [ ] **Step 7.1: Write failing tests**

Write `tests/gradnorm/test_detector_helpers.py`:
```python
import pytest
import torch

from projects.mmdet3d_plugin.models.sparse_detector import SparseDetector


def _tensor(v):
    return torch.tensor(float(v))


def test_aggregate_task_losses_basic():
    output = {
        "det_loss_cls": _tensor(1.0),
        "det_loss_box": _tensor(2.0),
        "det_loss_cns": _tensor(0.5),
        "det_loss_yns": _tensor(0.5),
        "map_loss_cls": _tensor(0.3),
        "map_loss_line": _tensor(0.7),
        "motion_loss_cls": _tensor(0.1),
        "motion_loss_reg": _tensor(0.2),
        "plan_loss_temp_cls": _tensor(0.4),
        "plan_loss_temp_reg": _tensor(0.6),
        "ego_loss_status": _tensor(0.9),
        "loss_dense_depth": _tensor(10.0),   # NOT aggregated
    }
    task_names = ["det", "map", "motion", "plan", "ego"]
    agg = SparseDetector._aggregate_task_losses(output, task_names)
    assert float(agg["det"].item()) == pytest.approx(4.0)
    assert float(agg["map"].item()) == pytest.approx(1.0)
    assert float(agg["motion"].item()) == pytest.approx(0.3)
    assert float(agg["plan"].item()) == pytest.approx(1.0)
    assert float(agg["ego"].item()) == pytest.approx(0.9)
    assert "depth" not in agg


def test_aggregate_asserts_when_prefix_missing():
    output = {"det_loss_cls": _tensor(1.0)}
    with pytest.raises(AssertionError):
        SparseDetector._aggregate_task_losses(output, ["det", "map"])


def test_rename_as_monitor_detaches_and_removes_loss_substring():
    x = torch.tensor(1.0, requires_grad=True)
    output = {
        "det_loss_cls": x * 2.0,
        "map_loss_line": x * 3.0,
        "loss_dense_depth": x * 5.0,     # pass-through
        "some_scalar": 7,                # pass-through
    }
    task_prefixes = {"det_loss_", "map_loss_"}
    out = SparseDetector._rename_as_monitor(output, task_prefixes)
    assert set(out.keys()) == {
        "monitor_det_cls", "monitor_map_line", "loss_dense_depth", "some_scalar"
    }
    # detached
    assert not out["monitor_det_cls"].requires_grad
    assert not out["monitor_map_line"].requires_grad
    # pass-through preserved
    assert out["loss_dense_depth"] is output["loss_dense_depth"]
    assert out["some_scalar"] == 7
    # 'loss' substring removed only in rename targets
    assert "loss" not in "monitor_det_cls"
    assert "loss" not in "monitor_map_line"
```

- [ ] **Step 7.2: Run to fail**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_detector_helpers.py -v
```
Expected: `AttributeError: type object 'SparseDetector' has no attribute '_aggregate_task_losses'`.

- [ ] **Step 7.3: Add the two staticmethod helpers to `SparseDetector`**

Open `projects/mmdet3d_plugin/models/sparse_detector.py`. At the top of the class body (right after `def __init__` closes, before `@auto_fp16` / `extract_feat`), add:
```python
    @staticmethod
    def _aggregate_task_losses(output, task_names):
        """Sum all ``{task}_loss_*`` items per task into one scalar each."""
        task_losses = {}
        for t in task_names:
            prefix = f"{t}_loss_"
            parts = [v for k, v in output.items() if k.startswith(prefix)]
            assert len(parts) > 0, (
                f"No loss key with prefix '{prefix}' found in head output"
            )
            stacked = torch.stack([p if torch.is_tensor(p) else torch.tensor(float(p)) for p in parts])
            task_losses[t] = stacked.sum()
        return task_losses

    @staticmethod
    def _rename_as_monitor(output, task_prefixes):
        """Rename ``{task}_loss_{name}`` keys to ``monitor_{task}_{name}`` and
        detach their tensors so mmcv ``_parse_losses`` (which sums keys
        containing ``'loss'``) and autograd both skip them.
        """
        new = {}
        for k, v in output.items():
            matched = next((p for p in task_prefixes if k.startswith(p)), None)
            if matched is not None:
                new_k = "monitor_" + k.replace("_loss_", "_", 1)
                new[new_k] = v.detach() if torch.is_tensor(v) else v
            else:
                new[k] = v
        return new
```

If `import torch` is missing at the top of the file, add it.

- [ ] **Step 7.4: Run to pass**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_detector_helpers.py -v
```
Expected: 3 passed.

- [ ] **Step 7.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/models/sparse_detector.py \
  tests/gradnorm/test_detector_helpers.py
```

---

## Task 8: CSV dumper (rank-0, iter-level)

**Files:**
- Create: `projects/mmdet3d_plugin/core/gradnorm/csv_dumper.py`
- Create: `tests/gradnorm/test_csv_dumper.py`

- [ ] **Step 8.1: Write failing tests**

Write `tests/gradnorm/test_csv_dumper.py`:
```python
import csv
import datetime as dt
import os

import pytest

from projects.mmdet3d_plugin.core.gradnorm.csv_dumper import GradNormCSVDumper


def _row_dict(path, step):
    return {
        "step": step,
        "epoch": 0,
        "iter_in_epoch": step,
        "phase_id": 3,
        "lr_model": 1e-4,
        "w_det": 1.2, "w_map": 2.1, "w_motion": 0.3, "w_plan": 0.9, "w_ego": 0.5,
        "grad_norm_det": 0.1, "grad_norm_map": 0.2, "grad_norm_motion": 0.3,
        "grad_norm_plan": 0.4, "grad_norm_ego": 0.5,
        "rt_det": 1.0, "rt_map": 1.1, "rt_motion": 0.9, "rt_plan": 1.0, "rt_ego": 1.0,
        "L_det": 0.1, "L_map": 0.2, "L_motion": 0.3, "L_plan": 0.4, "L_ego": 0.5,
        "L0_det": 0.1, "L0_map": 0.2, "L0_motion": 0.3, "L0_plan": 0.4, "L0_ego": 0.5,
        "weighted_L_det": 0.12, "weighted_L_map": 0.42, "weighted_L_motion": 0.09,
        "weighted_L_plan": 0.36, "weighted_L_ego": 0.25,
        "contribution_frac_det": 0.1, "contribution_frac_map": 0.3,
        "contribution_frac_motion": 0.1, "contribution_frac_plan": 0.3,
        "contribution_frac_ego": 0.2,
        "grad_norm_share_det": 0.1, "grad_norm_share_map": 0.2,
        "grad_norm_share_motion": 0.3, "grad_norm_share_plan": 0.2,
        "grad_norm_share_ego": 0.2,
        "w_ratio_det": 1.0, "w_ratio_map": 1.05, "w_ratio_motion": 0.95,
        "w_ratio_plan": 1.0, "w_ratio_ego": 1.0,
        "gn_loss": 0.01,
        "timestamp_iso": dt.datetime.utcnow().isoformat(),
    }


def test_dumper_writes_header_then_rows(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    d = GradNormCSVDumper(str(path), rank=0)
    d.append(_row_dict(str(path), 0))
    d.append(_row_dict(str(path), 1))

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert int(rows[0]["step"]) == 0
    assert int(rows[1]["step"]) == 1
    # All columns of row_dict present
    expected_cols = set(_row_dict(str(path), 0).keys())
    assert set(reader.fieldnames) == expected_cols


def test_dumper_rank_nonzero_is_noop(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    d = GradNormCSVDumper(str(path), rank=1)
    d.append(_row_dict(str(path), 0))
    assert not os.path.exists(path)


def test_dumper_rolls_existing_file(tmp_path):
    path = tmp_path / "gradnorm_log.csv"
    path.write_text("stale\n")
    GradNormCSVDumper(str(path), rank=0, roll_existing=True)
    # original should be renamed with timestamp suffix, path should not exist yet
    assert not path.exists()
    rolled = list(tmp_path.glob("gradnorm_log_*.csv"))
    assert len(rolled) == 1
```

- [ ] **Step 8.2: Run to fail**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_csv_dumper.py -v
```
Expected: `ModuleNotFoundError`.

- [ ] **Step 8.3: Implement dumper**

Write `projects/mmdet3d_plugin/core/gradnorm/csv_dumper.py`:
```python
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional


class GradNormCSVDumper:
    """Append-mode CSV writer used only on rank 0.

    Columns are inferred from the first appended row (order preserved via
    Python dict insertion order). If a pre-existing file is present and
    ``roll_existing`` is True, it is renamed to ``<stem>_<timestamp>.<ext>``
    before writing begins.
    """

    def __init__(
        self,
        path: str,
        rank: int = 0,
        roll_existing: bool = True,
    ) -> None:
        self._path = path
        self._rank = int(rank)
        self._active = self._rank == 0
        self._fieldnames: Optional[List[str]] = None

        if self._active and roll_existing and os.path.exists(path):
            stem, ext = os.path.splitext(path)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(path, f"{stem}_{stamp}{ext}")

    def append(self, row: Dict[str, object]) -> None:
        if not self._active:
            return
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        new_file = not os.path.exists(self._path)
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self._path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerow(row)
```

- [ ] **Step 8.4: Run to pass**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_csv_dumper.py -v
```
Expected: 3 passed.

- [ ] **Step 8.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/gradnorm/csv_dumper.py \
  tests/gradnorm/test_csv_dumper.py
```

---

## Task 9: `SparseDetector.__init__` & `forward_train` integration

**Files:**
- Modify: `projects/mmdet3d_plugin/models/sparse_detector.py`

- [ ] **Step 9.1: Add `gradnorm` kwarg and instance wiring**

Open `projects/mmdet3d_plugin/models/sparse_detector.py`. Find the `__init__` signature of `SparseDetector` and add a `gradnorm=None` keyword. Example (replace the existing `__init__` keyword list minimally — do not rewrite body):

Locate the first line of `__init__` that looks like `def __init__(self, ...)`. Immediately below the existing keyword arguments, add:
```python
        gradnorm=None,
```

Then at the **very end** of the `__init__` body (after existing initialization), add:
```python
        if gradnorm is not None:
            from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter
            from projects.mmdet3d_plugin.core.gradnorm.csv_dumper import (
                GradNormCSVDumper,
            )
            self.gradnorm = GradNormLossWeighter(**gradnorm)
            self._gradnorm_csv_path_template = "gradnorm_log.csv"
            self._gradnorm_csv_dumper = None  # lazy-init in _maybe_dump_gn_csv
        else:
            self.gradnorm = None
            self._gradnorm_csv_dumper = None
```

- [ ] **Step 9.2: Extend `forward_train`**

Replace the existing `forward_train` method body with the version below (keep the signature identical):
```python
    def forward_train(self, img, **data):
        feature_maps, depths = self.extract_feat(img, True, data)
        if "fut_img" in data and self.training:
            data = self.extract_fut_feat(img, feature_maps, data)
        model_outs = self.head(img, feature_maps, data)
        output = self.head.loss(model_outs, data)
        if depths is not None and "gt_depth" in data:
            output["loss_dense_depth"] = self.depth_branch.loss(
                depths, data["gt_depth"]
            )

        if self.gradnorm is None:
            return output

        task_names = self.gradnorm.task_names
        task_prefixes = {t + "_loss_" for t in task_names}

        task_losses = self._aggregate_task_losses(output, task_names)
        shared_params = self.head.onedecoder_head.collect_ffn_last_fc_params()
        weighted, gn_log = self.gradnorm(task_losses, shared_params)

        output = self._rename_as_monitor(output, task_prefixes)
        output["loss_gradnorm_total"] = weighted

        device = weighted.device
        for k, v in gn_log.items():
            output[f"gn_{k}"] = torch.as_tensor(
                v, device=device, dtype=torch.float32
            )

        self._maybe_dump_gn_csv(gn_log, task_losses, data)
        return output
```

- [ ] **Step 9.3: Add `_maybe_dump_gn_csv` helper**

At the bottom of the class (after `aug_test` or the last method), add:
```python
    def _maybe_dump_gn_csv(self, gn_log, task_losses, data):
        """Write one CSV row per iter on rank 0.

        ``gn_log`` is the dict returned by ``GradNormLossWeighter.forward``.
        ``task_losses`` are the pre-weight scalar tensors.
        ``data`` may contain ``meta`` with epoch/iter info (mmcv sets it).
        """
        try:
            import torch.distributed as dist
            rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        except Exception:
            rank = 0

        # Lazy dumper init (need work_dir from config; placed under CWD fallback)
        if self._gradnorm_csv_dumper is None and rank == 0:
            import os
            work_dir = os.environ.get("HIPAD_WORK_DIR") or os.getcwd()
            csv_path = os.path.join(work_dir, self._gradnorm_csv_path_template)
            from projects.mmdet3d_plugin.core.gradnorm.csv_dumper import (
                GradNormCSVDumper,
            )
            self._gradnorm_csv_dumper = GradNormCSVDumper(
                csv_path, rank=0, roll_existing=True
            )
        if rank != 0 or self._gradnorm_csv_dumper is None:
            return

        phase_id = int(gn_log.get("phase_id", 0))
        step = int(gn_log.get("step", 0))

        # Derived metrics (NaN when data not ready)
        task_names = self.gradnorm.task_names
        w_vec = self.gradnorm.w.detach().cpu()
        init_w_sum = float(self.gradnorm.init_w_sum.item())
        init_w_vec = w_vec  # approximation: we keep init values as buffer elsewhere if needed
        # Simpler: use config-stored init weights. We cached them on the weighter:
        # (see Task 1; we re-derive from init_w_sum and current w only when no
        # dedicated buffer exists). To avoid drift, store init_weights on the
        # weighter (Task 9.4 follow-up).

        # weighted_L_i is already in gn_log when active; else recompute from current w
        weighted_vals = []
        for n in task_names:
            if f"weighted_L_{n}" in gn_log:
                weighted_vals.append(float(gn_log[f"weighted_L_{n}"]))
            else:
                L = float(task_losses[n].detach().cpu().item())
                weighted_vals.append(float(w_vec[task_names.index(n)].item()) * L)
        w_sum_iter = sum(weighted_vals) if weighted_vals else 1.0
        contribution_frac = [v / w_sum_iter if w_sum_iter > 0 else float("nan")
                             for v in weighted_vals]

        grad_norm_vals = [gn_log.get(f"grad_norm_{n}", float("nan"))
                          for n in task_names]
        gn_sum = sum(v for v in grad_norm_vals if v == v)  # sum excluding NaN
        grad_norm_share = [(v / gn_sum) if (gn_sum and v == v) else float("nan")
                           for v in grad_norm_vals]

        w_ratio = [
            float(w_vec[i].item()) / float(self.gradnorm.init_weights_cached[i])
            if getattr(self.gradnorm, "init_weights_cached", None) is not None
            else float("nan")
            for i in range(len(task_names))
        ]

        # Epoch / iter / lr from mmcv runner metadata when available
        meta = data.get("meta") if isinstance(data, dict) else None
        epoch = -1
        iter_in_epoch = -1
        lr_model = float("nan")
        try:
            from mmcv.runner import get_dist_info
            # mmcv ≥1.x exposes runner state via `self._iter` etc; simplest is to
            # fall back to step_counter and ENV for epoch.
            epoch = int(os.environ.get("HIPAD_CURRENT_EPOCH", "-1"))
            iter_in_epoch = int(os.environ.get("HIPAD_CURRENT_ITER", str(step)))
            lr_model = float(os.environ.get("HIPAD_CURRENT_LR", "nan"))
        except Exception:
            pass

        from datetime import datetime
        row = {
            "step": step,
            "epoch": epoch,
            "iter_in_epoch": iter_in_epoch,
            "phase_id": phase_id,
            "lr_model": lr_model,
        }
        for i, n in enumerate(task_names):
            row[f"w_{n}"] = float(w_vec[i].item())
        for i, n in enumerate(task_names):
            row[f"grad_norm_{n}"] = grad_norm_vals[i]
        for i, n in enumerate(task_names):
            row[f"rt_{n}"] = gn_log.get(f"rt_{n}", float("nan"))
        for i, n in enumerate(task_names):
            row[f"L_{n}"] = gn_log.get(
                f"L_{n}", float(task_losses[n].detach().cpu().item())
            )
        for i, n in enumerate(task_names):
            row[f"L0_{n}"] = gn_log.get(f"L0_{n}", float("nan"))
        for i, n in enumerate(task_names):
            row[f"weighted_L_{n}"] = weighted_vals[i]
        for i, n in enumerate(task_names):
            row[f"contribution_frac_{n}"] = contribution_frac[i]
        for i, n in enumerate(task_names):
            row[f"grad_norm_share_{n}"] = grad_norm_share[i]
        for i, n in enumerate(task_names):
            row[f"w_ratio_{n}"] = w_ratio[i]
        row["gn_loss"] = gn_log.get("gn_loss", float("nan"))
        row["timestamp_iso"] = datetime.utcnow().isoformat()

        self._gradnorm_csv_dumper.append(row)
```

- [ ] **Step 9.4: Cache init_weights on the weighter for w_ratio derivation**

Open `projects/mmdet3d_plugin/core/gradnorm/weighter.py`. Inside `__init__`, just after `self.w = nn.Parameter(...)`, add:
```python
        self.init_weights_cached = list(float(v) for v in init_weights)
```

- [ ] **Step 9.5: Run all existing tests to confirm nothing regressed**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/ -v
```
Expected: all prior tests pass (12 passed).

- [ ] **Step 9.6: Smoke-check detector file compiles**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "import ast, pathlib; ast.parse(pathlib.Path('projects/mmdet3d_plugin/models/sparse_detector.py').read_text())" && echo "OK"
```
Expected: `OK`.

- [ ] **Step 9.7: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/models/sparse_detector.py \
  projects/mmdet3d_plugin/core/gradnorm/weighter.py
```

---

## Task 10: Training-runner metadata surface (epoch / iter / lr)

**Goal:** `_maybe_dump_gn_csv` currently reads epoch / iter / lr from env vars for portability. We wire them via a light-weight mmcv hook so the CSV shows real numbers.

**Files:**
- Create: `projects/mmdet3d_plugin/core/hooks/gradnorm_metadata_hook.py`
- Modify: `projects/mmdet3d_plugin/core/hooks/__init__.py`

- [ ] **Step 10.1: Inspect existing hooks directory to confirm `__init__.py` exports**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && cat projects/mmdet3d_plugin/core/hooks/__init__.py 2>&1 | head -30
```
Note the current exports; we'll add one line without disturbing them.

- [ ] **Step 10.2: Write hook**

Write `projects/mmdet3d_plugin/core/hooks/gradnorm_metadata_hook.py`:
```python
import os

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GradNormMetadataHook(Hook):
    """Publishes epoch / iter_in_epoch / current model lr into env vars so
    ``SparseDetector._maybe_dump_gn_csv`` can record them without threading
    the runner object through forward_train.
    """

    def before_train_iter(self, runner):
        os.environ["HIPAD_CURRENT_EPOCH"] = str(int(runner.epoch))
        os.environ["HIPAD_CURRENT_ITER"] = str(int(runner.iter))
        lr = None
        try:
            lr = runner.current_lr()[0]
        except Exception:
            pass
        if lr is not None:
            os.environ["HIPAD_CURRENT_LR"] = f"{float(lr):.6e}"
        os.environ["HIPAD_WORK_DIR"] = str(getattr(runner, "work_dir", os.getcwd()))
```

- [ ] **Step 10.3: Export hook from `__init__.py`**

Open `projects/mmdet3d_plugin/core/hooks/__init__.py`. Add (at the bottom, preserving existing content):
```python
from .gradnorm_metadata_hook import GradNormMetadataHook  # noqa: F401
```

- [ ] **Step 10.4: Smoke compile**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "from projects.mmdet3d_plugin.core.hooks import GradNormMetadataHook; print(GradNormMetadataHook)"
```
Expected: a class object printed, no import error.

- [ ] **Step 10.5: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  projects/mmdet3d_plugin/core/hooks/gradnorm_metadata_hook.py \
  projects/mmdet3d_plugin/core/hooks/__init__.py
```

---

## Task 11: Replace E9 config with the GradNorm-enabled version

**Files:**
- Modify (replace): `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py`

- [ ] **Step 11.1: Back up current E9 placeholder**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && cp projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py /tmp/E9_placeholder_backup.py
```

- [ ] **Step 11.2: Write replacement file**

Write `projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py`:
```python
# ──────────────────────────────────────────────────────────────
# E9: Stage2 training with GradNorm (ICML 2018) on E1 ckpt.
# Base: E2_E1_stage2_18ep.py
# Delta: `model.gradnorm` added; custom_hooks adds GradNormMetadataHook.
# Tasks (5): det, map, motion, plan, ego   (depth kept as fixed auxiliary)
# Shared W: last Linear.weight of every AsymmetricFFN in SparseOneDecoder
#           (6 tensors: 1 single-frame + 5 temporal)
# α=1.5, lr_w=2.5e-2, warmup=500 iter, pivot N=50 iter, update_every=1
# Data: nuscenes_infos_train_1_3_seed0.pkl (1/3 seed0 split)
# Load: ckpts/nusc_stage1_e1.pth
# See: docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md
# ──────────────────────────────────────────────────────────────

log_level = "INFO"
dist_params = dict(backend="nccl")

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"
work_dir = "work_dirs/exp/E9_E2_E1_stage2_18ep_GN"

version = 'trainval'
length = {'trainval': 28130, 'mini': 323}

num_gpus = 2
batch_size = 6
num_iters_per_epoch = int(length[version] // (num_gpus * batch_size))
num_epochs = 18
checkpoint_epoch_interval = 1

checkpoint_config = dict(interval=num_iters_per_epoch * checkpoint_epoch_interval,
                         max_keep_ckpts=-1)
wandb_project = "hipad"
wandb_name = "E9_E2_E1_stage2_18ep_GN"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(entity="e2ekd", project=wandb_project, name=wandb_name),
            by_epoch=False,
        ),
    ],
)
load_from = "/home/yongjae/e2e/HiP-AD/ckpts/nusc_stage1_e1.pth"
resume_from = None
workflow = [("train", 1)]
fp16 = dict(loss_scale=32.0)
input_shape = (704, 256)
num_cams = 6

# det & map
det_class_names = [
    "car", "truck", "construction_vehicle", "bus", "trailer", "barrier",
    "motorcycle", "bicycle", "pedestrian", "traffic_cone",
]
map_class_names = ["ped_crossing", "divider", "boundary"]

num_det_classes = len(det_class_names)
num_map_classes = len(map_class_names)

map_roi_size = (30, 60)
map_num_pts = 20

# traj
fut_ts = 12
fut_mode = 6
ego_fut_ts = 6
ego_fut_cmd = 3
ego_fut_mode = 6
ego_status_dims = 6

# model
embed_dims = 256
num_groups = 8
num_decoder = 6
num_single_frame_decoder = 1
use_deformable_func = True
strides = [4, 8, 16, 32]
num_levels = len(strides)
num_depth_layers = 3
drop_out = 0.1
decouple_attn = True
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

# temporal
temporal = True
temporal_det = True
temporal_map = True
temporal_ego = True
temporal_plan = True

# tasks
task_config = dict(with_onedecoder=True)

task_select = ["det", "map", "plan", "ego", "motion"]
query_select = ["det", "map", "plan", "ego"]

single_frame_layer = ["concat", "gnn", "inter_gnn", "norm", "split", "deformable",
                      "concat", "ffn", "norm", "split", "refine"]
temporal_frame_layer = ["concat", "temp_gnn", "gnn", "inter_gnn", "norm", "split",
                        "deformable", "concat", "ffn", "norm", "split", "refine"]

operation_order = single_frame_layer * num_single_frame_decoder + \
                  temporal_frame_layer * (num_decoder - num_single_frame_decoder)

# anchors
anchor_paths = {
    "det": "data_nusc/kmeans/kmeans_det_900.npy",
    "map": "data_nusc/kmeans/kmeans_map_100.npy",
    "motion": f"data_nusc/kmeans/kmeans_motion_{fut_mode}.npy",
}

plan_anchor_paths = f"data_nusc/kmeans/kmeans_plan_{ego_fut_mode}.npy"
plan_speed_refer = None
plan_anchor_refer = ("temp", "2hz")
plan_anchor_types = [("temp", "2hz")]


model = dict(
    type="SparseDetector",
    use_grid_mask=True,
    use_deformable_func=use_deformable_func,
    img_backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        frozen_stages=-1,
        norm_eval=False,
        style="pytorch",
        with_cp=True,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(type="BN", requires_grad=True),
        pretrained="ckpts/resnet50-19c8e357.pth",
    ),
    img_neck=dict(
        type="FPN",
        num_outs=num_levels,
        start_level=0,
        out_channels=embed_dims,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
        norm_cfg=dict(type="BN", requires_grad=True),
        no_norm_on_lateral=True,
        in_channels=[256, 512, 1024, 2048],
    ),
    depth_branch=dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
    ),
    head=dict(
        type="SparseHead",
        task_config=task_config,
        evaluate_bench2dive=False,
        onedecoder_head=dict(
            type="SparseOneDecoder",
            task_select=task_select,
            query_select=query_select,
            operation_order=operation_order,
            num_single_frame_decoder=num_single_frame_decoder,
            plan_speed_refer=plan_speed_refer,
            plan_anchor_refer=plan_anchor_refer,
            with_command_embed=True,
            with_target_point_embed=False,
            with_supervise_ego_status=True,
            num_command=ego_fut_cmd,
            with_ego_instance_feature=True,
            with_incremental_plan_refine=True,
            motion_anchor=anchor_paths["motion"],
            cls_threshold_to_reg=0.05,
            det_instance_bank=dict(
                type="InstanceBank",
                num_anchor=900,
                embed_dims=embed_dims,
                anchor=anchor_paths["det"],
                anchor_handler=dict(type="SparseBox3DKeyPointsGenerator"),
                num_temp_instances=600 if temporal_det else -1,
                confidence_decay=0.6,
                feat_grad=False,
                class_names=det_class_names,
                zero_velocity_classes=["barrier", "traffic_cone"],
            ),
            map_instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor=anchor_paths["map"],
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0 if temporal_map else -1,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            ego_instance_bank=dict(
                type="EgoInstanceBank",
                anchor_type="nus",
                embed_dims=embed_dims,
                num_temp_instances=1 if temporal_ego else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
                plan_anchor=plan_anchor_paths,
            ),
            plan_instance_bank=dict(
                type="PlanningInstanceBank",
                embed_dims=embed_dims,
                ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd,
                ego_fut_mode=ego_fut_mode,
                num_temp_mode=ego_fut_mode if temporal_plan else -1,
                feature_map_scale=(input_shape[1] / strides[-1], input_shape[0] / strides[-1]),
                anchor_paths=plan_anchor_paths,
                anchor_types=plan_anchor_types,
            ),
            det_anchor_encoder=dict(
                type="SparseBox3DEncoder",
                vel_dims=3,
                embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
                mode="cat" if decouple_attn else "add",
                output_fc=not decouple_attn,
                in_loops=1,
                out_loops=4 if decouple_attn else 2,
            ),
            map_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=map_num_pts,
                return_points_embed=True,
            ),
            plan_anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=ego_fut_ts,
                return_points_embed=True,
            ),
            custom_op=dict(type="CustomOperation"),
            temp_graph_model=dict(
                type="TemporalSeparateAttention",
                query_select=query_select,
                query_list=[["det"], ["map"], ["plan", "ego"]],
                key_list=[["det"], ["map"], ["det", "map"]],
                decouple_list=[True, False, False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ) if temporal else None,
            graph_model=dict(
                type="SeparateAttention",
                query_select=query_select,
                separate_list=[["det"], ["map"]],
                decouple_list=[True, False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims * 2,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ),
            inter_graph_model=dict(
                type="InteractiveAttention",
                query_select=query_select,
                query_list=[["plan", "ego"]],
                key_list=[["det", "map"]],
                decouple_list=[False],
                attn=[
                    dict(type="MultiheadFlashAttention", embed_dims=embed_dims,
                         num_heads=num_groups, batch_first=True, dropout=drop_out),
                ],
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            det_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=6,
                    fix_scale=[[0, 0, 0], [0.45, 0, 0], [-0.45, 0, 0],
                               [0, 0.45, 0], [0, -0.45, 0],
                               [0, 0, 0.45], [0, 0, -0.45]],
                ),
            ),
            map_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=map_num_pts,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            ego_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparseBox3DKeyPointsGenerator",
                    num_learnable_pts=12, fix_scale=[[0.45, 0, 0]],
                ),
            ),
            plan_deformable=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims, num_groups=num_groups, num_levels=num_levels,
                num_cams=6, attn_drop=0.15, use_deformable_func=use_deformable_func,
                use_camera_embed=True, residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims, num_sample=ego_fut_ts,
                    num_learnable_pts=3, fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=-1.84023,
                ),
            ),
            det_refine_layer=dict(
                type="SparseBox3DRefinementModule",
                embed_dims=embed_dims, num_cls=num_det_classes,
                refine_yaw=True, with_quality_estimation=True,
            ),
            map_refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims, num_sample=map_num_pts, num_cls=num_map_classes,
            ),
            ego_refine_layer=dict(
                type="EgoStatusRefinementModule",
                embed_dims=embed_dims, status_dims=ego_status_dims,
            ),
            plan_refine_layer=dict(
                type="SparsePlanAlignRefinementModule",
                embed_dims=embed_dims, ego_fut_ts=ego_fut_ts,
                ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
                anchor_types=plan_anchor_types,
            ),
            motion_refine_layer=dict(
                type="SparseMotionRefinementModule",
                embed_dims=embed_dims, fut_ts=fut_ts, fut_mode=fut_mode,
            ),
            det_sampler=dict(
                type="SparseBox3DTarget",
                num_dn_groups=0, num_temp_dn_groups=0,
                dn_noise_scale=[2.0] * 3 + [0.5] * 7,
                max_dn_gt=32, add_neg_dn=True,
                cls_weight=2.0, box_weight=0.25,
                reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
                cls_wise_reg_weights={
                    det_class_names.index("traffic_cone"):
                        [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                },
            ),
            map_sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type="HungarianLinesAssigner",
                    cost=dict(
                        type="MapQueriesCost",
                        cls_cost=dict(type="FocalLossCost", weight=1.0),
                        reg_cost=dict(type="LinesL1Cost", weight=10.0, beta=0.01, permute=True),
                    ),
                ),
                num_cls=num_map_classes, num_sample=map_num_pts, roi_size=map_roi_size,
            ),
            plan_sampler=dict(
                type="SparsePlanTarget",
                ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
            ),
            align_sampler=dict(
                type="AlignPlanTarget",
                ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd, ego_fut_mode=ego_fut_mode,
            ),
            motion_sampler=dict(type="SparseMotionTarget"),
            loss_det_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0),
            loss_det_reg=dict(type="SparseBox3DLoss",
                              loss_box=dict(type="L1Loss", loss_weight=0.25),
                              loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True),
                              loss_yawness=dict(type="GaussianFocalLoss")),
            loss_map_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
            loss_map_reg=dict(type="SparseLineLoss",
                              loss_line=dict(type="LinesL1Loss", loss_weight=10.0, beta=0.01),
                              num_sample=map_num_pts, roi_size=map_roi_size),
            loss_ego_status=dict(type="L1Loss", loss_weight=1.0),
            loss_plan_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.5),
            loss_plan_reg=dict(type="L1Loss", loss_weight=1.0),
            loss_motion_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.2),
            loss_motion_reg=dict(type="L1Loss", loss_weight=0.2),
            det_reg_weights=[2.0] * 3 + [1.0] * 7,
            map_reg_weights=[1.0] * 40,
            det_decoder=dict(type="SparseBox3DDecoder"),
            map_decoder=dict(type="SparsePoint3DDecoder"),
            plan_decoder=dict(type="SparsePlanDecoder",
                              ego_fut_ts=ego_fut_ts, ego_fut_cmd=ego_fut_cmd,
                              ego_fut_mode=ego_fut_mode, ego_vehicle="nus",
                              anchor_types=plan_anchor_types, anchor_refer=plan_anchor_refer,
                              speed_refer=plan_speed_refer, with_rescore=True),
            motion_decoder=dict(type="SparseMotionDecoder"),
        ),
    ),

    # ── GradNorm (NEW) ──
    gradnorm=dict(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[4.25, 11.0, 0.4, 1.5, 1.0],
        alpha=1.5,
        lr=2.5e-2,
        update_after_step=500,
        pivot_warmup_steps=50,
        update_every=1,
        clamp_min=1e-4,
    ),
)

# ================== data ========================
dataset_type = "NuScenes3DDataset"
data_root = "data_nusc/nuscenes/"
eval_data_root = "data_nusc/infos/nuscenes/"
anno_root = "data_nusc/infos/" if version == 'trainval' else "data_nusc/infos/mini/"
file_client_args = dict(backend="disk")

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="LoadPointsFromFile", coord_type="LIDAR",
         load_dim=5, use_dim=5, file_client_args=file_client_args),
    dict(type="ResizeCropFlipImage"),
    dict(type="MultiScaleDepthMapGenerator", downsample=strides[:num_depth_layers]),
    dict(type="BBoxRotation"),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=False,
         normalize=False, sample_num=map_num_pts, permute=True),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh", "gt_depth",
               "focal", "gt_bboxes_3d", "gt_labels_3d", "gt_map_labels",
               "gt_map_pts", "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_fut_trajs", "gt_ego_fut_masks", "gt_ego_fut_cmd",
               "gt_ego_fut_trajs_2hz", "gt_ego_fut_masks_2hz",
               "ego_status", "ego_status_mask"],
         meta_keys=["T_global", "T_global_inv", "timestamp", "instance_id"]),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="ResizeCropFlipImage"),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type="Collect",
         keys=["img", "timestamp", "projection_mat", "image_wh",
               "ego_status", "gt_ego_fut_cmd"],
         meta_keys=["T_global", "T_global_inv", "timestamp"]),
]

eval_pipeline = [
    dict(type="CircleObjectRangeFilter", class_dist_thred=[55] * len(det_class_names)),
    dict(type="InstanceNameFilter", classes=det_class_names),
    dict(type="VectorizeMap", roi_size=map_roi_size, simplify=True, normalize=False),
    dict(type="Collect",
         keys=["vectors", "gt_bboxes_3d", "gt_labels_3d",
               "gt_agent_fut_trajs", "gt_agent_fut_masks",
               "gt_ego_fut_trajs", "gt_ego_fut_masks",
               "gt_ego_fut_cmd", "fut_boxes"],
         meta_keys=["token", "timestamp"]),
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False,
    use_map=False, use_external=False,
)

nusc_version = "v1.0-trainval" if version == "trainval" else "v1.0-mini"
data_basic_config = dict(
    type=dataset_type, data_root=data_root, classes=det_class_names,
    map_classes=map_class_names, ego_status_dims=ego_status_dims,
    modality=input_modality, version=nusc_version, work_dir=work_dir,
)

eval_config = dict(
    **data_basic_config, eval_data_root=eval_data_root,
    ann_file=anno_root + "nuscenes_infos_val.pkl",
    pipeline=eval_pipeline, test_mode=True,
)

data_aug_conf = {
    "resize_lim": (0.40, 0.47),
    "final_dim": input_shape[::-1],
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (-5.4, 5.4),
    "H": 900, "W": 1600, "rand_flip": True, "rot3d_range": [0, 0],
}

data = dict(
    samples_per_gpu=batch_size, workers_per_gpu=batch_size,
    train=dict(
        **data_basic_config,
        ann_file=anno_root + "nuscenes_infos_train_1_3_seed0.pkl",
        pipeline=train_pipeline, test_mode=False,
        data_aug_conf=data_aug_conf, with_seq_flag=True,
        sequences_split_num=2, keep_consistent_seq_aug=True,
    ),
    val=dict(
        **data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline, data_aug_conf=data_aug_conf,
        test_mode=True, eval_config=eval_config,
    ),
    test=dict(
        **data_basic_config, ann_file=anno_root + "nuscenes_infos_val.pkl",
        pipeline=test_pipeline, data_aug_conf=data_aug_conf,
        test_mode=True, eval_config=eval_config,
    ),
)

# ================== training ========================
optimizer = dict(
    type="AdamW", lr=1e-4, weight_decay=0.001,
    paramwise_cfg=dict(custom_keys={"img_backbone": dict(lr_mult=0.5)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=25, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing", warmup="linear",
    warmup_iters=500, warmup_ratio=1.0 / 3, min_lr_ratio=1e-3,
)
runner = dict(type="IterBasedRunner",
              max_iters=num_iters_per_epoch * num_epochs)

# ================== eval ========================
eval_mode = dict(
    with_det=True, with_tracking=False, with_map=True,
    with_motion=True, with_planning=True,
    tracking_threshold=0.2, motion_threshhold=0.2,
)
evaluation = dict(
    interval=num_iters_per_epoch * checkpoint_epoch_interval * 2,
    jsonfile_prefix="val/", eval_mode=eval_mode, out_dir="val_vis",
)

custom_hooks = [
    dict(type="WandbValVisHook",
         vis_dir="val_vis/visual", max_images=8, interval=1, priority="LOWEST"),
    dict(type="GradNormMetadataHook", priority="LOWEST"),
]
```

- [ ] **Step 11.3: Smoke load the config**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "from mmcv import Config; c = Config.fromfile('projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py'); print('ok', c.model.gradnorm)"
```
Expected: `ok {'task_names': [...], 'init_weights': [...], 'alpha': 1.5, ...}`.

- [ ] **Step 11.4: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py
```

---

## Task 12: Detector-level dry-run smoke test (no real nuScenes data)

**Goal:** Verify `SparseDetector.forward_train` with `gradnorm` enabled produces the expected output-dict keys on a tiny synthetic input, without requiring the full nuScenes pipeline. This surfaces wiring bugs early.

**Files:**
- Create: `tests/gradnorm/test_detector_forward_train.py`

- [ ] **Step 12.1: Write dry-run test**

Write `tests/gradnorm/test_detector_forward_train.py`:
```python
"""Dry-run smoke test: calls SparseDetector.forward_train path logic with a
fake head.loss() output and a fake shared-params provider. Heavy model init
is NOT executed; we directly exercise the post-head branching.
"""

import types

import pytest
import torch

from projects.mmdet3d_plugin.models.sparse_detector import SparseDetector
from projects.mmdet3d_plugin.core.gradnorm import GradNormLossWeighter


class _FakeOneDecoder(torch.nn.Module):
    def __init__(self, n_shared=2, dim=4):
        super().__init__()
        self.shared = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(dim, dim, requires_grad=True))
             for _ in range(n_shared)]
        )

    def collect_ffn_last_fc_params(self):
        return list(self.shared)


class _FakeHead(torch.nn.Module):
    def __init__(self, fake_decoder):
        super().__init__()
        self.onedecoder_head = fake_decoder


def _make_output(decoder):
    x = torch.randn(decoder.shared[0].shape[-1])
    def dep(scale):
        return (decoder.shared[0] @ x).sum() * 1e-6 + float(scale)
    return {
        "det_loss_cls": dep(0.5),
        "det_loss_box": dep(0.3),
        "det_loss_cns": dep(0.1),
        "det_loss_yns": dep(0.1),
        "map_loss_cls": dep(0.4),
        "map_loss_line": dep(0.6),
        "motion_loss_cls": dep(0.1),
        "motion_loss_reg": dep(0.2),
        "plan_loss_temp_cls": dep(0.3),
        "plan_loss_temp_reg": dep(0.4),
        "ego_loss_status": dep(0.8),
    }


def test_forward_train_gradnorm_path_produces_expected_keys():
    # Build a GradNormLossWeighter with update_after_step=0, pivot=1
    weighter = GradNormLossWeighter(
        task_names=["det", "map", "motion", "plan", "ego"],
        init_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
        alpha=1.0, lr=1e-2,
        update_after_step=0, pivot_warmup_steps=1, update_every=1,
    )

    fake_decoder = _FakeOneDecoder(n_shared=2, dim=4)
    fake_head = _FakeHead(fake_decoder)

    # Build a "detector" by-pass: we only need an object exposing
    # self.gradnorm, self.head, self.depth_branch=None-like behavior, and
    # the helper methods. Compose via types.SimpleNamespace + bound methods.
    detector = types.SimpleNamespace()
    detector.gradnorm = weighter
    detector.head = fake_head
    detector.depth_branch = None
    detector._gradnorm_csv_dumper = None
    detector._gradnorm_csv_path_template = "gradnorm_log.csv"

    # Bind the staticmethods and instance methods from SparseDetector
    detector._aggregate_task_losses = SparseDetector._aggregate_task_losses
    detector._rename_as_monitor   = SparseDetector._rename_as_monitor
    # _maybe_dump_gn_csv uses `self.gradnorm.*`; we rebind manually
    detector._maybe_dump_gn_csv   = SparseDetector._maybe_dump_gn_csv.__get__(detector, types.SimpleNamespace)

    # Mock the head.loss + depth produce
    output = _make_output(fake_decoder)

    # Run the gradnorm branch logic manually (the forward_train body)
    task_names = detector.gradnorm.task_names
    task_prefixes = {t + "_loss_" for t in task_names}
    task_losses = detector._aggregate_task_losses(output, task_names)
    shared_params = detector.head.onedecoder_head.collect_ffn_last_fc_params()
    weighted, gn_log = detector.gradnorm(task_losses, shared_params)

    output = detector._rename_as_monitor(output, task_prefixes)
    output["loss_gradnorm_total"] = weighted
    for k, v in gn_log.items():
        output[f"gn_{k}"] = torch.as_tensor(
            v, device=weighted.device, dtype=torch.float32
        )

    # Assertions
    assert "loss_gradnorm_total" in output
    # Original task keys are gone (renamed)
    for k in ["det_loss_cls", "map_loss_line", "ego_loss_status"]:
        assert k not in output
    for k in ["monitor_det_cls", "monitor_map_line", "monitor_ego_status"]:
        assert k in output
    # gn_* logging
    assert any(k.startswith("gn_w_") for k in output)
    # shape sanity: weighted_loss is a scalar tensor
    assert output["loss_gradnorm_total"].ndim == 0
    # 'loss' substring present only in keys that mmcv _parse_losses should sum
    summable = [k for k in output if "loss" in k]
    assert set(summable) == {"loss_gradnorm_total"}  # depth omitted in this dry-run
```

- [ ] **Step 12.2: Run**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/test_detector_forward_train.py -v
```
Expected: 1 passed.

- [ ] **Step 12.3: Run the full test suite once more**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/ -v
```
Expected: all tests green (roughly 15 passed).

- [ ] **Step 12.4: Stage**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git add tests/gradnorm/test_detector_forward_train.py
```

---

## Task 13: End-to-end training dry-run (1 iter, mini data)

**Goal:** Execute the full mmcv pipeline for **one iteration** to verify the config loads, the model builds, data loads, forward-backward runs, and the CSV file is created with expected columns.

This task requires a functioning environment (GPU, nuScenes mini). If the environment is unavailable, mark Step 13.x as `skip-env` and document in the final commit.

**Files:**
- No file changes. Uses existing scripts.

- [ ] **Step 13.1: Make a 1-iter override config**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -c "
import pathlib
src = pathlib.Path('projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py').read_text()
src = src.replace('num_epochs = 18', 'num_epochs = 1')
src = src.replace('length = {\\'trainval\\': 28130, \\'mini\\': 323}',
                  'length = {\\'trainval\\': 28130, \\'mini\\': 323}')
src = src.replace('version = \\'trainval\\'', 'version = \\'mini\\'')
src = src.replace('data_root = \"data_nusc/nuscenes/\"',
                  'data_root = \"data_nusc/nuscenes/\"')
src = src.replace('work_dir = \"work_dirs/exp/E9_E2_E1_stage2_18ep_GN\"',
                  'work_dir = \"work_dirs/exp/E9_dryrun\"')
src = src.replace('update_after_step=500', 'update_after_step=2')
src = src.replace('pivot_warmup_steps=50', 'pivot_warmup_steps=2')
pathlib.Path('projects/configs/experiments/E9_dryrun.py').write_text(src)
print('wrote E9_dryrun.py')
"
```
Expected: `wrote E9_dryrun.py`.

- [ ] **Step 13.2: Launch 1-iter dry run (single GPU, no dist)**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && timeout 1800 python tools/train.py \
  projects/configs/experiments/E9_dryrun.py \
  --no-validate --cfg-options runner.max_iters=5 log_config.interval=1 \
  2>&1 | tee /tmp/gn_dryrun.log | tail -40
```
Expected: No Python exception. `GradNormMetadataHook` initialization line appears. Training log shows `loss_gradnorm_total`, `loss_dense_depth`, and `gn_*` keys. If GPU is unavailable, record "skip-env" and proceed.

- [ ] **Step 13.3: Inspect CSV**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && ls work_dirs/exp/E9_dryrun/ 2>&1; cat work_dirs/exp/E9_dryrun/gradnorm_log.csv | head -5 2>&1 || echo "no csv"
```
Expected: `gradnorm_log.csv` exists; first line is the header; at least one row appears.

- [ ] **Step 13.4: Cleanup dry-run artifacts**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && rm -f projects/configs/experiments/E9_dryrun.py && rm -rf work_dirs/exp/E9_dryrun
```

- [ ] **Step 13.5: Stage (nothing to stage from this task — verification only)**

No new files; skip staging.

---

## Task 14: Consolidate into a single commit

**Files:** all previously staged files

- [ ] **Step 14.1: Verify staged files**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && git status --short
```
Expected staged:
```
A  docs/superpowers/plans/2026-04-22-hipad-stage2-gradnorm-plan.md
A  docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md
A  projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py
A  projects/mmdet3d_plugin/core/gradnorm/__init__.py
A  projects/mmdet3d_plugin/core/gradnorm/csv_dumper.py
A  projects/mmdet3d_plugin/core/gradnorm/shared_params.py
A  projects/mmdet3d_plugin/core/gradnorm/weighter.py
A  projects/mmdet3d_plugin/core/hooks/gradnorm_metadata_hook.py
M  projects/mmdet3d_plugin/core/hooks/__init__.py
M  projects/mmdet3d_plugin/models/sparse_detector.py
M  projects/mmdet3d_plugin/models/sparse_onedecoder.py
A  tests/gradnorm/__init__.py
A  tests/gradnorm/test_csv_dumper.py
A  tests/gradnorm/test_detector_forward_train.py
A  tests/gradnorm/test_detector_helpers.py
A  tests/gradnorm/test_shared_params.py
A  tests/gradnorm/test_weighter.py
```

Also stage design + plan docs if not already:
```bash
cd /home/yongjae/e2e/HiP-AD && git add \
  docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md \
  docs/superpowers/plans/2026-04-22-hipad-stage2-gradnorm-plan.md
```

- [ ] **Step 14.2: Run the full test suite one final time**

Run:
```bash
cd /home/yongjae/e2e/HiP-AD && python -m pytest tests/gradnorm/ -v
```
Expected: all passed.

- [ ] **Step 14.3: Create the single consolidated commit (requires explicit user approval)**

**Important:** Per repo policy the assistant should NOT run `git commit` without the user's explicit "commit" instruction. Ask the user to approve before executing this step.

When approved, run:
```bash
cd /home/yongjae/e2e/HiP-AD && git commit -m "$(cat <<'EOF'
feat(stage2): integrate GradNorm loss balancing (ICML 2018) for HiP-AD

- Add GradNormLossWeighter (core/gradnorm/weighter.py) with:
  - Adam-based w update, sum-to-init-sum renormalization, clamp safety
  - fp32 gradient-norm computation to avoid fp16 underflow
  - Warmup gate (update_after_step=500) and pivot freeze via
    pivot_warmup_steps=50-iter mean of L_i(0)
  - DDP safety: all_reduce mean on w after update and on L0 at freeze
  - Phase-tagged log output consumed by CSV dumper
- Add collect_last_linear_weights helper + SparseOneDecoder method to
  expose the 6 AsymmetricFFN last-Linear weights as shared W
- Wire SparseDetector.forward_train to aggregate task loss by prefix,
  rename original keys to monitor_* so mmcv _parse_losses skips them,
  and surface loss_gradnorm_total + gn_* diagnostics
- GradNormMetadataHook publishes epoch/iter/lr via env vars for CSV dump
- E9_E2_E1_stage2_18ep_GN.py: replace placeholder with GradNorm-enabled
  config (α=1.5, lr_w=2.5e-2, init_weights preserve existing loss_weight
  ratios, 1/3 seed0 training split)
- Tests: unit coverage for weighter (constructor, warmup, pivot, main step),
  shared_params helper, csv_dumper, SparseDetector helpers, and a
  detector-level dry-run smoke test

Spec: docs/superpowers/specs/2026-04-22-hipad-stage2-gradnorm-design.md
Plan: docs/superpowers/plans/2026-04-22-hipad-stage2-gradnorm-plan.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: `git status --short` is empty afterwards (except possibly untracked unrelated files).

---

## Post-Implementation Checklist

- [ ] All `tests/gradnorm/` tests pass.
- [ ] `python -c "from mmcv import Config; Config.fromfile('projects/configs/experiments/E9_E2_E1_stage2_18ep_GN.py')"` succeeds.
- [ ] (If environment permits) 1-iter dry run produces `gradnorm_log.csv` with 5-task columns.
- [ ] Final commit is a single commit containing design doc, plan doc, implementation, and tests.

## Spec → Plan Coverage Matrix (Self-Review)

| Spec section | Covered in tasks |
|---|---|
| §1 Overview / scope | — (design-only, no code) |
| §2 Module structure | Tasks 1, 5, 6, 7, 8, 9, 10, 11 |
| §3 `GradNormLossWeighter` class (state + forward + DDP safety + sum-to-init-sum) | Tasks 1–4 |
| §4 SparseDetector integration (aggregate + rename + total loss plumbing) | Tasks 7, 9 |
| §4.4 `collect_ffn_last_fc_params` | Tasks 5, 6 |
| §5 E9 config delta | Task 11 |
| §6 Logging (wandb + CSV) | Tasks 8, 9.3, 10 |
| §6.5 validation metrics | — (reuses existing `evaluation` hook; no code change) |
| §7 Risks & rollback | — (rollback = `model.gradnorm=None` which the code already honors in Task 9.1/9.2) |
| Go/no-go criteria | Task 13 (dry run observations) |

If any spec requirement lacks a task, add it here before execution.
