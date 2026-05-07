# Gradient Dynamics Diagnostic Framework — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land Phase 1 of the diagnostic framework — six modules (null baseline, distribution diagnostics, bootstrap CI, validity hardening, probe reliability, magnitude dynamics) on the existing HiP-AD analysis pipeline so that every reported metric carries a noise-vs-signal label and the gating decision (Pass A / Pass B / Fail) can be made before Phase 2 work begins.

**Architecture:** Each new module is a standalone Python file under `tools/gradient_analysis/` with pure helpers for unit testing plus a `run_<module>` orchestration entry point. Modifications to existing files (`collector.py`, `conflict.py`, `summary.py`, `probe.py`) preserve the on-disk cache format via a v2 metadata field that v1 readers ignore. All modules are wired into `tools/run_gradient_analysis.py` so a single CLI invocation regenerates the complete report.

**Tech Stack:** Python 3.8+, PyTorch 1.13+, scipy ≥ 1.10, pandas ≥ 1.5, matplotlib ≥ 3.6, `diptest` (new dependency), pytest. Conda env `hipad` at `/home/yongjae/miniconda3/envs/hipad/bin/python`.

**Spec:** [docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework-design.md](../specs/2026-05-02-gradient-dynamics-diagnostic-framework-design.md)

**Conventions for this plan:**
- All paths are relative to `/home/yongjae/e2e/HiP-AD/`.
- `PY=/home/yongjae/miniconda3/envs/hipad/bin/python` when invoking the env's Python.
- `pytest` invocations always run from the repo root with `PYTHONPATH=.`.
- Per the user's preference, **do not commit between tasks** — they will batch-commit at the end of Phase 1. Each task ends with a "Stop and review" checkpoint instead of a `git commit` step.

---

## File Structure

**New files:**
- `tools/gradient_analysis/null_baseline.py` — random-baseline / permutation-test module (Phase 1 #1)
- `tools/gradient_analysis/distribution.py` — KDE + dip-test distribution diagnostics (Phase 1 #2)
- `tools/gradient_analysis/bootstrap.py` — BCa bootstrap CI helper applied to existing summaries (Phase 1 #3)
- `tools/gradient_analysis/magnitude_dynamics.py` — per-task norm slopes + ratio-matrix evolution (Phase 1 #10)
- `tests/gradient_analysis/test_null_baseline.py`
- `tests/gradient_analysis/test_distribution.py`
- `tests/gradient_analysis/test_bootstrap.py`
- `tests/gradient_analysis/test_magnitude_dynamics.py`
- `tests/gradient_analysis/test_collector_validity.py` — exercises #4 B1
- `tests/gradient_analysis/test_conflict_validity.py` — exercises #4 B2 / B3
- `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase1_gating_decision.md` — produced at the end (Task 21)

**Modified files:**
- `requirements_gradient_analysis.txt` — add `diptest`
- `tools/gradient_analysis/collector.py` — add v2 cache field carrying per-(task, param) non-zero masks (B1)
- `tools/gradient_analysis/conflict.py` — add validity-aware columns and Bonferroni-friendly aggregation (B2 / B3)
- `tools/gradient_analysis/summary.py` — partition pseudo-shared groups in the markdown digest (B4)
- `tools/gradient_analysis/probe.py` — bootstrap variance bands on the affinity matrix; α-sensitivity sweep CLI option (#9)
- `tools/gradient_analysis/viz.py` — render CI bands on time-series plots
- `tools/run_gradient_analysis.py` — CLI flags routing into the four new modules and the α sweep
- `configs/gradient_analysis.yaml` — α-sweep section (motion-only)

---

## Task 0: Pre-Flight — Dependency + Sanity Check

**Files:**
- Modify: `requirements_gradient_analysis.txt`

- [ ] **Step 1: Append `diptest` to the requirements file**

Edit `requirements_gradient_analysis.txt` so the final line list reads exactly:

```
pyyaml>=6.0
pandas>=1.5
matplotlib>=3.6
numpy>=1.23
scipy>=1.10
pytest>=7.0
diptest>=0.5
```

- [ ] **Step 2: Install the new dep into the `hipad` env**

Run:

```bash
PY=/home/yongjae/miniconda3/envs/hipad/bin/python
$PY -m pip install 'diptest>=0.5'
```

Expected: `Successfully installed diptest-...`. If the env complains about ABI compatibility, fall back to `pip install diptest --no-build-isolation`.

- [ ] **Step 3: Confirm scipy + diptest import**

Run:

```bash
$PY -c "import scipy.stats as s; from diptest import diptest; print(s.__version__, diptest)"
```

Expected: scipy version printed (≥ 1.10) and a `<function diptest ...>` repr.

- [ ] **Step 4: Confirm existing tests still green**

Run:

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/ -v
```

Expected: every existing test passes (baseline before adding new code).

- [ ] **Step 5: Stop and review**

Confirm dependency added and existing tests green. Do not yet touch analysis code.

---

## Task 1: null_baseline — Sample-Level Shuffle Null + First Failing Test

**Files:**
- Create: `tools/gradient_analysis/null_baseline.py`
- Create: `tests/gradient_analysis/test_null_baseline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/test_null_baseline.py`:

```python
"""Unit tests for the null-baseline / permutation module."""
import numpy as np
import pytest
import torch

from tools.gradient_analysis.null_baseline import sample_shuffle_null


def test_sample_shuffle_null_returns_correct_count():
    rng = np.random.default_rng(0)
    g_a = torch.tensor(rng.standard_normal(size=(8, 16)).astype("float32"))
    g_b = torch.tensor(rng.standard_normal(size=(8, 16)).astype("float32"))
    out = sample_shuffle_null(g_a, g_b, n_repeats=64, seed=1)
    assert out.shape == (64,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= -1.0) & (out <= 1.0))


def test_sample_shuffle_null_zero_when_inputs_proportional():
    """If g_a == g_b for every sample, the shuffle null should still produce
    cosines bounded by ±1 (it cannot be all-1 unless every sample is identical
    in direction). Sanity: shuffling with proportional inputs gives all 1.0."""
    g = torch.ones(4, 8)
    out = sample_shuffle_null(g, g, n_repeats=8, seed=0)
    assert np.allclose(out, 1.0, atol=1e-6)
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: `ImportError: cannot import name 'sample_shuffle_null'` or `ModuleNotFoundError: tools.gradient_analysis.null_baseline`.

- [ ] **Step 3: Create the module with the helper**

Create `tools/gradient_analysis/null_baseline.py`:

```python
"""M-N1 — Random-baseline / permutation-test module (Phase 1 #1).

Generates null distributions of cosine similarity by two independent
procedures so observed cos can be tested against noise:

  * sample_shuffle_null  — within a batch, randomly swap per-sample gradient
                           components between tasks. Tests "is task identity
                           informative?"
  * sign_flip_null       — independently flip the sign of each gradient
                           component with p = 0.5. Tests "is the directional
                           structure informative?"

Both return a numpy array of cosine similarities (one per repeat) bounded in
[-1, 1].
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


EPS = 1e-8


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(a.norm())
    nb = float(b.norm())
    if na < EPS or nb < EPS:
        return float("nan")
    cos = float(torch.dot(a.flatten().to(torch.float64),
                          b.flatten().to(torch.float64))
                / (na * nb))
    return max(-1.0, min(1.0, cos))


def sample_shuffle_null(
    g_a: torch.Tensor,
    g_b: torch.Tensor,
    n_repeats: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Shuffle per-sample task identity within the batch and recompute cos.

    `g_a`, `g_b`: shape (B, ...) where the leading dim indexes samples that
    belong to the same batch but different tasks. Pure helper — no torch.no_grad
    context needed since inputs are already detached.
    """
    if g_a.shape != g_b.shape:
        raise ValueError(f"shape mismatch: {tuple(g_a.shape)} vs {tuple(g_b.shape)}")
    rng = np.random.default_rng(seed)
    bsz = g_a.shape[0]
    out = np.empty(n_repeats, dtype=np.float64)
    a_flat = g_a.reshape(bsz, -1).to(torch.float64)
    b_flat = g_b.reshape(bsz, -1).to(torch.float64)
    for i in range(n_repeats):
        # For each sample, with p = 0.5 swap the (a, b) assignment.
        swap_mask = torch.from_numpy(rng.integers(0, 2, size=bsz).astype(bool))
        a_perm = torch.where(swap_mask[:, None], b_flat, a_flat)
        b_perm = torch.where(swap_mask[:, None], a_flat, b_flat)
        # Sum across samples to form a single batch-level gradient per task,
        # matching the way the observed cos is computed in M2.
        ag = a_perm.sum(dim=0)
        bg = b_perm.sum(dim=0)
        out[i] = _cosine(ag, bg)
    return out
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Stop and review**

`null_baseline.sample_shuffle_null` is callable and shape-correct. Sign-flip null comes next.

---

## Task 2: null_baseline — Sign-Flip Null

**Files:**
- Modify: `tools/gradient_analysis/null_baseline.py`
- Modify: `tests/gradient_analysis/test_null_baseline.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/gradient_analysis/test_null_baseline.py`:

```python
from tools.gradient_analysis.null_baseline import sign_flip_null


def test_sign_flip_null_returns_correct_count():
    rng = np.random.default_rng(2)
    g_a = torch.tensor(rng.standard_normal(size=(64,)).astype("float32"))
    g_b = torch.tensor(rng.standard_normal(size=(64,)).astype("float32"))
    out = sign_flip_null(g_a, g_b, n_repeats=128, seed=3)
    assert out.shape == (128,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= -1.0) & (out <= 1.0))


def test_sign_flip_null_centered_at_zero():
    """For random Gaussian inputs, the sign-flipped cosine distribution should
    be approximately centered at zero (no directional preference)."""
    rng = np.random.default_rng(4)
    g = torch.tensor(rng.standard_normal(size=(256,)).astype("float32"))
    out = sign_flip_null(g, g, n_repeats=2000, seed=5)
    # Mean should be close to zero. Loose tolerance for stochasticity.
    assert abs(float(out.mean())) < 0.05
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: `ImportError: cannot import name 'sign_flip_null'`.

- [ ] **Step 3: Add the helper**

Append to `tools/gradient_analysis/null_baseline.py`:

```python
def sign_flip_null(
    g_a: torch.Tensor,
    g_b: torch.Tensor,
    n_repeats: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Independently flip the sign of each component of (g_a, g_b) with p = 0.5
    and recompute cos.

    Inputs may be any shape; they are flattened. Returns shape (n_repeats,).
    """
    if g_a.shape != g_b.shape:
        raise ValueError(f"shape mismatch: {tuple(g_a.shape)} vs {tuple(g_b.shape)}")
    rng = np.random.default_rng(seed)
    a = g_a.flatten().to(torch.float64)
    b = g_b.flatten().to(torch.float64)
    n = a.numel()
    out = np.empty(n_repeats, dtype=np.float64)
    for i in range(n_repeats):
        sa = torch.from_numpy(rng.choice([-1.0, 1.0], size=n))
        sb = torch.from_numpy(rng.choice([-1.0, 1.0], size=n))
        out[i] = _cosine(a * sa, b * sb)
    return out
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: all four tests pass.

- [ ] **Step 5: Stop and review**

Both null procedures live alongside each other. Next: pair them with a statistical test.

---

## Task 3: null_baseline — Mann-Whitney U + Rank-Biserial Effect Size

**Files:**
- Modify: `tools/gradient_analysis/null_baseline.py`
- Modify: `tests/gradient_analysis/test_null_baseline.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/gradient_analysis/test_null_baseline.py`:

```python
from tools.gradient_analysis.null_baseline import test_against_null


def test_mann_whitney_significant_when_observed_shifted():
    rng = np.random.default_rng(7)
    null = rng.standard_normal(1000) * 0.05         # tight null near 0
    observed = rng.standard_normal(100) * 0.05 + 0.3  # shifted by 0.3
    res = test_against_null(observed, null)
    assert res.p_value < 1e-6
    assert abs(res.rank_biserial) > 0.5  # large effect


def test_mann_whitney_null_when_observed_matches_null():
    rng = np.random.default_rng(8)
    null = rng.standard_normal(1000) * 0.05
    observed = rng.standard_normal(100) * 0.05
    res = test_against_null(observed, null)
    assert res.p_value > 0.01
    assert abs(res.rank_biserial) < 0.2
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: `ImportError: cannot import name 'test_against_null'`.

- [ ] **Step 3: Add the helper**

Append to `tools/gradient_analysis/null_baseline.py`:

```python
from dataclasses import dataclass

from scipy import stats as _stats


@dataclass
class NullTestResult:
    observed_mean: float
    null_mean: float
    null_ci_lo: float        # 2.5 % percentile of null
    null_ci_hi: float        # 97.5 % percentile of null
    u_statistic: float
    p_value: float
    rank_biserial: float     # signed, ∈ [-1, 1]
    n_observed: int
    n_null: int


def test_against_null(
    observed: np.ndarray,
    null: np.ndarray,
) -> NullTestResult:
    """Mann-Whitney U two-sided + rank-biserial effect size.

    `observed` is the empirical distribution of cos across batches at one
    (group, pair). `null` is the corresponding null distribution generated by
    `sample_shuffle_null` or `sign_flip_null`.
    """
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    observed = observed[np.isfinite(observed)]
    null = null[np.isfinite(null)]
    if observed.size == 0 or null.size == 0:
        return NullTestResult(
            observed_mean=float("nan"), null_mean=float("nan"),
            null_ci_lo=float("nan"), null_ci_hi=float("nan"),
            u_statistic=float("nan"), p_value=float("nan"),
            rank_biserial=float("nan"),
            n_observed=int(observed.size), n_null=int(null.size),
        )
    u_stat, p = _stats.mannwhitneyu(observed, null, alternative="two-sided")
    # Rank-biserial r = 1 - 2U / (n1 * n2). Range [-1, 1].
    r = 1.0 - 2.0 * float(u_stat) / (float(observed.size) * float(null.size))
    return NullTestResult(
        observed_mean=float(observed.mean()),
        null_mean=float(null.mean()),
        null_ci_lo=float(np.percentile(null, 2.5)),
        null_ci_hi=float(np.percentile(null, 97.5)),
        u_statistic=float(u_stat),
        p_value=float(p),
        rank_biserial=float(r),
        n_observed=int(observed.size),
        n_null=int(null.size),
    )
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: all six tests pass.

- [ ] **Step 5: Stop and review**

Pure-helper layer is complete. Next: aggregate over (ckpt, group, pair) cells with Bonferroni correction.

---

## Task 4: null_baseline — Per-Cell Aggregator + Bonferroni Correction

**Files:**
- Modify: `tools/gradient_analysis/null_baseline.py`
- Modify: `tests/gradient_analysis/test_null_baseline.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/gradient_analysis/test_null_baseline.py`:

```python
import pandas as pd

from tools.gradient_analysis.null_baseline import run_null_baseline


def test_run_null_baseline_writes_csv_with_required_columns(tmp_path):
    """End-to-end smoke: a synthetic 3-batch cache, two tasks, two groups."""
    cached = []
    rng = np.random.default_rng(0)
    for b in range(3):
        shared = {}
        for task in ("a", "b"):
            shared[task] = {
                "g0": torch.tensor(rng.standard_normal(64).astype("float32")),
                "g1": torch.tensor(rng.standard_normal(64).astype("float32")),
            }
        cached.append({"batch_idx": b, "shared": shared})
    df = run_null_baseline(
        cached_batches=cached,
        tasks=["a", "b"],
        group_keys=["g0", "g1"],
        n_repeats=64,                # small for speed
        seed=42,
        bonferroni_family_size=None, # auto-derive
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "task_a", "task_b", "group", "null_kind",
        "observed_mean", "null_mean", "null_ci_lo", "null_ci_hi",
        "u_statistic", "p_value", "p_value_bonferroni", "rank_biserial",
        "passes_noise_threshold", "n_observed", "n_null",
    }
    assert expected_cols.issubset(set(df.columns))
    # bonferroni p must be >= raw p
    assert (df["p_value_bonferroni"] >= df["p_value"]).all()
    # Two null kinds (sample_shuffle, sign_flip) × pair × group = 4 rows.
    assert len(df) == 4
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: `ImportError: cannot import name 'run_null_baseline'`.

- [ ] **Step 3: Add the aggregator**

Append to `tools/gradient_analysis/null_baseline.py`:

```python
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


_NULL_KINDS = ("sample_shuffle", "sign_flip")
_RANK_BISERIAL_THRESHOLD = 0.1   # Cohen's small-effect convention
_RAW_ALPHA = 0.05


def _generate_null_for_batches(
    cached_batches: Iterable[Dict],
    task_a: str,
    task_b: str,
    group: str,
    null_kind: str,
    n_repeats: int,
    seed: int,
) -> np.ndarray:
    """Pool null draws across all batches for one (group, pair, kind) cell.

    Each batch contributes (n_repeats / n_batches) draws so the total null
    sample size is approximately n_repeats. This pooling mirrors how the
    observed cosine distribution is built (one cos per batch), giving the
    Mann-Whitney test comparable n on both sides up to constants.
    """
    batches = list(cached_batches)
    if not batches:
        return np.array([], dtype=np.float64)
    per_batch = max(1, n_repeats // len(batches))
    chunks: List[np.ndarray] = []
    for i, b in enumerate(batches):
        ga = b["shared"].get(task_a, {}).get(group)
        gb = b["shared"].get(task_b, {}).get(group)
        if ga is None or gb is None:
            continue
        if null_kind == "sample_shuffle":
            # ga / gb are flat group-level vectors here; sample-level shuffle
            # only makes sense if they were collected per-sample. The cached
            # M2 format stores group-summed gradients, so for null we fall
            # back to sign_flip when sample-level data is absent.
            chunks.append(sign_flip_null(ga, gb, n_repeats=per_batch, seed=seed + i))
        elif null_kind == "sign_flip":
            chunks.append(sign_flip_null(ga, gb, n_repeats=per_batch, seed=seed + i))
        else:
            raise ValueError(f"unknown null_kind: {null_kind}")
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def _observed_per_batch(
    cached_batches: Iterable[Dict],
    task_a: str,
    task_b: str,
    group: str,
) -> np.ndarray:
    out: List[float] = []
    for b in cached_batches:
        ga = b["shared"].get(task_a, {}).get(group)
        gb = b["shared"].get(task_b, {}).get(group)
        if ga is None or gb is None:
            continue
        c = _cosine(ga, gb)
        if np.isfinite(c):
            out.append(c)
    return np.asarray(out, dtype=np.float64)


def run_null_baseline(
    cached_batches: List[Dict],
    tasks: List[str],
    group_keys: List[str],
    n_repeats: int = 1000,
    seed: int = 0,
    bonferroni_family_size: Optional[int] = None,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run null-baseline tests over every (group, task pair, kind) cell.

    Bonferroni correction divides α by the family size, which defaults to
    (n_groups × n_pairs × n_kinds). Pass an explicit value to override.
    """
    pairs = list(combinations(tasks, 2))
    if bonferroni_family_size is None:
        bonferroni_family_size = max(1, len(group_keys) * len(pairs) * len(_NULL_KINDS))
    rows = []
    for a, b in pairs:
        for group in group_keys:
            observed = _observed_per_batch(cached_batches, a, b, group)
            for kind in _NULL_KINDS:
                null = _generate_null_for_batches(
                    cached_batches, a, b, group, kind, n_repeats, seed,
                )
                res = test_against_null(observed, null)
                p_bon = min(1.0, res.p_value * bonferroni_family_size)
                passes = bool(
                    abs(res.rank_biserial) >= _RANK_BISERIAL_THRESHOLD
                    and p_bon <= _RAW_ALPHA
                ) if np.isfinite(res.p_value) else False
                rows.append({
                    "task_a": a, "task_b": b, "group": group,
                    "null_kind": kind,
                    "observed_mean": res.observed_mean,
                    "null_mean": res.null_mean,
                    "null_ci_lo": res.null_ci_lo,
                    "null_ci_hi": res.null_ci_hi,
                    "u_statistic": res.u_statistic,
                    "p_value": res.p_value,
                    "p_value_bonferroni": p_bon,
                    "rank_biserial": res.rank_biserial,
                    "passes_noise_threshold": passes,
                    "n_observed": res.n_observed,
                    "n_null": res.n_null,
                })
    df = pd.DataFrame(rows)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_null_baseline.py -v
```

Expected: all seven tests pass.

- [ ] **Step 5: Stop and review**

`run_null_baseline` is end-to-end testable. Next: wire it into the CLI.

---

## Task 5: null_baseline — Wire into `run_gradient_analysis.py`

**Files:**
- Modify: `tools/run_gradient_analysis.py`

- [ ] **Step 1: Read the current CLI to understand the per-checkpoint loop**

Open `tools/run_gradient_analysis.py` and locate the per-checkpoint loop that loads cached `BatchGradients` and dispatches M2/M3/M5. Read the section that reads `gradient_analysis_results/ckpt_<tag>/` and routes module outputs.

- [ ] **Step 2: Add a `--modules` recognition for `M_N1` (null baseline)**

Modify the module dispatcher in `tools/run_gradient_analysis.py` so that when `"M_N1"` (or `"null_baseline"`) appears in `--modules`, it invokes `run_null_baseline` against the per-checkpoint cached batches and writes `gradient_analysis_results/ckpt_<tag>/null_baseline/null_baseline.csv`. Use the following exact code where the existing M2/M3/etc. branches live (replace `# ... existing module dispatch ...` with the new branch in addition to the existing ones):

```python
from tools.gradient_analysis.null_baseline import run_null_baseline

# ... after M2 has run and `cached_batches` is in memory for this checkpoint ...
if "M_N1" in modules or "null_baseline" in modules:
    out_dir = ckpt_out_dir / "null_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_null_baseline(
        cached_batches=cached_batches,
        tasks=cfg["tasks"],
        group_keys=group_keys_for_ckpt,
        n_repeats=cfg.get("null_baseline", {}).get("n_repeats", 1000),
        seed=cfg.get("seed", 42),
        out_path=out_dir / "null_baseline.csv",
    )
```

If the existing CLI uses an argparse default of "all" for `--modules`, ensure `M_N1` is included in the default. The exact insertion site depends on the file's current shape — read it carefully and place the new branch alongside the M2 branch.

- [ ] **Step 3: Add a config key under `configs/gradient_analysis.yaml`**

Open `configs/gradient_analysis.yaml` and append (preserving existing content):

```yaml
null_baseline:
  n_repeats: 1000
```

- [ ] **Step 4: Smoke run on the smallest existing checkpoint**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M2,M_N1 \
    --no-supplementary
```

Expected: completes without exception; produces `gradient_analysis_results/ckpt_1ep/null_baseline/null_baseline.csv`. Eyeball the CSV — `passes_noise_threshold` should not be uniformly True or uniformly False (some variation across cells).

- [ ] **Step 5: Stop and review**

Phase 1 #1 is end-to-end. Move on to distribution diagnostics.

---

## Task 6: distribution — KDE + Hartigan Dip Test (Module Stubs + First Test)

**Files:**
- Create: `tools/gradient_analysis/distribution.py`
- Create: `tests/gradient_analysis/test_distribution.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/test_distribution.py`:

```python
"""Unit tests for the distribution-diagnostics module."""
import numpy as np

from tools.gradient_analysis.distribution import (
    classify_shape,
    compute_distribution_features,
)


def test_compute_features_unimodal_centered():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(500) * 0.05  # tight unimodal at 0
    feats = compute_distribution_features(samples)
    assert -0.05 < feats["mean"] < 0.05
    assert feats["dip_p_value"] > 0.05  # cannot reject unimodality
    assert feats["shape_label"] in ("unimodal-near-0", "unimodal-pos", "unimodal-neg")


def test_compute_features_bimodal_detected():
    rng = np.random.default_rng(1)
    a = rng.standard_normal(250) * 0.05 - 0.5
    b = rng.standard_normal(250) * 0.05 + 0.5
    samples = np.concatenate([a, b])
    feats = compute_distribution_features(samples)
    # Dip test should reject unimodality at p < 0.05
    assert feats["dip_p_value"] < 0.05
    assert feats["shape_label"] == "bimodal"


def test_classify_shape_returns_one_of_known_labels():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal(500) * 0.05 + 0.3  # unimodal positive
    feats = compute_distribution_features(samples)
    assert feats["shape_label"] in {
        "unimodal-near-0", "unimodal-pos", "unimodal-neg",
        "bimodal", "heavy-tail",
    }


def test_classify_shape_handles_empty_array():
    feats = compute_distribution_features(np.array([], dtype=np.float64))
    assert feats["shape_label"] == "empty"
    assert np.isnan(feats["mean"])
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_distribution.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `tools/gradient_analysis/distribution.py`:

```python
"""M-N2 — Distribution diagnostics (Phase 1 #2).

Per (group, task pair) the cosine-similarity distribution across batches is
characterised by a KDE summary, a Hartigan dip test for unimodality, and tail
descriptors. A categorical `shape_label` ∈ {`unimodal-near-0`, `unimodal-pos`,
`unimodal-neg`, `bimodal`, `heavy-tail`, `empty`} flags whether mean-only
reporting is appropriate.

Pure helpers only — orchestration entry point lives in `run_distribution`
below.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


_DIP_ALPHA = 0.05
_HEAVY_TAIL_KURTOSIS = 6.0       # > 6 ⇒ tails heavier than t_5
_NEAR_ZERO_ABS_MEAN = 0.05


def compute_distribution_features(samples: np.ndarray) -> Dict[str, float]:
    """Compute mean, std, percentile bundle, kurtosis, dip p-value, and a
    categorical shape label for `samples`.

    Returns a dict with keys: mean, std, p1, p5, p50, p95, p99, kurtosis,
    dip_statistic, dip_p_value, shape_label, n.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    base = {
        "mean": float("nan"), "std": float("nan"),
        "p1": float("nan"), "p5": float("nan"), "p50": float("nan"),
        "p95": float("nan"), "p99": float("nan"),
        "kurtosis": float("nan"),
        "dip_statistic": float("nan"), "dip_p_value": float("nan"),
        "shape_label": "empty",
        "n": int(samples.size),
    }
    if samples.size == 0:
        return base
    base.update({
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)) if samples.size > 1 else 0.0,
        "p1": float(np.percentile(samples, 1)),
        "p5": float(np.percentile(samples, 5)),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
    })
    if samples.size > 3:
        from scipy.stats import kurtosis as _k
        base["kurtosis"] = float(_k(samples, fisher=False, bias=False))
    if samples.size >= 4:
        from diptest import diptest
        dip_stat, p = diptest(samples)
        base["dip_statistic"] = float(dip_stat)
        base["dip_p_value"] = float(p)
    base["shape_label"] = classify_shape(base)
    return base


def classify_shape(feats: Dict[str, float]) -> str:
    """Decide the shape label from raw features.

    Order of decisions (most specific first):
      1. dip p < 0.05 ⇒ `bimodal`
      2. kurtosis > 6 ⇒ `heavy-tail`
      3. |mean| > 0.05 and same sign as median ⇒ `unimodal-pos` / `unimodal-neg`
      4. otherwise ⇒ `unimodal-near-0`
    """
    if not np.isfinite(feats.get("mean", float("nan"))):
        return "empty"
    if np.isfinite(feats["dip_p_value"]) and feats["dip_p_value"] < _DIP_ALPHA:
        return "bimodal"
    if np.isfinite(feats["kurtosis"]) and feats["kurtosis"] > _HEAVY_TAIL_KURTOSIS:
        return "heavy-tail"
    m = feats["mean"]
    if abs(m) > _NEAR_ZERO_ABS_MEAN and np.sign(m) == np.sign(feats["p50"]):
        return "unimodal-pos" if m > 0 else "unimodal-neg"
    return "unimodal-near-0"
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_distribution.py -v
```

Expected: all four tests pass.

- [ ] **Step 5: Stop and review**

Pure-helper feature extractor is in. Next: aggregator + CLI hook.

---

## Task 7: distribution — Aggregator + KDE Figure

**Files:**
- Modify: `tools/gradient_analysis/distribution.py`
- Modify: `tests/gradient_analysis/test_distribution.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/gradient_analysis/test_distribution.py`:

```python
import torch

from tools.gradient_analysis.distribution import run_distribution


def test_run_distribution_writes_csv(tmp_path):
    cached = []
    rng = np.random.default_rng(0)
    for b in range(8):
        shared = {
            "a": {"g0": torch.tensor(rng.standard_normal(64).astype("float32"))},
            "b": {"g0": torch.tensor(rng.standard_normal(64).astype("float32"))},
        }
        cached.append({"batch_idx": b, "shared": shared})
    out_path = tmp_path / "distribution_report.csv"
    df = run_distribution(
        cached_batches=cached,
        tasks=["a", "b"],
        group_keys=["g0"],
        out_path=out_path,
        emit_kde_figures=False,
    )
    assert out_path.exists()
    assert {"task_a", "task_b", "group", "shape_label", "dip_p_value", "n"} \
        .issubset(set(df.columns))
    assert len(df) == 1
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_distribution.py::test_run_distribution_writes_csv -v
```

Expected: `ImportError: cannot import name 'run_distribution'`.

- [ ] **Step 3: Add the orchestration entry point**

Append to `tools/gradient_analysis/distribution.py`:

```python
from itertools import combinations

from .null_baseline import _cosine   # reuse the cached helper


def _observed_cos_per_batch(cached_batches, task_a, task_b, group):
    out = []
    for b in cached_batches:
        ga = b["shared"].get(task_a, {}).get(group)
        gb = b["shared"].get(task_b, {}).get(group)
        if ga is None or gb is None:
            continue
        c = _cosine(ga, gb)
        if np.isfinite(c):
            out.append(c)
    return np.asarray(out, dtype=np.float64)


def _emit_kde_figure(samples: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(samples, bins=30, density=True, alpha=0.4, label="empirical")
    if samples.size >= 2 and samples.std(ddof=1) > 1e-8:
        kde = gaussian_kde(samples, bw_method="silverman")
        xs = np.linspace(samples.min() - 0.05, samples.max() + 0.05, 200)
        ax.plot(xs, kde(xs), label="KDE")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_distribution(
    cached_batches: List[Dict],
    tasks: List[str],
    group_keys: List[str],
    out_path: Optional[Path] = None,
    emit_kde_figures: bool = True,
    figures_dir: Optional[Path] = None,
) -> pd.DataFrame:
    rows = []
    pairs = list(combinations(tasks, 2))
    for a, b in pairs:
        for group in group_keys:
            samples = _observed_cos_per_batch(cached_batches, a, b, group)
            feats = compute_distribution_features(samples)
            row = {"task_a": a, "task_b": b, "group": group, **feats}
            row["mean_is_misleading"] = bool(feats["shape_label"] in ("bimodal", "heavy-tail"))
            rows.append(row)
            if (
                emit_kde_figures
                and figures_dir is not None
                and feats["shape_label"] not in ("empty",)
                and samples.size > 0
            ):
                fig_path = Path(figures_dir) / f"kde_{a}_{b}_{group}.png"
                _emit_kde_figure(
                    samples,
                    title=f"{a} vs {b} — {group} ({feats['shape_label']})",
                    out_path=fig_path,
                )
    df = pd.DataFrame(rows)
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    return df
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_distribution.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Stop and review**

`run_distribution` matches the M2 dispatch shape. Next: wire it into the CLI alongside null baseline.

---

## Task 8: distribution — Wire into CLI

**Files:**
- Modify: `tools/run_gradient_analysis.py`

- [ ] **Step 1: Add the dispatch branch**

In `tools/run_gradient_analysis.py`, near the M_N1 branch added in Task 5, add the M_N2 branch:

```python
from tools.gradient_analysis.distribution import run_distribution

if "M_N2" in modules or "distribution" in modules:
    out_dir = ckpt_out_dir / "distribution"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_distribution(
        cached_batches=cached_batches,
        tasks=cfg["tasks"],
        group_keys=group_keys_for_ckpt,
        out_path=out_dir / "distribution_report.csv",
        emit_kde_figures=True,
        figures_dir=out_dir / "kde",
    )
```

Add `M_N2` to the default module set if applicable.

- [ ] **Step 2: Smoke run**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M2,M_N1,M_N2 \
    --no-supplementary
```

Expected: produces `gradient_analysis_results/ckpt_1ep/distribution/distribution_report.csv` and a `kde/` subdir with at least one `.png`. CSV's `shape_label` column should not be uniformly the same value.

- [ ] **Step 3: Stop and review**

#1 and #2 done. Move on to bootstrap CI.

---

## Task 9: bootstrap — BCa Helper + Apply to Conflict Summary

**Files:**
- Create: `tools/gradient_analysis/bootstrap.py`
- Create: `tests/gradient_analysis/test_bootstrap.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/test_bootstrap.py`:

```python
"""Unit tests for the bootstrap-CI helper."""
import numpy as np

from tools.gradient_analysis.bootstrap import bca_ci, augment_summary_with_ci


def test_bca_ci_normal_data():
    rng = np.random.default_rng(0)
    samples = rng.standard_normal(200) * 0.1 + 0.5
    lo, hi = bca_ci(samples, statistic=np.mean, n_resamples=1000, seed=1)
    assert lo < 0.5 < hi
    # tight CI: width less than 0.06
    assert hi - lo < 0.06


def test_bca_ci_handles_constant_input():
    samples = np.full(100, 0.3)
    lo, hi = bca_ci(samples, statistic=np.mean, n_resamples=200, seed=0)
    assert abs(lo - 0.3) < 1e-6
    assert abs(hi - 0.3) < 1e-6
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_bootstrap.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `tools/gradient_analysis/bootstrap.py`:

```python
"""M-N3 — Bootstrap CI helper (Phase 1 #3).

A single BCa bootstrap implementation that the rest of the pipeline calls into.
Adds {`<col>_ci_lo`, `<col>_ci_hi`} columns to existing summary DataFrames so
downstream consumers don't need to know how the CI was computed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import bootstrap


def bca_ci(
    samples: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (lo, hi) of the BCa bootstrap CI for `statistic` of `samples`.

    Constant-input edge case is handled explicitly because scipy's
    `bootstrap` raises `DegenerateDataWarning` and returns NaNs there.
    """
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    if np.allclose(samples, samples[0]):
        v = float(statistic(samples))
        return v, v
    rng = np.random.default_rng(seed)
    res = bootstrap(
        (samples,),
        statistic,
        method="BCa",
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        random_state=rng,
        vectorized=False,
    )
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def augment_summary_with_ci(
    df: pd.DataFrame,
    per_batch_df: pd.DataFrame,
    group_cols: List[str],
    value_cols: List[str],
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """For each `value_col`, look up the matching per-batch column and attach
    `<col>_ci_lo`, `<col>_ci_hi` to `df` keyed on `group_cols`.

    Caller responsibility: `value_cols` here are the *summary* columns
    (e.g. `mean_cos`); the per-batch column they were aggregated from is
    inferred by stripping a `mean_` / `median_` prefix.
    """
    out = df.copy()
    for vcol in value_cols:
        per_batch_col = vcol
        for prefix in ("mean_", "median_"):
            if vcol.startswith(prefix):
                per_batch_col = vcol[len(prefix):]
                break
        lo_col = f"{vcol}_ci_lo"
        hi_col = f"{vcol}_ci_hi"
        out[lo_col] = np.nan
        out[hi_col] = np.nan
        for idx, row in out.iterrows():
            mask = np.ones(len(per_batch_df), dtype=bool)
            for gc in group_cols:
                mask &= (per_batch_df[gc] == row[gc])
            samples = per_batch_df.loc[mask, per_batch_col].to_numpy()
            stat = np.mean if vcol.startswith("mean_") else np.median
            lo, hi = bca_ci(samples, statistic=stat, n_resamples=n_resamples, seed=seed + int(idx))
            out.at[idx, lo_col] = lo
            out.at[idx, hi_col] = hi
    return out
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_bootstrap.py -v
```

Expected: both tests pass.

- [ ] **Step 5: Stop and review**

`bca_ci` and `augment_summary_with_ci` are unit-tested. Apply next to the existing conflict summary CSV.

---

## Task 10: bootstrap — Apply to conflict, probe, gradnorm Summaries (CLI Hook)

**Files:**
- Modify: `tools/run_gradient_analysis.py`
- Modify: `tools/gradient_analysis/bootstrap.py`

- [ ] **Step 1: Add an integration test exercising the CLI plumbing**

Append to `tests/gradient_analysis/test_bootstrap.py`:

```python
import pandas as pd

from tools.gradient_analysis.bootstrap import augment_summary_with_ci


def test_augment_conflict_summary_attaches_ci():
    rng = np.random.default_rng(0)
    per_batch = pd.DataFrame({
        "group": ["g0"] * 100,
        "cos": rng.standard_normal(100) * 0.1 + 0.2,
    })
    summary = pd.DataFrame({"group": ["g0"], "mean_cos": [per_batch["cos"].mean()]})
    out = augment_summary_with_ci(
        summary, per_batch,
        group_cols=["group"],
        value_cols=["mean_cos"],
        n_resamples=500,
        seed=0,
    )
    assert "mean_cos_ci_lo" in out.columns
    assert out.at[0, "mean_cos_ci_lo"] < out.at[0, "mean_cos"] < out.at[0, "mean_cos_ci_hi"]
```

- [ ] **Step 2: Run test — verify pass (helper already implemented)**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_bootstrap.py -v
```

Expected: all three tests pass.

- [ ] **Step 3: Add bootstrap CI columns to existing summary writers**

In `tools/run_gradient_analysis.py`, after the M2 step writes `conflict_<a>_<b>_summary.csv`, run:

```python
from tools.gradient_analysis.bootstrap import augment_summary_with_ci

# After M2 has written its summary csvs:
if cfg.get("bootstrap", {}).get("enabled", True):
    n_resamples = cfg.get("bootstrap", {}).get("n_resamples", 2000)
    for f in (ckpt_out_dir / "conflict").glob("conflict_*_summary.csv"):
        per_batch_path = f.with_name(f.name.replace("_summary.csv", "_per_batch.csv"))
        if not per_batch_path.exists():
            continue
        summary = pd.read_csv(f)
        per_batch = pd.read_csv(per_batch_path)
        augmented = augment_summary_with_ci(
            summary, per_batch,
            group_cols=["group"],
            value_cols=["mean_cos", "median_cos", "mean_coop_mag", "mean_conf_mag"],
            n_resamples=n_resamples,
            seed=cfg.get("seed", 42),
        )
        augmented.to_csv(f, index=False)
```

Repeat the augmentation for `probe_per_batch.csv` aggregated to affinity matrices and for the `gradnorm/per_task_norm.csv` (mean_norm). The augmentation is value-column-dependent; copy the pattern above with appropriate `group_cols` and `value_cols`.

- [ ] **Step 4: Add bootstrap config to YAML**

Append to `configs/gradient_analysis.yaml`:

```yaml
bootstrap:
  enabled: true
  n_resamples: 2000
```

- [ ] **Step 5: Smoke run + eyeball CSV**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M2,M_N1,M_N2 \
    --no-supplementary
```

Open `gradient_analysis_results/ckpt_1ep/conflict/conflict_det_map_summary.csv` — confirm `mean_cos_ci_lo` and `mean_cos_ci_hi` columns are present and bracket `mean_cos`.

- [ ] **Step 6: Stop and review**

Bootstrap CI now augments existing summaries. Move on to the validity hardening (#4).

---

## Task 11: collector — v2 Cache Field Carrying Per-Param Non-Zero Mask (B1)

**Files:**
- Modify: `tools/gradient_analysis/collector.py`
- Create: `tests/gradient_analysis/test_collector_validity.py`

- [ ] **Step 1: Read collector.py:115–278 to understand BatchGradients**

Re-read `tools/gradient_analysis/collector.py` lines 115–278 — especially `BatchGradients`, `collect_batch`, and the per-group concat at lines 256–262. The plan adds a `nonzero_masks: Dict[str, Dict[str, torch.Tensor]]` field carrying per-(task, group_key) bool tensors of length `sum(p.numel() for p in group_params)`.

- [ ] **Step 2: Write the failing test**

Create `tests/gradient_analysis/test_collector_validity.py`:

```python
"""B1 — per-param non-zero mask carried alongside cached gradients."""
import torch

from tools.gradient_analysis.collector import BatchGradients


def test_batch_gradients_carries_v2_masks(tmp_path):
    bg = BatchGradients(
        batch_idx=0,
        shared={"a": {"g0": torch.randn(8)}},
        full_norm={"a": 1.0},
        shared_norm={"a": 1.0},
        loss_values={"a": 0.0},
        nonzero_masks={"a": {"g0": torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.bool)}},
    )
    bg.save(tmp_path)
    loaded = BatchGradients.load(tmp_path / "batch_00000.pt")
    assert "a" in loaded.nonzero_masks
    assert loaded.nonzero_masks["a"]["g0"].sum().item() == 5


def test_batch_gradients_v1_load_backward_compat(tmp_path):
    """A v1 .pt file (no `nonzero_masks` key) must load with empty masks."""
    legacy = {
        "batch_idx": 0,
        "shared": {"a": {"g0": torch.randn(4)}},
        "full_norm": {"a": 1.0},
        "shared_norm": {"a": 1.0},
        "loss_values": {"a": 0.0},
    }
    (tmp_path / "batch_00000.pt").parent.mkdir(parents=True, exist_ok=True)
    torch.save(legacy, tmp_path / "batch_00000.pt")
    loaded = BatchGradients.load(tmp_path / "batch_00000.pt")
    assert loaded.nonzero_masks == {}
```

- [ ] **Step 3: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_collector_validity.py -v
```

Expected: both tests fail (TypeError or AttributeError on `nonzero_masks`).

- [ ] **Step 4: Modify `BatchGradients`**

In `tools/gradient_analysis/collector.py`, replace the existing dataclass with:

```python
@dataclass
class BatchGradients:
    """Per-batch, per-task gradient bundle (v2)."""
    batch_idx: int
    shared: Dict[str, Dict[str, torch.Tensor]]
    full_norm: Dict[str, float]
    shared_norm: Dict[str, float]
    loss_values: Dict[str, float]
    nonzero_masks: Dict[str, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 2,
                "batch_idx": self.batch_idx,
                "shared": self.shared,
                "full_norm": self.full_norm,
                "shared_norm": self.shared_norm,
                "loss_values": self.loss_values,
                "nonzero_masks": self.nonzero_masks,
            },
            out_dir / f"batch_{self.batch_idx:05d}.pt",
        )

    @classmethod
    def load(cls, path: Path) -> "BatchGradients":
        d = torch.load(path, map_location="cpu")
        d.pop("schema_version", None)
        d.setdefault("nonzero_masks", {})
        return cls(**d)
```

Add `from dataclasses import dataclass, field` to the imports if `field` isn't already imported.

- [ ] **Step 5: Compute the mask in `collect_batch`**

In `collect_batch`, replace the per-group concat loop (around line 256–262) with:

```python
            shared_groups: Dict[str, torch.Tensor] = {}
            mask_groups: Dict[str, torch.Tensor] = {}
            id2idx = {id(p): k for k, p in enumerate(self.full_params)}
            for gk, params in self.shared_param_groups.items():
                parts: List[torch.Tensor] = []
                masks: List[torch.Tensor] = []
                for p in params:
                    g = fg[id2idx[id(p)]].detach()
                    parts.append(g.flatten())
                    masks.append((g.abs().flatten() > EPS_GRAD))
                if parts:
                    shared_groups[gk] = torch.cat(parts).cpu()
                    mask_groups[gk] = torch.cat(masks).cpu()
                else:
                    shared_groups[gk] = torch.empty(0)
                    mask_groups[gk] = torch.empty(0, dtype=torch.bool)
            shared[task] = shared_groups
            nonzero_masks[task] = mask_groups
```

Add `EPS_GRAD = 1e-12` at the top of `collector.py` if not present, and add `nonzero_masks: Dict[str, Dict[str, torch.Tensor]] = {}` to the locals at the top of `collect_batch`. Then pass `nonzero_masks=nonzero_masks` into the `BatchGradients(...)` constructor at the bottom of the method.

- [ ] **Step 6: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_collector_validity.py -v
```

Expected: both tests pass.

- [ ] **Step 7: Run the full existing collector test suite to confirm no regression**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_collector.py -v
```

Expected: pre-existing tests still pass.

- [ ] **Step 8: Stop and review**

The cache format now carries per-(task, group, param) non-zero masks. Sub-bucket consumption is the next step.

---

## Task 12: conflict — Validity-Aware Aggregation (B2 + B3)

**Files:**
- Modify: `tools/gradient_analysis/conflict.py`
- Create: `tests/gradient_analysis/test_conflict_validity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/test_conflict_validity.py`:

```python
"""B2 / B3 — validity-aware aggregation in summarize_pair."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.conflict import summarize_pair


def test_summarize_pair_emits_validity_columns():
    df = pd.DataFrame({
        "batch_idx": list(range(8)),
        "group": ["g0"] * 8,
        "cos":      [0.1, -0.2, np.nan, 0.0, 0.05, -0.1, np.nan, 0.2],
        "coop_mag": [0.1,  0.0,  0.0,   0.0, 0.05,  0.0,  0.0,   0.2],
        "conf_mag": [0.0,  0.2,  0.0,   0.0, 0.0,   0.1,  0.0,   0.0],
        "norm_a":   [1.0,  1.0,  0.0,   1.0, 1.0,   1.0,  1.0,   1.0],
        "norm_b":   [1.0,  1.0,  1.0,   0.0, 1.0,   1.0,  0.0,   1.0],
    })
    summary = summarize_pair(df)
    row = summary.iloc[0]
    assert row["n_total"] == 8
    assert row["n_valid"] == 5     # cos finite AND norm_a > 0 AND norm_b > 0
    assert row["n_pseudo_shared"] == 3
    assert 0.0 <= row["pseudo_shared_ratio"] <= 1.0
    # is_pseudo_shared_group — flag only for groups where ratio ≥ 0.5
    assert row["is_pseudo_shared_group"] in (True, False)
    # legacy column preserved
    assert "conflict_ratio_legacy" in summary.columns
    # main conflict_ratio uses n_valid as denominator
    assert abs(row["conflict_ratio"] - (2.0 / 5.0)) < 1e-6
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_conflict_validity.py -v
```

Expected: failure on missing columns / wrong denominator.

- [ ] **Step 3: Modify `summarize_pair`**

In `tools/gradient_analysis/conflict.py`, replace `summarize_pair` with:

```python
def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(task_pair, group) across batches with validity-aware
    counts.

    Validity rule: a row contributes to `n_valid` iff cos is finite *and*
    norm_a > EPS *and* norm_b > EPS.
    """
    EPS_LOCAL = EPS

    def _agg(sub: pd.DataFrame) -> pd.Series:
        n_total = int(len(sub))
        finite = sub["cos"].notna()
        nonzero = (sub["norm_a"] > EPS_LOCAL) & (sub["norm_b"] > EPS_LOCAL)
        valid = finite & nonzero
        n_valid = int(valid.sum())
        n_nan = int((~finite).sum())
        # pseudo-shared: cos was NaN solely because at least one norm was below EPS
        n_pseudo_shared = int((~finite & ~nonzero).sum())
        valid_sub = sub[valid]

        if n_valid > 0:
            mean_cos = float(valid_sub["cos"].mean())
            median_cos = float(valid_sub["cos"].median())
            std_cos = float(valid_sub["cos"].std())
            conflict_ratio = float((valid_sub["cos"] < 0).mean())
        else:
            mean_cos = float("nan"); median_cos = float("nan")
            std_cos = float("nan"); conflict_ratio = float("nan")
        # Legacy ratio: keep prior denominator (n_total) for one cycle.
        conflict_ratio_legacy = float((sub["cos"].fillna(False).astype(float) < 0).mean()) \
            if n_total > 0 else float("nan")
        pseudo_ratio = float(n_pseudo_shared) / n_total if n_total > 0 else float("nan")

        out = {
            "mean_cos": mean_cos,
            "std_cos": std_cos,
            "median_cos": median_cos,
            "n": n_total,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_pseudo_shared": n_pseudo_shared,
            "n_nan": n_nan,
            "pseudo_shared_ratio": pseudo_ratio,
            "is_pseudo_shared_group": pseudo_ratio >= 0.5,
            "conflict_ratio": conflict_ratio,
            "conflict_ratio_legacy": conflict_ratio_legacy,
            "mean_coop_mag": float(sub["coop_mag"].mean()) if n_total > 0 else float("nan"),
            "mean_conf_mag": float(sub["conf_mag"].mean()) if n_total > 0 else float("nan"),
        }

        # Conditional norm stats (kept from prior implementation)
        conf = valid_sub[valid_sub["cos"] < 0]
        coop = valid_sub[valid_sub["cos"] >= 0]

        def _cond(frame: pd.DataFrame, suffix: str) -> Dict[str, float]:
            if frame.empty:
                return {
                    f"mean_norm_a_{suffix}": float("nan"),
                    f"mean_norm_b_{suffix}": float("nan"),
                    f"mean_norm_ratio_{suffix}": float("nan"),
                    f"median_norm_ratio_{suffix}": float("nan"),
                    f"n_{suffix}": 0,
                }
            ratio = frame["norm_a"] / frame["norm_b"].where(frame["norm_b"] > EPS_LOCAL)
            return {
                f"mean_norm_a_{suffix}": float(frame["norm_a"].mean()),
                f"mean_norm_b_{suffix}": float(frame["norm_b"].mean()),
                f"mean_norm_ratio_{suffix}": float(ratio.mean()),
                f"median_norm_ratio_{suffix}": float(ratio.median()),
                f"n_{suffix}": int(len(frame)),
            }
        out.update(_cond(conf, "conflict"))
        out.update(_cond(coop, "coop"))
        return pd.Series(out)

    grouped = df.groupby("group").apply(_agg).reset_index()
    return grouped
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_conflict_validity.py -v
```

Expected: pass.

- [ ] **Step 5: Run existing conflict tests**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_conflict.py -v
```

Expected: existing tests still pass (legacy ratio preserved).

- [ ] **Step 6: Stop and review**

`summarize_pair` now reports validity counts; `conflict_ratio` uses the right denominator. Next: partition pseudo-shared in the markdown digest.

---

## Task 13: summary — Partition Pseudo-Shared Groups in Markdown Digest (B4)

**Files:**
- Modify: `tools/gradient_analysis/summary.py`
- Modify: `tools/gradient_analysis/conflict.py` (only if needed for grouping)

- [ ] **Step 1: Read `summary.py` to find the ranking section**

Open `tools/gradient_analysis/summary.py` and locate the section that produces "Highest / lowest conflict_ratio" tables.

- [ ] **Step 2: Modify the digest writer**

Replace the ranking section so it filters by `is_pseudo_shared_group` and emits a separate appendix. Insert this logic in place of the existing ranking block:

```python
def _write_conflict_rankings(buf, summary_df: pd.DataFrame) -> None:
    real = summary_df[~summary_df["is_pseudo_shared_group"].fillna(False)]
    pseudo = summary_df[summary_df["is_pseudo_shared_group"].fillna(False)]

    buf.write("\n## Highest conflict_ratio (real shared groups)\n\n")
    top = real.sort_values("conflict_ratio", ascending=False).head(10)
    buf.write(top.to_markdown(index=False))
    buf.write("\n\n## Lowest conflict_ratio (real shared groups)\n\n")
    low = real.sort_values("conflict_ratio", ascending=True).head(10)
    buf.write(low.to_markdown(index=False))

    if not pseudo.empty:
        buf.write("\n\n## Pseudo-shared groups (excluded from main analysis)\n\n")
        buf.write(
            "These groups had at least 50 % of batches where one task's gradient was below EPS. "
            "Their cosine values are not interpretable as a measure of inter-task interaction.\n\n"
        )
        buf.write(pseudo[["task_a", "task_b", "group", "n_total",
                          "n_pseudo_shared", "pseudo_shared_ratio"]]
                  .to_markdown(index=False))
```

Wherever `summary.py` previously inlined the ranking, call `_write_conflict_rankings(buf, summary_df)` instead. The `summary_df` argument should be the per-checkpoint conflict summary that already contains the `is_pseudo_shared_group` column (produced by Task 12).

- [ ] **Step 3: Smoke run on existing cache + eyeball the markdown**

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M2,M_N1,M_N2 \
    --no-supplementary
```

Open `gradient_analysis_results/summary_report.md` — confirm a "Pseudo-shared groups" appendix exists.

- [ ] **Step 4: Verify the M7 antisymmetric NaN cascade is resolved**

Run M7 in isolation:

```bash
PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --modules M7 \
    --no-supplementary
```

Open `gradient_analysis_results/summary_report.md` — the "Top asymmetric pairs" section's `antisym` and `abs_antisym` columns should now contain finite values for `det vs ego` (previously NaN). If they remain NaN, fall through to Task 14.

- [ ] **Step 5: Stop and review**

If Step 4 produced finite values, skip Task 14 and jump to Task 15. Otherwise continue with the dedicated debug task.

---

## Task 14: M7 NaN Targeted Debug (Allowed by Risk R6 — Skip if Task 13 Resolved It)

**Files:**
- Modify: `tools/gradient_analysis/asymmetry.py` (likely)
- Modify: `tools/gradient_analysis/gradnorm.py` (where `antisymmetric_frobenius` lives)

- [ ] **Step 1: Reproduce on a saved probe matrix**

```bash
PYTHONPATH=. $PY -c "
import pandas as pd, numpy as np
from tools.gradient_analysis.gradnorm import antisymmetric_frobenius
m = pd.read_csv('gradient_analysis_results/ckpt_1ep/probe/probe_matrix_1step_normalized.csv', index_col=0)
print(m)
print('antisym fro:', antisymmetric_frobenius(m.values))
"
```

If the printed matrix contains NaN cells (e.g. ego row), the cascade is `NaN ⊕ −NaN = NaN`. Fix by ignoring NaN cells.

- [ ] **Step 2: Patch `antisymmetric_frobenius`**

In `tools/gradient_analysis/gradnorm.py`, replace `antisymmetric_frobenius` with:

```python
def antisymmetric_frobenius(M: np.ndarray) -> float:
    """Frobenius norm of the antisymmetric part (M − M^T) / 2.

    NaN-aware: any cell whose (i, j) or (j, i) counterpart is NaN is dropped.
    Returns NaN only if every off-diagonal cell is invalid.
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    pairs: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a = M[i, j]
            b = M[j, i]
            if np.isfinite(a) and np.isfinite(b):
                pairs.append(((a - b) / 2.0) ** 2)
    if not pairs:
        return float("nan")
    return float(np.sqrt(2.0 * sum(pairs)))   # factor 2 because (i,j) and (j,i) both contribute
```

- [ ] **Step 3: Add a test**

Append to `tests/gradient_analysis/test_asymmetry.py` (or create the file if absent):

```python
import numpy as np

from tools.gradient_analysis.gradnorm import antisymmetric_frobenius


def test_antisymmetric_skips_nan_pairs():
    m = np.array([[0.0, 1.0, np.nan],
                  [-0.5, 0.0, 2.0],
                  [np.nan, -1.5, 0.0]])
    val = antisymmetric_frobenius(m)
    assert np.isfinite(val)
    # Only the (0,1) <-> (1,0) and (1,2) <-> (2,1) pairs are valid.
    # antisym components: (1.0 - (-0.5))/2 = 0.75; (2.0 - (-1.5))/2 = 1.75.
    expected = np.sqrt(2 * (0.75**2 + 1.75**2))
    assert abs(val - expected) < 1e-6
```

- [ ] **Step 4: Run the test**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_asymmetry.py -v
```

Expected: pass.

- [ ] **Step 5: Re-run M7 and confirm the report contains finite values**

```bash
PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --modules M7 \
    --no-supplementary
```

Open `gradient_analysis_results/summary_report.md` — verify `(det, ego)` antisym now reports a finite number instead of NaN.

- [ ] **Step 6: Stop and review**

Validity hardening complete. Move on to probe reliability.

---

## Task 15: probe — Bootstrap Variance Bands on the Affinity Matrix (#9 part 1)

**Files:**
- Modify: `tools/gradient_analysis/probe.py`

- [ ] **Step 1: Add a test exercising variance-band aggregation**

Append to `tests/gradient_analysis/test_probe.py`:

```python
import numpy as np
import pandas as pd

from tools.gradient_analysis.probe import aggregate_affinity_matrix_with_ci


def test_aggregate_affinity_matrix_with_ci_attaches_bands():
    rng = np.random.default_rng(0)
    rows = []
    for b in range(50):
        for s_task in ("a", "b"):
            for t_task in ("a", "b"):
                rows.append({
                    "batch_idx": b,
                    "source_task": s_task, "target_task": t_task,
                    "steps": 1, "variant": "raw", "layer": "_all",
                    "delta": rng.standard_normal() * 0.1 + (0.05 if s_task != t_task else -0.05),
                })
    df = pd.DataFrame(rows)
    mean_mat, lo_mat, hi_mat = aggregate_affinity_matrix_with_ci(
        df, steps=1, variant="raw", layer="_all", n_resamples=500, seed=0,
    )
    for mat in (mean_mat, lo_mat, hi_mat):
        assert mat.shape == (2, 2)
    # CI must bracket the mean
    assert (lo_mat.values <= mean_mat.values).all()
    assert (mean_mat.values <= hi_mat.values).all()
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_probe.py::test_aggregate_affinity_matrix_with_ci_attaches_bands -v
```

Expected: `ImportError: cannot import name 'aggregate_affinity_matrix_with_ci'`.

- [ ] **Step 3: Add the helper**

Append to `tools/gradient_analysis/probe.py`:

```python
from tools.gradient_analysis.bootstrap import bca_ci


def aggregate_affinity_matrix_with_ci(
    df: pd.DataFrame,
    steps: int,
    variant: str,
    layer: str = "_all",
    n_resamples: int = 2000,
    seed: int = 0,
):
    """Like `aggregate_affinity_matrix` but returns three matrices:
    (mean, ci_lo, ci_hi) of the per-batch ΔL distribution per (source, target).
    """
    sub = df[(df["steps"] == steps) & (df["variant"] == variant) & (df["layer"] == layer)]
    sources = sorted(sub["source_task"].unique())
    targets = sorted(sub["target_task"].unique())
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
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_probe.py -v
```

Expected: pass.

- [ ] **Step 5: Wire into the M3 writer**

In `run_m3` (also in `probe.py`), after the existing `mat.to_csv(...)` call, also write `*_ci_lo.csv` and `*_ci_hi.csv`. Replace the loop:

```python
    for layer in layers:
        suffix = "" if layer == "_all" else f"_{layer}"
        for s in steps_list:
            for v in variants:
                mean_mat, lo_mat, hi_mat = aggregate_affinity_matrix_with_ci(
                    df, s, v, layer, n_resamples=2000, seed=42,
                )
                if mean_mat.empty:
                    continue
                mean_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}.csv")
                lo_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}_ci_lo.csv")
                hi_mat.to_csv(out_dir / f"probe_matrix{suffix}_{s}step_{v}_ci_hi.csv")
```

- [ ] **Step 6: Stop and review**

Variance bands now ship with every probe matrix. Next: α-sensitivity.

---

## Task 16: probe — α-Sensitivity Sweep CLI Flag (#9 part 2)

**Files:**
- Modify: `tools/gradient_analysis/probe.py`
- Modify: `tools/run_gradient_analysis.py`
- Modify: `configs/gradient_analysis.yaml`

- [ ] **Step 1: Add a `run_alpha_sweep` orchestrator**

Append to `tools/gradient_analysis/probe.py`:

```python
def run_alpha_sweep(
    collector,
    dataloader,
    num_batches: int,
    alphas: List[float],
    sources: List[str],
    out_dir: Path,
    steps_list: List[int] = (1,),
    variants: List[str] = ("raw", "normalized"),
    target_layers: Optional[Sequence[str]] = None,
    freeze_matching: bool = True,
    forward_seed: Optional[int] = 42,
    reset_temporal_state: bool = True,
) -> pd.DataFrame:
    """Re-run the probe on a per-α basis, restricted to the given source tasks.

    Output: one CSV per α at `out_dir/sweep_alpha_<a>.csv` plus a combined
    `sweep_summary.csv` with columns: alpha, source_task, target_task,
    diag_violation_rate, mean_delta, mean_delta_ci_lo, mean_delta_ci_hi.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict] = []
    for a in alphas:
        rows: List[ProbeRow] = []
        prev_data = None
        # Re-iterate the dataloader. Caller is responsible for calling the
        # collector with a fresh iterator — here we assume `dataloader` is a
        # callable returning a fresh iterator, or already a fresh iterator.
        it = iter(dataloader) if not hasattr(dataloader, "__iter__") else iter(dataloader)
        for i, data in enumerate(it):
            if i >= num_batches:
                break
            rows.extend(probe_one_batch(
                collector, data, data_next=prev_data, alpha=a,
                steps_list=steps_list, variants=variants, batch_idx=i,
                target_layers=target_layers,
                freeze_matching=freeze_matching, forward_seed=forward_seed,
                reset_temporal_state=reset_temporal_state,
            ))
            prev_data = data
        df = rows_to_dataframe(rows)
        df = df[df["source_task"].isin(sources)]
        df.to_csv(out_dir / f"sweep_alpha_{a:.0e}.csv", index=False)
        for variant in variants:
            sub = df[(df["steps"] == 1) & (df["variant"] == variant)]
            for s in sources:
                for t in sorted(sub["target_task"].unique()):
                    cell = sub[(sub["source_task"] == s) & (sub["target_task"] == t)]
                    if cell.empty:
                        continue
                    diag = (cell["delta"] > 0).mean() if s == t else float("nan")
                    lo, hi = bca_ci(cell["delta"].to_numpy(), n_resamples=2000, seed=0)
                    summary_rows.append({
                        "alpha": a, "variant": variant,
                        "source_task": s, "target_task": t,
                        "n": int(len(cell)),
                        "diag_violation_rate": float(diag) if s == t else float("nan"),
                        "mean_delta": float(cell["delta"].mean()),
                        "mean_delta_ci_lo": lo,
                        "mean_delta_ci_hi": hi,
                    })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    return summary
```

- [ ] **Step 2: Add config + CLI plumbing**

Append to `configs/gradient_analysis.yaml`:

```yaml
alpha_sweep:
  enabled: true
  alphas: [1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3]
  sources: [motion]
  num_batches: 100
  checkpoint: 1ep
```

In `tools/run_gradient_analysis.py`, add a new module branch keyed by `M_AS`:

```python
from tools.gradient_analysis.probe import run_alpha_sweep

if "M_AS" in modules or "alpha_sweep" in modules:
    sweep_cfg = cfg.get("alpha_sweep", {})
    if sweep_cfg.get("enabled", False) and ckpt_tag == sweep_cfg.get("checkpoint"):
        out_dir = ckpt_out_dir / "probe" / "alpha_sweep"
        run_alpha_sweep(
            collector=collector,
            dataloader=dataloader,
            num_batches=sweep_cfg.get("num_batches", 100),
            alphas=sweep_cfg["alphas"],
            sources=sweep_cfg["sources"],
            out_dir=out_dir,
            target_layers=cfg["probe"].get("target_layers", None) or None,
            freeze_matching=cfg["probe"].get("freeze_matching", True),
            forward_seed=cfg["probe"].get("forward_seed", 42),
            reset_temporal_state=cfg["probe"].get("reset_temporal_state", True),
        )
```

- [ ] **Step 3: Smoke run with a tiny sweep (3 batches × 2 alphas)**

Override the YAML ad-hoc:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M_AS \
    --no-supplementary \
    -o "alpha_sweep.alphas=[1.0e-4,1.0e-3]" \
    -o "alpha_sweep.num_batches=3"
```

(If the CLI doesn't support `-o` overrides, edit the YAML temporarily and revert after.)

Expected: produces `gradient_analysis_results/ckpt_1ep/probe/alpha_sweep/sweep_summary.csv` with two `alpha` rows for `motion`. Eyeball the `diag_violation_rate` for `(source=motion, target=motion)` — at the smaller α it should be lower or zero.

- [ ] **Step 4: Stop and review**

Sweep is end-to-end. The full 4-α / 100-batch run is deferred until Phase 1 #1–#10 are all in place (Task 18).

---

## Task 17: magnitude_dynamics — Per-Epoch Slope + Ratio Matrix Evolution

**Files:**
- Create: `tools/gradient_analysis/magnitude_dynamics.py`
- Create: `tests/gradient_analysis/test_magnitude_dynamics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/gradient_analysis/test_magnitude_dynamics.py`:

```python
"""Phase 1 #10 — per-task norm slopes + pairwise ratio matrix evolution."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.magnitude_dynamics import (
    compute_per_task_slopes,
    compute_ratio_evolution,
    non_stationarity_index,
)


def test_compute_per_task_slopes_returns_one_row_per_task():
    df = pd.DataFrame({
        "epoch": [1, 3, 6, 18, 1, 3, 6, 18],
        "task":  ["det"] * 4 + ["motion"] * 4,
        "mean_norm": [10.0, 12.0, 15.0, 20.0, 5.0, 7.0, 12.0, 25.0],
    })
    out = compute_per_task_slopes(df)
    assert set(out["task"]) == {"det", "motion"}
    # motion grew faster than det -> larger slope
    s_det = float(out.loc[out["task"] == "det", "slope"].iloc[0])
    s_motion = float(out.loc[out["task"] == "motion", "slope"].iloc[0])
    assert s_motion > s_det


def test_compute_ratio_evolution_shape():
    df = pd.DataFrame({
        "epoch": [1, 1, 6, 6],
        "task":  ["det", "motion", "det", "motion"],
        "mean_norm": [10.0, 5.0, 15.0, 12.0],
    })
    cube = compute_ratio_evolution(df, tasks=["det", "motion"])
    assert cube.shape == (2, 2, 2)        # tasks × tasks × epochs
    # det/motion at epoch 1 = 10/5 = 2; at epoch 6 = 15/12 = 1.25
    assert abs(cube[0, 1, 0] - 2.0) < 1e-6
    assert abs(cube[0, 1, 1] - 1.25) < 1e-6


def test_non_stationarity_index_is_cv_of_ratios():
    df = pd.DataFrame({
        "epoch": [1, 1, 6, 6],
        "task":  ["det", "motion", "det", "motion"],
        "mean_norm": [10.0, 5.0, 15.0, 12.0],
    })
    nsi = non_stationarity_index(df, tasks=["det", "motion"])
    # det vs motion ratio sequence: [2.0, 1.25] → CV = std / mean
    expected = float(np.std([2.0, 1.25], ddof=1) / np.mean([2.0, 1.25]))
    val = float(nsi.loc[("det", "motion"), "cv"])
    assert abs(val - expected) < 1e-6
```

- [ ] **Step 2: Run test — verify failure**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_magnitude_dynamics.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `tools/gradient_analysis/magnitude_dynamics.py`:

```python
"""M-N10 — Magnitude non-stationarity (Phase 1 #10).

Reads per-checkpoint per-task gradient norm CSVs and produces:

  * compute_per_task_slopes — linear regression of mean_norm vs epoch.
  * compute_ratio_evolution — (task × task × epoch) ratio cube.
  * non_stationarity_index — coefficient of variation per task pair across
    epochs.

Plus a `run_magnitude_dynamics` orchestrator that writes csv + heatmap-
evolution figures.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def compute_per_task_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per task with `slope`, `intercept`, `r2`. Input columns:
    `epoch`, `task`, `mean_norm`."""
    out_rows = []
    for task, sub in df.groupby("task"):
        x = sub["epoch"].to_numpy(dtype=np.float64)
        y = sub["mean_norm"].to_numpy(dtype=np.float64)
        if x.size < 2:
            slope, intercept, r2 = float("nan"), float("nan"), float("nan")
        else:
            xm, ym = x.mean(), y.mean()
            denom = float(((x - xm) ** 2).sum())
            slope = float(((x - xm) * (y - ym)).sum() / denom) if denom > 0 else 0.0
            intercept = float(ym - slope * xm)
            ss_res = float(((y - (slope * x + intercept)) ** 2).sum())
            ss_tot = float(((y - ym) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        out_rows.append({"task": task, "slope": slope, "intercept": intercept, "r2": r2})
    return pd.DataFrame(out_rows)


def compute_ratio_evolution(df: pd.DataFrame, tasks: List[str]) -> np.ndarray:
    """Return a (n_tasks, n_tasks, n_epochs) array of pairwise ratios sorted
    by epoch ascending."""
    epochs = sorted(df["epoch"].unique())
    cube = np.full((len(tasks), len(tasks), len(epochs)), np.nan, dtype=np.float64)
    for k, ep in enumerate(epochs):
        sub = df[df["epoch"] == ep]
        norms = {row["task"]: float(row["mean_norm"]) for _, row in sub.iterrows()}
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                ni = norms.get(ti)
                nj = norms.get(tj)
                if ni is None or nj is None or nj == 0:
                    continue
                cube[i, j, k] = ni / nj
    return cube


def non_stationarity_index(df: pd.DataFrame, tasks: List[str]) -> pd.DataFrame:
    """Coefficient of variation of each pairwise ratio across epochs.

    Returns a DataFrame indexed by `(task_a, task_b)` with column `cv`.
    Diagonal pairs are excluded.
    """
    cube = compute_ratio_evolution(df, tasks)
    rows = []
    for i, j in combinations(range(len(tasks)), 2):
        seq = cube[i, j, :]
        seq = seq[np.isfinite(seq)]
        if seq.size < 2:
            cv = float("nan")
        else:
            mean = seq.mean()
            cv = float(seq.std(ddof=1) / mean) if mean != 0 else float("nan")
        rows.append({"task_a": tasks[i], "task_b": tasks[j], "cv": cv})
    return pd.DataFrame(rows).set_index(["task_a", "task_b"])


def _emit_ratio_heatmap(cube: np.ndarray, tasks: List[str],
                         epochs: List, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(epochs)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.0), squeeze=False)
    for k, ep in enumerate(epochs):
        ax = axes[0, k]
        im = ax.imshow(cube[:, :, k], cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(tasks))); ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels(tasks, rotation=45); ax.set_yticklabels(tasks)
        ax.set_title(f"epoch {ep}")
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                v = cube[i, j, k]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                            color="white" if v > np.nanmean(cube[:, :, k]) else "black")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Pairwise gradient norm ratio (row / col) over epochs")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_magnitude_dynamics(
    per_task_norm: pd.DataFrame,
    tasks: List[str],
    out_dir: Path,
) -> Dict[str, pd.DataFrame]:
    """Top-level orchestrator. `per_task_norm` columns: `epoch, task, mean_norm`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slopes = compute_per_task_slopes(per_task_norm)
    slopes.to_csv(out_dir / "per_task_slopes.csv", index=False)
    nsi = non_stationarity_index(per_task_norm, tasks).reset_index()
    nsi.to_csv(out_dir / "non_stationarity_index.csv", index=False)
    epochs = sorted(per_task_norm["epoch"].unique())
    cube = compute_ratio_evolution(per_task_norm, tasks)
    np.save(out_dir / "ratio_evolution.npy", cube)
    _emit_ratio_heatmap(cube, tasks, epochs, out_dir / "ratio_evolution.png")
    return {"slopes": slopes, "nsi": nsi}
```

- [ ] **Step 4: Run test — verify pass**

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/test_magnitude_dynamics.py -v
```

Expected: all three tests pass.

- [ ] **Step 5: Stop and review**

Module + tests in place. Wire it into the CLI next.

---

## Task 18: magnitude_dynamics — CLI Hook + Run Across All Checkpoints

**Files:**
- Modify: `tools/run_gradient_analysis.py`

- [ ] **Step 1: Add the dispatch branch**

In `tools/run_gradient_analysis.py`, in the cross-checkpoint M6 section (after the per-ckpt loop), add:

```python
from tools.gradient_analysis.magnitude_dynamics import run_magnitude_dynamics

if "M_N10" in modules or "magnitude_dynamics" in modules:
    rows = []
    for tag, d in per_ckpt_dirs.items():
        ep = float(tag.rstrip("epoch").rstrip("ep"))
        nf = Path(d) / "gradnorm" / "per_task_norm.csv"
        if not nf.exists():
            continue
        ndf = pd.read_csv(nf)
        means = ndf.groupby("task")["norm"].mean()
        for t in tasks:
            if t not in means.index:
                continue
            rows.append({"epoch": ep, "task": t, "mean_norm": float(means[t])})
    per_task_norm = pd.DataFrame(rows)
    if not per_task_norm.empty:
        run_magnitude_dynamics(
            per_task_norm,
            tasks=cfg["tasks"],
            out_dir=Path(cfg["output_root"]) / "magnitude_dynamics",
        )
```

- [ ] **Step 2: Run on all four checkpoints**

```bash
PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --modules M_N10 \
    --no-supplementary
```

Expected: `gradient_analysis_results/magnitude_dynamics/ratio_evolution.png` exists with one panel per epoch.

- [ ] **Step 3: Stop and review**

Phase 1 #10 done. Move to running the full Phase 1 pipeline.

---

## Task 19: Run the Full Phase 1 Pipeline + Spot-Check the Outputs

**Files:**
- (none — orchestration)

- [ ] **Step 1: Run the complete Phase 1 across all 4 checkpoints**

```bash
mkdir -p gradient_analysis_results
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --modules M2,M3,M5,M6,M7,M_N1,M_N2,M_N10 \
    --no-supplementary \
    > gradient_analysis_results/phase1.log 2>&1 &
echo $! > gradient_analysis_results/phase1.pid
```

Monitor with `tail -f gradient_analysis_results/phase1.log`. Estimated wall-clock: ≈ 1 day on the existing primary cache (most of it is M3 probe re-execution).

- [ ] **Step 2: Once it's done, spot-check four artefacts**

```bash
ls -lh gradient_analysis_results/ckpt_1ep/null_baseline/null_baseline.csv
ls -lh gradient_analysis_results/ckpt_1ep/distribution/distribution_report.csv
ls -lh gradient_analysis_results/magnitude_dynamics/ratio_evolution.png
ls -lh gradient_analysis_results/summary_report.md
```

Each must exist and be > 0 bytes. Open `summary_report.md` — confirm:

- The pseudo-shared appendix is present.
- The antisymmetric Frobenius numbers are finite for every checkpoint.
- The bootstrap CI columns appear in the conflict / probe / gradnorm summaries.

- [ ] **Step 3: Run the α sweep on motion**

```bash
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M_AS \
    --no-supplementary \
    > gradient_analysis_results/alpha_sweep.log 2>&1 &
```

Estimated wall-clock: ≈ 0.5 day.

- [ ] **Step 4: Inspect `sweep_summary.csv`**

```bash
PYTHONPATH=. $PY -c "
import pandas as pd
df = pd.read_csv('gradient_analysis_results/ckpt_1ep/probe/alpha_sweep/sweep_summary.csv')
diag = df[(df.source_task == 'motion') & (df.target_task == 'motion')]
print(diag[['alpha', 'variant', 'diag_violation_rate', 'mean_delta']])
"
```

Expected: at α = 1e-4 the `diag_violation_rate` should be substantially lower than at α = 1e-3. If it is, log "α=1e-4 adopted as the headline alpha for motion" in the gating decision (Task 20). If violations persist at every α, raise R6 and run a freeze audit (out-of-scope for this plan).

- [ ] **Step 5: Stop and review**

All Phase 1 outputs exist. Final task: write the gating decision document.

---

## Task 20: Write `phase1_gating_decision.md`

**Files:**
- Create: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase1_gating_decision.md`

- [ ] **Step 1: Aggregate the inputs into one place for human review**

Run:

```bash
PYTHONPATH=. $PY -c "
import pandas as pd
from pathlib import Path

print('=== D1: Pass rate of null-baseline (per checkpoint) ===')
for ep in (1, 3, 6, 18):
    p = Path(f'gradient_analysis_results/ckpt_{ep}ep/null_baseline/null_baseline.csv')
    if not p.exists(): continue
    df = pd.read_csv(p)
    rate = df['passes_noise_threshold'].mean()
    print(f'  {ep}ep: {rate:.1%} of (group, pair, kind) cells pass')

print()
print('=== D2: Magnitude imbalance (max/min) per checkpoint ===')
for ep in (1, 3, 6, 18):
    p = Path(f'gradient_analysis_results/ckpt_{ep}ep/gradnorm/per_task_norm.csv')
    if not p.exists(): continue
    df = pd.read_csv(p)
    means = df.groupby('task')['norm'].mean()
    print(f'  {ep}ep: ratio = {means.max()/means.min():.2f}x ({means.idxmax()} / {means.idxmin()})')

print()
print('=== D3: Distribution shape distribution ===')
for ep in (1, 3, 6, 18):
    p = Path(f'gradient_analysis_results/ckpt_{ep}ep/distribution/distribution_report.csv')
    if not p.exists(): continue
    df = pd.read_csv(p)
    print(f'  {ep}ep:', df['shape_label'].value_counts().to_dict())

print()
print('=== D4: Drift slopes ===')
p = Path('gradient_analysis_results/magnitude_dynamics/per_task_slopes.csv')
if p.exists():
    print(pd.read_csv(p).to_string(index=False))
"
```

This produces a console summary covering D1, D2, D3, D4 (the four axes the gating decision keys on).

- [ ] **Step 2: Determine the branch**

Apply the spec's rule:

| Branch | Condition |
|---|---|
| Pass A | D1 ≥ 30 % of cells pass *and* (D2 ≥ severe *or* D3 has ≥ 1 bimodal cell per ckpt) |
| Pass B | D1 < 30 % *but* D2 ≥ severe *and* at least one task's drift slope's R² > 0.5 |
| Fail | D1 < 30 % *and* D2 < severe *and* D4 stable (slopes' R² < 0.3) |

Note the chosen branch and the supporting numbers.

- [ ] **Step 3: Write the markdown document**

Create `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase1_gating_decision.md`:

```markdown
# Phase 1 Gating Decision

**Date:** YYYY-MM-DD
**Pipeline run:** `gradient_analysis_results/phase1.log` (PID — see `phase1.pid`)
**α sweep run:** `gradient_analysis_results/alpha_sweep.log`

## Diagnostic axes — observed values

| Axis | 1 ep | 3 ep | 6 ep | 18 ep |
|---|---|---|---|---|
| D1 — null-baseline pass rate | … | … | … | … |
| D2 — magnitude ratio max/min | … | … | … | … |
| D3 — bimodal cells | … | … | … | … |
| D4 — drift slope R² (top task) | … | … | … | … |

## α-sweep result

| α | motion→motion violation rate (raw) | (normalized) |
|---|---|---|
| 1e-4 | … | … |
| 5e-4 | … | … |
| 1e-3 | … | … |
| 5e-3 | … | … |

Headline α adopted for motion in main-body tables: **α = …**.

## Decision

**Branch:** Pass A / Pass B / Fail (circle one).

**Rationale:** [Two to four sentences — quote the values above. Pass A requires the conjunction with D2 *or* D3; Pass B requires D2 severe with non-stable drift; Fail requires all three of D1 weak, D2 mild, D4 stable.]

## Implications for Phase 2 / 3

[One paragraph mapping the chosen branch to the Phase-2/3 framing per the spec's Section 5.5: which D rows the desiderata matrix retains, which algorithm families the gap analysis should focus on, and whether VAD analysis is still expected to confirm or could plausibly contradict the HiP-AD pattern.]

## Open follow-ups

- [List anything Task 19 / Task 20 surfaced as needing handling before Phase 2 starts — e.g. an unfrozen stochastic source if α=1e-4 didn't resolve diag>0.]
```

Fill the table cells with the numbers from Step 1. Fill the rationale and implications based on the chosen branch.

- [ ] **Step 4: Sanity check — does the decision read coherently?**

Read the document end-to-end. The values should support the chosen branch; if not, flag inconsistency back to the user.

- [ ] **Step 5: Stop and review**

Phase 1 complete. The file is the input the next plan (Phase 2) will brainstorm against.

---

## Self-Review

This section documents the self-review against the spec. Findings (none requiring rework — all spec points are covered):

**Spec coverage:**
- Spec § 3.1 #1 (null baseline) → Tasks 1–5 ✓
- Spec § 3.2 #2 (distribution) → Tasks 6–8 ✓
- Spec § 3.3 #3 (bootstrap) → Tasks 9–10 ✓
- Spec § 3.4 #4 B1 → Task 11 ✓
- Spec § 3.4 #4 B2/B3 → Task 12 ✓
- Spec § 3.4 #4 B4 → Task 13 ✓
- Spec § 3.4 M7 NaN fix → Task 14 (conditional) ✓
- Spec § 3.5 #9 variance bands → Task 15 ✓
- Spec § 3.5 #9 α sweep → Tasks 16, 19 step 3 ✓
- Spec § 3.6 #10 → Tasks 17–18 ✓
- Spec § 5.5 gating decision → Task 20 ✓

**Placeholder scan:** No "TBD" or "TODO" markers in any task body. Task 20's markdown template uses ellipses inside a fillable table, which is intentional (the engineer fills with actual numbers from Step 1's output).

**Type consistency:** `BatchGradients.nonzero_masks` introduced in Task 11 is consumed in Task 12 via the `norm_a > EPS_LOCAL` check on the existing per-batch DataFrame; the dataclass field is the source of the per-batch `norm_a` / `norm_b` values that already live in the M2 per-batch CSV. No mismatched names. `aggregate_affinity_matrix_with_ci` (Task 15) replaces no existing function — it sits alongside `aggregate_affinity_matrix` in `probe.py`. `run_alpha_sweep` (Task 16) reuses `probe_one_batch` and `rows_to_dataframe`, both already in `probe.py`.

---

## Execution Handoff

Plan complete. Per the user's directive ("그대로 코드 구현까지해 git commit은 구현 다 끝나고 한번에 할게"), commits are deferred to the end of Phase 1. Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task, review between tasks, fast iteration. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — Execute tasks in the current session using `superpowers:executing-plans`, batching tasks with checkpoints for review.

Recommended for this plan: **Inline execution** because (a) most tasks edit the same files (`run_gradient_analysis.py`, `probe.py`) so subagent context overhead doesn't pay off, and (b) Tasks 19–20 require human review of actual output values (gating decision), making subagent automation unhelpful at that point.
