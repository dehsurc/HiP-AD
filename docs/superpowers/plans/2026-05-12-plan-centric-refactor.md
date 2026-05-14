# Plan-Centric Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `phase2_hipad_vad_analysis_visualization.ipynb` and underlying gradient-analysis modules so that "does each task loss update actually help planning" is the main analysis, while the existing global-conflict work becomes a Diagnostic appendix.

**Architecture:** Compute helpers, summary builders, schema detection, and template-CSV writers go into a new `tools/gradient_analysis/plan_centric.py` (pure functions, unit-tested). Experiment-side instrumentation extends `probe.py` to emit `grad_dot` + `step_size`, and a new `tools/gradient_analysis/query_sensitivity.py` runner uses a `HipadAdapter.get_task_queries` hook to dump `dL_plan/dQ_task` for HiP-AD. The notebook becomes a thin orchestrator: imports builders, runs them, plots/saves figures, and demotes Parts A–E to an Appendix.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, pytest, PyTorch (existing). Notebook editing via `nbformat`.

**Reference spec:** `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/2026-05-12-plan-centric-refactor-design.md`

**Repo root for paths below:** `/home/yongjae/e2e/HiP-AD`. All relative paths in this plan are relative to that root.

**Working branch:** continue on the current branch (`nusc/pcgrad`).

---

## Task 1: Module skeleton + ensure_columns

**Files:**
- Create: `tools/gradient_analysis/plan_centric.py`
- Create: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/gradient_analysis/test_plan_centric.py`:

```python
import warnings

import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis import plan_centric as pc


def test_ensure_columns_returns_true_when_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert pc.ensure_columns(df, ["a", "b"], "df") is True


def test_ensure_columns_returns_false_and_warns_when_missing():
    df = pd.DataFrame({"a": [1]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok = pc.ensure_columns(df, ["a", "b", "c"], "myframe")
    assert ok is False
    assert any("myframe" in str(w.message) and "b" in str(w.message) for w in caught)
```

- [ ] **Step 2: Run test, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: ImportError / ModuleNotFoundError on `plan_centric`.

- [ ] **Step 3: Create the module with EPS constant and ensure_columns**

Create `tools/gradient_analysis/plan_centric.py`:

```python
"""Plan-centric directed transfer analysis (Phase 2 refactor).

Compute helpers and summary builders for the post-conflict, plan-target
analysis. The matching notebook is
``docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/
phase2_hipad_vad_analysis_visualization.ipynb``; the design doc is in the
same directory dated 2026-05-12.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd

EPS = 1e-8


def ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> bool:
    """Warn (don't raise) when required columns are absent.

    Returns True iff all required columns are present. Caller decides whether
    to SKIP or continue.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        warnings.warn(
            f"[{df_name}] missing columns: {missing}",
            stacklevel=2,
        )
        return False
    return True
```

- [ ] **Step 4: Run test, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): scaffold module with ensure_columns helper"
```

---

## Task 2: bootstrap_ci

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/gradient_analysis/test_plan_centric.py`:

```python
def test_bootstrap_ci_returns_nan_for_small_n():
    lo, hi = pc.bootstrap_ci([1.0, 2.0])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_brackets_mean_for_normal_sample():
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=2.0, scale=0.5, size=200)
    lo, hi = pc.bootstrap_ci(sample, n_boot=500, seed=1)
    assert lo < 2.0 < hi
    assert hi - lo < 0.5


def test_bootstrap_ci_filters_nonfinite():
    sample = [1.0, 2.0, np.nan, np.inf, 3.0, 4.0, 5.0, 6.0]
    lo, hi = pc.bootstrap_ci(sample, n_boot=200, seed=0)
    assert np.isfinite(lo) and np.isfinite(hi)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.bootstrap_ci`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def bootstrap_ci(
    values,
    stat_fn=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``stat_fn`` over ``values``.

    Returns ``(nan, nan)`` when fewer than 5 finite samples are available.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boots[i] = stat_fn(v[idx[i]])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add bootstrap_ci helper"
```

---

## Task 3: practical_threshold

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_practical_threshold_floors_at_min_value():
    assert pc.practical_threshold([0.0, 0.0, 0.0]) == pytest.approx(1e-8)


def test_practical_threshold_scales_with_median_abs():
    series = [-10.0, -5.0, 0.0, 5.0, 10.0]  # median(|·|) = 5
    assert pc.practical_threshold(series, ratio=0.1) == pytest.approx(0.5)


def test_practical_threshold_ignores_nonfinite():
    series = [np.nan, np.inf, 4.0, -4.0, 0.0]
    assert pc.practical_threshold(series, ratio=0.1) == pytest.approx(0.4)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.practical_threshold`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def practical_threshold(
    series, ratio: float = 0.1, min_value: float = 1e-8
) -> float:
    """Return ``max(min_value, ratio * median(|series|))``.

    Noise-robust threshold used by helpful/harmful rate calculations so that
    arbitrarily small deltas are not counted as helpful.
    """
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(min_value)
    return float(max(min_value, ratio * float(np.median(np.abs(v)))))
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add practical_threshold helper"
```

---

## Task 4: standardize_probe_df

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def _make_probe(loss_before=True, rel_delta=True):
    cols = {
        "batch_idx": [0, 1],
        "source_task": ["det", "map"],
        "target_task": ["plan", "plan"],
        "steps": [1, 1],
        "variant": ["normalized", "normalized"],
        "layer": ["_all", "_all"],
        "grad_norm": [2.0, 3.0],
        "delta": [-0.1, 0.2],
    }
    if loss_before:
        cols["baseline_loss"] = [1.0, 2.0]
        cols["stepped_loss"] = [0.9, 2.2]
    if rel_delta:
        cols["rel_delta"] = [-0.1, 0.1]
    return pd.DataFrame(cols)


def test_standardize_probe_df_aliases_loss_columns():
    df = pc.standardize_probe_df(_make_probe())
    assert "loss_before" in df.columns
    assert "loss_after" in df.columns
    assert df.loc[0, "loss_before"] == pytest.approx(1.0)
    assert df.loc[1, "loss_after"] == pytest.approx(2.2)


def test_standardize_probe_df_computes_gain():
    df = pc.standardize_probe_df(_make_probe())
    assert df.loc[0, "gain"] == pytest.approx(0.1)
    assert df.loc[1, "gain"] == pytest.approx(-0.2)


def test_standardize_probe_df_computes_delta_rel_when_losses_present():
    df = pc.standardize_probe_df(_make_probe())
    # delta_rel = (after - before) / |before| = -0.1 / 1.0 = -0.1
    assert df.loc[0, "delta_rel"] == pytest.approx(-0.1)
    assert df.loc[0, "gain_rel"] == pytest.approx(0.1)


def test_standardize_probe_df_falls_back_to_rel_delta_when_no_losses():
    base = _make_probe(loss_before=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = pc.standardize_probe_df(base)
    assert df.loc[0, "delta_rel"] == pytest.approx(-0.1)
    # warning fired because losses were absent
    assert any("loss_before" in str(w.message) or "loss_after" in str(w.message)
               for w in caught)


def test_standardize_probe_df_emits_nan_when_no_losses_and_no_rel_delta():
    base = _make_probe(loss_before=False, rel_delta=False)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        df = pc.standardize_probe_df(base)
    assert df["delta_rel"].isna().all()
    assert df["gain_rel"].isna().all()
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.standardize_probe_df`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def standardize_probe_df(probe: pd.DataFrame) -> pd.DataFrame:
    """Add ``gain``, ``gain_rel``, ``delta_rel`` and ``loss_before/after`` aliases.

    The on-disk probe schema uses ``baseline_loss`` / ``stepped_loss``; the
    spec text uses ``loss_before`` / ``loss_after``. We expose both names so
    downstream code can pick either.
    """
    df = probe.copy()

    if "loss_before" not in df.columns and "baseline_loss" in df.columns:
        df["loss_before"] = df["baseline_loss"]
    if "loss_after" not in df.columns and "stepped_loss" in df.columns:
        df["loss_after"] = df["stepped_loss"]

    df["gain"] = -df["delta"]

    if {"loss_before", "loss_after"}.issubset(df.columns):
        df["delta_rel"] = (df["loss_after"] - df["loss_before"]) / (
            df["loss_before"].abs() + EPS
        )
        df["gain_rel"] = -df["delta_rel"]
    elif "rel_delta" in df.columns:
        df["delta_rel"] = df["rel_delta"]
        df["gain_rel"] = -df["rel_delta"]
        warnings.warn(
            "standardize_probe_df: loss_before/loss_after absent; "
            "falling back to existing rel_delta as delta_rel.",
            stacklevel=2,
        )
    else:
        df["delta_rel"] = np.nan
        df["gain_rel"] = np.nan
        warnings.warn(
            "standardize_probe_df: loss_before/loss_after and rel_delta "
            "both absent; delta_rel/gain_rel set to NaN.",
            stacklevel=2,
        )
    return df
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add standardize_probe_df with loss aliases and gain"
```

---

## Task 5: get_probe_base

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing test**

```python
def test_get_probe_base_filters_variant_and_steps():
    df = pd.DataFrame({
        "batch_idx": [0, 0, 0],
        "source_task": ["det", "det", "det"],
        "target_task": ["plan", "plan", "plan"],
        "steps": [1, 1, 2],
        "variant": ["normalized", "raw", "normalized"],
        "layer": ["_all", "_all", "_all"],
        "grad_norm": [1.0, 1.0, 1.0],
        "baseline_loss": [1.0, 1.0, 1.0],
        "stepped_loss": [0.9, 0.8, 0.7],
        "delta": [-0.1, -0.2, -0.3],
    })
    base = pc.get_probe_base(df, variant="normalized", steps=1)
    assert len(base) == 1
    assert "gain" in base.columns
    assert base.iloc[0]["gain"] == pytest.approx(0.1)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.get_probe_base`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def get_probe_base(
    probe: pd.DataFrame, variant: str = "normalized", steps: int = 1
) -> pd.DataFrame:
    """Standardize, then filter to ``(variant, steps)`` subset."""
    df = standardize_probe_df(probe)
    return df[(df["variant"] == variant) & (df["steps"] == steps)].copy()
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add get_probe_base subset helper"
```

---

## Task 6: build_plan_transfer_summary (Part F core)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing test**

```python
def _plan_target_fixture():
    rows = []
    rng = np.random.default_rng(0)
    for source in ["det", "map", "motion", "plan"]:
        # det → plan: helpful (delta < 0 mean)
        if source == "det":
            deltas = rng.normal(-0.05, 0.02, size=20)
        elif source == "map":
            deltas = rng.normal(0.0, 0.02, size=20)
        elif source == "motion":
            deltas = rng.normal(0.05, 0.02, size=20)
        else:
            deltas = rng.normal(-0.1, 0.02, size=20)  # self diag, large helpful
        for i, d in enumerate(deltas):
            rows.append({
                "model": "HiP-AD",
                "checkpoint": "1ep",
                "checkpoint_order": 1,
                "batch_idx": i,
                "source_task": source,
                "target_task": "plan",
                "steps": 1,
                "variant": "normalized",
                "layer": "dec3_ffn_0",
                "grad_norm": 1.0,
                "baseline_loss": 1.0,
                "stepped_loss": 1.0 + d,
                "delta": d,
            })
    return pd.DataFrame(rows)


def test_build_plan_transfer_summary_groups_and_signs():
    base = pc.standardize_probe_df(_plan_target_fixture())
    summary = pc.build_plan_transfer_summary(base)
    expected_cols = {
        "model", "checkpoint", "checkpoint_order", "layer", "source_task",
        "n", "mean_delta", "median_delta", "p05_delta", "p95_delta",
        "mean_gain", "median_gain",
        "helpful_rate", "harmful_rate",
        "practical_helpful_rate", "large_harm_rate",
        "std_delta", "sem_delta",
        "ci_lo_delta", "ci_hi_delta",
        "ci_lo_gain", "ci_hi_gain",
        "mean_grad_norm", "effect_size", "tau",
    }
    assert expected_cols.issubset(set(summary.columns))
    det_row = summary[summary["source_task"] == "det"].iloc[0]
    motion_row = summary[summary["source_task"] == "motion"].iloc[0]
    # det → plan: helpful (mean_gain > 0). motion → plan: harmful (mean_gain < 0).
    assert det_row["mean_gain"] > 0
    assert motion_row["mean_gain"] < 0
    # helpful_rate must be in [0, 1]
    assert (summary["helpful_rate"].between(0, 1)).all()
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.build_plan_transfer_summary`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
_MAIN_SOURCES = ("det", "map", "motion", "plan")
_GROUP_KEYS_FH = ["model", "checkpoint", "checkpoint_order", "layer", "source_task"]


def _ci_pair(values, seed_base: int) -> tuple[float, float]:
    return bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, seed=seed_base)


def build_plan_transfer_summary(base: pd.DataFrame) -> pd.DataFrame:
    """Part F. Plan-centric directed transfer summary.

    ``base`` must already be ``get_probe_base(..., variant="normalized",
    steps=1)`` output. Returns one row per ``(model, checkpoint, layer,
    source_task)`` restricted to ``target_task == "plan"``.
    """
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", "gain", "grad_norm"}
    if not ensure_columns(base, required, "plan_transfer_summary input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

    plan_df = base[
        (base["target_task"] == "plan")
        & (base["source_task"].isin(_MAIN_SOURCES))
    ].copy()
    if plan_df.empty:
        return pd.DataFrame(columns=_GROUP_KEYS_FH + [
            "n", "mean_delta", "median_delta", "p05_delta", "p95_delta",
            "mean_gain", "median_gain",
            "helpful_rate", "harmful_rate",
            "practical_helpful_rate", "large_harm_rate",
            "std_delta", "sem_delta",
            "ci_lo_delta", "ci_hi_delta",
            "ci_lo_gain", "ci_hi_gain",
            "mean_grad_norm", "effect_size", "tau",
        ])

    # tau is per (model, checkpoint) over the plan-target subset.
    tau_per_mc = (
        plan_df.groupby(["model", "checkpoint"])["delta"]
        .apply(lambda s: practical_threshold(s.values))
        .to_dict()
    )
    plan_df["tau"] = plan_df.apply(
        lambda r: tau_per_mc[(r["model"], r["checkpoint"])], axis=1
    )

    rows = []
    seed = 0
    for keys, sub in plan_df.groupby(_GROUP_KEYS_FH, sort=False):
        d = sub["delta"].to_numpy()
        g = sub["gain"].to_numpy()
        tau = float(sub["tau"].iloc[0])
        n = int(d.size)
        ci_lo_d, ci_hi_d = _ci_pair(d, seed)
        ci_lo_g, ci_hi_g = _ci_pair(g, seed + 1)
        seed += 2
        std_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        sem_d = std_d / np.sqrt(n) if n > 1 else float("nan")
        mean_g = float(np.mean(g))
        rows.append({
            **dict(zip(_GROUP_KEYS_FH, keys)),
            "n": n,
            "mean_delta": float(np.mean(d)),
            "median_delta": float(np.median(d)),
            "p05_delta": float(np.quantile(d, 0.05)),
            "p95_delta": float(np.quantile(d, 0.95)),
            "mean_gain": mean_g,
            "median_gain": float(np.median(g)),
            "helpful_rate": float(np.mean(d < 0)),
            "harmful_rate": float(np.mean(d > 0)),
            "practical_helpful_rate": float(np.mean(d < -tau)),
            "large_harm_rate": float(np.mean(d > tau)),
            "std_delta": std_d,
            "sem_delta": sem_d,
            "ci_lo_delta": ci_lo_d,
            "ci_hi_delta": ci_hi_d,
            "ci_lo_gain": ci_lo_g,
            "ci_hi_gain": ci_hi_g,
            "mean_grad_norm": float(sub["grad_norm"].mean()),
            "effect_size": mean_g / (std_d + EPS) if np.isfinite(std_d) else float("nan"),
            "tau": tau,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add build_plan_transfer_summary (Part F core)"
```

---

## Task 7: top_beneficial / top_harmful

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_top_beneficial_sorts_descending_by_mean_gain():
    summary = pd.DataFrame({
        "model": ["HiP-AD"] * 3,
        "checkpoint": ["1ep"] * 3,
        "layer": ["a", "b", "c"],
        "source_task": ["det"] * 3,
        "mean_gain": [0.1, 0.3, 0.05],
        "practical_helpful_rate": [0.5, 0.7, 0.4],
        "large_harm_rate": [0.0, 0.0, 0.0],
    })
    top = pc.top_beneficial(summary, k=2, by="mean_gain")
    assert list(top["layer"]) == ["b", "a"]


def test_top_harmful_sorts_ascending_by_mean_gain():
    summary = pd.DataFrame({
        "model": ["HiP-AD"] * 3,
        "checkpoint": ["1ep"] * 3,
        "layer": ["a", "b", "c"],
        "source_task": ["det"] * 3,
        "mean_gain": [-0.2, 0.0, -0.5],
        "practical_helpful_rate": [0.1, 0.2, 0.0],
        "large_harm_rate": [0.5, 0.0, 0.8],
    })
    top = pc.top_harmful(summary, k=2, by="mean_gain")
    assert list(top["layer"]) == ["c", "a"]
    top_harm = pc.top_harmful(summary, k=2, by="large_harm_rate")
    assert list(top_harm["layer"]) == ["c", "a"]
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.top_beneficial`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def top_beneficial(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows ranked by ``by`` descending (most beneficial first)."""
    return summary.sort_values(by, ascending=False).head(k).reset_index(drop=True)


def top_harmful(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows.

    For ``by="mean_gain"`` sorts ascending (worst gain first). For
    ``by="large_harm_rate"`` sorts descending. Other columns sort descending
    by default; flip your input if you need the opposite.
    """
    ascending = by == "mean_gain"
    return summary.sort_values(by, ascending=ascending).head(k).reset_index(drop=True)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 17 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add top_beneficial/top_harmful rankers"
```

---

## Task 8: build_asymmetry_summary + interpret_asymmetry_row (Part G)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def _asym_fixture(n=20):
    rng = np.random.default_rng(1)
    rows = []
    # det → plan: helpful;  plan → det: harmful  → asym > 0, "auxiliary useful, plan doesn't preserve det"
    # map → plan: helpful;  plan → map: helpful  → both positive, "mutually complementary"
    # motion → plan: ≈ 0;   plan → motion: ≈ 0   → "transfer 미미"
    plans = {
        ("det", "plan"): rng.normal(-0.05, 0.01, n),
        ("plan", "det"): rng.normal(+0.05, 0.01, n),
        ("map", "plan"): rng.normal(-0.05, 0.01, n),
        ("plan", "map"): rng.normal(-0.05, 0.01, n),
        ("motion", "plan"): rng.normal(0.0, 0.001, n),
        ("plan", "motion"): rng.normal(0.0, 0.001, n),
        # self-rows for plan_transfer_ratio denominator
        ("det", "det"): rng.normal(-0.2, 0.01, n),
        ("map", "map"): rng.normal(-0.2, 0.01, n),
        ("motion", "motion"): rng.normal(-0.05, 0.005, n),
    }
    for (s, t), deltas in plans.items():
        for i, d in enumerate(deltas):
            rows.append({
                "model": "HiP-AD", "checkpoint": "1ep", "checkpoint_order": 1,
                "batch_idx": i, "source_task": s, "target_task": t,
                "steps": 1, "variant": "normalized", "layer": "L0",
                "grad_norm": 1.0,
                "baseline_loss": 1.0, "stepped_loss": 1.0 + d, "delta": d,
            })
    return pd.DataFrame(rows)


def test_build_asymmetry_summary_signs_and_interpretation():
    base = pc.standardize_probe_df(_asym_fixture())
    summary = pc.build_asymmetry_summary(base)
    assert {"model", "checkpoint", "layer", "aux_task",
            "gain_A_to_plan", "gain_plan_to_A",
            "asymmetry", "plan_transfer_ratio",
            "helpful_A_to_plan", "helpful_plan_to_A",
            "interpretation"}.issubset(summary.columns)
    by_aux = summary.set_index("aux_task")
    assert by_aux.loc["det", "asymmetry"] > 0
    assert "보존하지 않는다" in by_aux.loc["det", "interpretation"]
    assert "상호 보완" in by_aux.loc["map", "interpretation"]
    assert "도달" in by_aux.loc["motion", "interpretation"] or \
           "미미" in by_aux.loc["motion", "interpretation"]


def test_interpret_asymmetry_row_covers_four_quadrants():
    tau = 0.01
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.05,
                                      "gain_plan_to_A": -0.05, "tau": tau})
    assert "보존하지" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.05,
                                      "gain_plan_to_A":  0.05, "tau": tau})
    assert "상호 보완" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan": -0.05,
                                      "gain_plan_to_A": -0.05, "tau": tau})
    assert "상호 간섭" in msg
    msg = pc.interpret_asymmetry_row({"gain_A_to_plan":  0.0,
                                      "gain_plan_to_A":  0.05, "tau": tau})
    assert "도달" in msg or "미미" in msg
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.build_asymmetry_summary` / `pc.interpret_asymmetry_row`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
_AUX_TASKS = ("det", "map", "motion")
_GROUP_KEYS_G = ["model", "checkpoint", "checkpoint_order", "layer", "aux_task"]


def interpret_asymmetry_row(row: dict) -> str:
    """Return one of four Korean messages based on (gain_A_to_plan, gain_plan_to_A, tau)."""
    g_ap = float(row["gain_A_to_plan"])
    g_pa = float(row["gain_plan_to_A"])
    tau = float(row.get("tau", 0.0))
    if abs(g_ap) <= tau:
        return "A→plan transfer가 미미. A task update가 planning에 잘 도달하지 않을 가능성."
    if g_ap > tau and g_pa < -tau:
        return ("A는 planning auxiliary로 유용하지만, planning update는 A "
                "task representation을 보존하지 않는다.")
    if g_ap > tau and g_pa > tau:
        return "A와 planning 사이에 상호 보완적 관계가 있다."
    if g_ap < -tau and g_pa < -tau:
        return "A와 planning은 해당 layer에서 상호 간섭 가능성이 있다."
    # Mixed (e.g. A→plan helpful, plan→A also small): default to "useful, but plan-side weak"
    return ("A는 planning auxiliary로 유용하지만, planning update의 A 방향 "
            "효과는 약하다.")


def build_asymmetry_summary(base: pd.DataFrame) -> pd.DataFrame:
    """Part G. Directed asymmetry around planning, per (model, ckpt, layer, aux_task)."""
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", "gain"}
    if not ensure_columns(base, required, "asymmetry_summary input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

    # tau per (model, ckpt) computed on plan-target subset
    plan_subset = base[base["target_task"] == "plan"]
    tau_per_mc = (
        plan_subset.groupby(["model", "checkpoint"])["delta"]
        .apply(lambda s: practical_threshold(s.values))
        .to_dict()
    )

    rows = []
    seed = 0
    grouper = ["model", "checkpoint", "checkpoint_order", "layer"]
    for keys, sub in base.groupby(grouper, sort=False):
        for aux in _AUX_TASKS:
            ap = sub[(sub["source_task"] == aux) & (sub["target_task"] == "plan")]
            pa = sub[(sub["source_task"] == "plan") & (sub["target_task"] == aux)]
            self_a = sub[(sub["source_task"] == aux) & (sub["target_task"] == aux)]
            if ap.empty and pa.empty:
                continue
            gain_ap = float(ap["gain"].mean()) if not ap.empty else float("nan")
            gain_pa = float(pa["gain"].mean()) if not pa.empty else float("nan")
            self_g = float(self_a["gain"].mean()) if not self_a.empty else float("nan")
            asym = gain_ap - gain_pa
            ratio = (gain_ap / (abs(self_g) + EPS)) if np.isfinite(self_g) else float("nan")
            helpful_ap = float((ap["delta"] < 0).mean()) if not ap.empty else float("nan")
            helpful_pa = float((pa["delta"] < 0).mean()) if not pa.empty else float("nan")
            # paired CI on the asymmetry: bootstrap per-batch difference of means
            # use simplest form — CI on (ap.delta).mean() and (pa.delta).mean() separately
            ci_lo_ap, ci_hi_ap = _ci_pair(ap["gain"].to_numpy(), seed)
            ci_lo_pa, ci_hi_pa = _ci_pair(pa["gain"].to_numpy(), seed + 1)
            seed += 2
            tau = tau_per_mc.get((keys[0], keys[1]), 0.0)
            r = {
                **dict(zip(grouper, keys)),
                "aux_task": aux,
                "n_A_to_plan": int(len(ap)),
                "n_plan_to_A": int(len(pa)),
                "n_self_A": int(len(self_a)),
                "gain_A_to_plan": gain_ap,
                "gain_plan_to_A": gain_pa,
                "self_gain_A": self_g,
                "asymmetry": asym,
                "plan_transfer_ratio": ratio,
                "helpful_A_to_plan": helpful_ap,
                "helpful_plan_to_A": helpful_pa,
                "ci_lo_ap": ci_lo_ap, "ci_hi_ap": ci_hi_ap,
                "ci_lo_pa": ci_lo_pa, "ci_hi_pa": ci_hi_pa,
                "tau": tau,
            }
            # Aliases requested by spec
            r["ci_lo_asym"] = ci_lo_ap - ci_hi_pa
            r["ci_hi_asym"] = ci_hi_ap - ci_lo_pa
            r["interpretation"] = interpret_asymmetry_row(r)
            rows.append(r)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add asymmetry summary and interpreter (Part G)"
```

---

## Task 9: build_effect_size_summary (Part H)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_build_effect_size_summary_columns_and_flags():
    base = pc.standardize_probe_df(_plan_target_fixture())
    summary = pc.build_effect_size_summary(base, target="plan")
    assert {"model", "checkpoint", "layer", "source_task",
            "n", "helpful_rate", "practical_helpful_rate", "large_harm_rate",
            "mean_gain", "median_gain", "std_delta", "effect_size",
            "ci_lo_gain", "ci_hi_gain", "ci_contains_zero",
            "flag_high_helpful_low_gain", "flag_positive_but_insig",
            "flag_high_practical_helpful", "flag_high_large_harm"}.issubset(summary.columns)
    # Bool columns must be 0/1 or bool dtype
    assert summary["ci_contains_zero"].dtype == bool or set(summary["ci_contains_zero"].unique()).issubset({True, False})
    # All rows must have target=plan source in MAIN_SOURCES (filter sanity)
    assert summary["source_task"].isin(["det", "map", "motion", "plan"]).all()
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.build_effect_size_summary`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def build_effect_size_summary(base: pd.DataFrame, target: str = "plan") -> pd.DataFrame:
    """Part H. Effect-size-aware 1-step probe summary for the given target."""
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", "gain"}
    if not ensure_columns(base, required, "effect_size_summary input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

    sub = base[(base["target_task"] == target)
               & (base["source_task"].isin(_MAIN_SOURCES))].copy()
    if sub.empty:
        return pd.DataFrame(columns=_GROUP_KEYS_FH + [
            "n", "helpful_rate", "practical_helpful_rate", "large_harm_rate",
            "mean_gain", "median_gain", "std_delta", "effect_size",
            "ci_lo_gain", "ci_hi_gain", "ci_contains_zero",
            "flag_high_helpful_low_gain", "flag_positive_but_insig",
            "flag_high_practical_helpful", "flag_high_large_harm", "tau",
        ])

    tau_per_mc = (
        sub.groupby(["model", "checkpoint"])["delta"]
        .apply(lambda s: practical_threshold(s.values))
        .to_dict()
    )

    rows = []
    seed = 1000
    for keys, g in sub.groupby(_GROUP_KEYS_FH, sort=False):
        d = g["delta"].to_numpy()
        gain = g["gain"].to_numpy()
        tau = float(tau_per_mc[(keys[0], keys[1])])
        n = int(d.size)
        helpful = float(np.mean(d < 0))
        practical_helpful = float(np.mean(d < -tau))
        large_harm = float(np.mean(d > tau))
        mean_g = float(np.mean(gain))
        std_d = float(np.std(d, ddof=1)) if n > 1 else float("nan")
        es = mean_g / (std_d + EPS) if np.isfinite(std_d) else float("nan")
        ci_lo_g, ci_hi_g = _ci_pair(gain, seed)
        seed += 1
        ci_contains_zero = bool((ci_lo_g <= 0 <= ci_hi_g))
        rows.append({
            **dict(zip(_GROUP_KEYS_FH, keys)),
            "n": n,
            "helpful_rate": helpful,
            "practical_helpful_rate": practical_helpful,
            "large_harm_rate": large_harm,
            "mean_gain": mean_g,
            "median_gain": float(np.median(gain)),
            "std_delta": std_d,
            "effect_size": es,
            "ci_lo_gain": ci_lo_g,
            "ci_hi_gain": ci_hi_g,
            "ci_contains_zero": ci_contains_zero,
            "flag_high_helpful_low_gain": bool(helpful >= 0.6 and abs(mean_g) < tau),
            "flag_positive_but_insig": bool(mean_g > 0 and ci_contains_zero),
            "flag_high_practical_helpful": bool(practical_helpful >= 0.5),
            "flag_high_large_harm": bool(large_harm >= 0.5),
            "tau": tau,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add build_effect_size_summary (Part H)"
```

---

## Task 10: detect_first_order_columns + build_first_order_residual_summary (Part I)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_detect_first_order_columns_returns_none_when_missing():
    df = pd.DataFrame({"delta": [0.0]})
    assert pc.detect_first_order_columns(df) is None


def test_detect_first_order_columns_finds_canonical_names():
    df = pd.DataFrame({"delta": [0.0], "grad_dot": [0.0], "step_size": [1e-3]})
    cols = pc.detect_first_order_columns(df)
    assert cols == {"grad_dot": "grad_dot", "step_size": "step_size"}


def test_first_order_residual_summary_pearson_one_when_pred_matches():
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({
        "model": ["HiP-AD"] * n, "checkpoint": ["1ep"] * n,
        "checkpoint_order": [1] * n, "layer": ["L0"] * n,
        "source_task": ["det"] * n, "target_task": ["plan"] * n,
        "step_size": [1e-3] * n,
        "grad_dot": rng.normal(0, 1, n),
    })
    df["delta"] = -df["step_size"] * df["grad_dot"]  # exact first-order
    base = pc.standardize_probe_df(
        df.assign(steps=1, variant="normalized", batch_idx=range(n),
                  grad_norm=1.0, baseline_loss=1.0,
                  stepped_loss=1.0 + df["delta"]))
    cols = {"grad_dot": "grad_dot", "step_size": "step_size"}
    summary = pc.build_first_order_residual_summary(base, cols)
    assert summary["pearson_actual_pred"].iloc[0] == pytest.approx(1.0, abs=1e-6)
    assert summary["mean_residual"].iloc[0] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.detect_first_order_columns`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
def detect_first_order_columns(probe: pd.DataFrame) -> dict | None:
    """Return the canonical first-order column names if both are present.

    Searches common aliases so probe CSVs from older runs auto-detect.
    """
    grad_dot_aliases = ("grad_dot", "dot", "grad_dot_source_target")
    step_size_aliases = ("step_size", "lr", "learning_rate", "probe_lr")
    gd = next((c for c in grad_dot_aliases if c in probe.columns), None)
    ss = next((c for c in step_size_aliases if c in probe.columns), None)
    if gd is None or ss is None:
        return None
    return {"grad_dot": gd, "step_size": ss}


def _safe_corr(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if kind == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    # spearman via rankdata
    from scipy.stats import rankdata
    rx = rankdata(x); ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def build_first_order_residual_summary(base: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Part I. First-order prediction vs actual 1-step delta.

    Adds ``pred_delta = -step_size * grad_dot`` and ``residual = delta - pred``.
    Returns one row per ``(model, checkpoint, layer, source_task, target_task)``.
    """
    gd = cols["grad_dot"]; ss = cols["step_size"]
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", gd, ss}
    if not ensure_columns(base, required, "first_order input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

    df = base.copy()
    df["pred_delta"] = -df[ss] * df[gd]
    df["residual"] = df["delta"] - df["pred_delta"]
    grouper = ["model", "checkpoint", "checkpoint_order", "layer",
               "source_task", "target_task"]
    rows = []
    for keys, g in df.groupby(grouper, sort=False):
        actual = g["delta"].to_numpy()
        pred = g["pred_delta"].to_numpy()
        resid = g["residual"].to_numpy()
        pearson = _safe_corr(actual, pred, "pearson")
        spearman = _safe_corr(actual, pred, "spearman")
        rows.append({
            **dict(zip(grouper, keys)),
            "n": int(actual.size),
            "mean_actual": float(np.mean(actual)),
            "mean_pred": float(np.mean(pred)),
            "mean_residual": float(np.mean(resid)),
            "std_residual": float(np.std(resid, ddof=1)) if actual.size > 1 else float("nan"),
            "pearson_actual_pred": pearson,
            "spearman_actual_pred": spearman,
            "r_squared": pearson ** 2 if np.isfinite(pearson) else float("nan"),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 23 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add first-order residual summary (Part I)"
```

---

## Task 11: load_or_template_query_sensitivity + build_query_sensitivity_summary (Part J data layer)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_load_or_template_query_sensitivity_writes_template_when_missing(tmp_path):
    path = tmp_path / "qs.csv"
    df, status = pc.load_or_template_query_sensitivity(path)
    assert df is None
    assert status == "template"
    assert path.exists()
    template = pd.read_csv(path)
    expected = {
        "model", "checkpoint", "checkpoint_order", "batch_idx", "scene_token",
        "layer", "task_query_type", "query_index", "query_norm",
        "grad_plan_wrt_query_norm",
    }
    assert expected.issubset(template.columns)


def test_load_or_template_query_sensitivity_reads_existing(tmp_path):
    path = tmp_path / "qs.csv"
    pd.DataFrame({
        "model": ["HiP-AD"], "checkpoint": ["1ep"], "checkpoint_order": [1],
        "batch_idx": [0], "scene_token": ["s"], "layer": ["L"],
        "task_query_type": ["det"], "query_index": [0],
        "query_norm": [1.0], "grad_plan_wrt_query_norm": [0.5],
    }).to_csv(path, index=False)
    df, status = pc.load_or_template_query_sensitivity(path)
    assert status == "loaded"
    assert len(df) == 1


def test_build_query_sensitivity_summary_basic():
    df = pd.DataFrame({
        "model": ["HiP-AD"] * 6,
        "checkpoint": ["1ep"] * 6,
        "checkpoint_order": [1] * 6,
        "layer": ["L0"] * 6,
        "task_query_type": ["det", "det", "map", "map", "motion", "motion"],
        "query_norm": [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
        "grad_plan_wrt_query_norm": [0.2, 0.4, 0.1, 0.1, 0.0, 0.0],
    })
    out = pc.build_query_sensitivity_summary(df)
    assert {"model", "checkpoint", "layer", "task_query_type",
            "mean_sensitivity", "median_sensitivity", "p05", "p95",
            "mean_query_norm", "sensitivity_normed_mean",
            "share_task"}.issubset(out.columns)
    by_task = out.set_index("task_query_type")
    assert by_task.loc["det", "share_task"] > by_task.loc["motion", "share_task"]
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.load_or_template_query_sensitivity`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
QUERY_SENSITIVITY_TEMPLATE_COLUMNS = [
    "model", "checkpoint", "checkpoint_order", "batch_idx", "scene_token",
    "layer", "task_query_type", "query_index", "query_norm",
    "grad_plan_wrt_query_norm",
]


def load_or_template_query_sensitivity(path):
    """Return (df, status) where status ∈ {"loaded", "template"}.

    Writes a header-only template CSV when ``path`` is missing.
    """
    from pathlib import Path
    p = Path(path)
    if p.exists():
        return pd.read_csv(p), "loaded"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=QUERY_SENSITIVITY_TEMPLATE_COLUMNS).to_csv(p, index=False)
    return None, "template"


def build_query_sensitivity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Part J. Mean sensitivity per (model, checkpoint, layer, task_query_type)."""
    if df is None or df.empty:
        return pd.DataFrame()
    required = {"model", "checkpoint", "layer", "task_query_type",
                "grad_plan_wrt_query_norm"}
    if not ensure_columns(df, required, "query_sensitivity input"):
        raise KeyError(f"missing required columns: {required - set(df.columns)}")

    grouper = ["model", "checkpoint", "checkpoint_order", "layer", "task_query_type"] \
        if "checkpoint_order" in df.columns else \
        ["model", "checkpoint", "layer", "task_query_type"]

    rows = []
    for keys, g in df.groupby(grouper, sort=False):
        s = g["grad_plan_wrt_query_norm"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size == 0:
            continue
        rec = {
            **dict(zip(grouper, keys)),
            "n": int(s.size),
            "mean_sensitivity": float(np.mean(s)),
            "median_sensitivity": float(np.median(s)),
            "p05": float(np.quantile(s, 0.05)),
            "p95": float(np.quantile(s, 0.95)),
        }
        if "query_norm" in g.columns:
            qn = g["query_norm"].to_numpy(dtype=float)
            rec["mean_query_norm"] = float(np.nanmean(qn))
            rec["sensitivity_normed_mean"] = float(np.nanmean(s / (qn + EPS)))
        else:
            rec["mean_query_norm"] = float("nan")
            rec["sensitivity_normed_mean"] = float("nan")
        rows.append(rec)
    out = pd.DataFrame(rows)

    # share_task = sum_sensitivity_task / sum_total_per_(model, ckpt, layer)
    sum_keys = [k for k in grouper if k != "task_query_type"]
    out["share_task"] = out.groupby(sum_keys)["mean_sensitivity"].transform(
        lambda s: s / (s.sum() + EPS)
    )
    return out
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add query-sensitivity loader/template/summary (Part J)"
```

---

## Task 12: load_or_template_elasticity + build_elasticity_summary + planning_safe_weight_range (Part K)

**Files:**
- Modify: `tools/gradient_analysis/plan_centric.py`
- Modify: `tests/gradient_analysis/test_plan_centric.py`

- [ ] **Step 1: Append failing tests**

```python
def test_load_or_template_elasticity_writes_template(tmp_path):
    path = tmp_path / "el.csv"
    df, status = pc.load_or_template_elasticity(path)
    assert df is None and status == "template" and path.exists()
    template = pd.read_csv(path)
    assert {"model", "run_id", "lambda_det", "lambda_map", "lambda_motion",
            "lambda_plan", "plan_l2"}.issubset(template.columns)


def test_build_elasticity_summary_long_format_and_directions():
    df = pd.DataFrame({
        "model": ["A"] * 6,
        "run_id": [f"r{i}" for i in range(6)],
        "lambda_det": [1.0, 0.5, 2.0, 1.0, 1.0, 1.0],
        "lambda_map": [1.0, 1.0, 1.0, 0.5, 2.0, 1.0],
        "lambda_motion": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "lambda_plan": [1.0] * 6,
        "det_metric": [0.30, 0.28, 0.32, 0.30, 0.30, 0.30],   # higher is better
        "map_metric": [0.50, 0.50, 0.50, 0.45, 0.55, 0.50],
        "motion_loss": [0.20] * 6,                              # lower is better (name match)
        "plan_l2": [1.0, 1.1, 0.9, 1.05, 0.95, 1.0],
        "is_baseline": [True, False, False, False, False, False],
    })
    summary = pc.build_elasticity_summary(df)
    assert {"model", "swept_task", "lambda_value", "log_lambda",
            "task_metric", "plan_metric",
            "relative_task_metric", "relative_plan_metric"}.issubset(summary.columns)
    # det run with lambda=2.0 should have positive relative_task_metric
    det2 = summary[(summary["swept_task"] == "det") & (summary["lambda_value"] == 2.0)]
    assert det2["relative_task_metric"].iloc[0] > 0


def test_planning_safe_weight_range_filters_by_tolerance():
    summary = pd.DataFrame({
        "model": ["A"] * 3,
        "swept_task": ["det"] * 3,
        "lambda_value": [0.5, 1.0, 2.0],
        "log_lambda": [np.log(0.5), 0.0, np.log(2.0)],
        "task_metric": [0.28, 0.30, 0.32],
        "plan_metric": [1.005, 1.000, 1.030],
        "relative_task_metric": [-0.067, 0.0, 0.067],
        "relative_plan_metric": [0.005, 0.0, 0.030],
    })
    safe = pc.planning_safe_weight_range(summary, tol=0.01)
    assert (safe["planning_safe"] == (safe["relative_plan_metric"] <= 0.01)).all()
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: AttributeError on `pc.load_or_template_elasticity`.

- [ ] **Step 3: Implement**

Append to `tools/gradient_analysis/plan_centric.py`:

```python
ELASTICITY_TEMPLATE_COLUMNS = [
    "model", "run_id", "seed", "checkpoint",
    "lambda_det", "lambda_map", "lambda_motion", "lambda_plan",
    "det_metric", "map_metric", "motion_metric",
    "plan_l2", "collision", "plan_metric_name",
    "is_baseline", "notes",
]


def load_or_template_elasticity(path):
    from pathlib import Path
    p = Path(path)
    if p.exists():
        return pd.read_csv(p), "loaded"
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=ELASTICITY_TEMPLATE_COLUMNS).to_csv(p, index=False)
    return None, "template"


def _metric_lower_is_better(name: str) -> bool:
    lname = name.lower()
    return any(tok in lname for tok in ("loss", "error", "ade", "fde",
                                        "plan_l2", "collision"))


_TASK_METRIC_FALLBACKS = {
    "det":    ["det_metric",    "det_score",    "det_map",    "det_loss"],
    "map":    ["map_metric",    "map_score",    "map_loss"],
    "motion": ["motion_metric", "motion_score", "motion_loss", "motion_ade", "motion_fde"],
}


def _resolve_task_metric(df: pd.DataFrame, task: str) -> str | None:
    for c in _TASK_METRIC_FALLBACKS[task]:
        if c in df.columns:
            return c
    return None


def build_elasticity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Part K. Build long-format elasticity summary from a runs table."""
    required = {"model", "lambda_det", "lambda_map", "lambda_motion",
                "lambda_plan", "plan_l2"}
    if not ensure_columns(df, required, "elasticity_summary input"):
        raise KeyError(f"missing required columns: {required - set(df.columns)}")

    # Identify baseline rows.
    if "is_baseline" in df.columns:
        baseline_mask = df["is_baseline"].astype(bool)
    else:
        baseline_mask = (
            (df["lambda_det"] == 1.0)
            & (df["lambda_map"] == 1.0)
            & (df["lambda_motion"] == 1.0)
            & (df["lambda_plan"] == 1.0)
        )

    rows = []
    plan_metric_col = "plan_l2"  # canonical; lower-is-better
    for model_name, model_df in df.groupby("model", sort=False):
        baseline = model_df[baseline_mask.reindex(model_df.index, fill_value=False)]
        if baseline.empty:
            continue
        plan_base = float(baseline[plan_metric_col].iloc[0])
        for task in ("det", "map", "motion", "plan"):
            lam_col = f"lambda_{task}"
            task_metric_col = (plan_metric_col if task == "plan"
                               else _resolve_task_metric(model_df, task))
            if task_metric_col is None:
                continue
            task_lib = _metric_lower_is_better(task_metric_col)
            task_base = float(baseline[task_metric_col].iloc[0])
            # Rows where only this lambda is swept
            others = [f"lambda_{o}" for o in ("det","map","motion","plan") if o != task]
            mask = (model_df[others] == 1.0).all(axis=1) & (model_df[lam_col] != 1.0)
            swept = model_df[mask]
            for _, r in swept.iterrows():
                lam = float(r[lam_col])
                task_metric = float(r[task_metric_col])
                plan_metric = float(r[plan_metric_col])
                # Relative improvement: positive = better.
                rel_task = ((task_base - task_metric) / (abs(task_base) + EPS)
                            if task_lib else
                            (task_metric - task_base) / (abs(task_base) + EPS))
                rel_plan = (plan_metric - plan_base) / (abs(plan_base) + EPS)
                # plan_l2 lower is better → positive rel_plan = worse
                rows.append({
                    "model": model_name,
                    "swept_task": task,
                    "lambda_value": lam,
                    "log_lambda": float(np.log(lam)),
                    "task_metric": task_metric,
                    "plan_metric": plan_metric,
                    "task_metric_name": task_metric_col,
                    "plan_metric_name": plan_metric_col,
                    "relative_task_metric": rel_task,
                    "relative_plan_metric": rel_plan,
                })
    summary = pd.DataFrame(rows)
    # Sequential elasticity per swept_task, sorted by lambda
    if not summary.empty:
        summary = summary.sort_values(["model", "swept_task", "lambda_value"])
        summary["elasticity"] = (
            summary.groupby(["model", "swept_task"])["task_metric"].diff()
            / summary.groupby(["model", "swept_task"])["log_lambda"].diff()
        )
    return summary.reset_index(drop=True)


def planning_safe_weight_range(
    elast_summary: pd.DataFrame, tol: float = 0.01
) -> pd.DataFrame:
    """Mark rows whose ``relative_plan_metric`` (worse-for-plan) is within tolerance."""
    out = elast_summary.copy()
    out["planning_safe"] = out["relative_plan_metric"] <= tol
    return out
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_plan_centric.py -x -q`
Expected: 29 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/plan_centric.py tests/gradient_analysis/test_plan_centric.py
git commit -m "feat(plan_centric): add elasticity loader/template/summary (Part K)"
```

---

## Task 13: probe.py — emit grad_dot + step_size

**Files:**
- Modify: `tools/gradient_analysis/probe.py:103-114` (ProbeRow dataclass)
- Modify: `tools/gradient_analysis/probe.py:146-274` (probe_one_batch)
- Create: `tests/gradient_analysis/test_probe_grad_dot.py`

- [ ] **Step 1: Write failing test**

Create `tests/gradient_analysis/test_probe_grad_dot.py`:

```python
"""Regression: probe_one_batch emits grad_dot and step_size on the 1-step path."""
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


class _Adapter:
    def split_losses(self, fwd, task):
        return fwd[task]
    def snapshot_temporal_state(self, model):
        return None
    def restore_temporal_state(self, model, snap):
        pass
    def freeze_stochastic_state(self):
        from contextlib import nullcontext
        return nullcontext()


class _Collector:
    """Minimal collector contract used by probe_one_batch."""

    def __init__(self):
        torch.manual_seed(0)
        self.shared = nn.Linear(4, 4)
        # losses are nn.Linear-driven so they depend on .shared params
        self.head = {"a": nn.Linear(4, 1), "b": nn.Linear(4, 1)}
        self.tasks = ["a", "b"]
        self.adapter = _Adapter()
        self.full_params = list(self.shared.parameters())
        self.shared_param_groups = OrderedDict(L0=self.full_params)

        self._x = torch.randn(8, 4)
        self._ya = torch.randn(8, 1)
        self._yb = torch.randn(8, 1)

    @property
    def model(self):
        return self.shared

    def forward_losses(self, data):
        h = self.shared(self._x)
        return {"a": (self.head["a"](h) - self._ya).pow(2).mean(),
                "b": (self.head["b"](h) - self._yb).pow(2).mean()}


def test_probe_one_batch_emits_grad_dot_and_step_size():
    import pytest
    from tools.gradient_analysis.probe import probe_one_batch, rows_to_dataframe

    col = _Collector()
    rows = probe_one_batch(
        col, data=None, alpha=1e-3, steps_list=[1],
        variants=["normalized", "raw"], batch_idx=0,
        target_layers=None, freeze_matching=False, forward_seed=None,
    )
    df = rows_to_dataframe(rows)
    assert "grad_dot" in df.columns
    assert "step_size" in df.columns

    # Self-pair: grad_dot ≈ ||g||²
    self_rows = df[(df["source_task"] == "a") & (df["target_task"] == "a")
                   & (df["variant"] == "normalized")]
    grad_norm = float(self_rows["grad_norm"].iloc[0])
    assert np.isfinite(self_rows["grad_dot"].iloc[0])
    assert self_rows["grad_dot"].iloc[0] == pytest.approx(grad_norm ** 2, rel=1e-4)
    # For normalized: step_size == alpha / ||g||
    assert self_rows["step_size"].iloc[0] == pytest.approx(1e-3 / grad_norm, rel=1e-6)
    raw_self = df[(df["source_task"] == "a") & (df["target_task"] == "a")
                  & (df["variant"] == "raw")]
    assert raw_self["step_size"].iloc[0] == pytest.approx(1e-3, rel=1e-9)
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_probe_grad_dot.py -x -q`
Expected: KeyError or AssertionError because `grad_dot` / `step_size` are not emitted yet.

- [ ] **Step 3: Modify ProbeRow**

Edit `tools/gradient_analysis/probe.py:102-114`. Replace the `ProbeRow` dataclass with:

```python
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
```

- [ ] **Step 4: Modify probe_one_batch to pre-compute target gradients and emit new fields**

Edit `tools/gradient_analysis/probe.py:188-274` (the body of `probe_one_batch` from the `ctx = ...` line through the end of the source loop). Replace it with:

```python
    ctx = adapter.freeze_stochastic_state() if freeze_matching else _NullCM()
    with ctx:
        # Baseline forward — no_grad. This is the call that records matchings
        # (det/map indices, motion/plan mode_idx) into the freeze queues.
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
            # Cost: T backward passes per layer per batch; no extra forwards.
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
                        # step_size depends on variant. We use the *pre-step*
                        # gradient norm for the normalized case (matches the
                        # action taken by apply_virtual_step below).
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

                        # Stepped forward — no_grad
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
```

Note: this rewrite hoists the source loop inside the layer loop and pre-computes
all task gradients once per layer. The pre-existing `for source in collector.tasks:`
inner loop is replaced by the gradient pre-compute + the inner source loop
shown above.

- [ ] **Step 5: Run probe tests, verify new test passes and existing ones still pass**

Run: `pytest tests/gradient_analysis/test_probe.py tests/gradient_analysis/test_probe_variance.py tests/gradient_analysis/test_probe_grad_dot.py -x -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tools/gradient_analysis/probe.py tests/gradient_analysis/test_probe_grad_dot.py
git commit -m "feat(probe): emit grad_dot and step_size on 1-step path"
```

---

## Task 14: HipadAdapter.get_task_queries + VAD NotImplementedError

**Files:**
- Modify: `tools/gradient_analysis/adapters/base.py`
- Modify: `tools/gradient_analysis/adapters/hipad.py`
- Modify: `tools/gradient_analysis/adapters/vad.py`
- Create: `tests/gradient_analysis/adapters/test_get_task_queries.py`

- [ ] **Step 1: Read existing adapter contract**

Run: `pytest --collect-only tests/gradient_analysis/adapters/ -q | head -20`
Then read: `tools/gradient_analysis/adapters/base.py`, `tools/gradient_analysis/adapters/hipad.py`. Identify the head module attribute that holds the task-specific query tensors for HiP-AD. Concrete attribute path is resolved here.

- [ ] **Step 2: Write failing test**

Create `tests/gradient_analysis/adapters/test_get_task_queries.py`:

```python
"""Adapter contract: get_task_queries returns graph-attached query tensors."""
import pytest


def test_base_adapter_get_task_queries_default_raises_not_implemented():
    from tools.gradient_analysis.adapters.base import BaseAdapter

    class _A(BaseAdapter):
        pass

    with pytest.raises(NotImplementedError):
        _A().get_task_queries(model=None, fwd_artifacts=None)


def test_vad_adapter_get_task_queries_raises_not_implemented():
    from tools.gradient_analysis.adapters.vad import VadAdapter
    with pytest.raises(NotImplementedError):
        VadAdapter().get_task_queries(model=None, fwd_artifacts=None)


def test_hipad_adapter_get_task_queries_returns_dict_of_tensors(hipad_smoke_model):
    """Requires the hipad smoke model fixture (skipped if unavailable)."""
    from tools.gradient_analysis.adapters.hipad import HipadAdapter
    adapter = HipadAdapter()
    queries = adapter.get_task_queries(model=hipad_smoke_model.model,
                                       fwd_artifacts=hipad_smoke_model.fwd)
    assert set(queries.keys()) >= {"det", "map", "motion", "plan"}
    for t, q in queries.items():
        assert q.requires_grad, f"{t} query must be graph-attached"
```

If `hipad_smoke_model` fixture does not exist, mark the third test with
`pytest.importorskip("...")` or `pytest.skip("hipad smoke model unavailable")`
inside the test body.

- [ ] **Step 3: Run, verify failure**

Run: `pytest tests/gradient_analysis/adapters/test_get_task_queries.py -x -q`
Expected: AttributeError on `get_task_queries` (method missing).

- [ ] **Step 4: Add the BaseAdapter default**

Edit `tools/gradient_analysis/adapters/base.py`. Add a method on `BaseAdapter`:

```python
    def get_task_queries(self, model, fwd_artifacts) -> "dict[str, torch.Tensor]":
        """Return task-specific query tensors that participate in L_plan's graph.

        Default implementation raises ``NotImplementedError``. Subclasses that
        support query-sensitivity analysis (currently HiP-AD only) override
        this and return a mapping ``{task_query_type: Tensor}`` where each
        Tensor still requires gradient (no ``.detach()``).
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_task_queries is not implemented"
        )
```

- [ ] **Step 5: Add the HiP-AD override**

Edit `tools/gradient_analysis/adapters/hipad.py`. Add:

```python
    def get_task_queries(self, model, fwd_artifacts) -> "dict[str, torch.Tensor]":
        """Return HiP-AD's per-task query tensors used inside the planner.

        ``fwd_artifacts`` is the output dict from ``forward_losses``; the
        runner is responsible for capturing the post-forward intermediate
        tensors and stashing them there. The concrete attribute paths used
        below depend on the HiP-AD model layout — see ``projects/mmdet3d_plugin``.
        """
        if fwd_artifacts is None or "task_queries" not in fwd_artifacts:
            raise NotImplementedError(
                "HipadAdapter.get_task_queries requires fwd_artifacts['task_queries']"
                " to be populated by the runner."
            )
        return dict(fwd_artifacts["task_queries"])
```

Edit `tools/gradient_analysis/adapters/vad.py`. Add:

```python
    def get_task_queries(self, model, fwd_artifacts) -> "dict[str, torch.Tensor]":
        raise NotImplementedError(
            "VadAdapter.get_task_queries is not implemented; "
            "query sensitivity analysis runs HiP-AD only."
        )
```

- [ ] **Step 6: Run, verify pass**

Run: `pytest tests/gradient_analysis/adapters/test_get_task_queries.py -x -q`
Expected: the BaseAdapter and VAD tests pass; the HiP-AD test is skipped if the
smoke fixture is unavailable.

- [ ] **Step 7: Commit**

```bash
git add tools/gradient_analysis/adapters/base.py tools/gradient_analysis/adapters/hipad.py tools/gradient_analysis/adapters/vad.py tests/gradient_analysis/adapters/test_get_task_queries.py
git commit -m "feat(adapters): add get_task_queries (HiP-AD via fwd_artifacts, VAD NIY)"
```

---

## Task 15: query_sensitivity.py runner

**Files:**
- Create: `tools/gradient_analysis/query_sensitivity.py`
- Create: `tests/gradient_analysis/test_query_sensitivity_runner.py`

- [ ] **Step 1: Write smoke test**

Create `tests/gradient_analysis/test_query_sensitivity_runner.py`:

```python
"""Smoke test for the query sensitivity runner CSV schema."""
from pathlib import Path

import pandas as pd
import torch
from torch import nn


class _Adapter:
    def split_losses(self, fwd, task):
        return fwd[task]
    def snapshot_temporal_state(self, model):
        return None
    def restore_temporal_state(self, model, snap):
        pass
    def freeze_stochastic_state(self):
        from contextlib import nullcontext
        return nullcontext()
    def get_task_queries(self, model, fwd_artifacts):
        return fwd_artifacts["task_queries"]


class _Collector:
    def __init__(self):
        torch.manual_seed(0)
        # Task queries are leaf tensors with requires_grad=True
        self.qs = {
            "det":    torch.randn(3, 2, requires_grad=True),
            "map":    torch.randn(3, 2, requires_grad=True),
            "motion": torch.randn(3, 2, requires_grad=True),
            "plan":   torch.randn(3, 2, requires_grad=True),
        }
        self.W = nn.Linear(2, 1)
        self.tasks = ["det", "map", "motion", "plan"]
        self.adapter = _Adapter()
    @property
    def model(self):
        return self.W
    def forward_losses(self, data):
        # plan loss depends on all task queries via the linear weight
        plan = self.W(self.qs["plan"]).pow(2).mean() \
             + self.W(self.qs["det"]).pow(2).mean() \
             + self.W(self.qs["map"]).pow(2).mean() \
             + self.W(self.qs["motion"]).pow(2).mean()
        return {"plan": plan, "_task_queries": dict(self.qs)}


def test_run_query_sensitivity_writes_expected_schema(tmp_path):
    from tools.gradient_analysis.query_sensitivity import run_query_sensitivity

    col = _Collector()
    out = tmp_path / "qs.csv"
    df = run_query_sensitivity(
        collector=col,
        dataloader=[None],   # one "batch"
        num_batches=1,
        out_dir=tmp_path,
        forward_seed=None,
        freeze_matching=False,
    )
    df_csv = pd.read_csv(out)
    expected = {"model", "checkpoint", "checkpoint_order", "batch_idx",
                "scene_token", "layer", "task_query_type", "query_index",
                "query_norm", "grad_plan_wrt_query_norm"}
    assert expected.issubset(df_csv.columns)
    # 4 tasks × 3 query rows each = 12 rows minimum
    assert len(df_csv) >= 12
```

- [ ] **Step 2: Run, verify failure**

Run: `pytest tests/gradient_analysis/test_query_sensitivity_runner.py -x -q`
Expected: ModuleNotFoundError on `query_sensitivity`.

- [ ] **Step 3: Implement runner**

Create `tools/gradient_analysis/query_sensitivity.py`:

```python
"""HiP-AD planning-query sensitivity dumper.

Computes ``||dL_plan/dQ_task||`` for each task-specific query tensor exposed
by ``adapter.get_task_queries``. VAD adapter raises NotImplementedError; the
runner catches it and writes an empty CSV with a SKIP message.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Callable, Optional

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
    """Adapter contract: prefer fwd['_task_queries'], else adapter.get_task_queries."""
    if isinstance(fwd, dict) and "_task_queries" in fwd:
        return fwd["_task_queries"]
    return adapter.get_task_queries(model=model, fwd_artifacts=fwd)


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
    sidecar = []  # for capture_vectors
    try:
        ctx = _maybe_freeze(collector.adapter, freeze_matching)
        for i, data in enumerate(dataloader):
            if i >= num_batches:
                break
            with ctx:
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
                    if gQ is None:
                        continue
                    # Per-token (flat across leading dims, last dim = feat)
                    q_flat = Q.detach().reshape(-1, Q.shape[-1])
                    g_flat = gQ.detach().reshape(-1, gQ.shape[-1])
                    q_norms = q_flat.norm(dim=-1).cpu().numpy()
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
                    if capture_vectors:
                        sidecar.append((i, task, g_flat.cpu().clone()))
    except NotImplementedError as e:
        print(f"[query_sensitivity] SKIP: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if capture_vectors:
        torch.save(sidecar, out_dir / "qs_vectors.pt")
    return df


def _seeded_forward(collector, data, seed):
    with _seed_scope(seed):
        return collector.forward_losses(data)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/gradient_analysis/test_query_sensitivity_runner.py -x -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/gradient_analysis/query_sensitivity.py tests/gradient_analysis/test_query_sensitivity_runner.py
git commit -m "feat(query_sensitivity): add HiP-AD planning-query gradient runner"
```

---

## Task 16: Notebook — extend data loading cell

**Files:**
- Modify: `docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb` (cell index 3, the data-loading cell)

- [ ] **Step 1: Write the cell-edit script**

Create a one-shot helper at `tools/_nb_edit.py`:

```python
"""Notebook editor used by the plan-centric refactor plan.

Usage: ``python tools/_nb_edit.py <command> [<args>]`` where commands are
defined per task in the plan. This script is intentionally local to the
refactor and not exported.
"""
import argparse
import json
from pathlib import Path

NB_PATH = Path("docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb")


def load_nb():
    return json.loads(NB_PATH.read_text())


def save_nb(nb):
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}


def cmd_task16():
    nb = load_nb()
    # Cell 3 is the data-loading cell (verified by reading the notebook).
    src = "".join(nb["cells"][3]["source"])
    addition = "\n\nfrom tools.gradient_analysis.plan_centric import (\n    ensure_columns,\n    bootstrap_ci,\n    practical_threshold,\n    standardize_probe_df,\n    get_probe_base,\n)\n\nprobe = standardize_probe_df(probe)\nbase = get_probe_base(probe, variant='normalized', steps=1)\nprint('base rows:', len(base))\n"
    if "standardize_probe_df(probe)" not in src:
        nb["cells"][3]["source"] = (src + addition).splitlines(keepends=True)
    save_nb(nb)
    print("task16 applied")


CMDS = {"task16": cmd_task16}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    args = ap.parse_args()
    CMDS[args.cmd]()
```

- [ ] **Step 2: Run the edit**

Run: `cd /home/yongjae/e2e/HiP-AD && python tools/_nb_edit.py task16`
Expected stdout: `task16 applied`.

- [ ] **Step 3: Verify the cell has the new import + standardize block**

Run: `python -c "import json,re; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); s=''.join(nb['cells'][3]['source']); print('OK' if 'standardize_probe_df(probe)' in s and 'get_probe_base(probe' in s else 'MISSING')"`
Expected output: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): import plan_centric helpers and build base subset"
```

---

## Task 17: Notebook — Part F (plan transfer)

**Files:**
- Modify: `tools/_nb_edit.py` (add `cmd_task17`)
- Modify: `docs/superpowers/specs/.../phase2_hipad_vad_analysis_visualization.ipynb`

- [ ] **Step 1: Extend the editor**

Append to `tools/_nb_edit.py` inside the `CMDS` block. Add this function above `CMDS`:

```python
def _insert_after(nb, index, cells):
    nb["cells"][index + 1:index + 1] = cells


def cmd_task17():
    nb = load_nb()
    # Find insertion point: immediately after the data-loading cell (index 3).
    md = md_cell("# Part F. Plan-centric directed transfer\n\n"
                 "Question: 각 task의 1-step update가 planning loss를 줄이는가?\n"
                 "Source: `det`, `map`, `motion`, `plan`. Target: `plan`.\n")
    code_compute = code_cell(
        "from pathlib import Path\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    build_plan_transfer_summary, top_beneficial, top_harmful,\n"
        ")\n"
        "OUT_DIR = Path('gradient_analysis_results/plan_centric')\n"
        "OUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "(OUT_DIR / 'figures/F').mkdir(parents=True, exist_ok=True)\n"
        "plan_transfer = build_plan_transfer_summary(base)\n"
        "plan_transfer.to_csv(OUT_DIR / 'plan_transfer_summary.csv', index=False)\n"
        "display(plan_transfer.head())\n"
    )
    code_heatmap = code_cell(
        "import matplotlib.pyplot as plt\n"
        "SRC_ORDER = ['det', 'map', 'motion', 'plan']\n"
        "for (model, ckpt), sub in plan_transfer.groupby(['model', 'checkpoint']):\n"
        "    for metric in ['mean_gain', 'practical_helpful_rate']:\n"
        "        pivot = sub.pivot_table(index='layer', columns='source_task', values=metric).reindex(columns=SRC_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(pivot))))\n"
        "        cmap = 'RdBu_r' if metric == 'mean_gain' else 'viridis'\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap=cmap)\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: target=plan, {metric}')\n"
        "        fig.colorbar(im, ax=ax)\n"
        "        fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/F/{model}_{ckpt}_{metric}.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_tables = code_cell(
        "top_ben = top_beneficial(plan_transfer, k=10, by='mean_gain')\n"
        "top_ben.to_csv(OUT_DIR / 'plan_transfer_top_beneficial.csv', index=False)\n"
        "display(top_ben)\n"
        "top_harm = top_harmful(plan_transfer, k=10, by='mean_gain')\n"
        "top_harm.to_csv(OUT_DIR / 'plan_transfer_top_harmful.csv', index=False)\n"
        "display(top_harm)\n"
    )
    _insert_after(nb, 3, [md, code_compute, code_heatmap, code_tables])
    save_nb(nb)
    print("task17 applied")


CMDS["task17"] = cmd_task17
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task17`
Expected stdout: `task17 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'build_plan_transfer_summary' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part F plan-centric directed transfer"
```

---

## Task 18: Notebook — Part G (asymmetry)

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
def cmd_task18():
    nb = load_nb()
    # Insert after the 4 cells we appended in Task 17 (data load was index 3,
    # Task 17 added 4 cells → next index is 3+4 = 7). Use a search to be robust.
    idx = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "build_plan_transfer_summary" in "".join(c["source"]))
    # Find the last Part-F cell (the tables cell)
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "plan_transfer_top_harmful" in "".join(c["source"]))
    md = md_cell("# Part G. Directed asymmetry around planning\n\n"
                 "`A → plan` vs `plan → A`, A ∈ {det, map, motion}.\n")
    code_compute = code_cell(
        "from tools.gradient_analysis.plan_centric import build_asymmetry_summary\n"
        "(OUT_DIR / 'figures/G').mkdir(parents=True, exist_ok=True)\n"
        "asym = build_asymmetry_summary(base)\n"
        "asym.to_csv(OUT_DIR / 'plan_asymmetry_summary.csv', index=False)\n"
        "asym[['model','checkpoint','layer','aux_task','asymmetry','plan_transfer_ratio']].to_csv(\n"
        "    OUT_DIR / 'plan_transfer_ratio_summary.csv', index=False)\n"
        "display(asym.head())\n"
    )
    code_heatmap = code_cell(
        "AUX_ORDER = ['det', 'map', 'motion']\n"
        "for (model, ckpt), sub in asym.groupby(['model', 'checkpoint']):\n"
        "    for metric in ['asymmetry', 'plan_transfer_ratio']:\n"
        "        pivot = sub.pivot_table(index='layer', columns='aux_task', values=metric).reindex(columns=AUX_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(pivot))))\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r')\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: {metric}')\n"
        "        fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/G/{model}_{ckpt}_{metric}.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_trend = code_cell(
        "for model, sub in asym.groupby('model'):\n"
        "    for metric in ['asymmetry', 'plan_transfer_ratio']:\n"
        "        fig, ax = plt.subplots(figsize=(7, 4))\n"
        "        for aux, sub2 in sub.groupby('aux_task'):\n"
        "            agg = sub2.groupby('checkpoint_order')[metric].mean()\n"
        "            ax.plot(agg.index, agg.values, marker='o', label=aux)\n"
        "        ax.axhline(0, color='gray', lw=0.5)\n"
        "        ax.set_xlabel('checkpoint_order'); ax.set_ylabel(metric)\n"
        "        ax.set_title(f'{model}: {metric} trend')\n"
        "        ax.legend(); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/G/{model}_{metric}_trend.png', dpi=120)\n"
        "        plt.show()\n"
    )
    code_interp = code_cell(
        "display(asym[['model','checkpoint','layer','aux_task','interpretation']].head(20))\n"
    )
    _insert_after(nb, end, [md, code_compute, code_heatmap, code_trend, code_interp])
    save_nb(nb)
    print("task18 applied")


CMDS["task18"] = cmd_task18
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task18`
Expected: `task18 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'build_asymmetry_summary' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part G directed asymmetry around plan"
```

---

## Task 19: Notebook — Part H (effect size)

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
def cmd_task19():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "asym[['model','checkpoint','layer','aux_task','interpretation']]" in "".join(c["source"]))
    md = md_cell("# Part H. Effect-size-aware 1-step probe\n\n"
                 "Helpful ratio alone overweights noisy near-zero deltas. "
                 "Effect size = mean_gain / std_delta.\n")
    code_compute = code_cell(
        "from tools.gradient_analysis.plan_centric import build_effect_size_summary\n"
        "(OUT_DIR / 'figures/H').mkdir(parents=True, exist_ok=True)\n"
        "effect = build_effect_size_summary(base, target='plan')\n"
        "effect.to_csv(OUT_DIR / 'effect_size_probe_summary.csv', index=False)\n"
        "effect[['model','checkpoint','layer','source_task','helpful_rate','mean_gain','effect_size']].to_csv(\n"
        "    OUT_DIR / 'helpful_vs_effect_size.csv', index=False)\n"
        "display(effect.head())\n"
    )
    code_scatter = code_cell(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "colors = {'det':'C0','map':'C1','motion':'C2','plan':'C3'}\n"
        "markers = {'HiP-AD':'o','VAD':'s'}\n"
        "for (model, src), sub in effect.groupby(['model','source_task']):\n"
        "    ax.scatter(sub['helpful_rate'], sub['mean_gain'],\n"
        "               c=colors.get(src,'gray'), marker=markers.get(model,'x'),\n"
        "               alpha=0.6, label=f'{model}/{src}')\n"
        "# label top-k by |mean_gain|\n"
        "top = effect.reindex(effect['mean_gain'].abs().sort_values(ascending=False).index).head(15)\n"
        "for _, r in top.iterrows():\n"
        "    ax.annotate(f\"{r['layer']}/{r['source_task'][:3]}\", (r['helpful_rate'], r['mean_gain']),\n"
        "                fontsize=7, alpha=0.7)\n"
        "ax.axhline(0, color='gray', lw=0.5); ax.axvline(0.5, color='gray', lw=0.5)\n"
        "ax.set_xlabel('helpful_rate'); ax.set_ylabel('mean_gain (plan)')\n"
        "ax.set_title('helpful_rate vs mean_gain (target=plan)')\n"
        "ax.legend(fontsize=7, ncol=2); fig.tight_layout()\n"
        "fig.savefig(OUT_DIR / 'figures/H/helpful_vs_gain_scatter.png', dpi=120)\n"
        "plt.show()\n"
    )
    code_ci = code_cell(
        "import numpy as np\n"
        "for model, sub in effect.groupby('model'):\n"
        "    fig, ax = plt.subplots(figsize=(8, 4))\n"
        "    pos = 0; xticks = []; xlabels = []\n"
        "    for src, sub2 in sub.groupby('source_task'):\n"
        "        top = sub2.reindex(sub2['mean_gain'].abs().sort_values(ascending=False).index).head(5)\n"
        "        for _, r in top.iterrows():\n"
        "            ax.errorbar(pos, r['mean_gain'],\n"
        "                        yerr=[[r['mean_gain']-r['ci_lo_gain']],[r['ci_hi_gain']-r['mean_gain']]],\n"
        "                        fmt='o', color=colors.get(src,'gray'))\n"
        "            xticks.append(pos); xlabels.append(f\"{src[:3]}/{r['layer'][:10]}\"); pos += 1\n"
        "        pos += 1\n"
        "    ax.axhline(0, color='gray', lw=0.5)\n"
        "    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, rotation=60, ha='right', fontsize=7)\n"
        "    ax.set_ylabel('mean_gain ± 95% CI'); ax.set_title(f'{model}: top influential layers')\n"
        "    fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/H/{model}_top_layers_ci.png', dpi=120)\n"
        "    plt.show()\n"
    )
    code_flags = code_cell(
        "flag_cols = ['flag_high_helpful_low_gain','flag_positive_but_insig',\n"
        "             'flag_high_practical_helpful','flag_high_large_harm']\n"
        "for fc in flag_cols:\n"
        "    flagged = effect[effect[fc]]\n"
        "    print(f'{fc}: {len(flagged)} rows')\n"
        "    if not flagged.empty:\n"
        "        display(flagged[['model','checkpoint','layer','source_task','mean_gain','helpful_rate']].head(10))\n"
    )
    _insert_after(nb, end, [md, code_compute, code_scatter, code_ci, code_flags])
    save_nb(nb)
    print("task19 applied")


CMDS["task19"] = cmd_task19
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task19`
Expected: `task19 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'build_effect_size_summary' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part H effect-size-aware probe analysis"
```

---

## Task 20: Notebook — Part I (first-order vs actual)

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
def cmd_task20():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "flag_high_large_harm" in "".join(c["source"]))
    md = md_cell("# Part I. First-order prediction vs actual 1-step effect\n\n"
                 "Checks whether actual ΔL matches the linear prediction "
                 "`-step_size · <g_src, g_tgt>`.\n")
    code_main = code_cell(
        "from tools.gradient_analysis.plan_centric import (\n"
        "    detect_first_order_columns, build_first_order_residual_summary,\n"
        ")\n"
        "(OUT_DIR / 'figures/I').mkdir(parents=True, exist_ok=True)\n"
        "fo_cols = detect_first_order_columns(probe)\n"
        "if fo_cols is None:\n"
        "    print('SKIPPED. Required columns for first-order analysis:')\n"
        "    print('- source_task\\n- target_task\\n- model\\n- checkpoint\\n- layer\\n- delta')\n"
        "    print('- grad_dot (or equivalent)\\n- step_size / lr / probe_lr')\n"
        "    first_order = None\n"
        "else:\n"
        "    first_order = build_first_order_residual_summary(base, fo_cols)\n"
        "    first_order.to_csv(OUT_DIR / 'first_order_vs_actual_summary.csv', index=False)\n"
        "    first_order[first_order['target_task']=='plan'].to_csv(\n"
        "        OUT_DIR / 'first_order_residual_by_layer.csv', index=False)\n"
        "    display(first_order.head())\n"
    )
    code_plots = code_cell(
        "import numpy as np\n"
        "if first_order is not None:\n"
        "    df_full = base.copy()\n"
        "    df_full['pred_delta'] = -df_full[fo_cols['step_size']] * df_full[fo_cols['grad_dot']]\n"
        "    df_full['residual'] = df_full['delta'] - df_full['pred_delta']\n"
        "    fig, ax = plt.subplots(figsize=(5, 5))\n"
        "    ax.scatter(df_full['pred_delta'], df_full['delta'], alpha=0.4, s=8)\n"
        "    lim = max(abs(df_full['pred_delta'].min()), abs(df_full['delta'].max()))\n"
        "    ax.plot([-lim, lim], [-lim, lim], 'r--', lw=0.5)\n"
        "    ax.set_xlabel('predicted -lr·<g_s,g_t>'); ax.set_ylabel('actual ΔL')\n"
        "    ax.set_title('Actual vs first-order prediction')\n"
        "    fig.tight_layout(); fig.savefig(OUT_DIR / 'figures/I/actual_vs_pred.png', dpi=120)\n"
        "    plt.show()\n"
        "    # residual heatmap (plan target)\n"
        "    plan_res = first_order[first_order['target_task'] == 'plan']\n"
        "    for (model, ckpt), sub in plan_res.groupby(['model', 'checkpoint']):\n"
        "        pivot = sub.pivot_table(index='layer', columns='source_task', values='mean_residual').reindex(columns=['det','map','motion','plan'])\n"
        "        fig, ax = plt.subplots(figsize=(5, max(3, 0.3*len(pivot))))\n"
        "        im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r')\n"
        "        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "        ax.set_title(f'{model} {ckpt}: residual (plan target)')\n"
        "        fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/I/{model}_{ckpt}_residual_plan.png', dpi=120)\n"
        "        plt.show()\n"
        "    # histogram\n"
        "    for model, sub in df_full.groupby('model'):\n"
        "        fig, ax = plt.subplots(figsize=(6, 3))\n"
        "        ax.hist(sub['residual'], bins=60)\n"
        "        ax.set_title(f'{model}: residual distribution'); ax.set_xlabel('actual - predicted')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/I/{model}_residual_hist.png', dpi=120)\n"
        "        plt.show()\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task20 applied")


CMDS["task20"] = cmd_task20
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task20`
Expected: `task20 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'detect_first_order_columns' in src and 'build_first_order_residual_summary' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part I first-order vs actual analysis"
```

---

## Task 21: Notebook — Part J (query sensitivity)

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
def cmd_task21():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "first_order is not None" in "".join(c["source"]))
    md = md_cell("# Part J. Planning sensitivity and query reachability\n\n"
                 "Loads `query_sensitivity.csv` if produced by the runner. "
                 "Missing file → template + SKIP.\n")
    code_main = code_cell(
        "import os\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    load_or_template_query_sensitivity, build_query_sensitivity_summary,\n"
        ")\n"
        "(OUT_DIR / 'figures/J').mkdir(parents=True, exist_ok=True)\n"
        "QUERY_SENSITIVITY_PATH = Path(os.environ.get(\n"
        "    'QUERY_SENSITIVITY_PATH', OUT_DIR / 'query_sensitivity.csv'))\n"
        "qs_df, status = load_or_template_query_sensitivity(QUERY_SENSITIVITY_PATH)\n"
        "if status == 'template':\n"
        "    print(f'SKIPPED. Template written to {QUERY_SENSITIVITY_PATH}.')\n"
        "    print('Populate via tools/gradient_analysis/query_sensitivity.run_query_sensitivity')\n"
        "    qs_summary = None\n"
        "else:\n"
        "    qs_summary = build_query_sensitivity_summary(qs_df)\n"
        "    qs_summary.to_csv(OUT_DIR / 'query_sensitivity_summary.csv', index=False)\n"
        "    display(qs_summary.head())\n"
    )
    code_plots = code_cell(
        "if qs_summary is not None and not qs_summary.empty:\n"
        "    TASK_ORDER = ['det','map','motion','plan']\n"
        "    for (model, ckpt), sub in qs_summary.groupby(['model','checkpoint']):\n"
        "        agg = sub.groupby('task_query_type')['mean_sensitivity'].mean().reindex(TASK_ORDER)\n"
        "        fig, ax = plt.subplots(figsize=(5, 3))\n"
        "        ax.bar(agg.index, agg.values)\n"
        "        ax.set_title(f'{model} {ckpt}: mean planning sensitivity by query task')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/J/{model}_{ckpt}_sens_bar.png', dpi=120)\n"
        "        plt.show()\n"
        "        pivot = sub.pivot_table(index='layer', columns='task_query_type',\n"
        "                                values='mean_sensitivity').reindex(columns=TASK_ORDER)\n"
        "        if not pivot.empty:\n"
        "            fig, ax = plt.subplots(figsize=(5, max(3, 0.3 * len(pivot))))\n"
        "            im = ax.imshow(pivot.values, aspect='auto', cmap='viridis')\n"
        "            ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)\n"
        "            ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)\n"
        "            ax.set_title(f'{model} {ckpt}: layer × task_query_type sensitivity')\n"
        "            fig.colorbar(im, ax=ax); fig.tight_layout()\n"
        "            fig.savefig(OUT_DIR / f'figures/J/{model}_{ckpt}_sens_heatmap.png', dpi=120)\n"
        "            plt.show()\n"
        "    # trend\n"
        "    for model, sub in qs_summary.groupby('model'):\n"
        "        fig, ax = plt.subplots(figsize=(6, 3))\n"
        "        for tq, sub2 in sub.groupby('task_query_type'):\n"
        "            agg = sub2.groupby('checkpoint_order')['mean_sensitivity'].mean()\n"
        "            ax.plot(agg.index, agg.values, marker='o', label=tq)\n"
        "        ax.set_xlabel('checkpoint_order'); ax.set_ylabel('mean_sensitivity')\n"
        "        ax.set_title(f'{model}: planning sensitivity over training')\n"
        "        ax.legend(); fig.tight_layout()\n"
        "        fig.savefig(OUT_DIR / f'figures/J/{model}_sens_trend.png', dpi=120)\n"
        "        plt.show()\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task21 applied")


CMDS["task21"] = cmd_task21
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task21`
Expected: `task21 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'load_or_template_query_sensitivity' in src and 'build_query_sensitivity_summary' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part J planning sensitivity / query reachability"
```

---

## Task 22: Notebook — Part K (elasticity)

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
def cmd_task22():
    nb = load_nb()
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "qs_summary is not None and not qs_summary.empty" in "".join(c["source"]))
    md = md_cell("# Part K. Task loss weight elasticity\n\n"
                 "Loads `task_weight_elasticity_runs.csv`. Missing file → "
                 "template + SKIP.\n")
    code_main = code_cell(
        "import os\n"
        "from tools.gradient_analysis.plan_centric import (\n"
        "    load_or_template_elasticity, build_elasticity_summary,\n"
        "    planning_safe_weight_range,\n"
        ")\n"
        "(OUT_DIR / 'figures/K').mkdir(parents=True, exist_ok=True)\n"
        "ELASTICITY_LOG_PATH = Path(os.environ.get(\n"
        "    'ELASTICITY_LOG_PATH', OUT_DIR / 'task_weight_elasticity_runs.csv'))\n"
        "el_df, status = load_or_template_elasticity(ELASTICITY_LOG_PATH)\n"
        "if status == 'template':\n"
        "    print(f'SKIPPED. Template written to {ELASTICITY_LOG_PATH}.')\n"
        "    elasticity = None\n"
        "else:\n"
        "    elasticity = build_elasticity_summary(el_df)\n"
        "    elasticity.to_csv(OUT_DIR / 'task_weight_elasticity_summary.csv', index=False)\n"
        "    safe = planning_safe_weight_range(elasticity, tol=0.01)\n"
        "    safe.to_csv(OUT_DIR / 'planning_safe_weight_range.csv', index=False)\n"
        "    display(elasticity.head())\n"
    )
    code_plots = code_cell(
        "if elasticity is not None and not elasticity.empty:\n"
        "    for swept, sub in elasticity.groupby('swept_task'):\n"
        "        fig, axes = plt.subplots(1, 2, figsize=(10, 3))\n"
        "        for model, sub2 in sub.groupby('model'):\n"
        "            sub2 = sub2.sort_values('log_lambda')\n"
        "            axes[0].plot(sub2['log_lambda'], sub2['task_metric'], 'o-', label=model)\n"
        "            axes[1].plot(sub2['log_lambda'], sub2['plan_metric'], 'o-', label=model)\n"
        "        axes[0].set_title(f'lambda_{swept} vs task_metric'); axes[0].legend()\n"
        "        axes[1].set_title(f'lambda_{swept} vs plan_metric'); axes[1].legend()\n"
        "        axes[0].set_xlabel('log(lambda)'); axes[1].set_xlabel('log(lambda)')\n"
        "        fig.tight_layout(); fig.savefig(OUT_DIR / f'figures/K/{swept}_lambda_curves.png', dpi=120)\n"
        "        plt.show()\n"
        "    fig, ax = plt.subplots(figsize=(6, 4))\n"
        "    for (swept, model), sub in elasticity.groupby(['swept_task','model']):\n"
        "        ax.scatter(sub['relative_task_metric'], sub['relative_plan_metric'],\n"
        "                   label=f'{model}/{swept}', s=30)\n"
        "    ax.axhline(0, color='gray', lw=0.5); ax.axvline(0, color='gray', lw=0.5)\n"
        "    ax.set_xlabel('relative task gain (+ better)')\n"
        "    ax.set_ylabel('relative plan degradation (+ worse for plan_l2)')\n"
        "    ax.set_title('Pareto: task gain vs plan degradation')\n"
        "    ax.legend(fontsize=7); fig.tight_layout()\n"
        "    fig.savefig(OUT_DIR / 'figures/K/pareto.png', dpi=120); plt.show()\n"
        "    display(safe[safe['planning_safe']])\n"
    )
    _insert_after(nb, end, [md, code_main, code_plots])
    save_nb(nb)
    print("task22 applied")


CMDS["task22"] = cmd_task22
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task22`
Expected: `task22 applied`.

- [ ] **Step 3: Verify**

Run: `python -c "import json; nb=json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); src=''.join(s for c in nb['cells'] for s in c['source']); assert 'load_or_template_elasticity' in src and 'planning_safe_weight_range' in src; print('OK')"`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): add Part K task loss weight elasticity"
```

---

## Task 23: Notebook — demote Parts A–E to Appendix

**Files:**
- Modify: `tools/_nb_edit.py`
- Modify: notebook

- [ ] **Step 1: Extend editor**

Append to `tools/_nb_edit.py`:

```python
# Header text mapping for demotion. Heading rewrite is exact-substring,
# so we anchor on the existing headings that the notebook uses today.
_DEMOTE_HEADERS = [
    ("# Part A. HiP-AD 세부 분석",                    "# Appendix.A. HiP-AD diagnostic (legacy)"),
    ("# Part B. VAD 세부 분석",                       "# Appendix.B. VAD diagnostic (legacy)"),
    ("# Part C. 1-step probe를 task 중심으로 다시 비교", "# Appendix.C. Task-pair helpful matrix (Appendix only)"),
    ("# Part D. Distribution과 correlation 보조 분석",  "# Appendix.D. Distribution / correlation (legacy)"),
    ("# Part E. 기존 `gradient_analysis_viz.ipynb` 스타일 추가 분석",
     "# Appendix.E. Affinity / heavy-tail / CI / self-violation (legacy)"),
]


def cmd_task23():
    nb = load_nb()
    # 1) Rewrite the headings to mark them as Appendix.
    for old, new in _DEMOTE_HEADERS:
        for c in nb["cells"]:
            if c["cell_type"] != "markdown":
                continue
            src = "".join(c["source"])
            if old in src:
                c["source"] = src.replace(old, new).splitlines(keepends=True)

    # 2) Add a single banner cell immediately after the Part K block
    #    explaining that the cells below are legacy.
    end = next(i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code"
               and "planning_safe_weight_range" in "".join(c["source"]))
    banner = md_cell(
        "# Appendix. Diagnostic background\n\n"
        "Below are the original Part A–E analyses (global conflict, null "
        "baseline, distribution, affinity, heavy-tail, self-violation). They "
        "are preserved verbatim for traceability but the main story now "
        "lives in Parts F–K above. Read these only when you need to verify "
        "background claims.\n"
    )
    if not any(c["cell_type"] == "markdown"
               and "Appendix. Diagnostic background" in "".join(c["source"])
               for c in nb["cells"][end + 1:end + 3]):
        _insert_after(nb, end, [banner])

    # 3) Update the cell 0 / cell 1 intro notes with one line on demotion.
    intro = nb["cells"][1]  # cell 1 is the "0. 처음 보는 사람" markdown
    intro_src = "".join(intro["source"])
    note = ("\n\n> **Note (2026-05-12):** This notebook has been refactored "
            "around plan-centric directed transfer (Parts F–K). The original "
            "Part A–E analyses are now in the Appendix.\n")
    if "Note (2026-05-12)" not in intro_src:
        intro["source"] = (intro_src + note).splitlines(keepends=True)

    save_nb(nb)
    print("task23 applied")


CMDS["task23"] = cmd_task23
```

- [ ] **Step 2: Apply**

Run: `python tools/_nb_edit.py task23`
Expected: `task23 applied`.

- [ ] **Step 3: Verify**

Run:

```bash
python <<'EOF'
import json
nb = json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb'))
src = "".join(s for c in nb['cells'] for s in c['source'])
required = [
    "Appendix.A. HiP-AD diagnostic",
    "Appendix.B. VAD diagnostic",
    "Appendix.C. Task-pair helpful matrix",
    "Appendix.D. Distribution / correlation",
    "Appendix.E. Affinity / heavy-tail",
    "Appendix. Diagnostic background",
    "Note (2026-05-12)",
]
missing = [r for r in required if r not in src]
print("MISSING:", missing if missing else "(none)")
EOF
```

Expected: `MISSING: (none)`.

- [ ] **Step 4: Run the whole notebook smoke (no kernel, just JSON well-formed)**

Run: `python -c "import json; json.load(open('docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb')); print('JSON OK')"`
Expected: `JSON OK`.

- [ ] **Step 5: Commit**

```bash
git add tools/_nb_edit.py docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
git commit -m "refactor(notebook): demote Parts A–E to Diagnostic appendix"
```

---

## Task 24: Final verification

**Files:** (read-only)

- [ ] **Step 1: Run the full test suite for the gradient_analysis package**

Run: `pytest tests/gradient_analysis/ -x -q`
Expected: all pass (existing tests + new ones added in Tasks 1–15).

- [ ] **Step 2: Run a notebook-execution smoke (only Parts F/G/H, which need no extra data)**

Run:

```bash
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=600 \
  --output /tmp/plan_centric_smoke.ipynb \
  docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/phase2_hipad_vad_analysis_visualization.ipynb
```

Expected: notebook executes start-to-finish. Parts I/J/K may emit `SKIPPED`
print statements when the probe has not been re-run yet — that is the
intended behavior, not an error.

- [ ] **Step 3: Inspect output artifacts**

Run: `ls gradient_analysis_results/plan_centric/`
Expected: at minimum
```
plan_transfer_summary.csv
plan_transfer_top_beneficial.csv
plan_transfer_top_harmful.csv
plan_asymmetry_summary.csv
plan_transfer_ratio_summary.csv
effect_size_probe_summary.csv
helpful_vs_effect_size.csv
query_sensitivity.csv         (template, empty rows)
task_weight_elasticity_runs.csv (template, empty rows)
figures/F/  figures/G/  figures/H/
```

- [ ] **Step 4: Final commit (artifacts + smoke output)**

If any documentation tweaks fell out of the smoke run, fix them and commit.
Otherwise no commit needed.

---

## Notes for the executor

- The notebook editing helper `tools/_nb_edit.py` is intentionally small and
  refactor-local. After Task 23 it can be deleted in a follow-up PR; the
  plan keeps it under version control so each Task is independently
  reproducible by re-running its `cmd_taskNN` step.
- Cell-index anchors in Tasks 18–23 use substring search rather than fixed
  indices because inserting cells shifts indices.
- Korean strings in the asymmetry interpretation messages must match the
  test fixtures byte-for-byte. If you change the wording, update both
  `interpret_asymmetry_row` and `test_build_asymmetry_summary_signs_and_interpretation`
  in the same commit.
- `_seeded_forward` inside `tools/gradient_analysis/query_sensitivity.py`
  duplicates probe.py's logic; both must stay in sync if either changes.
- Tests for the HiP-AD-specific `get_task_queries` are intentionally
  gated; the runner test in Task 15 uses a synthetic `_Adapter` and does
  not require the full HiP-AD model.
