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

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy import stats as _stats


EPS = 1e-8

_NULL_KINDS = ("batch_permutation", "sign_flip")
_RANK_BISERIAL_THRESHOLD = 0.1   # Cohen's small-effect convention
_RAW_ALPHA = 0.05


# --------------------------- pure helpers ---------------------------

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
    """[Legacy / Phase-2 use] Shuffle per-sample task identity within the
    batch and recompute cos.

    `g_a`, `g_b`: shape (B, ...) where the leading dim indexes samples that
    belong to the same batch but different tasks. The M2 cache currently
    stores group-summed gradients (no per-sample structure), so this function
    is unreachable from the production aggregator — it remains here for
    Phase 2 collectors that emit per-sample gradients.
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
        ag = a_perm.sum(dim=0)
        bg = b_perm.sum(dim=0)
        out[i] = _cosine(ag, bg)
    return out


def batch_permutation_null(
    grads_a: List[torch.Tensor],
    grads_b: List[torch.Tensor],
    n_repeats: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Pair task_a's gradient from batch i with task_b's from batch j (i != j)
    and recompute cos. The null asks: "if the task label were independent of
    which batch the gradient came from, what cos would we see?"

    `grads_a` / `grads_b` are lists of group-summed gradient tensors, one per
    batch. They must have the same length (= number of batches available for
    this (group, pair) cell) and matching tensor shapes.

    Returns an array of length `n_repeats` of cosines drawn from off-diagonal
    pairings. If the batch list has length 1 (no off-diagonal pairs available),
    returns a single-element NaN array — the caller will filter it out.
    """
    if len(grads_a) != len(grads_b):
        raise ValueError(f"batch count mismatch: {len(grads_a)} vs {len(grads_b)}")
    n_batches = len(grads_a)
    if n_batches < 2:
        return np.array([float("nan")], dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty(n_repeats, dtype=np.float64)
    # Pre-promote to float64 once.
    a_flat = [g.flatten().to(torch.float64) for g in grads_a]
    b_flat = [g.flatten().to(torch.float64) for g in grads_b]
    for k in range(n_repeats):
        i = int(rng.integers(0, n_batches))
        # Pick j != i — derangement of a single index.
        j = int(rng.integers(0, n_batches - 1))
        if j >= i:
            j += 1
        out[k] = _cosine(a_flat[i], b_flat[j])
    return out


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


# --------------------------- statistical test ---------------------------

@dataclass
class NullTestResult:
    observed_mean: float
    null_mean: float
    null_ci_lo: float
    null_ci_hi: float
    u_statistic: float
    p_value: float
    rank_biserial: float
    n_observed: int
    n_null: int


def compare_to_null(
    observed: np.ndarray,
    null: np.ndarray,
) -> NullTestResult:
    """Mann-Whitney U two-sided + rank-biserial effect size.

    Renamed from `test_against_null` to avoid pytest collection collision
    (pytest treats any top-level `test_*` callable as a test case).
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


# --------------------------- aggregator ---------------------------

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

    Two distinct null procedures supported:

    * `batch_permutation` — pair (g_a from batch i, g_b from batch j != i).
      One pooled draw set of size `n_repeats`; needs ≥ 2 batches.

    * `sign_flip` — per-batch sign flips on the group-summed gradients.
      Each batch contributes `n_repeats / n_batches` draws; works on a
      single batch.
    """
    batches = list(cached_batches)
    if not batches:
        return np.array([], dtype=np.float64)

    if null_kind == "batch_permutation":
        grads_a: List[torch.Tensor] = []
        grads_b: List[torch.Tensor] = []
        for b in batches:
            ga = b["shared"].get(task_a, {}).get(group)
            gb = b["shared"].get(task_b, {}).get(group)
            if ga is None or gb is None:
                continue
            # Skip cells where one side has zero norm — they contribute only
            # NaN cosines and would just bloat the output array.
            if float(ga.norm()) < EPS or float(gb.norm()) < EPS:
                continue
            grads_a.append(ga)
            grads_b.append(gb)
        if len(grads_a) < 2:
            return np.array([], dtype=np.float64)
        return batch_permutation_null(grads_a, grads_b, n_repeats=n_repeats, seed=seed)

    if null_kind == "sign_flip":
        per_batch = max(1, n_repeats // len(batches))
        chunks: List[np.ndarray] = []
        for i, b in enumerate(batches):
            ga = b["shared"].get(task_a, {}).get(group)
            gb = b["shared"].get(task_b, {}).get(group)
            if ga is None or gb is None:
                continue
            chunks.append(sign_flip_null(ga, gb, n_repeats=per_batch, seed=seed + i))
        if not chunks:
            return np.array([], dtype=np.float64)
        return np.concatenate(chunks)

    raise ValueError(f"unknown null_kind: {null_kind}")


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
        # Phase 1 #1 (HIGH fix 2026-05-03): the two null kinds are robustness
        # checks on the same hypothesis, NOT independent tests. Family size is
        # n_groups × n_pairs only — multiplying by n_kinds inflates the
        # correction and biases gating toward Pass B / Fail.
        bonferroni_family_size = max(1, len(group_keys) * len(pairs))
    rows = []
    for a, b in pairs:
        for group in group_keys:
            observed = _observed_per_batch(cached_batches, a, b, group)
            for kind in _NULL_KINDS:
                null = _generate_null_for_batches(
                    cached_batches, a, b, group, kind, n_repeats, seed,
                )
                res = compare_to_null(observed, null)
                p_bon = min(1.0, res.p_value * bonferroni_family_size) if np.isfinite(res.p_value) else float("nan")
                passes = bool(
                    abs(res.rank_biserial) >= _RANK_BISERIAL_THRESHOLD
                    and np.isfinite(p_bon)
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
