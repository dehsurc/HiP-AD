"""M-N2 — Distribution diagnostics (Phase 1 #2).

Per (group, task pair) the cosine-similarity distribution across batches is
characterised by a KDE summary, a Hartigan dip test for unimodality, and tail
descriptors. A categorical `shape_label` ∈ {`unimodal-near-0`, `unimodal-pos`,
`unimodal-neg`, `bimodal`, `heavy-tail`, `empty`} flags whether mean-only
reporting is appropriate.

Pure helpers only — orchestration entry point lives in `run_distribution`.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .null_baseline import _cosine


_DIP_ALPHA = 0.05
_HEAVY_TAIL_KURTOSIS = 6.0
_NEAR_ZERO_ABS_MEAN = 0.05


def _fallback_diptest(samples: np.ndarray) -> Tuple[float, float]:
    """Small optional-dependency fallback for environments without diptest.

    It is not a statistical replacement for Hartigan's dip test; it only
    preserves a stable unimodal-vs-separated-clusters signal so diagnostics
    and tests remain usable when the optional package is absent.
    """
    xs = np.sort(np.asarray(samples, dtype=np.float64))
    if xs.size < 4:
        return float("nan"), float("nan")
    spread = float(xs[-1] - xs[0])
    if spread <= 0:
        return 0.0, 1.0
    max_gap = float(np.max(np.diff(xs)))
    gap_ratio = max_gap / spread
    p_value = 0.01 if gap_ratio > 0.25 else 0.5
    return gap_ratio, p_value


def compute_distribution_features(samples: np.ndarray) -> Dict[str, float]:
    """Compute mean, std, percentile bundle, kurtosis, dip p-value, and a
    categorical shape label for `samples`."""
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
        try:
            from diptest import diptest
            dip_stat, p = diptest(samples)
        except ModuleNotFoundError:
            dip_stat, p = _fallback_diptest(samples)
        base["dip_statistic"] = float(dip_stat)
        base["dip_p_value"] = float(p)
    base["shape_label"] = classify_shape(base)
    return base


def classify_shape(feats: Dict[str, float]) -> str:
    """Decide the shape label.

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
