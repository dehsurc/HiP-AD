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


def get_probe_base(
    probe: pd.DataFrame, variant: str = "normalized", steps: int = 1
) -> pd.DataFrame:
    """Standardize, then filter to ``(variant, steps)`` subset."""
    df = standardize_probe_df(probe)
    return df[(df["variant"] == variant) & (df["steps"] == steps)].copy()


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


def top_beneficial(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows ranked by ``by`` descending (most beneficial first)."""
    return summary.sort_values(by, ascending=False).head(k).reset_index(drop=True)


def top_harmful(summary: pd.DataFrame, k: int = 10, by: str = "mean_gain") -> pd.DataFrame:
    """Top-``k`` rows.

    For ``by="mean_gain"`` sorts ascending (worst gain first). For
    ``by="large_harm_rate"`` sorts descending.
    """
    ascending = by == "mean_gain"
    return summary.sort_values(by, ascending=ascending).head(k).reset_index(drop=True)


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
    return ("A는 planning auxiliary로 유용하지만, planning update의 A 방향 "
            "효과는 약하다.")


def build_asymmetry_summary(base: pd.DataFrame) -> pd.DataFrame:
    """Part G. Directed asymmetry around planning, per (model, ckpt, layer, aux_task)."""
    required = {"model", "checkpoint", "checkpoint_order", "source_task",
                "target_task", "layer", "delta", "gain"}
    if not ensure_columns(base, required, "asymmetry_summary input"):
        raise KeyError(f"missing required columns: {required - set(base.columns)}")

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
            r["ci_lo_asym"] = ci_lo_ap - ci_hi_pa
            r["ci_hi_asym"] = ci_hi_ap - ci_lo_pa
            r["interpretation"] = interpret_asymmetry_row(r)
            rows.append(r)
    return pd.DataFrame(rows)


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


# ---------------------------------------------------------------------------
# Part I – first-order Taylor prediction vs actual 1-step delta
# ---------------------------------------------------------------------------

def detect_first_order_columns(probe: pd.DataFrame) -> dict | None:
    """Return canonical first-order column names if both are present."""
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
    from scipy.stats import rankdata
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def build_first_order_residual_summary(base: pd.DataFrame, cols: dict) -> pd.DataFrame:
    """Part I. First-order prediction vs actual 1-step delta."""
    gd = cols["grad_dot"]
    ss = cols["step_size"]
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


# ---------------------------------------------------------------------------
# Part J – query sensitivity loader / summary
# ---------------------------------------------------------------------------

QUERY_SENSITIVITY_TEMPLATE_COLUMNS = [
    "model", "checkpoint", "checkpoint_order", "batch_idx", "scene_token",
    "layer", "task_query_type", "query_index", "query_norm",
    "grad_plan_wrt_query_norm",
]


def load_or_template_query_sensitivity(path):
    """Load query-sensitivity CSV; write a template if the file does not exist."""
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

    grouper = (
        ["model", "checkpoint", "checkpoint_order", "layer", "task_query_type"]
        if "checkpoint_order" in df.columns
        else ["model", "checkpoint", "layer", "task_query_type"]
    )

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

    sum_keys = [k for k in grouper if k != "task_query_type"]
    out["share_task"] = out.groupby(sum_keys)["mean_sensitivity"].transform(
        lambda s: s / (s.sum() + EPS)
    )
    return out


# ---------------------------------------------------------------------------
# Part K – elasticity loader / summary / safe range
# ---------------------------------------------------------------------------

ELASTICITY_TEMPLATE_COLUMNS = [
    "model", "run_id", "seed", "checkpoint",
    "lambda_det", "lambda_map", "lambda_motion", "lambda_plan",
    "det_metric", "map_metric", "motion_metric",
    "plan_l2", "collision", "plan_metric_name",
    "is_baseline", "notes",
]


def load_or_template_elasticity(path):
    """Load elasticity CSV; write a template if the file does not exist."""
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
    plan_metric_col = "plan_l2"
    for model_name, model_df in df.groupby("model", sort=False):
        baseline = model_df[baseline_mask.reindex(model_df.index, fill_value=False)]
        if baseline.empty:
            continue
        plan_base = float(baseline[plan_metric_col].iloc[0])
        for task in ("det", "map", "motion", "plan"):
            lam_col = f"lambda_{task}"
            task_metric_col = (
                plan_metric_col if task == "plan"
                else _resolve_task_metric(model_df, task)
            )
            if task_metric_col is None:
                continue
            task_lib = _metric_lower_is_better(task_metric_col)
            task_base = float(baseline[task_metric_col].iloc[0])
            others = [f"lambda_{o}" for o in ("det", "map", "motion", "plan") if o != task]
            mask = (model_df[others] == 1.0).all(axis=1) & (model_df[lam_col] != 1.0)
            swept = model_df[mask]
            for _, r in swept.iterrows():
                lam = float(r[lam_col])
                task_metric = float(r[task_metric_col])
                plan_metric = float(r[plan_metric_col])
                rel_task = (
                    (task_base - task_metric) / (abs(task_base) + EPS)
                    if task_lib
                    else (task_metric - task_base) / (abs(task_base) + EPS)
                )
                rel_plan = (plan_metric - plan_base) / (abs(plan_base) + EPS)
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
