"""Planning-aligned task importance — post-hoc analysis on layer_conflict + probe CSVs.

For each aux task ``i ∈ {det, map, motion}`` we measure two complementary signals of
"does ``i`` help planning" at the ``inter_gnn`` shared-parameter scope and check whether
they agree:

  * **Alignment** ``A_i = cos(g_i, g_plan)`` at inter_gnn — read from
    ``layer_conflict/conflict_<i>_plan_per_batch.csv``. Positive = ``i``'s gradient
    pulls shared params the same direction planning does.
  * **Transfer**  ``T_i = -ΔL_plan`` after a 1-step virtual update in ``i``'s gradient
    direction at inter_gnn — read from ``probe/probe_per_batch.csv``
    (``target_task == plan``, ``source_task == i``).

Plus:
  * **Scene-dependence**: per-batch argmax winner + normalized entropy of the winner
    distribution, quantifying ``direction.md``'s claim that the important task varies
    by scene.
  * **Agreement**: Spearman rank corr, per-batch winner agreement, sign agreement
    between alignment (cheap proxy) and transfer (causal) — validates whether the
    proxy can drive Planning-Aligned Modular Distillation.
  * **motion det-collinearity**: motion has no query in HiP-AD (rides on det), so its
    gradient at shared params flows via the det pathway. We report
    ``cos(g_motion, g_det)`` and a flag when collinearity ≥ threshold.
  * **planning layer-sensitivity** (item 2 secondary): mean ``grad_norm`` of plan loss
    per inter_gnn layer (``source_task == plan`` rows of the probe CSV).

Design spec:
``docs/superpowers/specs/2026-05-23-planning-aligned-task-importance-design.md``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .plan_centric import EPS, bootstrap_ci

_DEFAULT_AUX: Tuple[str, ...] = ("det", "map", "motion")
_INTER_GNN_TOKEN = "inter_gnn"


# --------------------------------------------------------------------- helpers


def _detect_inter_gnn(values: Iterable[str]) -> List[str]:
    """Auto-detect inter_gnn-scoped group/layer keys from a series of strings."""
    return sorted({str(v) for v in values if v and _INTER_GNN_TOKEN in str(v)})


def _try_read_conflict_pair(
    layer_conflict_dir: Path, a: str, b: str,
) -> Optional[pd.DataFrame]:
    """Try ``conflict_{a}_{b}_per_batch.csv`` then ``conflict_{b}_{a}_per_batch.csv``."""
    for name in (f"conflict_{a}_{b}_per_batch.csv",
                 f"conflict_{b}_{a}_per_batch.csv"):
        p = layer_conflict_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if not df.empty:
                return df
    return None


def _safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_rate(values: np.ndarray, predicate) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(predicate(arr).mean()) if arr.size else float("nan")


def _normalized_entropy(counts: np.ndarray) -> float:
    """H(counts/sum) / log2(K). 0 = one bucket always wins; 1 = uniform."""
    counts = np.asarray(counts, dtype=float)
    K = counts.size
    if K <= 1:
        return 0.0
    total = counts.sum()
    if total <= 0:
        return float("nan")
    p = counts / total
    p = p[p > 0]
    H = float(-(p * np.log2(p)).sum())
    return H / float(np.log2(K))


# --------------------------------------------------------------------- loaders


def load_plan_alignment(
    layer_conflict_dir,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
    inter_gnn_groups: Optional[Sequence[str]] = None,
    plan_task: str = "plan",
) -> pd.DataFrame:
    """``cos(g_aux, g_plan)`` per batch at inter_gnn groups (long format).

    Pseudo-shared rows (either norm < EPS) are dropped to NaN cos so summaries treat
    them as missing rather than mean-biasing toward zero.
    """
    layer_conflict_dir = Path(layer_conflict_dir)
    frames: List[pd.DataFrame] = []
    for aux in aux_tasks:
        df = _try_read_conflict_pair(layer_conflict_dir, aux, plan_task)
        if df is None:
            continue
        keep = ["batch_idx", "group", "cos", "norm_a", "norm_b"]
        keep = [c for c in keep if c in df.columns]
        sub = df[keep].copy()
        if "norm_a" in sub.columns and "norm_b" in sub.columns:
            zero_norm = (sub["norm_a"].abs() < EPS) | (sub["norm_b"].abs() < EPS)
            sub.loc[zero_norm, "cos"] = np.nan
        sub.insert(0, "aux_task", aux)
        frames.append(sub[["aux_task", "batch_idx", "group", "cos"]])
    if not frames:
        return pd.DataFrame(columns=["aux_task", "batch_idx", "group", "cos"])
    out = pd.concat(frames, ignore_index=True)
    groups = list(inter_gnn_groups) if inter_gnn_groups else _detect_inter_gnn(out["group"])
    if groups:
        out = out[out["group"].isin(groups)].copy()
    return out.reset_index(drop=True)


def load_plan_transfer(
    probe_csv,
    variant: str = "raw",
    steps: int = 1,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
    inter_gnn_layers: Optional[Sequence[str]] = None,
    plan_task: str = "plan",
) -> pd.DataFrame:
    """1-step probe rows where target=plan, source∈aux_tasks, filtered to inter_gnn.

    Returns long df with ``gain = -delta`` so positive = helpful.

    Pcgrad-variant rows (``other_task != "_none"``) are excluded — the
    planning-aligned importance question (does aux help plan?) is defined on
    the single-source step, not on a paired projection. Their analysis lives
    in the dedicated projection-probe report.
    """
    cols = ["aux_task", "batch_idx", "layer", "delta", "gain", "grad_dot", "grad_norm"]
    p = Path(probe_csv)
    if not p.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    mask = (
        (df["target_task"] == plan_task)
        & (df["source_task"].isin(list(aux_tasks)))
        & (df["variant"] == variant)
        & (df["steps"] == steps)
    )
    if "other_task" in df.columns:
        mask &= df["other_task"].fillna("_none") == "_none"
    keep = ["source_task", "batch_idx", "layer", "delta", "grad_dot", "grad_norm"]
    keep = [c for c in keep if c in df.columns]
    sub = df.loc[mask, keep].copy().rename(columns={"source_task": "aux_task"})
    layers = list(inter_gnn_layers) if inter_gnn_layers else _detect_inter_gnn(sub.get("layer", pd.Series(dtype=str)))
    if layers and "layer" in sub.columns:
        sub = sub[sub["layer"].isin(layers)].copy()
    sub["gain"] = -sub["delta"]
    for c in cols:
        if c not in sub.columns:
            sub[c] = np.nan
    return sub[cols].reset_index(drop=True)


def load_task_collinearity(
    layer_conflict_dir,
    target: str = "motion",
    ref: str = "det",
    inter_gnn_groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """``cos(g_target, g_ref)`` per batch at inter_gnn groups (long format)."""
    df = _try_read_conflict_pair(Path(layer_conflict_dir), ref, target)
    if df is None:
        return pd.DataFrame(columns=["batch_idx", "group", "collinearity"])
    sub = df[["batch_idx", "group", "cos"]].rename(columns={"cos": "collinearity"})
    groups = list(inter_gnn_groups) if inter_gnn_groups else _detect_inter_gnn(sub["group"])
    if groups:
        sub = sub[sub["group"].isin(groups)].copy()
    return sub.reset_index(drop=True)


# --------------------------------------------------------------------- summaries


def build_importance_summary(
    align_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    collinearity_df: Optional[pd.DataFrame] = None,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
    collinear_threshold: float = 0.8,
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per aux_task with alignment, transfer, bootstrap CIs, motion flag."""
    coll_mean = float("nan")
    if collinearity_df is not None and not collinearity_df.empty:
        coll_mean = _safe_mean(collinearity_df["collinearity"].to_numpy())
    rows = []
    for i, aux in enumerate(aux_tasks):
        a = align_df.loc[align_df["aux_task"] == aux, "cos"].to_numpy() \
            if not align_df.empty else np.array([])
        g = transfer_df.loc[transfer_df["aux_task"] == aux, "gain"].to_numpy() \
            if not transfer_df.empty else np.array([])
        a_f = a[np.isfinite(a)]
        g_f = g[np.isfinite(g)]
        ci_a = bootstrap_ci(a_f, n_boot=n_boot, seed=seed + 2 * i) \
            if a_f.size else (float("nan"), float("nan"))
        ci_g = bootstrap_ci(g_f, n_boot=n_boot, seed=seed + 2 * i + 1) \
            if g_f.size else (float("nan"), float("nan"))
        is_motion = aux == "motion"
        rows.append({
            "aux_task": aux,
            "n_align": int(a_f.size),
            "n_transfer": int(g_f.size),
            "align_score": _safe_mean(a),
            "align_pos_rate": _safe_rate(a, lambda x: x > 0),
            "align_ci_lo": ci_a[0],
            "align_ci_hi": ci_a[1],
            "transfer_gain": _safe_mean(g),
            "helpful_rate": _safe_rate(g, lambda x: x > 0),
            "transfer_ci_lo": ci_g[0],
            "transfer_ci_hi": ci_g[1],
            "motion_det_collinearity": coll_mean if is_motion else float("nan"),
            "flag_det_collinear": bool(
                is_motion and np.isfinite(coll_mean) and coll_mean >= collinear_threshold
            ),
        })
    return pd.DataFrame(rows)


def build_scene_winner(
    metric_df: pd.DataFrame,
    value_col: str,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per batch: average ``value_col`` over scope per task, then argmax → winner.

    Returns ``(winner_per_batch, winner_distribution, normalized_entropy)``.
    """
    empty_wpb = pd.DataFrame(columns=["batch_idx", "winner_task", "top_value", "margin"])
    empty_dist = pd.DataFrame({
        "aux_task": list(aux_tasks),
        "win_count": [0] * len(aux_tasks),
        "win_frac": [0.0] * len(aux_tasks),
    })
    if metric_df is None or metric_df.empty or value_col not in metric_df.columns:
        return empty_wpb, empty_dist, float("nan")

    agg = (
        metric_df.groupby(["batch_idx", "aux_task"])[value_col].mean()
        .unstack("aux_task").reindex(columns=list(aux_tasks))
    )
    winners: List[Dict] = []
    for batch_idx, row in agg.iterrows():
        vals = row.dropna()
        if vals.empty:
            continue
        ordered = vals.sort_values(ascending=False)
        winners.append({
            "batch_idx": int(batch_idx),
            "winner_task": str(ordered.index[0]),
            "top_value": float(ordered.iloc[0]),
            "margin": float(ordered.iloc[0] - ordered.iloc[1])
                       if len(ordered) >= 2 else float("nan"),
        })
    wpb = pd.DataFrame(winners) if winners else empty_wpb.copy()
    if wpb.empty:
        counts = pd.Series([0] * len(aux_tasks), index=list(aux_tasks), dtype=int)
    else:
        counts = wpb["winner_task"].value_counts().reindex(list(aux_tasks), fill_value=0)
    total = int(counts.sum())
    dist = pd.DataFrame({
        "aux_task": list(aux_tasks),
        "win_count": counts.values.astype(int),
        "win_frac": (counts.values.astype(float) / total) if total > 0
                    else np.zeros(len(aux_tasks), dtype=float),
    })
    H = _normalized_entropy(counts.to_numpy(dtype=float))
    return wpb.reset_index(drop=True), dist, H


def build_alignment_transfer_agreement(
    align_df: pd.DataFrame,
    transfer_df: pd.DataFrame,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
) -> Dict[str, float]:
    """Spearman / winner / sign agreement between alignment and transfer at
    ``(batch_idx, aux_task)`` granularity (after averaging over inter_gnn scope).
    """
    nan_out = {
        "spearman": float("nan"),
        "per_batch_winner_agreement": float("nan"),
        "sign_agreement": float("nan"),
        "n_joined": 0,
    }
    if align_df.empty or transfer_df.empty:
        return nan_out
    a = (align_df.groupby(["batch_idx", "aux_task"])["cos"]
                 .mean().reset_index().rename(columns={"cos": "align_value"}))
    t = (transfer_df.groupby(["batch_idx", "aux_task"])["gain"]
                    .mean().reset_index().rename(columns={"gain": "transfer_value"}))
    j = (a.merge(t, on=["batch_idx", "aux_task"], how="inner")
          .dropna(subset=["align_value", "transfer_value"]))
    if j.empty:
        return nan_out
    j = j[j["aux_task"].isin(list(aux_tasks))]
    if j.empty:
        return nan_out

    rx = j["align_value"].rank().to_numpy()
    ry = j["transfer_value"].rank().to_numpy()
    if rx.size < 3 or rx.std() == 0 or ry.std() == 0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])

    sign_agree = float(
        (np.sign(j["align_value"]) == np.sign(j["transfer_value"])).mean()
    )

    aw, tw = {}, {}
    for b, sub in j.groupby("batch_idx"):
        if sub["align_value"].notna().any():
            aw[b] = str(sub.loc[sub["align_value"].idxmax(), "aux_task"])
        if sub["transfer_value"].notna().any():
            tw[b] = str(sub.loc[sub["transfer_value"].idxmax(), "aux_task"])
    common = sorted(set(aw) & set(tw))
    winner_agree = (
        float(np.mean([aw[b] == tw[b] for b in common])) if common else float("nan")
    )

    return {
        "spearman": spearman,
        "per_batch_winner_agreement": winner_agree,
        "sign_agreement": sign_agree,
        "n_joined": int(len(j)),
    }


def build_planning_layer_sensitivity(
    probe_csv,
    variant: str = "raw",
    steps: int = 1,
    inter_gnn_layers: Optional[Sequence[str]] = None,
    plan_task: str = "plan",
) -> pd.DataFrame:
    """Item 2 secondary: mean ``grad_norm`` of plan loss per inter_gnn layer.

    Replacement for the near-tautological query-based Part J. Reads ``source_task == plan``
    rows of the probe CSV (those rows record ``||g_plan||`` at the probe layer).
    """
    cols = ["layer", "mean_grad_norm", "n"]
    p = Path(probe_csv)
    if not p.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(p)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=cols)
    if df.empty:
        return pd.DataFrame(columns=cols)
    mask = (
        (df["source_task"] == plan_task)
        & (df["variant"] == variant)
        & (df["steps"] == steps)
    )
    if "other_task" in df.columns:
        mask &= df["other_task"].fillna("_none") == "_none"
    sub = df.loc[mask, ["layer", "grad_norm"]].copy().drop_duplicates(
        subset=["layer", "grad_norm"], keep="first"
    )
    # NOTE: probe writes one row per (source, target, layer, ...) so source=plan rows
    # repeat ||g_plan|| once per target. drop_duplicates above keeps a single entry per
    # (layer, grad_norm) within a batch_idx; harmless and avoids double-counting.
    layers = list(inter_gnn_layers) if inter_gnn_layers else _detect_inter_gnn(sub["layer"])
    if layers:
        sub = sub[sub["layer"].isin(layers)].copy()
    if sub.empty:
        return pd.DataFrame(columns=cols)
    out = (sub.groupby("layer")["grad_norm"]
              .agg(mean_grad_norm="mean", n="size")
              .reset_index()
              .sort_values("layer"))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------- runner


def run_planning_importance(
    layer_conflict_dir,
    probe_csv,
    out_dir,
    aux_tasks: Sequence[str] = _DEFAULT_AUX,
    variant: str = "raw",
    steps: int = 1,
    inter_gnn_groups: Optional[Sequence[str]] = None,
    inter_gnn_layers: Optional[Sequence[str]] = None,
    plan_task: str = "plan",
    collinear_ref: str = "det",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """Read layer_conflict + probe CSVs, write planning_importance/*.csv to ``out_dir``.

    Files written:
      - ``importance_summary.csv``
      - ``scene_winner_per_batch.csv``
      - ``scene_winner_distribution.csv``  (with ``normalized_entropy`` column)
      - ``alignment_transfer_agreement.csv``
      - ``planning_layer_sensitivity.csv``
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    align_df = load_plan_alignment(
        layer_conflict_dir, aux_tasks=aux_tasks,
        inter_gnn_groups=inter_gnn_groups, plan_task=plan_task,
    )
    transfer_df = load_plan_transfer(
        probe_csv, variant=variant, steps=steps, aux_tasks=aux_tasks,
        inter_gnn_layers=inter_gnn_layers, plan_task=plan_task,
    )
    coll_df = (
        load_task_collinearity(
            layer_conflict_dir, target="motion", ref=collinear_ref,
            inter_gnn_groups=inter_gnn_groups,
        ) if "motion" in aux_tasks else pd.DataFrame()
    )

    summary = build_importance_summary(
        align_df, transfer_df, coll_df, aux_tasks=aux_tasks,
        n_boot=n_boot, seed=seed,
    )
    summary.to_csv(out_dir / "importance_summary.csv", index=False)

    wpb_a, dist_a, H_a = build_scene_winner(align_df, "cos", aux_tasks=aux_tasks)
    wpb_t, dist_t, H_t = build_scene_winner(transfer_df, "gain", aux_tasks=aux_tasks)
    pd.concat([wpb_a.assign(metric="alignment"),
               wpb_t.assign(metric="transfer")],
              ignore_index=True).to_csv(out_dir / "scene_winner_per_batch.csv", index=False)
    pd.concat([dist_a.assign(metric="alignment", normalized_entropy=H_a),
               dist_t.assign(metric="transfer", normalized_entropy=H_t)],
              ignore_index=True).to_csv(out_dir / "scene_winner_distribution.csv", index=False)

    agreement = build_alignment_transfer_agreement(
        align_df, transfer_df, aux_tasks=aux_tasks,
    )
    pd.DataFrame([agreement]).to_csv(out_dir / "alignment_transfer_agreement.csv", index=False)

    layer_sens = build_planning_layer_sensitivity(
        probe_csv, variant=variant, steps=steps,
        inter_gnn_layers=inter_gnn_layers, plan_task=plan_task,
    )
    layer_sens.to_csv(out_dir / "planning_layer_sensitivity.csv", index=False)

    return {
        "summary": summary,
        "winner_distribution_alignment": dist_a,
        "winner_distribution_transfer": dist_t,
        "agreement": agreement,
        "layer_sensitivity": layer_sens,
        "entropy_alignment": H_a,
        "entropy_transfer": H_t,
    }
