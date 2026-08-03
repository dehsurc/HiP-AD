#!/usr/bin/env python
"""Per-sample TSV-gradient analysis (NO averaging).

Companion to tsv_gradient.py. That script averages N samples into one mean
gradient matrix Gbar_task per shared inter_gnn weight and SVDs *that* single
averaged object, yielding ONE number per (matrix, task-pair). This script does
the opposite: it computes the per-task gradient matrices for EACH of N samples
*individually* and reports the DISTRIBUTION of cross-task alignment across the N
samples -- i.e. N results (1 per sample), not 1 result (the mean).

Why the distribution, not the per-sample spectrum
-------------------------------------------------
A single-sample gradient of an attn/linear weight is  G = sum_t delta_t x_t^T
over token positions, so rank(G) <= (#tokens) and plan's single-sample G is
nearly rank-1. Its SVD *spectrum* therefore reflects token/activation
statistics, not task structure -- it is CONFOUNDED and NOT comparable to the
averaged spectrum (see docs/TSV_RUNBOOK.md, section 5, item 1). We still record
per-sample effrank for reference, flagged as confounded.

The meaningful per-sample object is the CROSS-TASK ALIGNMENT, which is
well-defined at any rank:

  flat_cos(plan, aux)  = <G_plan, G_aux>_F / (||G_plan|| ||G_aux||)
  mode0_align(aux)     = u0^T G_aux v0 / ||G_aux||   (u0,v0 = plan's top left/
                         right singular vectors; = cos(G_aux, plan's dominant
                         rank-1 descent direction), in [-1, 1])

The scientific question this answers: the averaged pipeline reported
cos(Gbar_plan, Gbar_aux) ~ 0. Is that ~0 because (a) every sample is genuinely
~orthogonal, or (b) a symmetric +/- cancellation where each sample couples
strongly but the signs average out? Only the per-sample distribution can tell
these apart. For reference we also recompute the averaged "cosine of the means"
here so the two live side by side in one run.

Reuses tsv_gradient.py's harness: same targets, same grad extraction, same
qkv split, batch_size=1, temporal state reset per sample.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA
from tools.gradient_analysis.tsv_gradient import (
    TASKS, AUX, EPS, resolve_targets, grad_matrices, split_qkv,
)


def pr_effrank(s) -> float:
    """Participation-ratio effective rank  (sum s)^2 / sum s^2."""
    s = np.asarray(s, dtype=float)
    return float((s.sum() ** 2) / ((s ** 2).sum() + 1e-30))


def per_sample_metrics(G: dict, name: str) -> list:
    """G = {task: 2D cpu-float tensor} for ONE sample & ONE weight matrix.

    Returns one row-dict per aux task. plan's SVD is taken here so u0/v0 (its
    dominant descent mode) can be reused across all aux comparisons.
    """
    Gp = G["plan"]
    # SVD of plan's single-sample gradient. full_matrices=False -> U:(m,k),
    # S:(k,), Vh:(k,n) with k=min(m,n). rank is confounded (<= #tokens) but
    # the TOP direction u0 v0^T is the sample's dominant plan descent mode.
    U, S, Vh = torch.linalg.svd(Gp, full_matrices=False)
    u0, v0 = U[:, 0], Vh[0, :]
    s0 = float(S[0])
    plan_fro = float(Gp.norm())
    plan_effrank = pr_effrank(S.numpy())
    rows = []
    for aux in AUX:
        if aux not in G:
            continue
        Ga = G[aux]
        aux_fro = float(Ga.norm())
        flat_cos = float((Gp.flatten() @ Ga.flatten()) / (plan_fro * aux_fro + EPS))
        mode0_proj = float(u0 @ Ga @ v0)              # <G_aux, u0 v0^T>_F
        mode0_align = mode0_proj / (aux_fro + EPS)    # cos(G_aux, plan top mode)
        aux_effrank = pr_effrank(torch.linalg.svdvals(Ga).numpy())
        rows.append(dict(
            matrix=name, aux=aux,
            flat_cos=flat_cos, mode0_align=mode0_align, mode0_proj=mode0_proj,
            plan_s0=s0, plan_fro=plan_fro, aux_fro=aux_fro,
            plan_effrank=plan_effrank, aux_effrank=aux_effrank,
        ))
    return rows


def _narrate_first_sample(sample_G: dict) -> None:
    """Print a detailed breakdown of what is computed, on the first sample."""
    names = list(sample_G.keys())
    print("\n[persample] ===== FIRST-SAMPLE COMPUTATION BREAKDOWN =====", flush=True)
    print(f"[persample] tracked matrices: {len(names)} (e.g. {names[:4]})", flush=True)
    ex = names[0]
    G = sample_G[ex]
    print(f"[persample] example weight '{ex}'  shape={tuple(G['plan'].shape)}", flush=True)
    print("[persample]   per-task gradient Frobenius norms ||dL_task/dW||:", flush=True)
    for t in TASKS:
        if t in G:
            print(f"[persample]     {t:>6s}: {float(G[t].norm()):.4e}", flush=True)
    Gp = G["plan"]
    S = torch.linalg.svdvals(Gp)
    print(f"[persample]   plan top-5 singular values: "
          f"{[round(float(x), 3) for x in S[:5]]}", flush=True)
    print(f"[persample]   plan per-sample effrank = {pr_effrank(S.numpy()):.2f}  "
          f"(<= token-count bound -> CONFOUNDED, see RUNBOOK section 5)", flush=True)
    for aux in AUX:
        if aux in G:
            fc = float((Gp.flatten() @ G[aux].flatten())
                       / (Gp.norm() * G[aux].norm() + EPS))
            print(f"[persample]   cos(plan, {aux}) on this sample = {fc:+.4f}", flush=True)
    print("[persample] =================================================\n", flush=True)


def _cos_of_means(mean_grads: dict) -> dict:
    """OLD metric: cos(mean G_plan, mean G_aux) over the concatenated weights.
    One number per aux -- the averaged-pipeline value, for side-by-side."""
    names = [n for n in mean_grads["plan"]
             if all(n in mean_grads[t] for t in TASKS)]
    gp = torch.cat([mean_grads["plan"][n].flatten() for n in names])
    out = {}
    for aux in AUX:
        ga = torch.cat([mean_grads[aux][n].flatten() for n in names])
        out[aux] = float((gp @ ga) / (gp.norm() * ga.norm() + EPS))
    return out


def analyze(gdf: pd.DataFrame, df: pd.DataFrame, mean_grads: dict,
            out: Path) -> None:
    """gdf = per-sample GLOBAL cosine (1 row per sample x aux; headline).
    df  = per (sample, matrix, aux) metrics (finer grain).
    """
    out.mkdir(parents=True, exist_ok=True)
    gdf.to_csv(out / "per_sample_global.csv", index=False)
    df.to_csv(out / "per_sample_metrics.csv", index=False)

    old_cos = _cos_of_means(mean_grads)

    def q(p):
        return lambda x: float(np.quantile(x, p))

    # HEADLINE: distribution over the N per-sample global cosines.
    summ = gdf.groupby("aux").agg(
        n=("flat_cos", "size"),
        cos_mean=("flat_cos", "mean"),
        cos_std=("flat_cos", "std"),
        cos_median=("flat_cos", "median"),
        cos_q05=("flat_cos", q(0.05)),
        cos_q95=("flat_cos", q(0.95)),
        cos_min=("flat_cos", "min"),
        cos_max=("flat_cos", "max"),
        frac_neg=("flat_cos", lambda x: float((x < 0).mean())),
        frac_strong=("flat_cos", lambda x: float((x.abs() > 0.1).mean())),
    ).reset_index()
    summ["cos_of_means_ref"] = summ["aux"].map(old_cos)
    summ.to_csv(out / "per_sample_summary.csv", index=False)

    # Per-matrix breakdown (which weights carry the per-sample coupling).
    bym = df.groupby(["matrix", "aux"]).agg(
        cos_mean=("flat_cos", "mean"), cos_std=("flat_cos", "std"),
        abs_cos_mean=("flat_cos", lambda x: float(x.abs().mean())),
        frac_neg=("flat_cos", lambda x: float((x < 0).mean())),
        align_mean=("mode0_align", "mean"),
    ).reset_index()
    bym.to_csv(out / "per_sample_by_matrix.csv", index=False)

    n = int(gdf["sample"].nunique())
    print(f"\n=== HEADLINE: per-sample GLOBAL cross-task cosine distribution (N={n}) ===")
    print("    (all inter_gnn weights concatenated -> ONE cos(plan,aux) per sample)")
    print(summ[["aux", "n", "cos_mean", "cos_std", "cos_median",
                "cos_q05", "cos_q95", "frac_neg", "frac_strong",
                "cos_of_means_ref"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  cos_mean         = mean over N samples of the per-sample cos(G_plan,G_aux)")
    print("  cos_of_means_ref = cos(mean G_plan, mean G_aux)  <- OLD averaged (~0) metric")
    print("  frac_neg         = fraction of samples with cos<0   (~0.5 => sign-balanced)")
    print("  frac_strong      = fraction of samples with |cos|>0.1 (per-sample coupling strength)")

    print("\n=== top per-matrix per-sample coupling (mean |cos| over samples) ===")
    top = bym.sort_values("abs_cos_mean", ascending=False).head(10)
    print(top[["matrix", "aux", "cos_mean", "abs_cos_mean", "frac_neg"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== per-sample effrank (CONFOUNDED, reference only) ===")
    er = df.groupby("aux").agg(plan_effrank=("plan_effrank", "mean"),
                               aux_effrank=("aux_effrank", "mean")).reset_index()
    print(er.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # Interpretation on the headline (global) distribution: does the averaged
    # ~0 hide strong per-sample coupling that cancels by sign?
    print("\n=== interpretation (per aux, on global per-sample cosine) ===")
    for _, r in summ.iterrows():
        m, fn, fs = r["cos_mean"], r["frac_neg"], r["frac_strong"]
        spread = r["cos_q95"] - r["cos_q05"]
        if fs >= 0.25 and abs(m) < 0.05:
            verdict = (f"SIGN-CANCELLATION: strong per-sample coupling "
                       f"(|cos|>0.1 in {fs:.0%} of samples) but mean~0, "
                       f"frac_neg={fn:.2f} -> +/- averages out")
        elif fs >= 0.25 and abs(m) >= 0.05:
            sign = "reinforcing(+)" if m > 0 else "opposing(-)"
            verdict = (f"CONSISTENT {sign} coupling: mean={m:+.3f}, "
                       f"|cos|>0.1 in {fs:.0%} of samples")
        elif spread < 0.15:
            verdict = "GENUINE near-orthogonal (small per-sample too)"
        else:
            verdict = (f"WEAK/mixed: mean={m:+.3f}, spread(q05..q95)={spread:.3f}, "
                       f"frac_neg={fn:.2f}")
        print(f"  plan<->{r['aux']:<6s}: {verdict}")


def run(ckpt_path, config, n_samples, out, target="inter_gnn",
        device="cuda:0", seed=42):
    cfg = dict(adapter="hipad", model_config=config, tasks=TASKS,
               device=device, seed=seed)
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(config))
    adapter = RGA.build_adapter(rt, cfg)
    print(f"[persample] checkpoint = {ckpt_path}", flush=True)
    model = RGA.load_model(rt, adapter, Path(ckpt_path), device, True)
    raw = model.module if hasattr(model, "module") else model
    targets = resolve_targets(raw, config, target)
    params = [p for _, p, _ in targets]
    print(f"[persample] target={target}: {len(targets)} weight matrices tracked; "
          f"collecting {n_samples} INDIVIDUAL samples (no averaging)", flush=True)

    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=seed)
    state_snap = adapter.snapshot_temporal_state(model)
    # accum kept ONLY to compute the averaged cosine-of-means reference.
    accum = {t: {} for t in TASKS}
    counts = {t: 0 for t in TASKS}
    rows = []          # per (sample, matrix, aux)
    global_rows = []   # per (sample, aux) -- concatenated-weight cosine (headline)
    t0, used, err = time.time(), 0, 0
    for i, data in enumerate(dl):
        if used >= n_samples:
            break
        try:
            adapter.restore_temporal_state(model, state_snap)
            losses = adapter.forward_losses(model, data)
            sample_G: dict = {}  # {matrix_name: {task: tensor}}
            for ti, task in enumerate(TASKS):
                L = adapter.split_losses(losses, task)
                if L is None:
                    continue
                gm = grad_matrices(L, params, retain=(ti < len(TASKS) - 1))
                for (nm, _, kind), M in zip(targets, gm):
                    if kind == "qkv":
                        items = split_qkv(nm, M)
                    elif kind == "conv":
                        items = [(nm, M.reshape(M.shape[0], -1))]
                    else:
                        items = [(nm, M)]
                    for sn, blk in items:
                        blk = blk.float().cpu()
                        sample_G.setdefault(sn, {})[task] = blk
                        a = accum[task].get(sn)
                        accum[task][sn] = blk if a is None else a + blk
                counts[task] += 1
            # matrices where ALL 4 tasks produced a gradient (needed for both
            # the per-matrix metrics and the concatenated global cosine).
            full_names = [sn for sn in sample_G
                          if all(t in sample_G[sn] for t in TASKS)]
            if not full_names:
                continue
            for sn in full_names:
                for r in per_sample_metrics(sample_G[sn], sn):
                    r["sample"] = used
                    rows.append(r)
            # global per-sample cosine: concatenate every tracked weight's grad
            # into one vector per task, then cos(plan, aux). This is the
            # "one result per sample" object (matches M2's flat-cosine scope).
            gp = torch.cat([sample_G[sn]["plan"].flatten() for sn in full_names])
            for aux in AUX:
                ga = torch.cat([sample_G[sn][aux].flatten() for sn in full_names])
                global_rows.append(dict(
                    sample=used, aux=aux,
                    flat_cos=float((gp @ ga) / (gp.norm() * ga.norm() + EPS)),
                    plan_norm=float(gp.norm()), aux_norm=float(ga.norm()),
                ))
            if used == 0:
                _narrate_first_sample(sample_G)
            used += 1
        except Exception as e:
            err += 1
            torch.cuda.empty_cache()
            print(f"[persample] ERR sample {i}: {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            if err > 30:
                print("[persample] >30 errors, abort", flush=True)
                break
            continue
        if used and used % 10 == 0:
            print(f"[persample] {used}/{n_samples} "
                  f"({(time.time()-t0)/60:.1f} min, err={err})", flush=True)

    print(f"[persample] collected {used} samples, counts={counts} err={err} "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)
    mean_grads = {t: {k: v / max(counts[t], 1) for k, v in accum[t].items()}
                  for t in TASKS}
    df = pd.DataFrame(rows)
    gdf = pd.DataFrame(global_rows)
    if df.empty or gdf.empty:
        print("[persample] no rows collected — nothing to analyze")
        return
    analyze(gdf, df, mean_grads, Path(out))
    torch.save({"per_sample_rows": rows, "global_rows": global_rows,
                "mean_grads": mean_grads, "counts": counts},
               Path(out) / "per_sample.pt")
    print(f"[persample] wrote -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default="ckpts/rev_nusc/70+stage2_18ep.pth")
    ap.add_argument("--config",
                    default="ckpts/rev_nusc/E2_E1_stage2_18ep_from70ep.py")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--target", default="inter_gnn",
                    choices=["inter_gnn", "backbone", "neck", "backbone_neck"])
    ap.add_argument("--out",
                    default="gradient_analysis_results/tsv_gradient/persample_18ep")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    run(a.ckpt_path, a.config, a.n, a.out, target=a.target)
