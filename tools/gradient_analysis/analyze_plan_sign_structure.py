#!/usr/bin/env python
"""Sign-structure analysis of aux->plan transfer (zero-compute, existing probe data).

Question this answers (2026-06-12): the per-batch aux->plan probe signal is
~2.5e-5 (well above the float32 floor) but signed-median ~ 0 because signs
split ~50/50 across batches. Is that split
  (a) pure noise  -> per-scene weighting is hopeless, only the integrated
      (training) net bias can decide who helps planning, or
  (b) scene-structured -> the SAME batches flip the SAME way across
      checkpoints/layers, so per-scene task weighting is a real lever.
Also: is the tiny net bias (median ~ -2.4e-7, "helps") statistically real?

Metric: analytic grad_dot (g_src . g_plan) — no finite-difference floor, no
curvature contamination. delta is kept as a cross-check.

Outputs -> gradient_analysis_results/plan_sign_structure/
  net_bias.csv             per (source, ckpt) median grad_dot + bootstrap CI
  sign_consistency.csv     cross-ckpt / cross-layer per-batch sign agreement
  cross_aux_corr.csv       Spearman corr between aux sources per batch
  conditioning.csv         grad_dot vs plan difficulty (baseline plan loss)
  cosine_tails.csv         conflict-cosine tail mass for *_plan pairs
  report.md                human-readable verdict
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results")
RUN = ROOT / "inter_gnn_b1000_pcgrad"
OUT = ROOT / "plan_sign_structure"
OUT.mkdir(parents=True, exist_ok=True)

CKPTS = ["1ep", "3ep", "9ep", "18ep"]
AUX = ["det", "map", "motion"]
N_BOOT = 2000
RNG = np.random.default_rng(42)

# ---------- load ----------
frames = []
for c in CKPTS:
    d = pd.read_csv(RUN / f"ckpt_{c}/probe/probe_per_batch.csv")
    d["ckpt"] = c
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["steps"] = df["steps"].astype(str)
df = df[(df["steps"] == "1") & (df["variant"] == "raw")]
ap = df[(df["source_task"].isin(AUX)) & (df["target_task"] == "plan")].copy()
print(f"aux->plan rows: {len(ap)}  (ckpts={CKPTS}, layers={sorted(ap['layer'].unique())})")

# ---------- 1. net bias: median grad_dot with bootstrap CI ----------
def boot_ci_median(x, n=N_BOOT):
    x = np.asarray(x)
    meds = np.median(RNG.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(meds, 2.5), np.percentile(meds, 97.5)

rows = []
for s in AUX:
    for c in CKPTS + ["ALL"]:
        sub = ap[ap.source_task == s] if c == "ALL" else ap[(ap.source_task == s) & (ap.ckpt == c)]
        for col in ["grad_dot", "delta"]:
            x = sub[col].values
            lo, hi = boot_ci_median(x)
            rows.append(dict(source=s, ckpt=c, metric=col, n=len(x),
                             median=np.median(x), ci_lo=lo, ci_hi=hi,
                             sig=("HELPS(plan↓)" if hi < 0 else
                                  "HURTS(plan↑)" if lo > 0 else "n.s.")
                             if col == "delta" else
                             ("ALIGNED" if lo > 0 else "OPPOSED" if hi < 0 else "n.s."),
                             pos_frac=(x > 0).mean()))
net = pd.DataFrame(rows)
net.to_csv(OUT / "net_bias.csv", index=False)

# ---------- 2. sign consistency across checkpoints / layers ----------
# batch_idx maps to the same scenes across ckpts (same dataloader seed); rely
# on inner-join so 1ep's skipped batches drop out.
cons_rows = []
for s in AUX:
    sub = ap[ap.source_task == s]
    # cross-ckpt: per (layer, batch_idx), sign across the 4 ckpts
    piv = sub.pivot_table(index=["layer", "batch_idx"], columns="ckpt",
                          values="grad_dot", aggfunc="first").dropna()
    signs = np.sign(piv[CKPTS].values)
    same_all = (np.abs(signs.sum(axis=1)) == len(CKPTS)).mean()
    p = (signs > 0).mean()  # overall positive rate
    null_same = p ** 4 + (1 - p) ** 4
    n = len(signs)
    k = int(same_all * n)
    pval = stats.binomtest(k, n, null_same, alternative="greater").pvalue
    cons_rows.append(dict(source=s, axis="cross_ckpt(4)", n_batches=n,
                          consistent_frac=same_all, null_frac=null_same,
                          excess=same_all - null_same, p_value=pval))
    # cross-layer within ckpt: per (ckpt, batch_idx), sign across 3 layers
    piv2 = sub.pivot_table(index=["ckpt", "batch_idx"], columns="layer",
                           values="grad_dot", aggfunc="first").dropna()
    sg = np.sign(piv2.values)
    same_l = (np.abs(sg.sum(axis=1)) == sg.shape[1]).mean()
    p2 = (sg > 0).mean()
    null_l = p2 ** 3 + (1 - p2) ** 3
    n2 = len(sg)
    pval2 = stats.binomtest(int(same_l * n2), n2, null_l, alternative="greater").pvalue
    cons_rows.append(dict(source=s, axis="cross_layer(3)", n_batches=n2,
                          consistent_frac=same_l, null_frac=null_l,
                          excess=same_l - null_l, p_value=pval2))
cons = pd.DataFrame(cons_rows)
cons.to_csv(OUT / "sign_consistency.csv", index=False)

# ---------- 3. cross-aux correlation (same scenes help/hurt together?) ----------
wide = ap.pivot_table(index=["ckpt", "layer", "batch_idx"], columns="source_task",
                      values="grad_dot", aggfunc="first").dropna()
corr_rows = []
for a, b in combinations(AUX, 2):
    rho, pv = stats.spearmanr(wide[a], wide[b])
    corr_rows.append(dict(pair=f"{a}~{b}", spearman_rho=rho, p_value=pv, n=len(wide)))
cc = pd.DataFrame(corr_rows)
cc.to_csv(OUT / "cross_aux_corr.csv", index=False)

# ---------- 4. conditioning on plan difficulty ----------
cond_rows = []
for s in AUX:
    sub = ap[ap.source_task == s]
    rho, pv = stats.spearmanr(sub["grad_dot"], sub["baseline_loss"])
    rho2, pv2 = stats.spearmanr(sub["grad_dot"].abs(), sub["baseline_loss"])
    cond_rows.append(dict(source=s, corr="grad_dot~plan_loss", rho=rho, p=pv))
    cond_rows.append(dict(source=s, corr="|grad_dot|~plan_loss", rho=rho2, p=pv2))
cond = pd.DataFrame(cond_rows)
cond.to_csv(OUT / "conditioning.csv", index=False)

# ---------- 5. cosine tails for *_plan pairs (M2 per-batch) ----------
tail_rows = []
for c in CKPTS:
    for f in sorted((RUN / f"ckpt_{c}" / "layer_conflict").glob("conflict_*_plan_per_batch.csv")) + \
             sorted((RUN / f"ckpt_{c}" / "layer_conflict").glob("conflict_plan_*_per_batch.csv")):
        d = pd.read_csv(f)
        col = "cosine" if "cosine" in d.columns else ("cos" if "cos" in d.columns else None)
        if col is None:
            continue
        x = d[col].dropna().values
        tail_rows.append(dict(ckpt=c, file=f.name, n=len(x), mean=x.mean(),
                              p_lt_m03=(x < -0.3).mean(), p_lt_m01=(x < -0.1).mean(),
                              p_gt_01=(x > 0.1).mean(), p_gt_03=(x > 0.3).mean()))
tails = pd.DataFrame(tail_rows)
if len(tails):
    tails.to_csv(OUT / "cosine_tails.csv", index=False)

# ---------- report ----------
L = ["# aux->plan sign-structure verdict", ""]
L += ["## 1. Net bias (median grad_dot, bootstrap 95% CI; ALIGNED = helps plan direction)", ""]
L.append(net[net.metric == "grad_dot"].to_string(index=False))
L += ["", "## 1b. Same, finite-difference delta cross-check (HELPS = plan loss down)", ""]
L.append(net[net.metric == "delta"].to_string(index=False))
L += ["", "## 2. Per-batch sign consistency (scene-structure test)",
      "consistent_frac >> null_frac (small p) = the SAME scenes flip the same way = structured.", ""]
L.append(cons.to_string(index=False))
L += ["", "## 3. Cross-aux Spearman (do the same scenes help/hurt for all aux tasks?)", ""]
L.append(cc.to_string(index=False))
L += ["", "## 4. Conditioning on plan difficulty (baseline plan loss)", ""]
L.append(cond.to_string(index=False))
if len(tails):
    L += ["", "## 5. Plan-pair cosine tails (M2 per-batch)", ""]
    L.append(tails.groupby("ckpt")[["p_lt_m03", "p_lt_m01", "p_gt_01", "p_gt_03"]]
             .mean().to_string())
(OUT / "report.md").write_text("\n".join(L))
print(f"wrote -> {OUT}")
print("\n".join(L[:40]))
