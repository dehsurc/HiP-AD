#!/usr/bin/env python
"""Per-batch aux->plan ALIGNMENT vs LOSS-DELTA relationship (the "look at the 1000
individually before averaging" analysis).

Motivation (2026-06-18): the existing pipeline reports averaged grad_dot and averaged
delta. This script instead keeps every per-batch (alignment, delta) pair and answers:
  Q1. Is grad_dot<->delta just the 1st-order identity delta ~= -alpha*grad_dot? (tautology check)
  Q2. Does NORMALIZED alignment cos(g_aux,g_plan) -- which is NOT pinned to delta by the
      identity (delta depends on cos AND the gradient norms) -- predict per-batch delta?
  Q3. Per batch, does aux help or hurt plan, and do the signs CANCEL across the 1000?
      (mean~0 is "cancellation", not "no effect" -- show pos_frac and mean|.| vs |mean|.)
  Q4. Is |alignment| larger on plan-hard scenes? (conditioning on plan baseline_loss)

Sign convention: virtual update = gradient DESCENT on the source loss, so
  delta_plan = L_plan(after) - L_plan(before) ~= -step_size * <g_src, g_plan> = -step_size*grad_dot.
  grad_dot>0 (ALIGNED)  -> delta<0  -> plan loss DOWN  -> source HELPS plan.
  grad_dot<0 (OPPOSED)  -> delta>0  -> plan loss UP    -> source HURTS plan.

Data:  inter_gnn_b1000_pcgrad/ckpt_{1,3,9,18}ep/probe/probe_per_batch.csv
Filter: target_task=='plan', variant=='raw', steps==1, source in {det,map,motion}.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/yongjae/e2e/HiP-AD-pcgrad")
RUN = ROOT / "gradient_analysis_results/inter_gnn_b1000_pcgrad"
OUTFIG = ROOT / "docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/figures"
OUTCSV = RUN / "align_delta_perbatch"
OUTFIG.mkdir(parents=True, exist_ok=True)
OUTCSV.mkdir(parents=True, exist_ok=True)

CKPTS = ["1ep", "3ep", "9ep", "18ep"]
AUX = ["det", "map", "motion"]
SCOL = {"det": "#1f77b4", "map": "#d62728", "motion": "#2ca02c"}
RNG = np.random.default_rng(42)
EPS = 1e-12


def boot_ci_median(x, n=2000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    meds = np.median(RNG.choice(x, size=(n, len(x)), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ---------- load + build cosine via per-task gradient norms ----------
raw_frames = []
for ck in CKPTS:
    f = RUN / f"ckpt_{ck}/probe/probe_per_batch.csv"
    if not f.exists():
        print(f"[warn] missing {f}")
        continue
    d = pd.read_csv(f)
    d["ckpt"] = ck
    raw_frames.append(d)
if not raw_frames:
    sys.exit("no probe_per_batch.csv found")
full = pd.concat(raw_frames, ignore_index=True)
full["steps"] = full["steps"].astype(str)
full = full[(full.steps == "1") & (full.variant == "raw")].copy()

# pick layer: prefer full-param update "_all", else the most frequent layer
layers = full["layer"].value_counts()
LAYER = "_all" if "_all" in layers.index else layers.index[0]
full = full[full.layer == LAYER].copy()
print(f"[info] layer={LAYER}  rows={len(full)}  ckpts={sorted(full.ckpt.unique())}")

# per (ckpt, batch_idx, source_task) gradient norm  (||g_task|| at this layer)
norm = (full.groupby(["ckpt", "batch_idx", "source_task"])["grad_norm"]
        .first().rename("gnorm").reset_index())
norm_plan = norm[norm.source_task == "plan"][["ckpt", "batch_idx", "gnorm"]] \
    .rename(columns={"gnorm": "gnorm_plan"})

ap = full[(full.source_task.isin(AUX)) & (full.target_task == "plan")].copy()
ap = ap.merge(norm.rename(columns={"gnorm": "gnorm_src"}),
              on=["ckpt", "batch_idx", "source_task"], how="left")
ap = ap.merge(norm_plan, on=["ckpt", "batch_idx"], how="left")
ap["cos"] = ap["grad_dot"] / (ap["gnorm_src"] * ap["gnorm_plan"] + EPS)
ap = ap.replace([np.inf, -np.inf], np.nan).dropna(subset=["grad_dot", "delta", "cos"])
print(f"[info] aux->plan paired rows={len(ap)}  per-source: "
      + ", ".join(f"{s}={int((ap.source_task==s).sum())}" for s in AUX))

STEP = float(ap.step_size.iloc[0])

# ---------- per-(source,ckpt) "stats over the 1000 batches" ----------
rows = []
for s in AUX:
    for ck in CKPTS + ["ALL"]:
        sub = ap[ap.source_task == s] if ck == "ALL" else ap[(ap.source_task == s) & (ap.ckpt == ck)]
        if len(sub) < 2:
            continue
        gd = sub.grad_dot.values
        dl = sub.delta.values
        cs = sub.cos.values
        pos = (gd > 0).mean()                       # P(aligned) -> ~0.5 == sign cancellation
        signed_mean = gd.mean()
        abs_mean = np.abs(gd).mean()
        # cancellation factor: how much the per-batch magnitude survives averaging
        cancel = abs(signed_mean) / (abs_mean + EPS)
        # does the (noisy) delta sign agree with the grad_dot verdict?  HELP=delta<0
        agree = ((gd > 0) == (dl < 0)).mean()
        lo, hi = boot_ci_median(gd)
        verdict = "ALIGNED/helps" if lo > 0 else "OPPOSED/hurts" if hi < 0 else "n.s.(cancels)"
        # non-tautological: does normalized cos predict delta? (rank corr, sign-flipped)
        rho_cos, p_cos = stats.spearmanr(cs, dl)
        rows.append(dict(source=s, ckpt=ck, n=len(sub),
                         pos_frac=pos, signed_mean_gd=signed_mean, abs_mean_gd=abs_mean,
                         cancel_ratio=cancel, median_gd=np.median(gd), ci_lo=lo, ci_hi=hi,
                         sign_agree_gd_delta=agree, spearman_cos_delta=rho_cos, p_cos_delta=p_cos,
                         mean_cos=cs.mean(), absmean_cos=np.abs(cs).mean(), verdict=verdict))
stat = pd.DataFrame(rows)
stat.to_csv(OUTCSV / "stats.csv", index=False)

# ---------- conditioning: |grad_dot| vs plan difficulty ----------
cond = []
for s in AUX:
    sub = ap[ap.source_task == s]
    rho, p = stats.spearmanr(np.abs(sub.grad_dot), sub.baseline_loss)
    cond.append(dict(source=s, corr="|grad_dot|~plan_baseline_loss", rho=rho, p=p, n=len(sub)))
cond = pd.DataFrame(cond)
cond.to_csv(OUTCSV / "conditioning.csv", index=False)

# ===================== FIGURE =====================
fig, ax = plt.subplots(2, 3, figsize=(20, 11))

# A. grad_dot vs delta -- the 1st-order TAUTOLOGY
a = ax[0, 0]
for s in AUX:
    g = ap[ap.source_task == s]
    a.scatter(g.grad_dot, g.delta, s=3, alpha=0.2, color=SCOL[s], label=s, rasterized=True)
xs = np.linspace(ap.grad_dot.min(), ap.grad_dot.max(), 100)
a.plot(xs, -STEP * xs, "k--", lw=1.6, label=f"identity: delta=-{STEP}*grad_dot")
r_taut, _ = stats.pearsonr(ap.grad_dot, ap.delta)
a.set(xlabel="grad_dot = <g_aux,g_plan>", ylabel="delta = dL_plan (1 step)",
      title=f"A. grad_dot vs delta = TAUTOLOGY (Pearson r={r_taut:.3f})\n"
            f"-> same 1st-order quantity, no new info")
a.axhline(0, color=".5", lw=.5); a.axvline(0, color=".5", lw=.5); a.legend(fontsize=8, markerscale=3)

# B. cos vs delta -- NON-tautological (delta also depends on the norms)
b = ax[0, 1]
for s in AUX:
    g = ap[ap.source_task == s]
    b.scatter(g.cos, g.delta, s=3, alpha=0.2, color=SCOL[s], rasterized=True)
rho_all, _ = stats.spearmanr(ap.cos, ap.delta)
b.set(xlabel="cos(g_aux,g_plan)  (normalized alignment)", ylabel="delta = dL_plan (1 step)",
      title=f"B. cos vs delta = NON-tautological (Spearman rho={rho_all:.3f})\n"
            f"normalized alignment vs actual plan change")
b.axhline(0, color=".5", lw=.5); b.axvline(0, color=".5", lw=.5)

# C. |grad_dot| vs plan difficulty
c = ax[0, 2]
for s in AUX:
    g = ap[ap.source_task == s]
    c.scatter(g.baseline_loss, np.abs(g.grad_dot), s=3, alpha=0.2, color=SCOL[s], rasterized=True)
c.set(xlabel="plan baseline_loss (scene difficulty)", ylabel="|grad_dot| (interaction magnitude)",
      title="C. interaction MAGNITUDE vs plan difficulty\n"
            + " ; ".join(f"{r.source}:rho={r.rho:.2f}" for _, r in cond.iterrows()))
c.set_yscale("log")

# D. sign split per (source x ckpt) -- the CANCELLATION story
d = ax[1, 0]
width = 0.2
xticks = np.arange(len(CKPTS))
for i, s in enumerate(AUX):
    fr = [ (ap[(ap.source_task==s)&(ap.ckpt==ck)].grad_dot > 0).mean() if
           len(ap[(ap.source_task==s)&(ap.ckpt==ck)]) else np.nan for ck in CKPTS]
    d.bar(xticks + (i-1)*width, fr, width, color=SCOL[s], label=s)
d.axhline(0.5, color="k", ls="--", lw=1.2, label="0.5 = perfect cancellation")
d.set(xticks=xticks, xticklabels=CKPTS, ylim=(0.3, 0.7),
      ylabel="P(grad_dot>0) = frac batches where aux HELPS plan",
      title="D. per-batch sign split ~ 0.5 = signs CANCEL\n(why the average is ~0)")
d.legend(fontsize=8)

# E. distribution of per-batch grad_dot (signed) -> symmetric around 0
e = ax[1, 1]
for s in AUX:
    g = ap[ap.source_task == s].grad_dot.values
    lim = np.percentile(np.abs(g), 99)
    e.hist(np.clip(g, -lim, lim), bins=80, histtype="step", lw=1.6, color=SCOL[s],
           density=True, label=f"{s} (mean={g.mean():.1e}, mean|.|={np.abs(g).mean():.1e})")
e.axvline(0, color="k", lw=.8)
e.set(xlabel="per-batch grad_dot (signed)", ylabel="density",
      title="E. per-batch dist ~ symmetric about 0\nmean|.| >> |mean| => real but cancelling")
e.legend(fontsize=7.5)

# F. cancellation ratio |mean|/mean|.| per (source x ckpt) -- closer to 0 = more cancelling
f = ax[1, 2]
for i, s in enumerate(AUX):
    cr = [ stat[(stat.source==s)&(stat.ckpt==ck)].cancel_ratio.values[0]
           if len(stat[(stat.source==s)&(stat.ckpt==ck)]) else np.nan for ck in CKPTS]
    f.plot(xticks, cr, "o-", color=SCOL[s], label=s)
f.set(xticks=xticks, xticklabels=CKPTS, ylabel="|mean| / mean|.|  (0 = total cancellation)",
      title="F. cancellation ratio per checkpoint\n(how much per-batch signal survives averaging)")
f.legend(fontsize=8)

fig.suptitle("Per-batch aux->plan ALIGNMENT vs DELTA (1000-batch joint analysis): "
             "grad_dot~delta is a tautology; the real story is per-batch SIGN CANCELLATION",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTFIG / "align_delta_perbatch.png", dpi=120)
print(f"[ok] wrote {OUTFIG/'align_delta_perbatch.png'}")

# ---------- printed verdict ----------
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== Q1 tautology: grad_dot vs delta Pearson r =", round(r_taut, 4),
      "(near identity line) ; Q2 cos vs delta Spearman rho =", round(rho_all, 4), "===")
print("\n=== per-(source,ckpt) stats over the 1000 batches ===")
show = stat[["source", "ckpt", "n", "pos_frac", "signed_mean_gd", "abs_mean_gd",
             "cancel_ratio", "sign_agree_gd_delta", "spearman_cos_delta", "verdict"]]
print(show.to_string(index=False))
print("\n=== conditioning ===")
print(cond.to_string(index=False))
print(f"\n[ok] stats -> {OUTCSV/'stats.csv'} ; conditioning -> {OUTCSV/'conditioning.csv'}")
