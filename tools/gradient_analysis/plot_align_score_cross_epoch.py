#!/usr/bin/env python
"""ALIGN_SCORE cross-epoch -- the gradient-alignment mirror of tg_l2_cross_epoch.png.

TG measures the EFFECT on plan L2 of removing each aux task (retraining).
This measures the gradient ALIGNMENT of each aux task with plan, on the SAME
checkpoints, so the two figures can be placed side by side.

Left panel  (mirror of TG_L2 bars): net signed alignment = mean per-batch
  <g_aux, g_plan>.  Sign convention matches TG: >0 helps plan (grad_dot>0 ->
  delta_plan<0 -> plan loss down), <0 hurts.
Right panel (the cancellation context, mirror of TG drift): per-batch
  |<g_aux,g_plan>| mean -- the magnitude that SURVIVES per sample but cancels
  in the signed mean on the left.

Source: gradient_analysis_results/inter_gnn_b1000_pcgrad/align_delta_perbatch/stats.csv
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATS = "/home/yongjae/e2e/HiP-AD-pcgrad/gradient_analysis_results/inter_gnn_b1000_pcgrad/align_delta_perbatch/stats.csv"
OUT = "/home/yongjae/e2e/HiP-AD-pcgrad/docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/figures/align_score_cross_epoch.png"

# match TG checkpoints (1/3/9ep) for direct side-by-side comparison
CKPTS = ["1ep", "3ep", "9ep"]
AUX = ["det", "map", "motion"]
COL = {"det": "#1f77b4", "map": "#ff7f0e", "motion": "#2ca02c"}  # tab:blue/orange/green == TG

rows = list(csv.DictReader(open(STATS)))


def get(src, ck, col):
    for r in rows:
        if r["source"] == src and r["ckpt"] == ck:
            return float(r[col])
    return float("nan")


fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.5))
x = np.arange(len(CKPTS))
w = 0.25

# ---- LEFT: net signed alignment (analog of TG_L2) ----
for i, s in enumerate(AUX):
    vals = [get(s, ck, "signed_mean_gd") for ck in CKPTS]
    bars = axL.bar(x + (i - 1) * w, vals, w, color=COL[s], label=s, edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        axL.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
axL.axhline(0, color="k", lw=1)
axL.set(xticks=x, xticklabels=CKPTS, xlabel="Checkpoint",
        ylabel="net align = mean_batch <g_aux , g_plan>  (signed)")
axL.set_title("Net aux->plan ALIGNMENT (signed): epoch x task\n(>0 helps . <0 hurts)  -- gradient mirror of TG_L2")
axL.legend(title="Aux Task")

# ---- RIGHT: per-sample magnitude (the cancellation context) ----
for i, s in enumerate(AUX):
    vals = [get(s, ck, "abs_mean_gd") for ck in CKPTS]
    bars = axR.bar(x + (i - 1) * w, vals, w, color=COL[s], label=s, edgecolor="k", linewidth=0.4)
    for b, v in zip(bars, vals):
        axR.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), ha="center", va="bottom", fontsize=8)
axR.set(xticks=x, xticklabels=CKPTS, xlabel="Checkpoint",
        ylabel="per-batch |<g_aux , g_plan>|  (mean |.|)")
axR.set_title("Per-sample alignment MAGNITUDE\n(real signal -- left panel's signed mean cancels it to ~0)")
axR.legend(title="Aux Task")

fig.suptitle("ALIGN_SCORE cross-epoch (mirror of TG): net align ~0 & sign-flips like TG_L2, "
             "but |align| is 5-30x larger  =>  REAL-BUT-CANCELLING",
             fontweight="bold", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=120)
print("[ok] wrote", OUT)

# ---- printed comparison table vs TG_L2 ----
TG_L2 = {  # from tg_l2_cross_epoch.png
    ("det", "1ep"): +0.026, ("map", "1ep"): -0.085, ("motion", "1ep"): -0.051,
    ("det", "3ep"): +0.004, ("map", "3ep"): -0.028, ("motion", "3ep"): -0.008,
    ("det", "9ep"): -0.005, ("map", "9ep"): -0.033, ("motion", "9ep"): -0.013,
}
print("\nckpt  task   net_align   |align|    mean_cos   pos_frac  | TG_L2   sign_match")
for ck in CKPTS:
    for s in AUX:
        na = get(s, ck, "signed_mean_gd"); ab = get(s, ck, "abs_mean_gd")
        mc = get(s, ck, "mean_cos"); pf = get(s, ck, "pos_frac"); tg = TG_L2[(s, ck)]
        match = "same" if (na > 0) == (tg > 0) else "OPPOSITE"
        print(f"{ck:5s} {s:7s} {na:+.5f}  {ab:.4f}   {mc:+.5f}   {pf:.3f}    | {tg:+.3f}   {match}")
