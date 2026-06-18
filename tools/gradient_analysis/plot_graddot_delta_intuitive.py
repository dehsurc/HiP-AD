"""Intuitive: per-batch 1-step probe, cos(grad_dot) and delta MOVE TOGETHER."""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/yongjae/e2e/HiP-AD-pcgrad")
RES = ROOT / "gradient_analysis_results/inter_gnn_b1000_pcgrad"
OUT = ROOT / "docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/figures"
SRC = ["det", "map", "motion"]
fr = []
for ck in ["1ep", "3ep", "9ep", "18ep"]:
    d = pd.read_csv(RES / f"ckpt_{ck}/probe/probe_per_batch.csv")
    d = d[(d.target_task == "plan") & (d.variant == "raw") & (d.steps == 1) & (d.source_task.isin(SRC))]
    fr.append(d)
df = pd.concat(fr, ignore_index=True)
gd = df.grad_dot.values
imp = -df.delta.values

fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))

a = ax[0]
q = np.quantile(gd, np.linspace(0, 1, 21)); q[-1] += 1e-12
idx = np.clip(np.digitize(gd, q[1:-1]), 0, 19)
cx, cy, ce = [], [], []
for k in range(20):
    m = idx == k
    if m.sum() < 5:
        continue
    cx.append(gd[m].mean()); cy.append(imp[m].mean()); ce.append(imp[m].std() / np.sqrt(m.sum()))
cx, cy, ce = map(np.array, (cx, cy, ce))
a.scatter(gd, imp, s=3, alpha=0.06, color="0.6", rasterized=True, zorder=1)
a.errorbar(cx, cy, yerr=ce, fmt="o-", color="#d62728", lw=2.4, ms=6, capsize=3, zorder=3,
           label="mean per grad_dot bin (+/-SE)")
a.axhline(0, color="k", lw=0.7); a.axvline(0, color="k", lw=0.7)
a.set_xlabel("grad_dot = g_aux . g_plan   (cos; >0 = aligned)", fontsize=11)
a.set_ylabel("plan improvement = -delta_plan_loss   (>0 = plan loss down)", fontsize=11)
a.set_title("A. Higher cos -> larger 1-step plan improvement\n"
            "(binned into 20 grad_dot quantiles -> clean monotone)", fontsize=12)
a.legend(fontsize=10, loc="upper left")

b = ax[1]
aligned = gd > 0
improves = imp > 0
M = np.array([[np.sum(aligned & improves), np.sum(aligned & ~improves)],
              [np.sum(~aligned & improves), np.sum(~aligned & ~improves)]], float)
Mp = M / M.sum() * 100
agree = (Mp[0, 0] + Mp[1, 1])
b.imshow(Mp, cmap="Blues", vmin=0, vmax=Mp.max())
for i in range(2):
    for j in range(2):
        b.text(j, i, f"{Mp[i,j]:.1f}%\n({int(M[i,j])})", ha="center", va="center",
               fontsize=15, fontweight="bold" if i == j else "normal",
               color="white" if Mp[i, j] > Mp.max() * 0.6 else "black")
b.set_xticks([0, 1]); b.set_xticklabels(["plan improves\n(delta<0)", "plan worsens\n(delta>0)"], fontsize=10)
b.set_yticks([0, 1]); b.set_yticklabels(["aligned\n(cos>0)", "opposed\n(cos<0)"], fontsize=10)
b.set_title(f"B. Signs agree {agree:.0f}% of the time\n(diagonal = cos and delta give same verdict)", fontsize=12)
for i in range(2):
    b.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#d62728", lw=3))

fig.suptitle("per-batch 1-step probe: cos(grad_dot) and delta move together (same 1st-order coupling)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "graddot_delta_intuitive.png", dpi=135)
print("wrote", OUT / "graddot_delta_intuitive.png", "| agree=%.1f%%" % agree)
