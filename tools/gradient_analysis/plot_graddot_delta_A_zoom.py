"""Panel A of graddot_delta_intuitive, zoomed: grad_dot in [-2,2], improvement in
[-0.005, 0.005], outliers removed, 20 grad_dot quantile bins recomputed on the window."""
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

print("step_size unique values:", sorted(df.step_size.unique()))
print("step_size per source:", df.groupby("source_task").step_size.unique().to_dict())
STEP = float(df.step_size.iloc[0])

gd = df.grad_dot.values
imp = -df.delta.values
XL, YL = (-2.0, 2.0), (-0.005, 0.005)
keep = (gd >= XL[0]) & (gd <= XL[1]) & (imp >= YL[0]) & (imp <= YL[1])
print(f"kept {keep.sum()}/{len(gd)} ({100*keep.mean():.1f}%) after removing outliers outside x{XL} y{YL}")
gd, imp = gd[keep], imp[keep]

fig, a = plt.subplots(figsize=(7.8, 6.2))
q = np.quantile(gd, np.linspace(0, 1, 21)); q[-1] += 1e-12
idx = np.clip(np.digitize(gd, q[1:-1]), 0, 19)
cx, cy, ce = [], [], []
for k in range(20):
    m = idx == k
    if m.sum() < 5:
        continue
    cx.append(gd[m].mean()); cy.append(imp[m].mean()); ce.append(imp[m].std() / np.sqrt(m.sum()))
cx, cy, ce = map(np.array, (cx, cy, ce))

a.scatter(gd, imp, s=4, alpha=0.08, color="0.6", rasterized=True, zorder=1)
a.errorbar(cx, cy, yerr=ce, fmt="o-", color="#d62728", lw=2.4, ms=6, capsize=3, zorder=3,
           label="mean per grad_dot bin (+/-SE)")
xs = np.linspace(*XL, 100)
a.plot(xs, STEP * xs, "k--", lw=1.3, alpha=0.7, zorder=2, label=f"1st-order: imp = {STEP:g} x grad_dot")
a.axhline(0, color="k", lw=0.7); a.axvline(0, color="k", lw=0.7)
a.set_xlim(XL); a.set_ylim(YL)
a.set_xlabel("grad_dot = g_aux . g_plan   (>0 = aligned)", fontsize=11)
a.set_ylabel("plan improvement = -delta_plan_loss   (>0 = plan loss down)", fontsize=11)
a.set_title("A (zoom). Higher grad_dot -> larger 1-step plan improvement\n"
            "20 grad_dot quantile bins; outliers removed (x in [-2,2], y in [-0.005,0.005])", fontsize=11)
a.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "graddot_delta_A_zoom.png", dpi=140)
print("wrote", OUT / "graddot_delta_A_zoom.png")
