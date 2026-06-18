"""cos(=analytic grad_dot) vs delta(=finite-difference ΔL_plan): same 1st-order coupling,
two ways. delta sits near the float32 floor (noisier) but agrees on robust sign patterns.

Data: inter_gnn_b1000_pcgrad/ckpt_{1,3,9,18}ep/probe/probe_per_batch.csv
Filter: target_task=='plan', variant=='raw', steps==1, source in {det,map,motion}.
1st-order identity:  delta ≈ -step_size * grad_dot  (step_size=0.001, gradient descent on g_source).
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/yongjae/e2e/HiP-AD-pcgrad")
RES = ROOT / "gradient_analysis_results/inter_gnn_b1000_pcgrad"
OUT = ROOT / "docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/figures"
OUT.mkdir(parents=True, exist_ok=True)
CKPTS = ["1ep", "3ep", "9ep", "18ep"]
SRC = ["det", "map", "motion"]
SCOL = {"det": "#1f77b4", "map": "#d62728", "motion": "#2ca02c"}

frames = []
for ck in CKPTS:
    d = pd.read_csv(RES / f"ckpt_{ck}/probe/probe_per_batch.csv")
    d = d[(d.target_task == "plan") & (d.variant == "raw") & (d.steps == 1) & (d.source_task.isin(SRC))].copy()
    d["ckpt"] = ck
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
STEP = float(df.step_size.iloc[0])           # 0.001
ULP = 1.2e-7                                  # float32 ULP near O(1) plan loss
print(f"pooled rows={len(df)}  step_size={STEP}")

fig, ax = plt.subplots(1, 3, figsize=(19, 5.6))

# ---- Panel A: per-batch scatter grad_dot vs delta (the same coupling) ----
a = ax[0]
for s in SRC:
    g = df[df.source_task == s]
    a.scatter(g.grad_dot, g.delta, s=4, alpha=0.25, color=SCOL[s], label=s, rasterized=True)
xs = np.linspace(df.grad_dot.min(), df.grad_dot.max(), 100)
a.plot(xs, -STEP * xs, "k--", lw=1.8, label=f"1st-order: δ = −{STEP}·grad_dot")
r, _ = pearsonr(df.grad_dot, df.delta)
a.set_xlabel("grad_dot  =  g_aux · g_plan   (analytic, floor-free)")
a.set_ylabel("delta  =  ΔL_plan after 1 step  (finite difference)")
a.set_title(f"A. Same first-order coupling, two ways\nPearson r = {r:.3f}  (δ ≈ −step·grad_dot)", fontsize=12)
a.legend(fontsize=8.5, loc="upper right", markerscale=2)
a.axhline(0, color="0.5", lw=0.6); a.axvline(0, color="0.5", lw=0.6)

# ---- Panel B: delta is floor-limited / quantized ----
b = ax[1]
absd = df.delta.abs().replace(0, np.nan).dropna()
b.hist(absd, bins=np.logspace(-7.5, -2.5, 60), color="#888", edgecolor="k", lw=0.3)
b.set_xscale("log")
med = absd.median()
b.axvline(ULP, color="red", lw=2, ls="--", label=f"float32 ULP ≈ {ULP:.1e}")
b.axvline(med, color="blue", lw=2, label=f"median |δ| = {med:.1e}  (×{med/ULP:.0f} floor)")
b.set_xlabel("|delta|  (log scale)")
b.set_ylabel("# of per-batch measurements")
b.set_title("B. delta is near the float32 FLOOR → noisy\n(grad_dot is the clean, floor-free version)", fontsize=12)
b.legend(fontsize=9, loc="upper left")

# ---- Panel C: verdict agreement across (source × ckpt) ----
c = ax[2]
med_tab = df.groupby(["source_task", "ckpt"]).agg(gd=("grad_dot", "median"), dl=("delta", "median")).reset_index()
for s in SRC:
    g = med_tab[med_tab.source_task == s]
    c.scatter(g.gd, g.dl, s=80, color=SCOL[s], edgecolor="k", zorder=3, label=s)
xs = np.linspace(med_tab.gd.min() * 1.2, med_tab.gd.max() * 1.2, 50)
c.plot(xs, -STEP * xs, "k--", lw=1.5, label=f"δ = −{STEP}·grad_dot")
# annotate the robust det-1ep cell
det1 = med_tab[(med_tab.source_task == "det") & (med_tab.ckpt == "1ep")].iloc[0]
c.annotate("det @1ep\ngrad_dot<0 (OPPOSED)\nδ>0 (HURTS) — agree",
           xy=(det1.gd, det1.dl), xytext=(det1.gd * 0.2, det1.dl * 1.4),
           fontsize=8.5, color="#1f77b4",
           arrowprops=dict(arrowstyle="->", color="#1f77b4"))
rc, _ = pearsonr(med_tab.gd, med_tab.dl)
c.axhline(0, color="0.5", lw=0.6); c.axvline(0, color="0.5", lw=0.6)
c.set_xlabel("median grad_dot  (per source×ckpt)")
c.set_ylabel("median delta")
c.set_title(f"C. Same VERDICT across 12 cells\nr = {rc:.3f}: both flag det@1ep OPPOSED/HURTS", fontsize=12)
c.legend(fontsize=8.5, loc="upper right")

fig.suptitle("grad_dot (cos) and delta measure the SAME first-order aux→plan coupling — delta just noisier (floor-limited)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / "grad_dot_vs_delta.png", dpi=130)
print("wrote", OUT / "grad_dot_vs_delta.png")
print("Panel A Pearson r =", round(r, 3), "| Panel C cell-median r =", round(rc, 3))
print("median|delta| =", f"{med:.2e}", "= x", round(med / ULP), "floor")
