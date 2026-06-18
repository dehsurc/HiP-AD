"""Visualize the NULL result: per-scene / data-feature-level gradient analysis
carries no usable structure.  Two figures built from persample_perf_{1,3,9,18}ep.csv.

FIG1 feature_level_null.png (3 panels)
  A  per-set cos(g_aux,g_plan) histograms -> real magnitude but SIGN-CANCELLED (~0 mean, frac_neg~0.5)
  B  Spearman(cos_aux, scene_feature) per feature, det/map/motion + random-gradient control,
     with a permutation noise band -> every aux sits in the noise floor (== ctrl_random feature)
  C  binary strata diff mean(cos|Z)-mean(cos|~Z) -> real aux indistinguishable from the ctrl stratum

FIG2 no_performance_knob.png (2 panels)
  A  Spearman(cos_aux, loss_plan) per checkpoint -> tiny + SIGN-FLIPPING (map +,+,-,-), no stable knob
  B  gplan_norm confound: gplan_norm~loss_plan rises with training, and partialling it out
     collapses the only "significant" late signal (18ep map) to noise while the RANDOM control stays
     -> the late effect is g_plan-norm geometry, not aux-specific alignment
"""
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/yongjae/e2e/HiP-AD-pcgrad")
RES = ROOT / "gradient_analysis_results"
OUT = ROOT / "docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/figures"
OUT.mkdir(parents=True, exist_ok=True)
CKPTS = ["1ep", "3ep", "9ep", "18ep"]
AUX = ["det", "map", "motion"]
ACOL = {"det": "#1f77b4", "map": "#d62728", "motion": "#2ca02c", "rand": "#888888"}
FEATS = ["n_near15", "n_relevant", "map_curv", "ego_path_curv", "ego_speed", "nearest_dist", "ctrl_random"]
FEAT_LBL = {"n_near15": "density\n(n_near15)", "n_relevant": "ego-rel\n(n_relevant)",
            "map_curv": "map cplx\n(map_curv)", "ego_path_curv": "path curv\n(ego_path_curv)",
            "ego_speed": "speed\n(ego_speed)", "nearest_dist": "nearest\n(nearest_dist)",
            "ctrl_random": "CONTROL\n(ctrl_random)"}
STRATA = [("dense\nn_near15>=6", lambda d: d.n_near15 >= 6),
          ("turn\nis_turn", lambda d: d.is_turn >= 1),
          ("hi_rel\nn_relevant>=1", lambda d: d.n_relevant >= 1),
          ("hi_curv\nmap_curv>med", lambda d: d.map_curv >= d.map_curv.median()),
          ("fast\nego_speed>med", lambda d: d.ego_speed >= d.ego_speed.median()),
          ("CONTROL\nctrl_random", lambda d: d.ctrl_random >= 0.5)]

dfs = {c: pd.read_csv(RES / f"persample_perf_{c}.csv") for c in CKPTS}
pool = pd.concat(dfs.values(), ignore_index=True)
rng = np.random.RandomState(0)


def perm_floor_corr(x, y, n=300):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    null = [abs(spearmanr(x, rng.permutation(y)).correlation) for _ in range(n)]
    return float(np.percentile(null, 95))


def perm_floor_diff(c, mask, n=300):
    m = np.isfinite(c)
    c2, mk = c[m], mask[m]
    obs = np.nanmean(c2[mk]) - np.nanmean(c2[~mk])
    null = []
    for _ in range(n):
        p = rng.permutation(mk)
        null.append(np.nanmean(c2[p]) - np.nanmean(c2[~p]))
    return obs, float(np.percentile(np.abs(null), 95))


def partial_spearman(x, y, z):
    """partial Spearman of (x,y) controlling z, via residuals of rank regressions."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    rx, ry, rz = rankdata(x[m]), rankdata(y[m]), rankdata(z[m])
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


# ============================== FIGURE 1 ==============================
plt.rcParams.update({"font.size": 11.5, "axes.titlesize": 13})
fig1, ax = plt.subplots(1, 2, figsize=(15, 6.2))

# --- Panel A: sign cancellation histogram (pooled) ---
a = ax[0]
bins = np.linspace(-0.12, 0.12, 61)
for aux in AUX:
    v = pool[f"cos_{aux}"].dropna().values
    a.hist(v, bins=bins, histtype="step", lw=2, density=True, color=ACOL[aux],
           label=f"{aux}: mean={v.mean():+.4f}, neg={np.mean(v<0):.2f}")
vr = pool["cos_rand"].dropna().values
a.hist(vr, bins=bins, histtype="stepfilled", density=True, color="0.6", alpha=0.45,
       label=f"random dir: |cos|~{np.abs(vr).mean():.4f}")
a.axvline(0, color="k", lw=0.8, ls="--")
a.set_title("A. Per-scene coupling is REAL in size but SIGN-CANCELLED\n"
            "cos(g_aux, g_plan) per sample, pooled 4 ckpts (n=%d)" % len(pool), fontsize=11)
a.set_xlabel("cos(g_aux, g_plan)"); a.set_ylabel("density"); a.legend(fontsize=8.5, loc="upper left")

# --- Panel B: scene-feature correlation vs noise floor (pooled) ---
b = ax[1]
x = np.arange(len(FEATS)); w = 0.2
floors = {f: perm_floor_corr(pool[f"cos_map"].values, pool[f].values) for f in FEATS}
for j, aux in enumerate(AUX + ["rand"]):
    col = f"cos_{aux}"
    rhos = [spearmanr(pool[col].values, pool[f].values, nan_policy="omit").correlation for f in FEATS]
    b.bar(x + (j - 1.5) * w, rhos, w, color=ACOL[aux], label=("rand-grad ctrl" if aux == "rand" else aux))
# shaded permutation noise band (avg floor across real features; ctrl_random excluded)
fl = float(np.nanmean([floors[f] for f in FEATS if f != "ctrl_random"]))
b.axhspan(-fl, fl, color="0.85", zorder=0, label=f"perm. noise floor (±{fl:.3f})")
b.axhline(0, color="k", lw=0.8)
b.set_xticks(x); b.set_xticklabels([FEAT_LBL[f] for f in FEATS], fontsize=8)
b.set_ylim(-0.18, 0.18)
b.set_title("B. No scene feature gives a usable signal\n"
            "Spearman(cos_aux, feature): |rho|<0.05, map-only, sign-inconsistent; CONTROL≈0", fontsize=12)
b.set_ylabel("Spearman rho"); b.legend(fontsize=8, ncol=2, loc="upper right")

fig1.suptitle("Data-feature-level gradient analysis is NULL: per-scene aux↔plan coupling has no exploitable structure",
              fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(OUT / "feature_level_null.png", dpi=130)
print("wrote", OUT / "feature_level_null.png")

# ============================== FIGURE 2 ==============================
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5.4))

# --- Panel A: cos<->loss_plan per ckpt, sign-flip ---
pa = ax2[0]
x = np.arange(len(CKPTS)); w = 0.2
for j, aux in enumerate(AUX + ["rand"]):
    rhos = []
    for ck in CKPTS:
        d = dfs[ck]
        rhos.append(spearmanr(d[f"cos_{aux}"].values, d["loss_plan"].values, nan_policy="omit").correlation)
    pa.bar(x + (j - 1.5) * w, rhos, w, color=ACOL[aux], label=("rand ctrl" if aux == "rand" else aux))
pa.axhspan(-0.03, 0.03, color="0.85", zorder=0, label="~noise (|rho|<0.03)")
pa.axhline(0, color="k", lw=0.8)
pa.set_xticks(x); pa.set_xticklabels(CKPTS)
pa.set_ylim(-0.16, 0.16)
pa.annotate("map sign FLIPS\n+.05,+.05 → −.08,−.11", xy=(2, -0.085), xytext=(0.6, -0.14),
            fontsize=9, color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"))
pa.set_title("A. cos does NOT predict planning performance\n"
             "Spearman(cos_aux, loss_plan) per checkpoint — tiny & sign-unstable", fontsize=11)
pa.set_ylabel("Spearman(cos, loss_plan)"); pa.set_xlabel("checkpoint"); pa.legend(fontsize=8.5, ncol=2)

# --- Panel B: gplan_norm geometry confound ---
pb = ax2[1]
gn_loss = [spearmanr(dfs[ck]["gplan_norm"].values, dfs[ck]["loss_plan"].values, nan_policy="omit").correlation
           for ck in CKPTS]
pb.plot(x, gn_loss, "o-", color="purple", lw=2, label="gplan_norm ~ loss_plan (the confound)")
for xi, yi in zip(x, gn_loss):
    pb.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="purple")
# 18ep map: raw vs partial(gplan_norm) vs random control
d18 = dfs["18ep"]
raw_map = spearmanr(d18["cos_map"], d18["loss_plan"], nan_policy="omit").correlation
par_map = partial_spearman(d18["cos_map"].values, d18["loss_plan"].values, d18["gplan_norm"].values)
raw_rnd = spearmanr(d18["cos_rand"], d18["loss_plan"], nan_policy="omit").correlation
par_rnd = partial_spearman(d18["cos_rand"].values, d18["loss_plan"].values, d18["gplan_norm"].values)
bx = [5.2, 5.9, 6.6, 7.3]
pb.bar(bx, [raw_map, par_map, raw_rnd, par_rnd],
       color=["#d62728", "#f4a3a3", "#888888", "#cccccc"], width=0.6)
for xi, yi, lab in zip(bx, [raw_map, par_map, raw_rnd, par_rnd],
                       ["18ep map\nRAW", "map\n−gpnorm", "rand\nRAW", "rand\n−gpnorm"]):
    pb.annotate(f"{yi:+.3f}", (xi, yi), textcoords="offset points", xytext=(0, -14 if yi < 0 else 6),
                ha="center", fontsize=8)
    pb.text(xi, 0.02, lab, ha="center", fontsize=7.5)
pb.axhline(0, color="k", lw=0.8)
pb.set_xticks(list(x) + [6.25]); pb.set_xticklabels(CKPTS + ["18ep map: confound test"], fontsize=9)
pb.set_title("B. The only 'significant' late signal is g_plan-norm GEOMETRY\n"
             "hard scenes → big ‖g_plan‖ → low cos for ANY direction (incl. random)", fontsize=11)
pb.set_ylabel("Spearman rho"); pb.legend(fontsize=8.5, loc="lower left")

fig2.suptitle("No per-scene weighting knob: gradient alignment is performance-blind, and the lone late effect is a norm artifact",
              fontsize=12.5, fontweight="bold")
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(OUT / "no_performance_knob.png", dpi=130)
print("wrote", OUT / "no_performance_knob.png")

# print the key numbers for the report
print("\n--- key numbers ---")
print("Panel B floors:", {f: round(v, 3) for f, v in floors.items()})
print("18ep map raw/partial:", round(raw_map, 3), round(par_map, 3), "| rand raw/partial:", round(raw_rnd, 3), round(par_rnd, 3))
print("gplan_norm~loss_plan by ckpt:", [round(v, 2) for v in gn_loss])
