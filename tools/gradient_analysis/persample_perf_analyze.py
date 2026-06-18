"""Analyze per-set aux<->plan cos vs per-set planning performance (loss_plan).

Two axes (the user's question: don't average -> look at the distribution + the
cos<->performance relationship):

(A) CANCELLATION: per-set cos_aux distribution.
    mean(cos)  ~ the value that got averaged to ~0 (sign cancellation)
    mean|cos|  ~ the real per-set magnitude that survives
    frac_neg   ~ how sign-balanced (≈0.5 => cancels)
    ratio mean|cos| / mean|cos_rand|  => per-set coupling vs random-direction floor

(B) cos <-> PERFORMANCE: does per-set alignment relate to planning quality?
    Spearman corr(cos_aux, loss_plan), with corr(cos_rand, loss_plan) as the
    confound control (a large-residual set has a specific g_plan direction
    regardless of aux; cos_rand absorbs that).  Also stratify loss_plan into
    quartiles and report mean cos_aux + frac(cos>0) per quartile -- i.e. on the
    HARD-to-plan sets (high loss), is aux gradient ALIGNED (cos>0, up-weighting
    would help) or CONFLICTING (cos<0)?

READ: corr_aux >> corr_rand (and a monotone quartile trend) => per-set alignment
carries a planning-performance signal => a per-set/conditional weighting knob
exists.  corr_aux ~ corr_rand ~ 0 => even per-set, the sign is performance-blind.
"""
import argparse, glob, os
import numpy as np, pandas as pd
try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

AUX = ["det", "map", "motion"]


def boot_ci(x, y, fn, n=2000, seed=1):
    rng = np.random.RandomState(seed)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 20:
        return np.nan, np.nan, np.nan
    base = fn(x, y)
    vals = [fn(x[idx], y[idx]) for idx in (rng.randint(0, len(x), len(x)) for _ in range(n))]
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sp(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20 or spearmanr is None:
        return np.nan
    return spearmanr(x[m], y[m]).correlation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="gradient_analysis_results/persample_perf_*.csv")
    a = ap.parse_args()
    files = sorted(f for f in glob.glob(a.glob) if "_fp16" not in f and "test" not in f)
    for f in files:
        d = pd.read_csv(f)
        ck = d["ckpt"].iloc[0] if "ckpt" in d and len(d) else os.path.basename(f)
        print(f"\n{'='*90}\n{os.path.basename(f)}  ckpt={ck}  n={len(d)}\n{'='*90}")

        # (A) cancellation
        print("(A) per-set cos distribution (cancellation):")
        print(f"  {'aux':7} {'mean(cos)':>11} {'mean|cos|':>11} {'frac_neg':>9} {'|cos|/|rand|':>13}")
        randabs = d["cos_rand"].abs().mean() if "cos_rand" in d else np.nan
        for aux in AUX:
            c = d[f"cos_{aux}"].dropna()
            if not len(c):
                continue
            ratio = c.abs().mean() / randabs if randabs and np.isfinite(randabs) else np.nan
            print(f"  {aux:7} {c.mean():+11.4f} {c.abs().mean():11.4f} {(c<0).mean():9.3f} {ratio:12.2f}x")
        print(f"  (cos_rand: mean={d['cos_rand'].mean():+.4f} mean|.|={randabs:.4f})" if "cos_rand" in d else "")

        # (B) cos <-> performance
        if "loss_plan" not in d:
            print("(B) no loss_plan column -> skip"); continue
        lp = d["loss_plan"].values
        print(f"\n(B) cos <-> loss_plan (Spearman; loss_plan mean={np.nanmean(lp):.3f} "
              f"sd={np.nanstd(lp):.3f}):")
        print(f"  {'aux':7} {'corr(cos,loss)':>15} {'95% CI':>22}")
        for aux in AUX + ["rand"]:
            col = f"cos_{aux}"
            if col not in d:
                continue
            r, lo, hi = boot_ci(d[col].values, lp, sp)
            tag = "  <-- control" if aux == "rand" else ""
            print(f"  {aux:7} {r:+15.4f}   [{lo:+.3f},{hi:+.3f}]{tag}")

        # quartile stratification by planning difficulty
        try:
            d = d.assign(_q=pd.qcut(d["loss_plan"], 4, labels=["Q1_easy","Q2","Q3","Q4_hard"]))
        except Exception:
            continue
        print("\n  loss_plan quartile -> mean cos_aux  /  frac(cos>0):")
        print(f"  {'quartile':10} " + " ".join(f"{aux:>16}" for aux in AUX))
        for q in ["Q1_easy","Q2","Q3","Q4_hard"]:
            sub = d[d._q == q]
            cells = []
            for aux in AUX:
                c = sub[f"cos_{aux}"].dropna()
                cells.append(f"{c.mean():+.3f}/{(c>0).mean():.2f}" if len(c) else "   na")
            print(f"  {q:10} " + " ".join(f"{x:>16}" for x in cells))
    print("\nREAD: (B) corr_aux CI excluding 0 AND |corr_aux|>|corr_rand|, plus a monotone")
    print("      quartile trend => per-set alignment predicts planning quality (weighting knob).")


if __name__ == "__main__":
    main()
