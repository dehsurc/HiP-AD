"""Q4 analysis: is the cross-task FUNCTIONAL coupling (same-sample) significantly
above the shuffled cross-sample null, and above what the loss gradient sees?

Per (ckpt, aux):
  cosfunc_same  : same-sample cross-task functional alignment (cancellation-invariant ||K|| proxy)
  cosshuf       : cross-sample null (shared inter_gnn subspace baseline)
  |cos_loss|    : what the loss gradient actually sees (the cancelling contraction)
  floor         : random-param dimensional floor
Paired test (same - shuf > 0) via bootstrap CI + Wilcoxon.
READ: same >> shuf (paired sig) AND shuf ~ |cos_loss| => functional coupling is
real & sample-specific, but the loss gradient only probes the generic shared
subspace => "coupling exists functionally, gradient cancellation hides it".
"""
import argparse, glob, os
import numpy as np, pandas as pd
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def boot_ci(d, n=4000, seed=1):
    d = np.asarray(d, float); d = d[np.isfinite(d)]
    if len(d) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="gradient_analysis_results/ntk_*.csv")
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(a.glob)) if "smoke" not in f]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} samples from {[os.path.basename(f) for f in files]}")
    order = {"1ep": 1, "3ep": 3, "9ep": 9, "18ep": 18}
    for ck in sorted(df.ckpt.unique(), key=lambda c: order.get(c, 99)):
        d = df[df.ckpt == ck]
        print(f"\n{'='*86}\nCKPT {ck}  n={len(d)}  floor={d.cos_paramfloor.abs().mean():.4f}\n{'='*86}")
        print(f"{'aux':7} {'cosfunc_same':>13} {'cosshuf':>9} {'|cos_loss|':>11} "
              f"{'same-shuf (95%CI)':>26} {'ratio s/sh':>10} {'wilcoxon p':>11}")
        for aux in ["det", "map", "motion"]:
            same = d[f"cosfunc_absmean_{aux}"]
            shuf = d.get(f"cosshuf_absmean_{aux}")
            loss = d[f"cos_loss_{aux}"].abs()
            if shuf is None:
                continue
            paired = pd.concat([same, shuf], axis=1).dropna()
            dd = (paired.iloc[:, 0] - paired.iloc[:, 1]).values
            m, lo, hi = boot_ci(dd)
            ratio = same.mean() / shuf.mean() if shuf.mean() > 0 else np.nan
            try:
                p = wilcoxon(dd, alternative="greater").pvalue if (wilcoxon and len(dd) > 10) else np.nan
            except Exception:
                p = np.nan
            sig = "***" if (p == p and p < 0.001) else "**" if (p == p and p < 0.01) else "*" if (p == p and p < 0.05) else ""
            print(f"{aux:7} {same.mean():13.4f} {shuf.mean():9.4f} {loss.mean():11.4f} "
                  f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] {ratio:9.2f}x {str(round(p,4)) if p==p else 'na':>10}{sig}")
    print("\nREAD: cosfunc_same significantly > cosshuf (CI excludes 0, p<0.05) = real sample-specific")
    print("      functional coupling beyond the shared subspace; and cosshuf ~ |cos_loss| means the")
    print("      loss gradient only saw the generic part => functional coupling hidden by cancellation.")


if __name__ == "__main__":
    main()
