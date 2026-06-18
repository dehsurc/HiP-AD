"""Does conditioning on a scene feature UN-CANCEL the per-sample aux<->plan coupling?

Reads persample_*.csv.  For each (ckpt, aux):
  CANCEL  : signed_mean vs mean_abs of per-sample cos (confirm bs=1 cancellation).
  COND    : for each scene stratum Z, mean(cos | Z) vs mean(cos | ~Z) -> diff + perm p.
            A significant, sign-meaningful diff (NOT seen for the random control) =>
            the sign is STRUCTURED by Z => Z is a gradient-level weighting knob.
  CORR    : Spearman(cos, feature) + permutation floor.
The random-direction control (cos_rand) must show NO conditional structure (lie detector).
"""
import argparse, glob, os
import numpy as np, pandas as pd

AUX = ["det", "map", "motion", "rand"]          # rand = lie-detector control
# (feature, predicate) binary strata
STRATA = [
    ("dense",   lambda d: d.n_near15 >= 6),
    ("turn",    lambda d: d.is_turn >= 1),
    ("hi_rel",  lambda d: d.n_relevant >= 1),
    ("hi_curv", lambda d: d.map_curv >= d.map_curv.median()),
    ("fast",    lambda d: d.ego_speed >= d.ego_speed.median()),
    ("ctrl",    lambda d: d.ctrl_random >= 0.5),   # nonsense control stratum
]
CORR_FEATS = ["n_near15", "n_relevant", "map_curv", "ego_path_curv", "ego_speed", "nearest_dist", "ctrl_random"]


def perm_diff_p(vals, mask, n=2000, seed=1):
    """two-group mean diff vs permutation null of the grouping."""
    vals = np.asarray(vals, float); mask = np.asarray(mask, bool)
    ok = np.isfinite(vals)
    vals, mask = vals[ok], mask[ok]
    if mask.sum() < 10 or (~mask).sum() < 10:
        return np.nan, np.nan
    obs = vals[mask].mean() - vals[~mask].mean()
    rng = np.random.RandomState(seed)
    null = np.empty(n)
    for k in range(n):
        m = rng.permutation(mask)
        null[k] = vals[m].mean() - vals[~m].mean()
    p = (np.abs(null) >= abs(obs)).mean()
    return obs, p


def spearman_perm(x, y, n=2000, seed=2):
    x = pd.Series(x, dtype=float); y = pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    if len(x) < 30:
        return np.nan, np.nan
    rho = x.corr(y, method="spearman")
    rng = np.random.RandomState(seed)
    null = np.array([x.corr(pd.Series(rng.permutation(y.values)), method="spearman") for _ in range(n)])
    return rho, float(np.nanpercentile(np.abs(null), 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="gradient_analysis_results/persample_*.csv")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} samples from {[os.path.basename(f) for f in files]}")
    order = {"1ep": 1, "3ep": 3, "9ep": 9, "18ep": 18}
    for ck in sorted(df.ckpt.unique(), key=lambda c: order.get(c, 99)):
        d = df[df.ckpt == ck].copy()
        nnan = d.cos_det.isna().mean()
        print(f"\n{'='*92}\nCKPT {ck}   n={len(d)}  (NaN-plan frac={nnan:.2f})\n{'='*92}")
        for aux in AUX:
            cc = f"cos_{aux}"
            if cc not in d:
                continue
            v = d[cc].dropna()
            sm, ma = v.mean(), v.abs().mean()
            ratio = abs(sm) / ma if ma > 0 else np.nan
            fn = (v < 0).mean()
            print(f"\n-- aux={aux:6} | CANCEL: signed_mean={sm:+.4f} mean_abs={ma:.4f} "
                  f"ratio={ratio:.3f} frac_neg={fn:.2f}")
            # conditional strata
            for name, pred in STRATA:
                try:
                    mask = pred(d).values
                except Exception:
                    continue
                obs, p = perm_diff_p(d[cc].values, mask)
                if obs != obs:
                    continue
                star = "***" if p < 0.005 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                flip = " SIGN-FLIP" if (np.nanmean(d[cc].values[mask]) * np.nanmean(d[cc].values[~mask]) < 0) else ""
                print(f"     {name:8} in={np.nanmean(d[cc].values[mask]):+.4f} out={np.nanmean(d[cc].values[~mask]):+.4f} "
                      f"diff={obs:+.4f}{star:<3} p={p:.3f}{flip}")
            # continuous correlations
            cells = []
            for f in CORR_FEATS:
                if f not in d:
                    continue
                rho, fl = spearman_perm(d[cc].values, d[f].values)
                s = "*" if (rho == rho and abs(rho) > fl) else " "
                cells.append(f"{f}={rho:+.3f}{s}")
            print(f"     CORR: " + "  ".join(cells))
    print("\nREAD: a stratum diff/CORR that is significant for det/map/motion but NOT for aux=rand")
    print("      and ideally NOT for the 'ctrl' stratum / ctrl_random feature = STRUCTURED sign")
    print("      => that feature is the gradient-level conditional-weighting knob (idea 2 reborn).")


if __name__ == "__main__":
    main()
