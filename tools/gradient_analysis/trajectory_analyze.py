"""Derive Q1/Q2/Q3 from the k-step trajectory CSVs and contrast aux vs random.

Reads gradient_analysis_results/trajectory_*.csv (one per checkpoint) and reports,
per (ckpt, source, layer):
  CANCELLATION : per-batch dot_step(j=0) signed-mean vs mean-abs. If |mean|<<mean-abs,
                 the 'null cosine' is per-sample cancellation, NOT true orthogonality.
  Q3 Z(k)      : transference 1 - L_plan(k)/L_plan(0), aux vs rand (aux-rand = aux-specific).
  Q3 decomp    : first-order path-sum vs residual share of DeltaL_plan(k).
  Q2-C         : mixed Hessian C = <g_plan, H_plan step>/ (finite diff), aux vs rand, signed.
  Q1 curv0     : directional curvature ghat_aux^T H_plan ghat_aux (even component), signed-mean+std.
"""
import argparse, glob, os
import numpy as np, pandas as pd


def per_batch_traj(g, alpha):
    """g: rows for one (ckpt,source,layer,batch,arm) sorted by j. Returns dict."""
    g = g.sort_values("j")
    L = g["L_plan"].values
    dot = g["dot_step"].values
    sq = g["gplan_sq"].values
    cross = g["gplan_cross_prev"].values   # cross[j] = <g_plan_{j-1}, g_plan_j>
    k = len(L) - 1
    deltaL = L[k] - L[0]
    Z = 1 - L[k] / L[0] if abs(L[0]) > 1e-9 else np.nan
    path_sum = -alpha * np.nansum(dot[:k])                 # first-order predicted DeltaL
    residual = deltaL - path_sum
    # C(j) = -(cross[j+1] - sq[j]) / alpha , j=0..k-1
    Cs = []
    for j in range(k):
        if j + 1 < len(cross) and np.isfinite(cross[j + 1]):
            Cs.append(-(cross[j + 1] - sq[j]) / alpha)
    C = np.nanmean(Cs) if Cs else np.nan
    return dict(dot0=dot[0], Z=Z, deltaL=deltaL, path_sum=path_sum, residual=residual, C=C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="gradient_analysis_results/trajectory_*.csv")
    ap.add_argument("--alpha", type=float, default=0.01)
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows from {len(files)} files: {[os.path.basename(f) for f in files]}")
    order = {"1ep": 1, "3ep": 3, "6ep": 6, "9ep": 9, "18ep": 18}
    cks = sorted(df["ckpt"].unique(), key=lambda c: order.get(c, 99))

    for ck in cks:
        for src in sorted(df["source"].unique()):
            for lay in sorted(df["layer"].unique()):
                sub = df[(df.ckpt == ck) & (df.source == src) & (df.layer == lay)]
                traj = sub[sub.arm.isin(["aux", "rand"])]
                if traj.empty:
                    continue
                rec = {"aux": [], "rand": []}
                for (arm, b), g in traj.groupby(["arm", "batch"]):
                    rec[arm].append(per_batch_traj(g, a.alpha))
                A = pd.DataFrame(rec["aux"]); R = pd.DataFrame(rec["rand"])
                if A.empty:
                    continue
                nb = len(A)
                # cancellation diagnostic on aux dot0
                d0 = A["dot0"].values
                signed = np.nanmean(d0); absm = np.nanmean(np.abs(d0))
                cancel = abs(signed) / absm if absm > 1e-12 else np.nan
                # curv0 (aux only)
                c0 = sub[sub.arm == "curv0"]["curv"].values
                curv_m = np.nanmean(c0); curv_s = np.nanstd(c0)
                # aggregates
                def m(dfx, col):
                    return np.nanmean(dfx[col].values) if (dfx is not None and not dfx.empty) else np.nan
                print(f"\n== {ck:>4} | src={src:6} | {lay} | nb={nb} ==")
                print(f"  CANCEL  dot0: signed_mean={signed:+.4f}  mean|.|={absm:.4f}  "
                      f"ratio={cancel:.3f}  (small ratio => cancellation, not orthogonality)")
                print(f"  Q3 Z(k) aux={m(A,'Z'):+.5f}  rand={m(R,'Z'):+.5f}  "
                      f"aux-rand={m(A,'Z')-m(R,'Z'):+.5f}")
                print(f"  Q3 dec  DeltaL aux={m(A,'deltaL'):+.5f}  path_sum={m(A,'path_sum'):+.5f}  "
                      f"residual={m(A,'residual'):+.5f}  (residual share => beyond-first-order)")
                print(f"  Q2 C    aux={m(A,'C'):+.4f}  rand={m(R,'C'):+.4f}  "
                      f"aux-rand={m(A,'C')-m(R,'C'):+.4f}  (signed; survives where dot cancels?)")
                print(f"  Q1 curv0 mean={curv_m:+.4f}  std={curv_s:.4f}  "
                      f"(directional sharpness of L_plan along g_{src})")


if __name__ == "__main__":
    main()
