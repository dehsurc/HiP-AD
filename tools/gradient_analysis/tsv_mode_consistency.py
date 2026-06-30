#!/usr/bin/env python
"""Proper TSV singular-vector consistency of per-task gradients across epochs.

Answers: using the TSV singular vectors (not flat cosine), is plan's top-k
singular subspace less stable across 1/3/9/18ep than the aux tasks'?

Per task, per epoch-pair, on each inter_gnn mean-gradient matrix Gbar=U S Vᵀ:
  U_cos_k = mean principal-angle cos of top-k LEFT singular subspaces  (output space)
  V_cos_k = same for top-k RIGHT singular subspaces                    (input space)
  mode1_cos = |cos| of the rank-1 leading mode vec(u1 v1ᵀ) across epochs
              (sign-invariant: SVD flips u,v together so u v ᵀ is well-defined)
All sign/rotation-invariant — the correct way to compare singular vectors.
Pure post-processing of mean_grads.pt; same 64 fixed scenes at every epoch.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = "gradient_analysis_results/tsv_gradient"
EPS = [("1ep", "s2_1ep"), ("3ep", "s2_3ep"), ("9ep", "s2_9ep"), ("18ep", "s2_final")]
TASKS = ["det", "map", "motion", "plan"]
KS = [1, 2, 4, 8]


def load(tag):
    return torch.load(Path(ROOT) / tag / "mean_grads.pt", map_location="cpu")["mean_grads"]


def svd(M):
    U, S, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    return U, S, Vh  # Vh rows = right singular vectors


def subcos(Ak, Bk):
    return float(torch.linalg.svdvals(Ak.T @ Bk).mean())


def main():
    out = Path(ROOT) / "mode_consistency"; out.mkdir(parents=True, exist_ok=True)
    G = {lab: load(tag) for lab, tag in EPS}
    mats = list(G["18ep"]["plan"].keys())
    labels = [l for l, _ in EPS]
    consec = [("1ep", "3ep"), ("3ep", "9ep"), ("9ep", "18ep")]

    # precompute SVDs
    SV = {lab: {t: {m: svd(G[lab][t][m]) for m in mats} for t in TASKS} for lab in labels}

    rows = []
    for t in TASKS:
        for li, lj in [(labels[i], labels[j]) for i in range(len(labels)) for j in range(i + 1, len(labels))]:
            rec = dict(task=t, pair=f"{li}->{lj}", consecutive=(li, lj) in consec)
            for k in KS:
                rec[f"U_cos_k{k}"] = np.mean([subcos(SV[li][t][m][0][:, :k], SV[lj][t][m][0][:, :k]) for m in mats])
                rec[f"V_cos_k{k}"] = np.mean([subcos(SV[li][t][m][2][:k].T, SV[lj][t][m][2][:k].T) for m in mats])
            # leading rank-1 mode consistency
            m1 = []
            for m in mats:
                Ui, _, Vhi = SV[li][t][m]; Uj, _, Vhj = SV[lj][t][m]
                a = torch.outer(Ui[:, 0], Vhi[0]).flatten()
                b = torch.outer(Uj[:, 0], Vhj[0]).flatten()
                m1.append(abs(float(a @ b / (a.norm() * b.norm() + 1e-30))))
            rec["mode1_cos"] = np.mean(m1)
            rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "tsv_mode_consistency.csv", index=False)

    n = G["18ep"]["plan"][mats[0]].shape[0]
    floor = {k: np.sqrt(k / n) for k in KS}
    print(f"\n=== TSV singular-vector consistency across epochs (mean over {len(mats)} inter_gnn mats, n={n}) ===")
    print("per-task mean over the 3 CONSECUTIVE pairs (1->3,3->9,9->18):")
    c = df[df.consecutive].groupby("task").agg(
        U_cos_k1=("U_cos_k1", "mean"), U_cos_k4=("U_cos_k4", "mean"),
        V_cos_k1=("V_cos_k1", "mean"), V_cos_k4=("V_cos_k4", "mean"),
        mode1_cos=("mode1_cos", "mean")).reindex(TASKS)
    print(c.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nrandom floors: U/V_cos_k1~{floor[1]:.3f}, k4~{floor[4]:.3f}, mode1_cos~{np.sqrt(1/(n*n))*n:.3f}(≈{1/np.sqrt(min(n,n)):.3f} scale)")
    print("\nper-task mean over ALL 6 epoch-pairs (TSV subspace) vs the flat-cos number reported before:")
    a = df.groupby("task").agg(U_cos_k4=("U_cos_k4", "mean"), V_cos_k4=("V_cos_k4", "mean"),
                               mode1_cos=("mode1_cos", "mean")).reindex(TASKS)
    print(a.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
