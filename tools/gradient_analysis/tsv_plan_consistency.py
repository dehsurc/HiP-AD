#!/usr/bin/env python
"""Plan gradient direction-consistency across stage-2 epochs (1/3/9/18ep).

Loads the saved mean gradient matrices (mean_grads.pt written by tsv_gradient.py)
at each epoch and asks: does plan keep pulling the shared inter_gnn weights along
the SAME low-rank axis, or does the direction rotate during training?

Two complementary notions of "same direction":
  flat_cos(ep_i, ep_j)  = <Gbar(ep_i), Gbar(ep_j)>_F / norms   (signed; the
        overall descent direction. 1=identical pull, 0=unrelated, <0=reversed)
  subspace_cos_k        = mean principal-angle cos of top-k LEFT singular
        subspaces (sign-invariant; is the low-rank axis itself stable?)
For reference each is compared to plan-vs-aux consistency and a random floor.
Pure post-processing, no GPU.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = "gradient_analysis_results/tsv_gradient"
EPS = [("1ep", "s2_1ep"), ("3ep", "s2_3ep"), ("9ep", "s2_9ep"), ("18ep", "s2_final")]
TASKS = ["det", "map", "motion", "plan"]
KS = [1, 2, 4, 8]


def load(tag):
    d = torch.load(Path(ROOT) / tag / "mean_grads.pt", map_location="cpu")
    return d["mean_grads"]


def flat_cos(A, B):
    a, b = A.flatten().double(), B.flatten().double()
    return float(a @ b / (a.norm() * b.norm() + 1e-30))


def subspace_cos(A, B, k):
    Ua = torch.linalg.svd(A.double(), full_matrices=False).U[:, :k]
    Ub = torch.linalg.svd(B.double(), full_matrices=False).U[:, :k]
    return float(torch.linalg.svdvals(Ua.T @ Ub).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/plan_consistency")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    G = {lab: load(tag) for lab, tag in EPS}
    mats = list(G["18ep"]["plan"].keys())
    n_dim = G["18ep"]["plan"][mats[0]].shape[0]
    labels = [lab for lab, _ in EPS]

    rows = []
    for task in TASKS:
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                li, lj = labels[i], labels[j]
                fc = np.mean([flat_cos(G[li][task][m], G[lj][task][m]) for m in mats])
                rec = dict(task=task, pair=f"{li}->{lj}", flat_cos=fc)
                for k in KS:
                    rec[f"subspace_cos_k{k}"] = np.mean(
                        [subspace_cos(G[li][task][m], G[lj][task][m], k) for m in mats])
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(out / "direction_consistency.csv", index=False)

    # cross-task at fixed epoch (is plan's axis the same as aux's? expect no)
    cross = []
    for lab in labels:
        for aux in ["det", "map", "motion"]:
            fc = np.mean([flat_cos(G[lab]["plan"][m], G[lab][aux][m]) for m in mats])
            sc = np.mean([subspace_cos(G[lab]["plan"][m], G[lab][aux][m], 4) for m in mats])
            cross.append(dict(epoch=lab, pair=f"plan~{aux}", flat_cos=fc, subspace_cos_k4=sc))
    pd.DataFrame(cross).to_csv(out / "plan_vs_aux_same_epoch.csv", index=False)

    rnd1, rnd4 = np.sqrt(1 / n_dim), np.sqrt(4 / n_dim)
    print(f"\n=== PLAN direction consistency across epochs (mean over {len(mats)} inter_gnn matrices, n={n_dim}) ===")
    p = df[df.task == "plan"]
    print(p[["pair", "flat_cos", "subspace_cos_k1", "subspace_cos_k2", "subspace_cos_k4"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nconsecutive only (1->3, 3->9, 9->18):")
    cons = p[p.pair.isin(["1ep->3ep", "3ep->9ep", "9ep->18ep"])]
    print(cons[["pair", "flat_cos", "subspace_cos_k1", "subspace_cos_k4"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nis PLAN more direction-consistent than aux? (mean flat_cos over all 6 epoch-pairs)")
    print(df.groupby("task")["flat_cos"].mean().to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\n(random floors: flat_cos~0, subspace_cos_k1~{rnd1:.3f}, k4~{rnd4:.3f})")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
