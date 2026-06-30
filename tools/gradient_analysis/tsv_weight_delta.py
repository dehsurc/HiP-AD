#!/usr/bin/env python
"""TSV premise check on HiP-AD — SVD of the *stage-2 weight delta* (a real,
joint task vector) on the shared inter_gnn / temp_gnn 2D weights.

This is the cheap companion to the per-task gradient-SVD plan: before spending
GPU on per-task gradients we test whether TSV's central empirical claim — "layer
task matrices are often low-rank" — holds for *this* model's shared weights.

Object analysed: deltaW = W(70+stage2_*) - W(70ep) per shared 2D weight matrix.
in_proj_weight (3n, n) is split into Q/K/V blocks (each n x n) because they are
packed and have unrelated row-spaces. Pure weight arithmetic — no forward pass.

Metrics per matrix (all scale-invariant on the spectrum s, descending):
  pr_effrank   participation ratio (sum s)^2 / sum s^2     -> "how many modes"
  stable_rank  ||M||_F^2 / s_max^2
  energy_topk  cumulative sum s^2 / total at k in {1,4,8,16,32}
  pr_norm      pr_effrank / n           (1.0 = uses every mode like noise)
  rand_pr_norm same metric for an iid-Gaussian matrix of identical shape
               -> the "no structure" reference. delta is low-rank only if
               pr_norm << rand_pr_norm.
  rel_update   ||deltaW||_F / ||W_init||_F
Also a cross-checkpoint cross-check: cosine of the top-k LEFT singular subspace
of deltaW(early) vs deltaW(final) (is the update direction set early?).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CKPT_DIR = Path("/home/yongjae/e2e/HiP-AD/ckpts/rev_nusc")
INIT = "70ep.pth"                       # stage-2 initialisation (stage-1 end)
STAGES = {                              # tag -> file (deltas measured vs INIT)
    "s2_1ep": "70+stage2_1ep.pth",
    "s2_3ep": "70+stage2_3ep.pth",
    "s2_9ep": "70+stage2_9ep.pth",
    "s2_final": "70+stage2.pth",
}
CONFIG = "projects/configs/hipad_nusc_stage2.py"
KS = [1, 4, 8, 16, 32]


def load_sd(name):
    sd = torch.load(CKPT_DIR / name, map_location="cpu")
    return sd.get("state_dict", sd)


def gnn_layer_indices(config):
    from mmcv import Config
    oo = Config.fromfile(config).operation_order
    return ([i for i, o in enumerate(oo) if o == "inter_gnn"],
            [i for i, o in enumerate(oo) if o == "temp_gnn"])


def named_matrices(sd, layer_idx, op):
    """Yield (name, 2D weight tensor) for one decoder layer, splitting in_proj
    into Q/K/V blocks."""
    pref = f"head.onedecoder_head.layers.{layer_idx}."
    out = []
    for k, v in sd.items():
        if not k.startswith(pref) or v.dim() != 2:
            continue
        short = f"{op}{layer_idx}." + k[len(pref):]
        if k.endswith("in_proj_weight") and v.shape[0] == 3 * v.shape[1]:
            n = v.shape[1]
            for j, blk in enumerate(["q", "k", "v"]):
                out.append((short.replace("in_proj_weight", f"in_proj_{blk}"),
                            v[j * n:(j + 1) * n].contiguous()))
        else:
            out.append((short, v))
    return out


def spectrum_metrics(M):
    s = torch.linalg.svdvals(M.float()).cpu().numpy()
    s = np.sort(s)[::-1]
    n = len(s)
    e = s ** 2
    tot = e.sum()
    pr = (s.sum() ** 2) / (e.sum() + 1e-30)
    rec = {
        "n": n,
        "fro": float(np.sqrt(tot)),
        "smax": float(s[0]),
        "pr_effrank": float(pr),
        "pr_norm": float(pr / n),
        "stable_rank": float(tot / (s[0] ** 2 + 1e-30)),
    }
    cum = np.cumsum(e) / (tot + 1e-30)
    for k in KS:
        rec[f"energy_top{k}"] = float(cum[min(k, n) - 1])
    return rec, s


def rand_pr_norm(shape, trials=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(trials):
        M = torch.randn(*shape, generator=g)
        s = torch.linalg.svdvals(M).numpy()
        vals.append((s.sum() ** 2) / ((s ** 2).sum()) / len(s))
    return float(np.mean(vals))


def topk_subspace_cos(Ma, Mb, k):
    """mean cos of principal angles between top-k left singular subspaces."""
    Ua = torch.linalg.svd(Ma.float(), full_matrices=False).U[:, :k]
    Ub = torch.linalg.svd(Mb.float(), full_matrices=False).U[:, :k]
    sv = torch.linalg.svdvals(Ua.T @ Ub)
    return float(sv.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gradient_analysis_results/tsv_weight_delta")
    ap.add_argument("--config", default=CONFIG)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ig, tg = gnn_layer_indices(args.config)
    layer_set = [("inter_gnn", i) for i in ig] + [("temp_gnn", i) for i in tg]
    init = load_sd(INIT)

    rows = []
    spectra = {}
    deltas_final = {}   # name -> deltaW(final) for cross-ckpt subspace test
    deltas_early = {}
    for tag, fname in STAGES.items():
        sd = load_sd(fname)
        for op, li in layer_set:
            for name, Wf in named_matrices(sd, li, op):
                Wi = init[name.replace(f"{op}{li}.", f"head.onedecoder_head.layers.{li}.")
                          .replace("in_proj_q", "in_proj_weight")
                          .replace("in_proj_k", "in_proj_weight")
                          .replace("in_proj_v", "in_proj_weight")]
                # reconstruct the matching init block for q/k/v
                if "in_proj_" in name and name.split(".")[-1] in ("in_proj_q", "in_proj_k", "in_proj_v"):
                    n = Wf.shape[0]
                    blk = {"in_proj_q": 0, "in_proj_k": 1, "in_proj_v": 2}[name.split(".")[-1]]
                    Wi = Wi[blk * n:(blk + 1) * n]
                dW = (Wf - Wi)
                rec, s = spectrum_metrics(dW)
                rec.update(dict(ckpt=tag, op=op, layer=li, matrix=name,
                                rel_update=rec["fro"] / (Wi.float().norm().item() + 1e-30),
                                rand_pr_norm=rand_pr_norm(tuple(dW.shape))))
                rows.append(rec)
                spectra[(tag, name)] = s
                if tag == "s2_final":
                    deltas_final[name] = dW
                if tag == "s2_1ep":
                    deltas_early[name] = dW

    df = pd.DataFrame(rows)
    cols = ["ckpt", "op", "layer", "matrix", "n", "rel_update", "pr_effrank",
            "pr_norm", "rand_pr_norm", "stable_rank"] + [f"energy_top{k}" for k in KS]
    df = df[cols]
    df.to_csv(out / "weight_delta_spectrum.csv", index=False)

    # cross-checkpoint: is the final update subspace already set at 1ep?
    sub_rows = []
    for name in deltas_final:
        if name in deltas_early:
            for k in [4, 8, 16]:
                sub_rows.append(dict(matrix=name, k=k,
                                     cos_top_k_1ep_vs_final=topk_subspace_cos(
                                         deltas_early[name], deltas_final[name], k)))
    sub = pd.DataFrame(sub_rows)
    sub.to_csv(out / "subspace_early_vs_final.csv", index=False)

    # ---- console summary ----
    fin = df[df.ckpt == "s2_final"]
    print("\n=== stage-2 weight-delta SVD (final vs 70ep) — shared GNN matrices ===")
    g = fin.groupby("op").agg(
        n_mat=("matrix", "size"),
        rel_update=("rel_update", "mean"),
        pr_norm=("pr_norm", "mean"),
        rand_pr_norm=("rand_pr_norm", "mean"),
        e_top8=("energy_top8", "mean"),
        e_top32=("energy_top32", "mean"),
        stable_rank=("stable_rank", "mean"),
    )
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nper inter_gnn matrix (final):")
    show = fin[fin.op == "inter_gnn"][["matrix", "n", "rel_update", "pr_norm",
                                       "rand_pr_norm", "energy_top8", "energy_top32",
                                       "stable_rank"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\npr_norm trajectory across stage-2 (inter_gnn mean):")
    traj = (df[df.op == "inter_gnn"].groupby("ckpt")
            .agg(pr_norm=("pr_norm", "mean"), rel_update=("rel_update", "mean"),
                 e_top8=("energy_top8", "mean")))
    traj = traj.reindex([t for t in STAGES if t in traj.index])
    print(traj.to_string(float_format=lambda x: f"{x:.3f}"))
    if len(sub):
        print("\ntop-k left-subspace cos, deltaW(1ep) vs deltaW(final), inter_gnn:")
        m = sub[sub.matrix.str.contains("inter")].groupby("k")["cos_top_k_1ep_vs_final"].mean()
        print(m.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
