#!/usr/bin/env python
"""Spectrum of per-task gradients on SHARED inter_gnn weights vs each task's own
PRIVATE refine-head weights -- same forward, same 64 scenes.

Answers: (1) is plan's low-rank-ness on shared weights mechanical (few ego
tokens) -> then its private ego_refine head should be equally low-rank; or
structural to the shared bottleneck. (2) do tasks lean on shared capacity at
all, or is the real work private? -> compare gradient magnitude (Frobenius)
private vs shared.

Singular VECTORS are not comparable across the two (different param spaces);
only the spectrum shape (pr_norm / energy / stable_rank) and magnitude are.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA

TASKS = ["det", "map", "motion", "plan"]
PRIVATE_HEAD = {"det": "det_refine", "map": "map_refine",
                "motion": "motion_refine", "plan": "plan_refine"}
KS = [1, 8, 32]


def inter_gnn_params(raw, config):
    from mmcv import Config
    oo = Config.fromfile(config).operation_order
    ig = [i for i, o in enumerate(oo) if o == "inter_gnn"]
    name2p = dict(raw.named_parameters())
    out = []
    for i in ig:
        for suf in ["in_proj_weight", "out_proj.weight"]:
            n = f"head.onedecoder_head.layers.{i}.attns.0.attn.{suf}"
            if n in name2p:
                out.append((f"ig{i}.{suf}", name2p[n]))
    return out


def private_params(raw):
    """task -> list of (name, 2D weight param) in that task's refine head."""
    name2p = dict(raw.named_parameters())
    res = {}
    for task, head in PRIVATE_HEAD.items():
        key = f"head.onedecoder_head.{head}."
        res[task] = [(n.replace("head.onedecoder_head.", ""), p)
                     for n, p in name2p.items()
                     if n.startswith(key) and p.dim() == 2]
    return res


def spec(M):
    M = M.float()
    if M.dim() == 4:
        M = M.reshape(M.shape[0], -1)
    s = torch.linalg.svdvals(M).cpu().numpy()
    s = np.sort(s)[::-1]; e = s ** 2; tot = e.sum() + 1e-30
    pr = (s.sum() ** 2) / tot
    cum = np.cumsum(e) / tot
    rec = dict(n=len(s), fro=float(np.sqrt(tot)), pr_norm=float(pr / len(s)),
               stable_rank=float(tot / (s[0] ** 2 + 1e-30)))
    for k in KS:
        rec[f"energy_top{k}"] = float(cum[min(k, len(s)) - 1])
    return rec


def run(ckpt_path, config, n_samples, out, device="cuda:0", seed=42):
    cfg = dict(adapter="hipad", model_config=config, tasks=TASKS, device=device, seed=seed)
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(config))
    adapter = RGA.build_adapter(rt, cfg)
    model = RGA.load_model(rt, adapter, Path(ckpt_path), device, True)
    raw = model.module if hasattr(model, "module") else model
    shared = inter_gnn_params(raw, config)
    priv = private_params(raw)
    print(f"[svp] shared={len(shared)} inter_gnn mats; private mats per task: "
          + ", ".join(f"{t}:{len(priv[t])}" for t in TASKS), flush=True)

    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=seed)
    snap = adapter.snapshot_temporal_state(model)
    # accumulators: ("shared"/"private", task, matname) -> sum grad
    acc = {}; cnt = {t: 0 for t in TASKS}
    t0, used, err = time.time(), 0, 0
    for i, data in enumerate(dl):
        if used >= n_samples:
            break
        try:
            adapter.restore_temporal_state(model, snap)
            losses = adapter.forward_losses(model, data)
            ok = False
            for ti, task in enumerate(TASKS):
                L = adapter.split_losses(losses, task)
                if L is None:
                    continue
                tgt = shared + priv[task]
                g = torch.autograd.grad(L, [p for _, p in tgt], retain_graph=(ti < len(TASKS) - 1),
                                        allow_unused=True)
                for (nm, p), gi in zip(tgt, g):
                    if gi is None:
                        continue
                    scope = "shared" if nm.startswith("ig") else "private"
                    key = (scope, task, nm)
                    M = gi.detach().float().cpu()
                    acc[key] = M if key not in acc else acc[key] + M
                cnt[task] += 1; ok = True
            if ok:
                used += 1
        except Exception as e:
            err += 1; torch.cuda.empty_cache()
            print(f"[svp] ERR {i}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if err > 30:
                break
            continue
        if used and used % 20 == 0:
            print(f"[svp] {used}/{n_samples} ({(time.time()-t0)/60:.1f}m)", flush=True)

    rows = []
    for (scope, task, nm), M in acc.items():
        rec = spec(M / max(cnt[task], 1)); rec.update(scope=scope, task=task, matrix=nm)
        rows.append(rec)
    df = pd.DataFrame(rows)
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out) / "shared_vs_private_spectrum.csv", index=False)

    g = df.groupby(["task", "scope"]).agg(
        n_mat=("matrix", "size"), pr_norm=("pr_norm", "mean"),
        energy_top8=("energy_top8", "mean"), stable_rank=("stable_rank", "mean"),
        fro=("fro", "mean")).reindex(pd.MultiIndex.from_product([TASKS, ["shared", "private"]]))
    print("\n=== per-task gradient spectrum: SHARED inter_gnn vs PRIVATE refine head (18ep, n={}) ===".format(used))
    print(g.to_string(float_format=lambda x: f"{x:.3f}"))
    # magnitude ratio private/shared
    print("\nmean |grad| Frobenius ratio  private / shared  (>1 = task pushes its private head harder):")
    for t in TASKS:
        try:
            fs = df[(df.task == t) & (df.scope == "shared")].fro.mean()
            fp = df[(df.task == t) & (df.scope == "private")].fro.mean()
            print(f"  {t:>7}: shared={fs:.3f}  private={fp:.3f}  ratio={fp/ (fs+1e-9):.2f}")
        except Exception:
            pass
    print(f"wrote -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default="ckpts/rev_nusc/70+stage2_18ep.pth")
    ap.add_argument("--config", default="ckpts/rev_nusc/E2_E1_stage2_18ep_from70ep.py")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="gradient_analysis_results/tsv_gradient/shared_vs_private_18ep")
    a = ap.parse_args()
    run(a.ckpt_path, a.config, a.n, a.out)
