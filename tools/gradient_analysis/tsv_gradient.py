#!/usr/bin/env python
"""Task Singular *Gradients* (TSV-analog) on HiP-AD shared inter_gnn weights.

NOT the TSV paper object (that SVDs the accumulated weight delta deltaW per task;
we have no per-task finetunes, only a joint stage-2 delta -- see
tsv_weight_delta.py). Here the per-task object is the *mean loss-gradient matrix*
on each shared inter_gnn 2D weight:

    Gbar_task = (1/N) sum_b  dL_task(batch_b) / dW          (W = a 2D weight)

Mean over batches is deliberate: a single-batch grad of a linear/attn weight is
sum_t delta_t x_t^T over token positions, so its rank is bounded by token count
and its spectrum reflects activation statistics, not task structure. Averaging
cancels the per-token noise and leaves the *systematic* descent direction, whose
SVD is the thing worth interpreting.

For each matrix we SVD Gbar_plan = U S V^T and report, on plan's own basis:
  effrank/energy of plan's gradient (does the TSV low-rank premise hold here?)
  mode_contrib[aux,r] = s_r * (u_r^T Gbar_aux v_r)   (signed; <0 = aux opposes
                          plan's r-th descent mode, >0 = aux reinforces it)
  subspace_overlap   top-k(U_plan) vs top-k(U_aux)   (do tasks share modes?)
  flat_cos(plan,aux) = <Gbar_plan,Gbar_aux>_F / norms  (the existing ~0 metric,
                          to show WHERE in the spectrum that average sits)

in_proj_weight (3n,n) is split into Q/K/V (n,n) blocks. Reuses the RGA harness;
batch_size=1, temporal state reset per sample (matches ntk_coupling).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA

EPS = 1e-12
TASKS = ["det", "map", "motion", "plan"]
AUX = ["det", "map", "motion"]
KS = [1, 4, 8, 16, 32]


def inter_gnn_targets(raw_model, config):
    """Return list of (name, param, kind) for inter_gnn 2D weights."""
    from mmcv import Config
    oo = Config.fromfile(config).operation_order
    ig = [i for i, o in enumerate(oo) if o == "inter_gnn"]
    want = []
    for i in ig:
        want.append(f"head.onedecoder_head.layers.{i}.attns.0.attn.in_proj_weight")
        want.append(f"head.onedecoder_head.layers.{i}.attns.0.attn.out_proj.weight")
    out = []
    name2p = dict(raw_model.named_parameters())
    for n in want:
        if n not in name2p:
            continue
        short = n.replace("head.onedecoder_head.layers.", "ig").replace(".attns.0.attn", "")
        out.append((short, name2p[n], "qkv" if n.endswith("in_proj_weight") else "plain"))
    return out


def conv_targets(raw_model, which):
    """All 4D conv weights under img_backbone / img_neck (kind='conv', reshaped
    to (out_ch, -1) at SVD time). These are the *shared bottleneck* every task
    backprops through -- where capacity competition must physically happen."""
    mods = []
    if "backbone" in which:
        mods.append(("bb", "img_backbone"))
    if "neck" in which:
        mods.append(("neck", "img_neck"))
    out = []
    for tag, attr in mods:
        m = getattr(raw_model, attr, None)
        if m is None:
            continue
        for n, p in m.named_parameters():
            if p.requires_grad and p.dim() == 4:
                out.append((f"{tag}/{n.replace('.weight', '')}", p, "conv"))
    return out


def resolve_targets(raw_model, config, which):
    if which == "inter_gnn":
        return inter_gnn_targets(raw_model, config)
    return conv_targets(raw_model, which)


def grad_matrices(task_loss, params, retain):
    g = torch.autograd.grad(task_loss, params, retain_graph=retain, allow_unused=True)
    return [gi.detach() if gi is not None else torch.zeros_like(p)
            for gi, p in zip(g, params)]


def split_qkv(name, M):
    """in_proj (3n,n) -> 3 named (n,n) blocks; else passthrough."""
    if M.shape[0] == 3 * M.shape[1]:
        n = M.shape[1]
        base = name.replace(".in_proj_weight", "")
        return [(f"{base}.in_proj_q", M[:n]), (f"{base}.in_proj_k", M[n:2 * n]),
                (f"{base}.in_proj_v", M[2 * n:])]
    return [(name.replace(".weight", ""), M)]


def spectrum(s):
    s = np.sort(np.asarray(s, float))[::-1]
    e = s ** 2; tot = e.sum() + 1e-30
    pr = (s.sum() ** 2) / tot
    cum = np.cumsum(e) / tot
    rec = {"pr_effrank": float(pr), "pr_norm": float(pr / len(s)),
           "stable_rank": float(tot / (s[0] ** 2 + 1e-30))}
    for k in KS:
        rec[f"energy_top{k}"] = float(cum[min(k, len(s)) - 1])
    return rec


def analyze(mean_grads, out):
    """mean_grads: {task: {matrix_name: tensor}} -> CSVs + summary."""
    out.mkdir(parents=True, exist_ok=True)
    spec_rows, contrib_rows, overlap_rows = [], [], []
    names = list(mean_grads["plan"].keys())
    if not names:
        print("[tsv-grad] no plan gradients collected — nothing to analyze")
        return
    for name in names:
        Gp = mean_grads["plan"][name].float()
        U, S, Vh = torch.linalg.svd(Gp, full_matrices=False)
        spec_rows.append(dict(matrix=name, task="plan", fro=float(Gp.norm()),
                              **spectrum(S.cpu().numpy())))
        for aux in AUX:
            Ga = mean_grads[aux][name].float()
            spec_rows.append(dict(matrix=name, task=aux, fro=float(Ga.norm()),
                                  **spectrum(torch.linalg.svdvals(Ga).cpu().numpy())))
            flat_cos = float((Gp.flatten() @ Ga.flatten()) /
                             (Gp.norm() * Ga.norm() + EPS))
            # mode contributions on plan's basis
            UtGaV = U.T @ Ga @ Vh.T               # (r,r); diag = u_r^T Ga v_r
            diag = torch.diagonal(UtGaV)
            contrib = (S * diag).cpu().numpy()    # s_r * proj
            energy = float((S ** 2).sum())
            neg_energy = float((contrib[contrib < 0] ** 2).sum() /
                               ((contrib ** 2).sum() + 1e-30))
            for r in range(len(S)):
                contrib_rows.append(dict(matrix=name, aux=aux, mode=r,
                                         s_plan=float(S[r]), proj=float(diag[r]),
                                         contrib=float(contrib[r])))
            # top-k left-subspace overlap
            Ua = torch.linalg.svd(Ga, full_matrices=False).U
            for k in [4, 8, 16, 32]:
                sv = torch.linalg.svdvals(U[:, :k].T @ Ua[:, :k])
                overlap_rows.append(dict(matrix=name, aux=aux, k=k,
                                         subspace_cos=float(sv.mean()),
                                         flat_cos=flat_cos, neg_energy_ratio=neg_energy))
    spec = pd.DataFrame(spec_rows); spec.to_csv(out / "gradient_spectrum.csv", index=False)
    con = pd.DataFrame(contrib_rows); con.to_csv(out / "plan_mode_contrib.csv", index=False)
    ov = pd.DataFrame(overlap_rows); ov.to_csv(out / "subspace_overlap.csv", index=False)

    spec["grp"] = spec["matrix"].str.split(".").str[0].str.split("/").str[0] \
        if spec["matrix"].str.contains("/").any() else "inter_gnn"
    print("\n=== per-task gradient spectrum (mean over matrices, by group) ===")
    g = spec.groupby(["grp", "task"]).agg(
        n_mat=("matrix", "size"), pr_norm=("pr_norm", "mean"),
        energy_top8=("energy_top8", "mean"), stable_rank=("stable_rank", "mean")).reset_index()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== aux contribution on plan's top-8 descent modes (s_r * u_r^T Ga v_r) ===")
    top = con[con["mode"] < 8].groupby("aux").agg(
        mean_contrib=("contrib", "mean"),
        pos_frac=("contrib", lambda x: float((x > 0).mean())),
        sum_contrib=("contrib", "sum"))
    print(top.to_string(float_format=lambda x: f"{x:.4g}"))
    print("\n=== flat cos & negative-energy ratio & subspace overlap (k=8) ===")
    o8 = ov[ov.k == 8].groupby("aux").agg(
        flat_cos=("flat_cos", "mean"), subspace_cos_k8=("subspace_cos", "mean"),
        neg_energy_ratio=("neg_energy_ratio", "mean"))
    n_dim = float(np.mean([mean_grads["plan"][m].shape[0] for m in names]))
    print(o8.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"(random top-8 subspace_cos reference ~ sqrt(8/mean_n={n_dim:.0f}) = {np.sqrt(8/n_dim):.3f})")


def run(ckpt_path, config, n_samples, out, target="inter_gnn", device="cuda:0", seed=42):
    cfg = dict(adapter="hipad", model_config=config, tasks=TASKS,
               device=device, seed=seed)
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(config))
    adapter = RGA.build_adapter(rt, cfg)
    print(f"[tsv-grad] checkpoint = {ckpt_path}", flush=True)
    model = RGA.load_model(rt, adapter, Path(ckpt_path), device, True)
    raw = model.module if hasattr(model, "module") else model
    targets = resolve_targets(raw, config, target)
    params = [p for _, p, _ in targets]
    print(f"[tsv-grad] target={target}: {len(targets)} weight matrices tracked", flush=True)

    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=seed)
    state_snap = adapter.snapshot_temporal_state(model)
    accum = {t: {} for t in TASKS}; counts = {t: 0 for t in TASKS}
    t0, used, err = time.time(), 0, 0
    for i, data in enumerate(dl):
        if used >= n_samples:
            break
        try:
            adapter.restore_temporal_state(model, state_snap)
            losses = adapter.forward_losses(model, data)
            ok = False
            for ti, task in enumerate(TASKS):
                L = adapter.split_losses(losses, task)
                if L is None:
                    continue
                gm = grad_matrices(L, params, retain=(ti < len(TASKS) - 1))
                for (nm, _, kind), M in zip(targets, gm):
                    if kind == "qkv":
                        items = split_qkv(nm, M)
                    elif kind == "conv":
                        items = [(nm, M.reshape(M.shape[0], -1))]
                    else:
                        items = [(nm, M)]
                    for sn, blk in items:
                        a = accum[task].get(sn)
                        accum[task][sn] = blk.float().cpu() if a is None else a + blk.float().cpu()
                counts[task] += 1; ok = True
            if ok:
                used += 1
        except Exception as e:
            err += 1; torch.cuda.empty_cache()
            print(f"[tsv-grad] ERR sample {i}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            if err > 30:
                print("[tsv-grad] >30 errors, abort", flush=True); break
            continue
        if used and used % 20 == 0:
            print(f"[tsv-grad] {used}/{n_samples} ({(time.time()-t0)/60:.1f} min, err={err})", flush=True)

    mean_grads = {t: {k: v / max(counts[t], 1) for k, v in accum[t].items()} for t in TASKS}
    print(f"[tsv-grad] collected counts={counts} err={err} in {(time.time()-t0)/60:.1f} min", flush=True)
    analyze(mean_grads, Path(out))
    torch.save({"mean_grads": mean_grads, "counts": counts},
               Path(out) / "mean_grads.pt")
    print(f"[tsv-grad] wrote -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default="ckpts/rev_nusc/70+stage2_18ep.pth")
    ap.add_argument("--config",
                    default="ckpts/rev_nusc/E2_E1_stage2_18ep_from70ep.py")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--target", default="inter_gnn",
                    choices=["inter_gnn", "backbone", "neck", "backbone_neck"])
    ap.add_argument("--out", default="gradient_analysis_results/tsv_gradient/s2_final")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    run(a.ckpt_path, a.config, a.n, a.out, target=a.target)
