"""Unified k-step trajectory probe (Q1 + Q2-C + Q3 in ONE pass).

Bridges the gap between the NULL single-step coupling (<g_aux,g_plan> ~ 0) and
the non-null one-epoch TG effect (+0.19) by walking a k-step virtual trajectory
along the aux gradient and watching the planning loss + planning gradient.

From one trajectory per (checkpoint, batch, source, layer), post-processing yields:
  Q3 transference   Z(k)        = 1 - L_plan(theta_k)/L_plan(theta_0)
  Q3 decomposition  first-order path sum S(k) = -alpha * sum_j <step_dir_j, g_plan_j>
                    residual(k) = DeltaL_plan(k) - S(k)   (accumulated/curvature part)
  Q2-C mixed Hess.  C(j) = <g_plan_j, H_plan step_dir_j>  via finite difference
                    H_plan step_dir_j ~ -(g_plan_{j+1}-g_plan_j)/alpha
                    -> C(j) ~ -( <g_plan_j,g_plan_{j+1}> - ||g_plan_j||^2 )/alpha
  Q1 curvature      curv0 = [L_plan(theta+a*ghat) + L_plan(theta-a*ghat) - 2 L_plan(theta)]/a^2
                    lin0  = [L_plan(theta+a*ghat) - L_plan(theta-a*ghat)]/(2a)  (~ null dot, sanity)

Every quantity is run for BOTH an `aux` arm (step along g_source, recomputed each
step) and a norm-matched `rand` arm (fixed random direction) so aux-minus-rand
isolates the aux-SPECIFIC signal (the control all 3 critics demanded).

Reuses the verified probe primitives: frozen Hungarian matching + frozen mode
argmin + frozen temporal state + forward_seed (clean deltas), apply_virtual_step,
snapshot/restore_params, compute_task_full_gradient.
"""
import argparse, contextlib, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA
from tools.gradient_analysis.collector import compute_task_full_gradient
from tools.gradient_analysis.probe import (
    apply_virtual_step, snapshot_params, restore_params, _flat_norm, _seed_scope, _resolve_param_set,
)

EPS = 1e-8


def _dot(a, b):
    if not a or not b:
        return float("nan")
    return float(sum((x * y).sum() for x, y in zip(a, b)).item())


def trajectory_one_batch(collector, data, source, layer, alpha, k, target,
                         freeze_matching, forward_seed, reset_temporal_state,
                         rand_seed, batch_idx):
    adapter = collector.adapter
    state_snap = adapter.snapshot_temporal_state(collector.model) if reset_temporal_state else None

    def fwd():
        if state_snap is not None:
            adapter.restore_temporal_state(collector.model, state_snap)
        if forward_seed is not None:
            with _seed_scope(forward_seed):
                return collector.forward_losses(data)
        return collector.forward_losses(data)

    params, layer_label = _resolve_param_set(collector, layer)
    rows = []
    ctx = adapter.freeze_stochastic_state() if freeze_matching else contextlib.nullcontext()
    with ctx:
        for arm in ("aux", "rand"):
            snap = snapshot_params(params)
            prev_gplan = None
            rand_dir = None
            for j in range(k + 1):
                losses = fwd()
                Lp = adapter.split_losses(losses, target)
                Ls = adapter.split_losses(losses, source)
                if Lp is None:
                    break
                Lp_val = float(Lp.item())
                g_plan = compute_task_full_gradient(Lp, params, retain_graph=True)
                # need g_src this step? aux: yes (it is the step). rand: only at j==0 for norm-match.
                need_src = (arm == "aux") or (arm == "rand" and j == 0)
                g_src = (compute_task_full_gradient(Ls, params, retain_graph=False)
                         if (need_src and Ls is not None) else None)
                # free the retained graph if g_src didn't
                if g_src is None:
                    del losses, Lp, Ls
                gpn = _flat_norm(g_plan)
                gsn = _flat_norm(g_src) if g_src else float("nan")

                if arm == "aux":
                    step_dir = g_src
                else:
                    if rand_dir is None:
                        with _seed_scope(rand_seed + batch_idx):
                            rand_dir = [torch.randn_like(p) for p in params]
                        rn = _flat_norm(rand_dir)
                        sc = (gsn / rn) if (rn > EPS and np.isfinite(gsn)) else 1.0
                        rand_dir = [r * sc for r in rand_dir]
                    step_dir = rand_dir

                dot_step = _dot(step_dir, g_plan) if step_dir is not None else float("nan")
                cross = _dot(prev_gplan, g_plan) if prev_gplan is not None else float("nan")
                rows.append(dict(batch=batch_idx, arm=arm, source=source, layer=layer_label, j=j,
                                 L_plan=Lp_val, gplan_norm=gpn, gplan_sq=gpn * gpn,
                                 gsrc_norm=gsn, dot_step=dot_step, gplan_cross_prev=cross))
                prev_gplan = [g.detach().clone() for g in g_plan]
                if j == k or step_dir is None:
                    break
                apply_virtual_step(params, step_dir, alpha=alpha, normalize=False)
            restore_params(params, snap)

        # ---- Q1 symmetric curvature at theta_0 (forward-only even/odd split) ----
        snap = snapshot_params(params)
        l0 = fwd()
        Ls0 = adapter.split_losses(l0, source)
        Lp0 = adapter.split_losses(l0, target)
        if Ls0 is not None and Lp0 is not None:
            Lp0v = float(Lp0.item())
            g0 = compute_task_full_gradient(Ls0, params, retain_graph=False)
            gsn0 = _flat_norm(g0)
            ghat = [g / max(gsn0, EPS) for g in g0]
            apply_virtual_step(params, ghat, alpha=alpha, normalize=False)    # theta - a*ghat
            with torch.no_grad():
                Lm = float(adapter.split_losses(fwd(), target).item())
            restore_params(params, snap)
            apply_virtual_step(params, ghat, alpha=-alpha, normalize=False)   # theta + a*ghat
            with torch.no_grad():
                Lpl = float(adapter.split_losses(fwd(), target).item())
            restore_params(params, snap)
            curv = (Lpl + Lm - 2 * Lp0v) / (alpha * alpha)
            lin = (Lpl - Lm) / (2 * alpha)
            rows.append(dict(batch=batch_idx, arm="curv0", source=source, layer=layer_label, j=0,
                             L_plan=Lp0v, gsrc_norm=gsn0, curv=curv, lin=lin))
        restore_params(params, snap)
    return rows


def run(config, ckpt_tag, sources, layers, alpha, k, num_batches, out_csv, smoke):
    cfg = yaml.safe_load(open(config))
    cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    mcfg = Config.fromfile(cfg["model_config"])
    RGA._load_plugins(mcfg)
    adapter = RGA.build_adapter(rt, cfg)
    ckpt = RGA._resolve_ckpt_path(cfg, ckpt_tag)
    print(f"[traj] loading {ckpt}", flush=True)
    model = RGA.load_model(rt, adapter, ckpt, cfg["device"], cfg.get("fp16", True))
    groups = list(dict.fromkeys(list(cfg["shared_param_groups"]) + list(layers)))
    collector = rt["GradientCollector"](model=model, adapter=adapter,
                                        shared_layer_names=groups, device=cfg["device"])
    dl = rt["build_dataloader"](adapter, batch_size=cfg["primary"]["batch_size"],
                                shuffle=True, seed=cfg["seed"])
    pc = cfg["probe"]
    fm, fs, rts = pc.get("freeze_matching", True), pc.get("forward_seed", 42), pc.get("reset_temporal_state", True)
    nb = 2 if smoke else num_batches
    rows = []
    t0 = time.time()
    for i, data in enumerate(dl):
        if i >= nb:
            break
        for src in sources:
            for lay in layers:
                rows += trajectory_one_batch(collector, data, src, lay, alpha, k, "plan",
                                             fm, fs, rts, 1234, i)
        print(f"[traj] {ckpt_tag} batch {i+1}/{nb} done  ({(time.time()-t0)/60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df["ckpt"] = ckpt_tag
    df.to_csv(out_csv, index=False)
    print(f"[traj] wrote {out_csv}  rows={len(df)}  total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="1ep")
    ap.add_argument("--sources", default="det,map,motion")
    ap.add_argument("--layers", default="dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0")
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--num-batches", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = a.out or f"gradient_analysis_results/trajectory_{a.ckpt}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(a.config, a.ckpt, a.sources.split(","), a.layers.split(","),
        a.alpha, a.k, a.num_batches, out, a.smoke)
