"""Q4 — cross-task functional (NTK) coupling, cancellation-invariant.

The loss-gradient dot <g_aux,g_plan> = u_L^T K v_L is ONE contraction of the
cross-task kernel K = J_plan J_aux^T (output Jacobians w.r.t. inter_gnn params)
against the loss-residual directions u_L,v_L — and that contraction sign-cancels
across samples.  K itself need not.  We compare, per sample:
  cos_loss  = cos(g_aux, g_plan)               (loss-direction contraction; cancels)
  cos_func  = cos(J_plan^T u, J_aux^T v)  for RANDOM unit u,v in task-output space
              (random-direction contractions of the SAME K)
If mean|cos_func| >> |cos_loss|, the functional kernel is structured but the loss
residuals hit its small/null directions => "functional coupling exists, gradient
cancels it".  If comparable, the coupling is genuinely as weak as the loss dot.

A norm-matched RANDOM-PARAM control (cos of two random param vectors) gives the
high-dim Frobenius floor; cos_func is evidence only as EXCESS over it.

Reuses get_task_queries (graph-attached refine[-1] input queries) + the persample
loader.  NO freeze_matching (per-sample independent).
"""
import argparse, sys, time, contextlib
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools" / "viewer"))
import tools.run_gradient_analysis as RGA
from tools.gradient_analysis.collector import compute_task_full_gradient
from tools.gradient_analysis.probe import _flat_norm, _seed_scope, _resolve_param_set
from persample_coupling import load_train_index, extract_token, to_microsec_key, feats_for  # reuse

EPS = 1e-8
AUX = ["det", "map", "motion"]


def _gdot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)).item())


def _grad_to(scalar, params):
    g = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    return [gi if gi is not None else torch.zeros_like(p) for gi, p in zip(g, params)]


def _flat(g):
    return torch.cat([x.detach().flatten() for x in g])


def _cos_flat(a, b):
    na, nb = a.norm(), b.norm()
    return float((a @ b / (na * nb)).item()) if (na > EPS and nb > EPS) else np.nan


def ntk_one_sample(adapter, model, data, params, R, rng_seed, prev_a_plan):
    """Returns (rec, a_plan_flat_for_next_shuffle).  prev_a_plan = previous
    sample's flattened J_plan^T u (for the SHUFFLED cross-sample null)."""
    fwd = adapter.forward_losses(model, data)
    tq = fwd.get("task_queries") if isinstance(fwd, dict) else None
    if not tq or "plan" not in tq:
        return None, prev_a_plan
    Lp = adapter.split_losses(fwd, "plan")
    if Lp is None:
        return None, prev_a_plan
    f_plan = tq["plan"]
    g_plan = _grad_to(Lp, params); gpn = _flat_norm(g_plan)
    rec = {}
    a_plan_keep = None
    with _seed_scope(rng_seed):
        for aux in AUX:
            if aux not in tq:
                continue
            f_aux = tq[aux]
            La = adapter.split_losses(fwd, aux)
            g_aux = _grad_to(La, params) if La is not None else None
            if g_aux is not None:
                gan = _flat_norm(g_aux)
                rec[f"cos_loss_{aux}"] = (_gdot(g_aux, g_plan) / (gan * gpn)
                                          if gan > EPS and gpn > EPS else np.nan)
            cfs, shuf = [], []
            for r in range(R):
                u = torch.randn_like(f_plan); v = torch.randn_like(f_aux)
                a = _flat(_grad_to((u * f_plan).sum(), params))
                b = _flat(_grad_to((v * f_aux).sum(), params))
                cfs.append(_cos_flat(a, b))                       # same-sample cross-task
                if prev_a_plan is not None:
                    shuf.append(_cos_flat(prev_a_plan, b))        # plan(prev sample) x aux(this)
                if a_plan_keep is None:
                    a_plan_keep = a                               # stash for next sample's shuffle
            cfs = np.array(cfs, float)
            rec[f"cosfunc_absmean_{aux}"] = float(np.nanmean(np.abs(cfs)))
            rec[f"cosfunc_mean_{aux}"] = float(np.nanmean(cfs))
            if shuf:
                rec[f"cosshuf_absmean_{aux}"] = float(np.nanmean(np.abs(shuf)))
        r1 = [torch.randn_like(p) for p in params]; r2 = [torch.randn_like(p) for p in params]
        rec["cos_paramfloor"] = _gdot(r1, r2) / (_flat_norm(r1) * _flat_norm(r2) + EPS)
    return rec, (a_plan_keep if a_plan_keep is not None else prev_a_plan)


def run(config, ckpt_tag, layers, R, n_samples, out_csv, ckpt_path=None):
    import yaml
    cfg = yaml.safe_load(open(config)); cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(cfg["model_config"]))
    adapter = RGA.build_adapter(rt, cfg)
    ckpt = Path(ckpt_path) if ckpt_path else RGA._resolve_ckpt_path(cfg, ckpt_tag)
    print(f"[ntk] checkpoint = {ckpt}", flush=True)
    model = RGA.load_model(rt, adapter, ckpt, cfg["device"], cfg.get("fp16", True))
    groups = list(dict.fromkeys(list(cfg["shared_param_groups"]) + list(layers)))
    collector = rt["GradientCollector"](model=model, adapter=adapter, shared_layer_names=groups, device=cfg["device"])
    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=cfg["seed"])
    params = []
    for L in layers:
        params += _resolve_param_set(collector, L)[0]
    train_idx, sec_idx = load_train_index()
    rts = cfg["probe"].get("reset_temporal_state", True)
    state_snap = adapter.snapshot_temporal_state(model) if rts else None
    rows, t0, miss, nerr = [], time.time(), 0, 0
    prev_a = None
    for i, data in enumerate(dl):
        if i >= n_samples:
            break
        try:
            if state_snap is not None:
                adapter.restore_temporal_state(model, state_snap)
            rec, prev_a = ntk_one_sample(adapter, model, data, params, R, 1000 + i, prev_a)
        except Exception as e:
            print(f"[ntk] ERR sample {i}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            nerr += 1; torch.cuda.empty_cache()
            if nerr > 40:
                print("[ntk] >40 errors, stop", flush=True); break
            continue
        if rec is None:
            miss += 1; continue
        tok = extract_token(data); info = train_idx.get(to_microsec_key(tok)) if tok else None
        if info is None and tok is not None:
            info = sec_idx.get(int(tok))
        if info is not None:
            rec.update(feats_for(info))
        rec["token"] = tok
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[ntk] {ckpt_tag} {i+1}/{n_samples}  ({(time.time()-t0)/60:.1f} min, miss={miss}, err={nerr})", flush=True)
    df = pd.DataFrame(rows); df["ckpt"] = ckpt_tag
    df.to_csv(out_csv, index=False)
    print(f"[ntk] wrote {out_csv} rows={len(df)} miss={miss} err={nerr} {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="1ep")
    ap.add_argument("--layers", default="dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0")
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-path", default=None, help="direct .pth (overrides --ckpt tag)")
    a = ap.parse_args()
    out = a.out or f"gradient_analysis_results/ntk_{a.ckpt}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(a.config, a.ckpt, a.layers.split(","), a.R, a.n, out, ckpt_path=a.ckpt_path)
