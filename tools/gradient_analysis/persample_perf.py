"""Per-set (batch_size=1) aux<->plan gradient cos AND per-set planning PERFORMANCE.

Goal (user's idea): the original probe averaged <g_aux,g_plan> over 1000-2000
bs=6 sets and got ~0 (sign cancellation).  Here we DO NOT average: we record,
per set, the gradient cos for each aux AND the planning loss on that same set,
then study the DISTRIBUTION across sets and the cos<->performance relationship.

Differences vs persample_coupling.py (audited 2026-06-16):
  * loads the model in **fp32** (fp16=False).  fp16 backward through the planning
    head NaN-overflows at mid checkpoints (3ep 22%, 9ep 68% of plan grads were
    NaN under fp16) — that silently selection-biased earlier mid-ckpt results.
    fp32 removes it.  (couple_finetune.py already used fp32.)
  * records per-set PERFORMANCE: plan_loss (= split_losses('plan')) plus each aux
    loss, so we can correlate cos(g_aux,g_plan) with plan_loss.
  * keeps the norm-matched RANDOM-direction control (cos_rand) as the confound
    check: cos_aux<->plan_loss is only meaningful as EXCESS over cos_rand<->plan_loss
    (a sample with large residual has a specific g_plan direction regardless of aux).

Everything else (inter_gnn pooled params, fixed forward_seed, temporal reset, no
freeze_matching, degenerate-GT try/except, token->train-info match) is unchanged
from the verified persample loader.
"""
import argparse, sys, time, pickle, contextlib
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "viewer"))
import tools.run_gradient_analysis as RGA
from tools.gradient_analysis.collector import compute_task_full_gradient
from tools.gradient_analysis.probe import _flat_norm, _seed_scope, _resolve_param_set
from persample_coupling import (load_train_index, extract_token, to_microsec_key,
                                feats_for, _dot)

EPS = 1e-8
AUX = ["det", "map", "motion"]
TASKS = ["det", "map", "motion", "plan"]


def run(config, ckpt_tag, layers, n_samples, out_csv, fp16=False, rand_seed=1234):
    cfg = __import__("yaml").safe_load(open(config))
    cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(cfg["model_config"]))
    adapter = RGA.build_adapter(rt, cfg)
    ckpt = RGA._resolve_ckpt_path(cfg, ckpt_tag)
    print(f"[psp] loading {ckpt}  (fp16={fp16})", flush=True)
    model = RGA.load_model(rt, adapter, ckpt, cfg["device"], fp16)
    groups = list(dict.fromkeys(list(cfg["shared_param_groups"]) + list(layers)))
    collector = rt["GradientCollector"](model=model, adapter=adapter,
                                        shared_layer_names=groups, device=cfg["device"])
    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=cfg["seed"])
    params = []
    for L in layers:
        params += _resolve_param_set(collector, L)[0]
    print(f"[psp] inter_gnn pooled params: {len(params)} tensors", flush=True)

    fs = cfg["probe"].get("forward_seed", 42)
    rts = cfg["probe"].get("reset_temporal_state", True)
    train_idx, sec_idx = load_train_index()
    state_snap = adapter.snapshot_temporal_state(model) if rts else None

    rows, t0, miss, n_err, n_nan = [], time.time(), 0, 0, 0
    for i, data in enumerate(dl):
        if i >= n_samples:
            break
        try:
            if state_snap is not None:
                adapter.restore_temporal_state(model, state_snap)
            with _seed_scope(fs):
                losses = adapter.forward_losses(model, data)
            tl = {t: adapter.split_losses(losses, t) for t in TASKS}
            g = {}
            for ti, t in enumerate(TASKS):
                g[t] = (compute_task_full_gradient(tl[t], params, retain_graph=(ti < len(TASKS) - 1))
                        if tl[t] is not None else None)
        except Exception as e:
            print(f"[psp] ERR sample {i}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            n_err += 1; torch.cuda.empty_cache()
            if n_err > 60:
                print("[psp] >60 errors, stop", flush=True); break
            continue

        gp = g["plan"]
        gpn = _flat_norm(gp) if gp is not None else np.nan
        # NaN guard: a NaN plan grad (fp16 overflow, or degenerate loss) makes the
        # whole row meaningless. Count it explicitly instead of silently dropping.
        if gp is None or not np.isfinite(gpn) or gpn <= EPS:
            n_nan += 1
            if (i + 1) % 50 == 0:
                print(f"[psp] {ckpt_tag} {i+1}/{n_samples} (miss={miss} nan={n_nan} err={n_err})", flush=True)
            continue

        rec = {"token": extract_token(data), "gplan_norm": gpn}
        # per-set performance: planning loss (and aux losses) on THIS set
        for t in TASKS:
            rec[f"loss_{t}"] = float(tl[t].item()) if tl[t] is not None else np.nan
        for aux in AUX:
            ga = g[aux]
            if ga is None:
                rec[f"dot_{aux}"] = rec[f"cos_{aux}"] = rec[f"{aux}_norm"] = np.nan
                continue
            gan = _flat_norm(ga); dot = _dot(ga, gp)
            rec[f"{aux}_norm"] = gan
            rec[f"dot_{aux}"] = dot
            rec[f"cos_{aux}"] = dot / (gan * gpn) if (gan > EPS and gpn > EPS) else np.nan
        # norm-matched random-direction control (confound lie-detector)
        with _seed_scope(rand_seed + i):
            r = [torch.randn_like(p) for p in params]
        rn = _flat_norm(r)
        sc = (rec.get("det_norm") or gpn) / rn if rn > EPS else 1.0
        r = [x * sc for x in r]
        rec["dot_rand"] = _dot(r, gp)
        rec["cos_rand"] = rec["dot_rand"] / (_flat_norm(r) * gpn) if gpn > EPS else np.nan

        # scene features (non-fatal; only for optional conditioning)
        tok = rec["token"]; key = to_microsec_key(tok)
        info = train_idx.get(key) if key is not None else None
        if info is None and tok is not None:
            info = sec_idx.get(int(tok))
        if info is not None:
            try: rec.update(feats_for(info))
            except Exception: pass
        else:
            miss += 1
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f"[psp] {ckpt_tag} {i+1}/{n_samples}  rows={len(rows)} "
                  f"(miss={miss} nan={n_nan} err={n_err}, {(time.time()-t0)/60:.1f}m)", flush=True)

    df = pd.DataFrame(rows); df["ckpt"] = ckpt_tag
    df.to_csv(out_csv, index=False)
    print(f"[psp] wrote {out_csv} rows={len(df)} nan={n_nan} err={n_err} miss_tok={miss} "
          f"({(time.time()-t0)/60:.1f}m)  nan_rate={n_nan/max(1,i+1):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="9ep")
    ap.add_argument("--layers", default="dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--fp16", action="store_true", help="load fp16 (default fp32; fp16 NaNs at mid ckpts)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"gradient_analysis_results/persample_perf_{a.ckpt}{'_fp16' if a.fp16 else ''}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(a.config, a.ckpt, a.layers.split(","), a.n, out, fp16=a.fp16)
