"""Per-sample (batch_size=1) aux<->plan gradient coupling, labeled by scene feature.

Tests whether the per-sample <g_aux, g_plan> SIGN is STRUCTURED (predicted by a
scene variable) or RANDOM.  The b1000 probe showed the batch-MEAN coupling
cancels (cos~0.001) while per-batch |dot|~0.07 — i.e. coupling exists but is
sign-balanced.  If conditioning on a scene feature un-cancels it (conditional
mean flips sign across strata), that feature is the gradient-level knob for a
conditional task-weighting method (idea 2 reborn).  If the sign is random,
the coupling is unexploitable.

Reuses the verified loaders (run_gradient_analysis) + compute_task_full_gradient,
restricted to inter_gnn params, with frozen matching + forward seed for clean
gradients.  Records, per train sample:
  token, <g_aux,g_plan>, cos, ||g_aux||, ||g_plan||  for aux in det/map/motion
  + a norm-matched RANDOM-direction control dot (lie detector)
  + scene features (density / ego-relevance / maneuver / map complexity).
"""
import argparse, sys, time, pickle
from pathlib import Path

import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "viewer"))
import tools.run_gradient_analysis as RGA
from tools.gradient_analysis.collector import compute_task_full_gradient
from tools.gradient_analysis.probe import _flat_norm, _seed_scope, _resolve_param_set
from strata_analysis import scene_features            # tools/viewer
from relevance_analysis import relevance_feats        # tools/viewer

EPS = 1e-8
TRAIN_PKL = str(ROOT / "data/infos/nuscenes_infos_train.pkl")


def _dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)).item())


def load_train_index():
    """Index train infos by integer timestamp (img_metas has no token; nuScenes
    timestamps are unique microsecond ints, so they key the sample uniquely)."""
    d = pickle.load(open(TRAIN_PKL, "rb"))
    infos = d["infos"] if isinstance(d, dict) else d
    micro = {int(e["timestamp"]): e for e in infos}                 # exact microsecond
    sec = {int(int(e["timestamp"]) // 1_000_000): e for e in infos}  # second-floor fallback
    return micro, sec


def extract_token(data):
    """Return the integer timestamp of a (batch_size=1) sample for info lookup."""
    m = data.get("img_metas")
    while hasattr(m, "data"):
        m = m.data
    while isinstance(m, (list, tuple)) and len(m):
        m = m[0]
    ts = None
    if isinstance(m, dict):
        ts = m.get("timestamp")
    if ts is None:                          # fall back to top-level timestamp
        ts = data.get("timestamp")
        while hasattr(ts, "data"):
            ts = ts.data
        if isinstance(ts, (list, tuple)) and len(ts):
            ts = ts[0]
    if ts is None:
        return None
    if hasattr(ts, "item"):
        ts = ts.item()
    return float(ts)          # raw (meta is SECONDS, possibly with sub-second; keep precision)


def to_microsec_key(ts):
    """meta timestamp (seconds, maybe float) -> microsecond int matching the pkl."""
    if ts is None:
        return None
    return int(round(ts * 1e6)) if ts < 1e12 else int(round(ts))


def feats_for(info):
    f = scene_features(info)
    try:
        rf = relevance_feats(info)
        f["n_relevant"] = rf["n_relevant"]
        f["min_dist_path"] = rf["min_dist_path"]
    except Exception:
        f["n_relevant"] = np.nan; f["min_dist_path"] = np.nan
    return f


def run(config, ckpt_tag, layers, n_samples, out_csv, rand_seed=1234):
    cfg = yaml = __import__("yaml").safe_load(open(config))
    cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(cfg["model_config"]))
    adapter = RGA.build_adapter(rt, cfg)
    ckpt = RGA._resolve_ckpt_path(cfg, ckpt_tag)
    print(f"[psc] loading {ckpt}", flush=True)
    model = RGA.load_model(rt, adapter, ckpt, cfg["device"], cfg.get("fp16", True))
    groups = list(dict.fromkeys(list(cfg["shared_param_groups"]) + list(layers)))
    collector = rt["GradientCollector"](model=model, adapter=adapter,
                                        shared_layer_names=groups, device=cfg["device"])
    dl = rt["build_dataloader"](adapter, batch_size=1, shuffle=True, seed=cfg["seed"])
    # pooled inter_gnn params
    params = []
    for L in layers:
        params += _resolve_param_set(collector, L)[0]
    print(f"[psc] inter_gnn pooled params: {len(params)} tensors", flush=True)

    fs = cfg["probe"].get("forward_seed", 42)
    rts = cfg["probe"].get("reset_temporal_state", True)
    train_idx, sec_idx = load_train_index()
    print(f"[psc] train_idx n={len(train_idx)} (sec_idx n={len(sec_idx)})", flush=True)
    state_snap = adapter.snapshot_temporal_state(model) if rts else None
    import contextlib
    # NO freeze_matching: that pins sample-0's Hungarian indices and replaying
    # them on a DIFFERENT sample causes a CUDA index-out-of-bounds. Each sample
    # gets its own fresh matching (we only need per-sample gradients, no stepping).
    ctx = contextlib.nullcontext()

    rows, t0, miss = [], time.time(), 0
    rng = np.random.RandomState(rand_seed)
    with ctx:
        n_err = 0
        for i, data in enumerate(dl):
            if i >= n_samples:
                break
            tasks = ["det", "map", "motion", "plan"]
            try:
                if state_snap is not None:
                    adapter.restore_temporal_state(model, state_snap)
                with _seed_scope(fs):
                    losses = adapter.forward_losses(model, data)
                g = {}
                for ti, t in enumerate(tasks):
                    tl = adapter.split_losses(losses, t)
                    g[t] = (compute_task_full_gradient(tl, params, retain_graph=(ti < len(tasks) - 1))
                            if tl is not None else None)
            except Exception as e:                      # degenerate-GT sample (bs=1 exposes it)
                gt = {}
                for k, v in data.items():
                    if "gt" not in k:
                        continue
                    vv = v
                    while hasattr(vv, "data"):
                        vv = vv.data
                    while isinstance(vv, (list, tuple)) and len(vv):
                        vv = vv[0]
                    gt[k] = tuple(vv.shape) if hasattr(vv, "shape") else (
                        len(vv) if hasattr(vv, "__len__") else type(vv).__name__)
                print(f"[psc] ERR sample {i}: {type(e).__name__}: {str(e)[:90]} | gt={gt}", flush=True)
                n_err += 1
                torch.cuda.empty_cache()
                if n_err > 40:
                    print("[psc] >40 errors, stopping early", flush=True)
                    break
                continue
            tok = extract_token(data)
            key = to_microsec_key(tok)
            info = train_idx.get(key) if key is not None else None
            if info is None and tok is not None:        # fallback: match by integer second
                info = sec_idx.get(int(tok))
            if i < 3:
                print(f"[psc] sample {i}: tok={tok!r} key={key} exact={key in train_idx} "
                      f"sec={int(tok) in sec_idx if tok else None}", flush=True)
            if info is None:
                miss += 1
                continue
            gp = g["plan"]; gpn = _flat_norm(gp)
            rec = {"token": tok, "gplan_norm": gpn}
            for aux in ["det", "map", "motion"]:
                ga = g[aux]
                if ga is None or gp is None:
                    rec[f"dot_{aux}"] = np.nan; rec[f"cos_{aux}"] = np.nan; rec[f"{aux}_norm"] = np.nan
                    continue
                gan = _flat_norm(ga); dot = _dot(ga, gp)
                rec[f"dot_{aux}"] = dot; rec[f"{aux}_norm"] = gan
                rec[f"cos_{aux}"] = dot / (gan * gpn) if (gan > EPS and gpn > EPS) else np.nan
            # random control matched to ||g_det||
            with _seed_scope(rand_seed + i):
                r = [torch.randn_like(p) for p in params]
            rn = _flat_norm(r)
            sc = rec.get("det_norm", gpn) / rn if rn > EPS else 1.0
            r = [x * sc for x in r]
            rec["dot_rand"] = _dot(r, gp)
            rec["cos_rand"] = rec["dot_rand"] / (_flat_norm(r) * gpn) if gpn > EPS else np.nan
            rec.update(feats_for(info))
            rows.append(rec)
            if (i + 1) % 50 == 0:
                print(f"[psc] {ckpt_tag} {i+1}/{n_samples}  ({(time.time()-t0)/60:.1f} min, miss={miss})", flush=True)
    df = pd.DataFrame(rows); df["ckpt"] = ckpt_tag
    df.to_csv(out_csv, index=False)
    print(f"[psc] wrote {out_csv}  rows={len(df)}  miss_token={miss}  {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="1ep")
    ap.add_argument("--layers", default="dec0_inter_gnn_0,dec1_inter_gnn_0,dec2_inter_gnn_0")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"gradient_analysis_results/persample_{a.ckpt}.csv"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    run(a.config, a.ckpt, a.layers.split(","), a.n, out)
