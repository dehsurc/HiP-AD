"""D3 smoke — TAG-style lookahead aux->plan affinity (the MEASUREMENT part of D3).

For each aux a in {det,map,motion}, at a fixed checkpoint:
  1. forward batch B -> per-task losses (one graph)
  2. baseline plan loss on a HELD-OUT batch B' (no_grad)
  3. take ONE step on L_a only:  theta' = theta - eta * grad(L_a)   (plain SGD, all params)
  4. plan loss on B' at theta'  (no_grad)
  5. restore theta exactly (add the step back); affinity delta = L_plan(B';theta') - L_plan(B';theta)
Average over K (B,B') pairs.  Z[a->plan] = -mean(delta)/mean(L_plan_base)  (>0 => aux helps plan).

This is the CHEAP, performance-coupled affinity (loss-delta, not gradient cosine).
HELD-OUT B' makes it a transfer signal (not the same-batch dot that sign-cancels).
Sanity: ranking/sign should agree with leave-one-out TG (aux helps plan, front-loaded).

NOT a training method — just verifies the probe works + gives ballpark numbers + cost.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA

AUX = ["det", "map", "motion"]


def plan_loss(adapter, model, data):
    with torch.no_grad():
        out = adapter.forward_losses(model, data)
        tl = adapter.split_losses(out, "plan")
        return float(tl.item()) if tl is not None else np.nan


def run(config, ckpt_tag, K, eta, bs):
    import yaml
    cfg = yaml.safe_load(open(config)); cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(cfg["model_config"]))
    adapter = RGA.build_adapter(rt, cfg)
    ckpt = RGA._resolve_ckpt_path(cfg, ckpt_tag)
    print(f"[tag] {ckpt_tag} = {ckpt}  (fp32, eta={eta}, K={K}, bs={bs})", flush=True)
    model = RGA.load_model(rt, adapter, ckpt, cfg["device"], False)   # fp32
    raw = model.module if hasattr(model, "module") else model
    raw.train()
    dl = rt["build_dataloader"](adapter, batch_size=bs, shuffle=True, seed=cfg["seed"])
    params = [p for p in raw.parameters() if p.requires_grad]
    chk0 = float(sum(p.detach().double().sum() for p in params).item())

    delta = {a: [] for a in AUX}
    pred = {a: [] for a in AUX}     # first-order prediction -eta*<g_a,g_plan(B')>
    base = []
    t0, prev, n, nerr = time.time(), None, 0, 0
    for i, data in enumerate(dl):
        if n >= K:
            break
        if prev is None:
            prev = data; continue
        B, Bp = prev, data; prev = data
        try:
            Lp0 = plan_loss(adapter, model, Bp)
            if not np.isfinite(Lp0):
                continue
            # grad of plan on held-out B' (for first-order cross-check)
            outp = adapter.forward_losses(model, Bp)
            gplan = torch.autograd.grad(adapter.split_losses(outp, "plan"), params,
                                        retain_graph=False, allow_unused=True)
            gplan = [g if g is not None else None for g in gplan]
            del outp
            # one forward on B -> all aux losses share the graph
            out_b = adapter.forward_losses(model, B)
            lb = {a: adapter.split_losses(out_b, a) for a in AUX}
            present = [a for a in AUX if lb[a] is not None]
            # compute ALL aux grads BEFORE touching params: an in-place step
            # bumps param version counters and invalidates a still-retained graph.
            grads = {}
            for idx, a in enumerate(present):
                grads[a] = torch.autograd.grad(
                    lb[a], params, retain_graph=(idx < len(present) - 1), allow_unused=True)
            del out_b, lb
            for a in present:
                g = grads[a]
                # first-order prediction of plan-loss change: -eta*<g_a, g_plan(B')>
                dot = sum(float((gi * gp).sum().item())
                          for gi, gp in zip(g, gplan) if (gi is not None and gp is not None))
                pred[a].append(-eta * dot)
                with torch.no_grad():                       # theta -= eta*g
                    for p, gi in zip(params, g):
                        if gi is not None:
                            p.sub_(eta * gi)
                Lp1 = plan_loss(adapter, model, Bp)
                with torch.no_grad():                       # restore theta += eta*g
                    for p, gi in zip(params, g):
                        if gi is not None:
                            p.add_(eta * gi)
                delta[a].append(Lp1 - Lp0)
            del grads, gplan
            base.append(Lp0); n += 1
            if n % 10 == 0:
                print(f"[tag] {n}/{K}  ({(time.time()-t0)/60:.1f}m)", flush=True)
        except Exception as e:
            nerr += 1; torch.cuda.empty_cache()
            print(f"[tag] ERR {i}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            if nerr > 20:
                break
            continue

    chk1 = float(sum(p.detach().double().sum() for p in params).item())
    print(f"\n[tag] restore check: |Δparams|={abs(chk1-chk0):.3e} (should be ~0)", flush=True)
    Lpm = float(np.mean(base)) if base else np.nan
    print(f"[tag] n={n}  mean plan_loss(base)={Lpm:.4f}\n", flush=True)
    print(f"{'aux':7} {'meanΔ(actual)':>14} {'SE':>9} {'Z=-Δ/Lp':>9} {'meanΔ(1st-order)':>16}")
    for a in AUX:
        d = np.array(delta[a], float); d = d[np.isfinite(d)]
        p1 = np.array(pred[a], float); p1 = p1[np.isfinite(p1)]
        if not len(d):
            print(f"{a:7} (none)"); continue
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        Z = -d.mean() / Lpm if Lpm else np.nan
        print(f"{a:7} {d.mean():>+14.5f} {se:>9.5f} {Z:>+9.4f} {p1.mean():>+16.5f}")
    print("\nREAD: Z>0 => stepping that aux LOWERS held-out plan loss (helps planning).")
    print("      Compare ranking/sign to leave-one-out TG. SE<<|meanΔ| => distinguishable from 0.")
    print("      actual vs 1st-order: if they match, single-step is linear (≈ the dot); if actual")
    print("      is cleaner/larger, the finite+held-out transfer carries signal the per-sample dot missed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="3ep")
    ap.add_argument("--K", type=int, default=40)
    ap.add_argument("--eta", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=2)
    a = ap.parse_args()
    run(a.config, a.ckpt, a.K, a.eta, a.bs)
