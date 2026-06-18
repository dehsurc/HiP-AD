"""Stage A — causal test: does STRENGTHENING aux<->plan functional coupling improve planning?

3 arms (manual fine-tune loop, reuses the analysis adapter — no model surgery):
  baseline : standard loss only (lambda=0)
  coupling : loss + lambda * (-cos(pool(Q_plan), pool(Q_aux)))   [pull plan & aux query
             features together -> raises functional coupling K]
  control  : loss + lambda * (-cos(pool(Q_plan), pool(Q_aux[shuffled-in-batch])))
             [matched magnitude, no real pairing -> Kurin regularization control]

After fine-tuning each arm we (separately) re-measure cosfunc_same (did coupling rise?)
and planning collision/L2 (did it help?).  coupling > control on planning AND coupling
raised cosfunc_same  =>  causal support that functional coupling helps planning.
"""
import argparse, sys, time
from pathlib import Path
import torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA

AUX = ["det", "map", "motion"]


def coupling_loss(Q, arm):
    if arm == "baseline" or "plan" not in Q:
        return None
    qp = Q["plan"]
    qp = qp.reshape(qp.shape[0], -1, qp.shape[-1]).mean(1)   # (B, C)
    qp = F.normalize(qp, dim=-1)
    terms = []
    for a in AUX:
        if a not in Q:
            continue
        qa = Q[a]
        qa = qa.reshape(qa.shape[0], -1, qa.shape[-1]).mean(1)  # (B, C)
        qa = F.normalize(qa, dim=-1)
        if arm == "control":
            qa = qa.roll(1, dims=0)                              # shuffle pairing in batch
        terms.append((qp * qa).sum(-1).mean())                  # mean cos over batch
    if not terms:
        return None
    return -torch.stack(terms).mean()                           # maximize cos


def run(config, ckpt_tag, arm, lam, n_iters, lr, bs, out_ckpt, log_every=20):
    import yaml
    cfg = yaml.safe_load(open(config)); cfg["device"] = "cuda:0"
    rt = RGA._import_runtime()
    from mmcv import Config
    RGA._load_plugins(Config.fromfile(cfg["model_config"]))
    adapter = RGA.build_adapter(rt, cfg)
    model = RGA.load_model(rt, adapter, RGA._resolve_ckpt_path(cfg, ckpt_tag), cfg["device"], False)
    raw = model.module if hasattr(model, "module") else model
    raw.train()
    dl = rt["build_dataloader"](adapter, batch_size=bs, shuffle=True, seed=cfg["seed"])
    params = [p for p in raw.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    print(f"[ft] arm={arm} lambda={lam} lr={lr} bs={bs} iters={n_iters} params={len(params)}", flush=True)

    t0, it = time.time(), 0
    run_std, run_cpl = 0.0, 0.0
    while it < n_iters:
        for data in dl:
            if it >= n_iters:
                break
            out = adapter.forward_losses(model, data)
            std = sum(v for k, v in out.items() if "loss" in k and torch.is_tensor(v) and v.numel() == 1)
            Q = out.get("task_queries", {})
            cpl = coupling_loss(Q, arm)
            total = std + (lam * cpl if cpl is not None else 0.0)
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 25.0)
            opt.step()
            run_std += float(std.item()); run_cpl += float(cpl.item()) if cpl is not None else 0.0
            if (it + 1) % log_every == 0:
                print(f"[ft] {arm} it {it+1}/{n_iters}  std={run_std/log_every:.4f}  "
                      f"cpl={run_cpl/log_every:+.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
                run_std, run_cpl = 0.0, 0.0
            it += 1
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": raw.state_dict(), "meta": {"arm": arm, "lambda": lam, "iters": n_iters}}, out_ckpt)
    print(f"[ft] saved {out_ckpt}  ({(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="9ep")
    ap.add_argument("--arm", choices=["baseline", "coupling", "control"], default="baseline")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"work_dirs/exp/couple/{a.arm}.pth"
    run(a.config, a.ckpt, a.arm, a.lam, a.iters, a.lr, a.bs, out)
