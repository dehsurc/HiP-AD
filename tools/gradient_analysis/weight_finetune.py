"""D3 method test — budget-preserving per-aux LOSS REWEIGHTING fine-tune.

Reweights aux losses by per-aux multipliers w (det,map,motion); plan/ego stay at 1.
total = base_total + sum_aux (w_aux - 1) * L_aux
where base_total = the normal training loss (all components at weight 1).
Budget-preserving: sum(w_aux) is held = n_aux across arms (uniform = 1,1,1), so this
is a REDISTRIBUTION of a fixed aux budget, not "more aux" (clean vs Stage-A confound).

Arms (same start ckpt, same optimizer, only w differs):
  uniform   : 1,1,1                      (control == standard baseline)
  affinity  : from measured lookahead aux->plan transfer (map/motion up, det floor)
  shuffle   : the SAME weight multiset assigned to the WRONG aux (controls for
              "is it the specific assignment, or just any non-uniform weighting")
"""
import argparse, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import tools.run_gradient_analysis as RGA

AUX = ["det", "map", "motion"]


def run(config, ckpt_tag, arm, weights, n_iters, lr, bs, out_ckpt, wd, log_every=20):
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
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    w = {a: float(x) for a, x in zip(AUX, weights)}
    print(f"[wft] arm={arm} w={w} lr={lr} wd={wd} bs={bs} iters={n_iters}", flush=True)

    t0, it = time.time(), 0
    run_base, run_tot = 0.0, 0.0
    while it < n_iters:
        for data in dl:
            if it >= n_iters:
                break
            out = adapter.forward_losses(model, data)
            base = sum(v for k, v in out.items()
                       if "loss" in k and torch.is_tensor(v) and v.numel() == 1)
            total = base
            for a in AUX:
                La = adapter.split_losses(out, a)
                if La is not None and abs(w[a] - 1.0) > 1e-9:
                    total = total + (w[a] - 1.0) * La
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 25.0)
            opt.step()
            run_base += float(base.item()); run_tot += float(total.item())
            if (it + 1) % log_every == 0:
                print(f"[wft] {arm} it {it+1}/{n_iters}  base={run_base/log_every:.4f}  "
                      f"tot={run_tot/log_every:.4f}  ({(time.time()-t0)/60:.1f}m)", flush=True)
                run_base, run_tot = 0.0, 0.0
            it += 1
    Path(out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": raw.state_dict(),
                "meta": {"arm": arm, "weights": w, "iters": n_iters, "ckpt": ckpt_tag}}, out_ckpt)
    print(f"[wft] saved {out_ckpt}  ({(time.time()-t0)/60:.1f}m)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gradient_analysis_b2000.yaml")
    ap.add_argument("--ckpt", default="1ep")
    ap.add_argument("--arm", default="uniform")
    ap.add_argument("--w", default="1,1,1", help="det,map,motion multipliers")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--wd", type=float, default=0.001)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    weights = [float(x) for x in a.w.split(",")]
    out = a.out or f"work_dirs/exp/weight/{a.arm}.pth"
    run(a.config, a.ckpt, a.arm, weights, a.iters, a.lr, a.bs, out, a.wd)
