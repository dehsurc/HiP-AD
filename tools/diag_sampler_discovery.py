"""Quick check: does ModelStateSnapshot._sampler_holders actually find the
BaseTargetWithDenoising instances inside a built SparseDetector?"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> int:
    import yaml
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet.models import build_detector

    from tools.gradient_analysis.compat import (
        apply_use_reentrant_false, apply_index_put_fix,
    )
    from tools.gradient_analysis.temporal_state import (
        _sampler_holders, _bank_holders, ModelStateSnapshot, _raw_model,
    )
    from tools.run_gradient_analysis import _load_plugins, set_seeds

    apply_use_reentrant_false()
    apply_index_put_fix()

    with open("configs/gradient_analysis.yaml") as f:
        cfg_ana = yaml.safe_load(f)
    set_seeds(cfg_ana["seed"], False)

    model_cfg = Config.fromfile(cfg_ana["model_config"])
    _load_plugins(model_cfg)
    model = build_detector(model_cfg.model,
                           train_cfg=model_cfg.get("train_cfg"),
                           test_cfg=model_cfg.get("test_cfg"))
    model.init_weights()
    ckpt_path = Path(cfg_ana["ckpt_root"]) / cfg_ana["checkpoints"]["1ep"]
    load_checkpoint(model, str(ckpt_path), map_location="cpu")
    device = cfg_ana["device"]
    device_id = int(device.split(":")[1]) if ":" in device else 0
    model = model.to(device)
    model = MMDataParallel(model, device_ids=[device_id])

    raw = _raw_model(model)
    print("== Walking module tree for *sampler* attributes ==")
    found = []
    for name, mod in raw.named_modules():
        for attr in list(mod.__dict__.keys()):
            if "sampler" in attr:
                val = getattr(mod, attr, None)
                cn = type(val).__name__ if val is not None else "(None)"
                found.append((name, attr, cn, id(val)))
                print(f"  {name}.{attr} = {cn}  (id={id(val)})")

    print(f"\nfound {len(found)} sampler-named attributes")

    print("\n== _sampler_holders() returns ==")
    holders = _sampler_holders(model)
    for h in holders:
        print(f"  {type(h).__name__}  (id={id(h)})  dn_metas={getattr(h, 'dn_metas', '<no attr>')!r}")
    print(f"\n_sampler_holders found {len(holders)} BaseTargetWithDenoising instances")

    print("\n== _bank_holders() returns ==")
    banks = _bank_holders(model)
    for b in banks:
        print(f"  {type(b).__name__}  (id={id(b)})")
    print(f"\n_bank_holders found {len(banks)} InstanceBank-like objects")

    print("\n== ModelStateSnapshot capture summary ==")
    snap = ModelStateSnapshot(model)
    by_kind = {"run_step": 0, "bank": 0, "sampler": 0}
    for k in snap._initial:
        if k == "run_step":
            by_kind["run_step"] += 1
            continue
        oid, attr = k
        if any(id(b) == oid for b in banks):
            by_kind["bank"] += 1
        elif any(id(h) == oid for h in holders):
            by_kind["sampler"] += 1
    print(f"  total snapshot keys: {len(snap._initial)}")
    print(f"  by kind: {by_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
