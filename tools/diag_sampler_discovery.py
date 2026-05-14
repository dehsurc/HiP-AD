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
    from tools.gradient_analysis.adapters.hipad import (
        _ModelStateSnapshot as ModelStateSnapshot,
        _bank_holders,
        _raw_model,
        _sampler_holders,
    )
    from tools.run_gradient_analysis import (
        _import_runtime,
        _resolve_ckpt_path,
        build_adapter,
        set_seeds,
    )

    with open("configs/gradient_analysis.yaml") as f:
        cfg_ana = yaml.safe_load(f)
    set_seeds(cfg_ana["seed"], False)

    rt = _import_runtime()
    adapter = build_adapter(rt, cfg_ana, "hipad")
    ckpt_path = _resolve_ckpt_path(cfg_ana, "1ep")
    model = adapter.build_model(ckpt=ckpt_path, device=cfg_ana["device"])

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
