#!/usr/bin/env python
"""Write a before/after weight manifest for the TG run (for later viz tooling).

The fine-tuned (AFTER) checkpoints are already saved by mmcv at
<TG_ROOT>/<variant>/iter_<FINAL_ITER>.pth; the BEFORE weights are the shared
starting checkpoint ($CKPT). This script just records a clean, machine-readable
pairing so a downstream visualization site can load (before, after_full,
after_<ablation>) per scene without hunting the work_dirs.

Driven by env vars set in tools/anal_tg.sh. No model deps (stdlib only).
"""
import json
import os
import re
from pathlib import Path

TG_ROOT = Path(os.environ["TG_ROOT"])
START_CKPT = os.environ["CKPT"]
CKPT_TAG = os.environ.get("CKPT_TAG", "")
LR = os.environ.get("LR", "config-default")
FINAL_ITER = os.environ["FINAL_ITER"]
VARIANTS = os.environ["VARIANTS"].split()
CFG_DIR = Path("projects/configs/stage2_tg")


def read_ablate_tasks(variant: str):
    """Parse `ablate_tasks=[...]` from the variant config (regex, no mmcv)."""
    cfg = CFG_DIR / f"{variant}.py"
    if not cfg.exists():
        return None
    m = re.search(r"ablate_tasks\s*=\s*\[([^\]]*)\]", cfg.read_text())
    if not m:
        return None
    inner = m.group(1).strip()
    return [t.strip().strip("'\"") for t in inner.split(",") if t.strip()]


manifest = {
    "ckpt_tag": CKPT_TAG,
    "lr": LR,
    "final_iter": int(FINAL_ITER),
    "before_ckpt": os.path.realpath(START_CKPT),  # shared starting weights
    "variants": {},
}
for v in VARIANTS:
    after = TG_ROOT / v / f"iter_{FINAL_ITER}.pth"
    manifest["variants"][v] = {
        "ablate_tasks": read_ablate_tasks(v),
        "after_ckpt": os.path.realpath(after) if after.exists() else None,
        "config": str(CFG_DIR / f"{v}.py"),
        "eval_log": str(TG_ROOT / "eval" / f"{v}.log"),
    }

out = TG_ROOT / "weights" / "manifest.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(manifest, indent=2))
n_after = sum(1 for x in manifest["variants"].values() if x["after_ckpt"])
print(f"[tg-weights] manifest -> {out}  "
      f"(before=1, after={n_after}/{len(VARIANTS)})")
