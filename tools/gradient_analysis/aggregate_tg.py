#!/usr/bin/env python
"""Aggregate Transfer-Gain eval logs into a CSV + markdown report.

Parses the planning / det / map metric lines that NuScenes3DDataset.evaluate
prints (see nuscenes_3d_dataset.py: ``L2: x.xxxx``, ``obj_box_col: x.xxx%``)
from each ``<variant>.log`` under --eval-dir, then computes

    TG(task) = P(tg_no_<task>) - P(tg_full)

for every planning metric P. Sign convention (lower-is-better metrics):
TG > 0  => removing the task HURT planning => the task was helping.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# metric name -> regex on the eval log (last occurrence wins)
PATTERNS = {
    "L2": re.compile(r"\bL2:\s*([0-9.]+)"),
    "obj_box_col_pct": re.compile(r"\bobj_box_col:\s*([0-9.]+)%"),
    "obj_col_pct": re.compile(r"\bobj_col:\s*([0-9.]+)%"),
    # side-effect (non-planning) references, scraped when present
    "det_mAP": re.compile(r"\bmAP:\s*([0-9.]+)"),
    "det_NDS": re.compile(r"\bNDS:\s*([0-9.]+)"),
}
PLAN_METRICS = ["L2", "obj_box_col_pct", "obj_col_pct"]


def parse_log(path: Path) -> dict:
    text = path.read_text(errors="ignore")
    row = {}
    for name, pat in PATTERNS.items():
        hits = pat.findall(text)
        if hits:
            row[name] = float(hits[-1])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    eval_dir, out_dir = Path(args.eval_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in sorted(eval_dir.glob("*.log")):
        row = parse_log(log)
        if not row:
            print(f"[aggregate_tg] WARN: no metrics parsed from {log}")
            continue
        rows.append({"variant": log.stem, **row})
    if not rows:
        raise SystemExit("[aggregate_tg] no eval logs parsed — nothing to do")

    df = pd.DataFrame(rows).set_index("variant")
    df.to_csv(out_dir / "tg_metrics_raw.csv")

    lines = ["# Transfer Gain (leave-one-aux-out) — planning verdict", ""]
    lines.append("## Raw metrics")
    lines.append(df.to_string())
    lines.append("")

    if "tg_full" in df.index:
        full = df.loc["tg_full"]
        tg_rows = []
        for v in df.index:
            if not v.startswith("tg_no_"):
                continue
            task = v.replace("tg_no_", "")
            rec = {"task": task}
            for m in PLAN_METRICS:
                if m in df.columns and pd.notna(df.loc[v].get(m)) and pd.notna(full.get(m)):
                    rec[f"TG_{m}"] = df.loc[v][m] - full[m]
            tg_rows.append(rec)
        if tg_rows:
            tg = pd.DataFrame(tg_rows).set_index("task")
            tg.to_csv(out_dir / "tg_summary.csv")
            lines.append("## TG = P(no_task) − P(tg_full)   (>0 ⇒ task was helping plan)")
            lines.append(tg.to_string())
            lines.append("")
        if "pretrain_ref" in df.index and "L2" in df.columns:
            drift = full["L2"] - df.loc["pretrain_ref"]["L2"]
            lines.append(
                f"## Drift check: tg_full L2 − pretrain_ref L2 = {drift:+.4f} "
                "(how much the fine-tune itself moves planning; TG differences "
                "should be read against this scale)"
            )
    else:
        lines.append("**WARN**: tg_full missing — TG deltas not computed.")

    (out_dir / "tg_report.md").write_text("\n".join(lines))
    print(f"[aggregate_tg] wrote {out_dir/'tg_metrics_raw.csv'}, "
          f"{out_dir/'tg_summary.csv'}, {out_dir/'tg_report.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
