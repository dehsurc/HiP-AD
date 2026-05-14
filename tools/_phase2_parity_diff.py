"""Phase 2 T19 -- numeric parity gate for HiP-AD adapter path vs Phase 1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PHASE1 = Path("gradient_analysis_results_phase1")
PHASE2 = Path("gradient_analysis_results_phase2_hipad")

# Tolerances per file family.
COSINE_ATOL = 1e-6           # M1 conflict cells
NORM_RELTOL = 1e-5            # mean / shared norms
COUNT_EXACT = True            # n_total / n_valid / n_pseudo_shared exact
CI_ATOL = 1e-3                # bootstrap CI (resamples differ by RNG)


def main() -> int:
    failures: list[str] = []
    for p1 in sorted(PHASE1.rglob("*.csv")):
        rel = p1.relative_to(PHASE1)
        p2 = PHASE2 / rel
        if not p2.exists():
            failures.append(f"MISSING in phase2: {rel}")
            continue
        df1 = pd.read_csv(p1)
        df2 = pd.read_csv(p2)
        if list(df1.columns) != list(df2.columns):
            failures.append(f"COLUMN MISMATCH: {rel}")
            continue
        if len(df1) != len(df2):
            failures.append(f"ROW COUNT MISMATCH: {rel} ({len(df1)} vs {len(df2)})")
            continue
        for col in df1.columns:
            if not np.issubdtype(df1[col].dtype, np.number):
                if not df1[col].equals(df2[col]):
                    failures.append(f"STRING COL DIFF: {rel}::{col}")
                continue
            a = df1[col].to_numpy(dtype=np.float64)
            b = df2[col].to_numpy(dtype=np.float64)
            if "cos" in col or "antisym" in col:
                if not np.allclose(a, b, atol=COSINE_ATOL, equal_nan=True):
                    delta = np.nanmax(np.abs(a - b))
                    failures.append(f"COSINE DIFF: {rel}::{col} max|d|={delta:.3e}")
            elif "n_" in col:
                if not (a == b).all():
                    failures.append(f"COUNT DIFF: {rel}::{col}")
            elif "ci_" in col or "bootstrap" in col:
                if not np.allclose(a, b, atol=CI_ATOL, equal_nan=True):
                    failures.append(f"CI DIFF: {rel}::{col}")
            else:
                rel_diff = np.abs(a - b) / np.maximum(np.abs(a), 1e-12)
                if np.nanmax(rel_diff) > NORM_RELTOL:
                    failures.append(f"NORM RELTOL DIFF: {rel}::{col}")
    if failures:
        print("PARITY GATE FAILED:")
        for f in failures[:50]:
            print("  -", f)
        if len(failures) > 50:
            print(f"  ... ({len(failures) - 50} more)")
        return 1
    print("PARITY GATE PASS (all CSVs match within documented tolerances)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
