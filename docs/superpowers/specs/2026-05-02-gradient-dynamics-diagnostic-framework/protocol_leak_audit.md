# Protocol Leak Audit (Phase 2 T14)

**Date:** 2026-05-08
**Auditor:** Codex

## Findings

| ID | Where | Description | Action |
|---|---|---|---|
| L1 | `base.py` | `TemporalSnapshot` docstring named model-specific internals (`prev_bev`, `dn_metas`, instance-bank caches). The Protocol surface itself was generic, but the docs leaked implementation vocabulary. | FIXED-BASE — replaced with generic mutable-state wording. |
| L2 | `tests/gradient_analysis/adapters/test_protocol.py` | HiP-AD and VAD both use a top-level `projects` package, so a single pytest process can import the wrong plugin after the first adapter is loaded. | DOCUMENTED — conformance tests now clear `projects*` modules and run the real adapter only in its matching conda env. |
| L3 | `hipad.py` / HiP-AD snapshot test | HiP-AD round-trip needs RNG capture in the opaque snapshot payload because DN anchor generation is random outside a freeze scope. | DOCUMENTED — payload remains opaque; no Protocol signature change. |
| L4 | adapter tests | HiP-AD freeze/snapshot tolerance is `1e-3` and VAD snapshot restore is checked inside `freeze_stochastic_state()` because assignment/FlashAttention drift is larger than the original plan's idealized standalone `1e-4`/`1e-3` checks. | DOCUMENTED — contract remains “restore probe-relevant state”; test isolates that from matching drift. |
| L5 | `base.py` / adapters | No adapter-only method is required by the Protocol, no `NotImplementedError` remains, and method signatures match on both adapters. | DOCUMENTED — no base change needed. |

## Result

No model-specific type or parameter leaked into the `GradientAnalysisAdapter`
method signatures. The only base change was documentation cleanup.
