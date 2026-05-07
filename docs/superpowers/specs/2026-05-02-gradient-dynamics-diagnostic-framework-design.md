# Gradient Dynamics Diagnostic Framework — Design Spec

**Status:** approved (2026-05-02)
**Owner:** yongjae
**Predecessor:** [2026-04-20 MTL Gradient Analysis](2026-04-20-mtl-gradient-analysis-design.md)
**Successor:** Novel-algorithm design spec — to be brainstormed after Phase 3 of this spec produces the desiderata matrix and gap analysis (intentionally not pre-scoped here so the algorithm responds to the data).

---

## 1. Goals

By the end of this spec the project state shall satisfy:

1. The same diagnostic pipeline runs on **both HiP-AD and VAD** through a single `GradientAnalysisAdapter` Protocol.
2. Random-baseline / distribution / bootstrap analyses are integrated such that **every reported metric carries a "noise vs signal" label**.
3. The current M7 antisymmetric-Frobenius NaN bug is resolved.
4. Per-task gradient magnitude **time series with drift slopes** are first-class outputs.
5. A **desiderata matrix** maps observed diagnostic patterns to the algorithm properties they require, with HiP-AD and VAD evidence in each cell.
6. A **gap analysis** identifies the union of properties no existing algorithm family satisfies — the quantitative motivation for a future novel algorithm.
7. A **Phase-1 gating decision** record explicitly states which scenario branch (Pass A / Pass B / Fail) the data put us in, with the supporting metrics.

## 2. Non-Goals

The following are explicitly **out of scope** for this spec (they belong to follow-up specs):

- Novel-algorithm prototype, implementation, or training experiments.
- A practitioner-style comparison table of existing MTL algorithms (PCGrad / GradNorm / CAGrad / etc.). The gap analysis is a "what is missing" table, not a benchmark.
- Adapters for UniAD, SparseDrive, MapTR, or other models beyond HiP-AD + VAD.
- Higher-order analyses (Hessian eigenvectors, NTK overlap, Fisher information).
- Causality regression linking gradient signals to final evaluation metrics.

## 3. Phase 1 — Diagnostic Reliability (six modules)

Phase 1 introduces no new training runs. All work is post-processing on the existing HiP-AD primary cache (4 ckpts × 100 batches) plus one motion-only α-sweep that costs ≈ 0.5 GPU-day.

### 3.1 #1 Random Baseline / Permutation Test — `tools/gradient_analysis/null_baseline.py`

For each (checkpoint, group, task pair) a null distribution of cosine similarity is produced by two independent permutation procedures:

- **Sample-level shuffle.** Within a batch, swap the per-sample gradient components between tasks at random; recompute cos. Repeats: 1000.
- **Sign-flip null.** Flip the sign of each gradient component independently with p = 0.5; recompute cos. Repeats: 1000.

A two-sample test (Mann-Whitney U) compares observed cos against each null. The output is `null_baseline.csv` with columns: observed mean, null mean, null 95 % interval, U statistic, p-value, rank-biserial effect size, and a `passes_noise_threshold` boolean (rank-biserial |r| ≥ 0.1 — Cohen's small-effect convention — *and* p ≤ 0.05 with Bonferroni correction across the (group, pair) cells in the same checkpoint).

This module satisfies the 2026-03-25 debate's blocking objection: until every reported cos is paired with such a label, no downstream framing is justifiable.

### 3.2 #2 Distribution Diagnostics — `tools/gradient_analysis/distribution.py`

Per (group, pair) distributions of cos across the 100 batches are characterised by:

- KDE plot (Gaussian kernel, Silverman bandwidth).
- Hartigan dip test for unimodality (`diptest`).
- Tail descriptors: kurtosis, percentile bundle (1, 5, 95, 99 %).
- A `shape_label` ∈ {`unimodal-near-0`, `unimodal-pos`, `unimodal-neg`, `bimodal`, `heavy-tail`}.

Cells flagged `bimodal` carry a `mean_is_misleading` flag downstream so summary tables and figures suppress mean-only reporting in favour of the modes.

### 3.3 #3 Bootstrap CI — `tools/gradient_analysis/bootstrap.py`

Every mean / median / conflict-ratio in `conflict_<a>_<b>_summary.csv` and the probe / gradnorm summaries is augmented with a BCa bootstrap 95 % CI (n_resamples = 2000). The CI is also overlaid as a band on every cross-checkpoint time-series plot. CIs that overlap across checkpoints are flagged so readers do not over-interpret them as drift.

### 3.4 #4 Shared-Parameter Validity Hardening (B1 + B2 + B3 + B4)

The current `collector.py` concatenates every parameter in a shared group regardless of whether each task's gradient actually flows through it; unreachable parameters appear as zeros via `allow_unused=True`. Combined with the unconditional NaN guard in `cosine_similarity`, this pushes the conflict-ratio statistic systematically toward zero whenever a "shared" group is in fact one-task-only.

The hardening pass:

- **B1** — `collector.py` records a per-parameter, per-task non-zero mask alongside the existing concatenated tensor (one bit per parameter, written as a v2 cache field). For pair (a, b) analysis, sub-buckets are computed on demand by intersecting the two masks; new group keys take the form `<group>__<a>_<b>`.
- **B2** — `conflict.summarize_pair` returns four counts per group: `n_total`, `n_valid`, `n_pseudo_shared` (one side below `EPS`), `n_nan`. `conflict_ratio` is recomputed against `n_valid`. The legacy ratio is preserved as `conflict_ratio_legacy` for one cycle so prior plots reproduce.
- **B3** — Summary CSVs gain `pseudo_shared_ratio = n_pseudo_shared / n_total` and a derived `is_pseudo_shared_group` (≥ 0.5).
- **B4** — `summary.py` partitions the markdown digest: pseudo-shared groups move to a "Not a shared parameter group" appendix; the main "Highest / lowest conflict_ratio" rankings exclude them.

The M7 antisymmetric-Frobenius NaN cascade is verified to disappear after this pass; if it persists, a dedicated debug step is allowed (≤ 0.5 day).

### 3.5 #9 Probe Reliability — Variance Bands and α-Sensitivity

The current `probe_per_batch.csv` carries a single ΔL per batch with no spread information. Two additions:

- **Variance bands.** A bootstrap 95 % CI is added to each affinity-matrix cell. Cells whose CI crosses zero are flagged `sign_uncertain` and the markdown summary reports them separately.
- **α-sensitivity sweep, motion-only.** The motion source case is rerun across α ∈ {1 × 10⁻⁴, 5 × 10⁻⁴, 1 × 10⁻³, 5 × 10⁻³} on the same 100-batch slice; the diag > 0 violation rate is plotted as a function of α. Outcomes branch as follows:
    - Violations vanish as α → 10⁻⁴ → step-too-large was the cause; α = 10⁻⁴ is adopted for every motion-source affinity table reported in this spec's main body (raw and normalized variants alike).
    - Violations persist → an unfrozen stochastic source remains; a separate freeze audit is triggered (≤ 0.5-day debug step, allowed by R6 mitigation).

The `raw` variant for motion is demoted to an appendix; main-body affinity matrices use `normalized` only.

### 3.6 #10 Magnitude Non-Stationarity — `tools/gradient_analysis/magnitude_dynamics.py`

The current per-task norm CSV is a single mean per checkpoint. This module adds:

- Per-epoch slope (linear regression of mean norm against epoch index).
- Pairwise norm-ratio matrix evolution: a `(task × task × epoch)` array.
- A non-stationarity index: coefficient of variation of each pairwise ratio across epochs.

The deliverables are `magnitude_dynamics.csv` and a heatmap evolution figure (one panel per epoch) — the artefacts that turn the existing 7.5 × motion / map ratio into something one can report as a *trajectory* rather than a snapshot.

### 3.7 Phase 1 cost summary

| Module | Code | Extra forwards |
|---|---|---|
| #1 null baseline | ≈ 150 LoC | 0 |
| #2 distribution | ≈ 100 LoC | 0 |
| #3 bootstrap | ≈ 80 LoC | 0 |
| #4 hardening | ≈ 230 LoC | 0 |
| #9 probe variance + α | ≈ 80 LoC | motion × 4 α × 4 ckpts × 100 batches ≈ 0.5 GPU-day |
| #10 magnitude dynamics | ≈ 100 LoC | 0 |

External dependency additions: `scipy`, `diptest`.

## 4. Phase 2 — Adapter Abstraction + HiP-AD / VAD

### 4.1 Adapter Protocol — `tools/gradient_analysis/adapters/base.py`

Nine abstract operations capture every model-specific behaviour the M1–M8 modules currently inline against HiP-AD:

```python
class GradientAnalysisAdapter(Protocol):
    @property
    def tasks(self) -> list[str]: ...

    def build_model(self, ckpt: Path, device: str) -> nn.Module: ...
    def build_dataloader(self, batch_size: int, seed: int) -> DataLoader: ...
    def forward_losses(self, model, data) -> dict[str, Tensor]: ...
    def split_losses(self, loss_dict, task: str) -> Tensor | None: ...
    def shared_param_groups(self, model, group_names: list[str]) -> dict[str, list[Parameter]]: ...
    def selective_eval_types(self) -> tuple[type, ...]: ...

    @contextmanager
    def freeze_stochastic_state(self) -> Iterator: ...

    def snapshot_temporal_state(self, model) -> TemporalSnapshot: ...
```

The Protocol is defined while writing **both** adapters in parallel; writing HiP-AD first risks the interface inheriting HiP-AD assumptions that VAD then forces a rewrite of.

### 4.2 HiP-AD Adapter — `adapters/hipad.py`

Migrates the existing `_import_hipad_utils()`, `FrozenMatching`, `ModelStateSnapshot`, and `_selective_eval` infrastructure into the new Protocol. The on-disk `BatchGradients` cache format is preserved so the existing 100-batch primary run remains usable — no re-collection is required for HiP-AD.

### 4.3 VAD Adapter — `adapters/vad.py`

VAD-specific concerns:

- **Tasks:** `det, map, motion, plan` (VAD folds ego-status into plan).
- **Loss-key conventions:** `loss_cls`, `loss_bbox`, `loss_pts_*`, `loss_traj`, `loss_plan_*` — different prefixes from HiP-AD; `split_losses` owns the mapping.
- **Shared parameter groups:** the BEVFormer encoder (self-attn, cross-attn, FFN, LayerNorm) and `temporal_self_attention`. The HiP-AD names (`gnn`, `inter_gnn`, `ffn`, `norm`, `fc_before`, `fc_after`) do not transfer; `shared_param_groups` translates a canonical name set into VAD module attributes.
- **Stochastic-state freeze:** patches the VAD detection / map / motion / plan target-assign methods. The exact class set is established during adapter implementation; structurally simpler than HiP-AD's six-class patch surface because VAD does not split detection into a decoupled `SparseBox3D` head.
- **Temporal state:** snapshot / restore of `prev_bev_queue` (BEV-feature ring buffer) — the analogue of HiP-AD's InstanceBank.
- **Model construction:** `init_detector` from mmdet3d (HiP-AD uses `mmdet.build_detector`).

### 4.4 Module refactor

Every M1–M8 module now consumes only an adapter; HiP-AD imports are removed. `tests/gradient_analysis/` gains a `MockAdapter` that exercises the Protocol against toy models, replacing the model-specific test fixtures.

### 4.5 VAD primary run

Once the adapter passes Protocol tests, four VAD checkpoints (suggested: epoch 1 / 10 / 30 / 60 from `/home/yongjae/e2e/VAD/data/ckpts/`) are analysed at the same `batch_size = 6, num_batches = 100` profile as HiP-AD. Estimated wall-clock: ≈ 1.5 GPU-days.

### 4.6 Phase 2 cost summary

| Item | Code | GPU |
|---|---|---|
| Protocol definition | ≈ 80 LoC | — |
| HiP-AD adapter (refactor) | ≈ 350 LoC | — |
| VAD adapter (new) | ≈ 350 LoC | — |
| M1–M8 adapter consumer refactor | ≈ 200 LoC | — |
| MockAdapter test scaffolding | ≈ 100 LoC | — |
| VAD primary run | — | ≈ 1.5 day |

## 5. Phase 3 — Desiderata Matrix + Gap Analysis + Gating

### 5.1 Diagnostic axes (matrix rows)

Seven empirical axes computed from Phase 1 + 2 outputs. Each axis carries a categorical grade so that two models can be placed in the same cell:

| Axis | Source | Grades |
|---|---|---|
| D1. Conflict signal | #1 random baseline | significant / weak / null |
| D2. Magnitude imbalance | #10 magnitude dynamics | severe (> 5 ×) / moderate (2 – 5 ×) / mild (< 2 ×) |
| D3. Cosine distribution shape | #2 distribution | unimodal-near-0 / unimodal-pos / unimodal-neg / bimodal / heavy-tail |
| D4. Temporal drift | #10 + M6 | increasing / decreasing / stable |
| D5. Layer concentration | M3 per-layer probe | shallow / mid / deep / spread |
| D6. Sample-level hidden by averaging | M2 per-batch vs supplementary bs = 1 | hidden / visible |
| D7. Pseudo-shared ratio | #4 B3 | low / moderate / high |

### 5.2 Algorithm-property requirements (matrix columns)

Each diagnostic axis implies a property an algorithm must possess to address it:

| Property | Required when |
|---|---|
| P1 magnitude rescaling | D2 = severe |
| P2 direction modification | D1 = significant *and* D3 carries negative mass |
| P3 per-pair treatment | D3 / D5 patterns disagree across pairs |
| P4 per-layer treatment | D5 = concentrated |
| P5 temporal scheduling | D4 ≠ stable |
| P6 sample-level treatment | D6 = hidden |
| P7 bimodal-aware aggregation | D3 = bimodal |

### 5.3 Cells

For each (Di, Pj) cell the report records:

- A `requirement` value ∈ {`required`, `not-required`, `conditional`}.
- One-line justification anchored to the diagnostic measurement.
- Direct hyperlinks to the source rows in the Phase 1 / Phase 2 CSVs.
- Both the HiP-AD and VAD value of Di so a single cell shows two evidence points.

### 5.4 Gap analysis — `gap_analysis.md`

Five algorithm families (PCGrad-like / GradNorm-like / CAGrad-MGDA-like / Aligned-MTL-like / loss-reweighting) are scored against P1–P7. The deliverable is a single table whose columns are the five families and whose rows are P1–P7; each cell ∈ {✓, ✗, △}. The argument that motivates a future novel algorithm is then the assertion that the union of P-properties required by the data is not contained in any single family's row of ✓s.

If the union *is* contained in some family — that is, an existing algorithm already satisfies every required property — the spec records this as a negative result and the next-spec scope is reconsidered.

### 5.5 Gating — `phase1_gating_decision.md`

Phase 1 outputs are inspected before Phase 2 code work begins; the document records which branch the data placed us in:

- **Pass A.** D1 = significant *and* either D2 or D3 shows a clear pattern. Phase 2 / 3 proceed as written.
- **Pass B.** D1 = null but D2 = severe and D4 ≠ stable. Phase 3 re-frames around magnitude + temporal axes. D1 (conflict signal), D3 (cosine distribution shape), and D7 (pseudo-shared ratio) are dropped because the case for direction-modifying interventions collapses without a non-null D1; D7's role was specifically to discount false D1 readings, which is moot here. The gap analysis pivots toward GradNorm and uncertainty-weighting families.
- **Fail.** All of D1, D2, D4 register null. The spec terminates with the conclusion that gradient conflict and magnitude imbalance are not meaningful training bottlenecks for either model — itself a publishable negative result, and the trigger to re-scope downstream work.

All three branches are written into the Phase 3 templates so that whichever scenario obtains, the report has a coherent shape.

### 5.6 Phase 3 cost summary

| Item | Code / time |
|---|---|
| Desiderata matrix automation | ≈ 150 LoC |
| Gap analysis authoring | ≈ 1 day human |
| Gating decision automation | ≈ 80 LoC |
| Paper-ready figure polish | ≈ 0.5 day human |

## 6. Timeline

Sequential single-engineer estimate; Phase 2 GPU work can overlap Phase 2 code work.

| Stage | Code | GPU | Cumulative |
|---|---|---|---|
| Phase 1 modules #1, #2, #3, #10 | 2 – 3 days | 0 | 3 days |
| Phase 1 #4 hardening | 1 – 2 days | 0 | 5 days |
| Phase 1 #9 α sweep | 0.5 day | 0.5 day | 6 days |
| Phase 1 gating decision | 0.5 day | 0 | 6.5 days |
| Phase 2 Protocol + both adapters | 4 – 5 days | 0 | 11 days |
| Phase 2 VAD primary run | (background) | 1.5 days | 12.5 days |
| Phase 2 module refactor | 1 – 2 days | 0 | 14 days |
| Phase 3 matrix + gap | 1.5 – 2 days | 0 | 16 days |

Calendar: ≈ 2.5 weeks (≈ 2 weeks if Phase 2 GPU runs alongside coding). Total GPU occupancy: ≈ 2 days.

## 7. File Layout

```
tools/gradient_analysis/
├── adapters/                            (new)
│   ├── __init__.py
│   ├── base.py                          (Protocol)
│   ├── hipad.py                         (refactor of existing utilities)
│   └── vad.py                           (new)
├── null_baseline.py                     (new — Phase 1 #1)
├── distribution.py                      (new — Phase 1 #2)
├── bootstrap.py                         (new — Phase 1 #3)
├── magnitude_dynamics.py                (new — Phase 1 #10)
├── desiderata.py                        (new — Phase 3)
├── collector.py                         (refactor — Phase 1 #4 B1)
├── conflict.py                          (refactor — Phase 1 #4 B2 / B3)
├── probe.py                             (refactor — Phase 1 #9)
├── summary.py                           (refactor — Phase 1 #4 B4)
└── (existing M5 / M6 / M7 / M8 / landscape — adapter-consumer changes only)

tests/gradient_analysis/
├── adapters/test_protocol.py            (new — MockAdapter)
├── test_null_baseline.py
├── test_distribution.py
├── test_bootstrap.py
└── (existing tests retained, refactored to MockAdapter where applicable)

docs/superpowers/specs/
└── 2026-05-02-gradient-dynamics-diagnostic-framework-design.md   (this spec)

docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/
├── desiderata_matrix.csv
├── desiderata_matrix.md
├── gap_analysis.md
└── phase1_gating_decision.md
```

## 8. Risks

| # | Risk | Probability | Impact | Mitigation |
|---|---|---|---|---|
| R1 | VAD config or dataset incompatibility surfaces only at first forward | medium | medium (1 – 2 days added to Phase 2) | 0.5-day VAD smoke-test spike before Phase 2 code work begins |
| R2 | Phase 1 gating flips to Pass B / Fail and Phase 2 / 3 framing must be swapped mid-spec | medium | medium | All three scenarios are authored into the spec ahead of time (Approach C safeguard) |
| R3 | Existing 100-batch cache breaks under #4 B1 refactor | medium | large (1.5-day re-collection) | Keep a v1 loader path; add the per-parameter non-zero mask as a v2 field |
| R4 | α-sensitivity sweep needs to extend beyond motion | low | small (0.5 day) | Default scope is motion only; extend if violation pattern reproduces elsewhere |
| R5 | Adapter Protocol leaks HiP-AD assumptions despite parallel authoring | medium | large | Define the Protocol *during* both adapter implementations, not after one |
| R6 | M7 antisymmetric NaN root cause is not eliminated by B1–B4 | low | small | A focused debug step (≤ 0.5 day) is permitted after the hardening lands |
| R7 | Hartigan dip test lacks power at n = 100 | low | small | Optional extension to n_batch = 200 on HiP-AD (≈ 0.5 day) |

## 9. Success Criteria

The spec is considered complete only when every item below has explicit evidence:

1. Phase 1's six modules implemented; their tests pass; the existing HiP-AD report is regenerated against them.
2. The M7 antisymmetric Frobenius value is finite for every reported pair.
3. The adapter Protocol is defined; HiP-AD and VAD adapters both pass MockAdapter-driven Protocol tests.
4. The VAD primary run has completed and produced the same set of Phase 1 artefacts as HiP-AD.
5. `desiderata_matrix.md` is written; every cell carries a one-line justification and links to its source rows.
6. `gap_analysis.md` is written; the table of five algorithm families against P1–P7 is filled, and the union argument (or its negation) is explicit.
7. `phase1_gating_decision.md` records the chosen branch and the metrics that justified it.

## 10. Open Questions / Hooks for the Next Spec

- The novel algorithm prototype derives from the desiderata matrix and gap analysis produced here. The next spec is brainstormed *after* Phase 3 results are in hand — the algorithm design must respond to the data, not the inverse.
- Whether to extend coverage to UniAD or SparseDrive is deferred until a working novel algorithm exists; cross-model generality is the third spec's concern, not this one's.
- Causality regression linking gradient signals to evaluation metrics is intentionally postponed; it requires multiple training runs of the eventual novel algorithm, which is beyond Phase 3's scope.
