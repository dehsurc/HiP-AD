# Phase 1 Gating Decision

**Status:** filled (2026-05-04)
**Pipeline run log:** `/tmp/phase1.log`
**Output root:** `gradient_analysis_results_phase1/`
**Spec:** [2026-05-02 design](../2026-05-02-gradient-dynamics-diagnostic-framework-design.md)
**Plan:** [Phase 1 plan](../../plans/2026-05-02-gradient-dynamics-diagnostic-framework-phase1.md)

Implementation status:
- 62 unit tests pass; smoke verified end-to-end before this run
- CRITICAL fix (sample_shuffle bypass → batch_permutation null) applied 2026-05-03
- HIGH fixes applied: Bonferroni `n_kinds` factor removed, n_pseudo_shared math (`~nonzero`) corrected, `conflict_ratio_ci_lo/hi` added, `run_alpha_sweep` switched to `dl_factory` pattern

---

## Diagnostic axes — observed values

### D1 — Null-baseline pass rate

Fraction of `(group, pair, kind)` cells where the observed cosine is distinguishable from the permutation null at rank-biserial |r| ≥ 0.1 *and* Bonferroni-corrected p ≤ 0.05.

| Checkpoint | Pass rate |
|---|---|
| 1ep | **8.75 %** |
| 3ep | **8.39 %** |
| 6ep | **7.77 %** |
| 18ep | **8.84 %** |

All four checkpoints sit near the same low-pass band (≈ 8 %). Below the 30 % "significant" gate. **D1 verdict: weak (close to null).**

### D2 — Magnitude imbalance (max/min of mean per-task norms)

Computed from `dynamics/norm_dynamics.csv`.

| Checkpoint | det | map | motion | plan | max/min ratio | Top / Bottom | Severity |
|---|---|---|---|---|---|---|---|
| 1ep | 1.996 | 1.212 | 0.238 | 1.289 | **8.39×** | det / motion | severe (> 5×) |
| 3ep | 2.013 | 1.046 | 0.401 | 1.701 | **5.02×** | det / motion | severe (just above 5×) |
| 6ep | 2.195 | 1.207 | 0.447 | 1.749 | **4.91×** | det / motion | moderate (just under 5×) |
| 18ep | 2.627 | 1.166 | 0.617 | 1.434 | **4.26×** | det / motion | moderate |

Imbalance **shrinks monotonically** as training progresses (8.4× → 4.3×). The dominant axis is det vs motion across every checkpoint.

**D2 verdict: severe at 1ep / 3ep, moderate at 6ep / 18ep.** Severe condition met at the early checkpoints — the gating rule for Pass B requires "D2 = severe" at any checkpoint, satisfied here.

### D3 — Cosine distribution shape

Counts per `shape_label` from `ckpt_*/distribution/distribution_report.csv`.

| Checkpoint | empty | unimodal-near-0 | unimodal-pos | unimodal-neg | bimodal | heavy-tail |
|---|---|---|---|---|---|---|
| 1ep | 266 | 247 | 33 | 0 | **0** | 14 |
| 3ep | 266 | 243 | 33 | 0 | **0** | 18 |
| 6ep | 266 | 240 | 24 | 1 | **0** | 29 |
| 18ep | 266 | 204 | 33 | 0 | **0** | 57 |

`empty` cells are pseudo-shared groups (one task's gradient is below EPS — `inter_gnn`, `fc_*`). The remaining cells are overwhelmingly `unimodal-near-0`. **Zero bimodal cells across all four checkpoints.** Heavy-tail count climbs over training (14 → 57) — the tails fatten, but no distribution splits into two modes.

**D3 verdict: no bimodal signal anywhere.** The distribution-shape axis does not contribute to gating.

### D4 — Temporal drift (per-task slope of mean norm vs epoch)

From `magnitude_dynamics/per_task_slopes.csv`.

| Task | Slope | R² | Drift verdict |
|---|---|---|---|
| **det** | +0.0383 | **0.989** | strong increasing |
| map | +0.0009 | 0.008 | stable |
| **motion** | +0.0190 | **0.863** | strong increasing |
| plan | −0.0030 | 0.011 | stable |

Two of four tasks (det, motion) clear the R² > 0.5 threshold convincingly; their gradient norms grow monotonically across the four-checkpoint trajectory. Map and plan are essentially flat. **D4 verdict: not stable — strong monotone drift on det and motion.**

Cross-reference to non-stationarity (CV of pairwise ratios, `magnitude_dynamics/non_stationarity_index.csv`):

| Pair | CV |
|---|---|
| map ↔ motion | **0.453** |
| motion ↔ plan | **0.385** |
| det ↔ motion | 0.329 |
| det ↔ plan | 0.204 |
| map ↔ plan | 0.186 |
| det ↔ map | 0.133 |

Every motion-involving pair has CV > 0.3 — motion's norm relative to other tasks is highly non-stationary, confirming the drift verdict.

---

## α-sensitivity sweep result

motion → motion `diag_violation_rate` from `ckpt_1ep/probe/alpha_sweep/sweep_summary.csv`:

| α | raw violation rate | normalized violation rate | mean Δloss (raw) | mean Δloss (normalized) |
|---|---|---|---|---|
| 1e-4 | 0.87 | 0.66 | +0.0142 | +0.0016 |
| 5e-4 | 1.00 | 0.89 | +0.0871 | +0.0123 |
| 1e-3 | 1.00 | 0.94 | +0.1918 | +0.0261 |
| 5e-3 | 1.00 | 1.00 | +0.7020 | +0.1396 |

Diag-violation rate stays **≥ 66 %** even at α = 10⁻⁴ (normalized variant — the lowest available). Reducing α did not eliminate the violations. Per spec § 3.5 / Risk R6, this **triggers a freeze-audit follow-up**: an unfrozen stochastic source still exists in the motion probe path beyond what `freeze_matching` and `forward_seed` already pin.

Headline α adopted for motion in main-body tables: **α = 10⁻⁴, normalized variant** (66 % violations — highest-quality available, but with the explicit caveat that the residual unfrozen source taints the motion-source affinity numbers across the board, not just at this α).

---

## Decision

**Branch: Pass B** ☑

### Rule application

| Branch | Required conditions | Met? |
|---|---|---|
| Pass A | D1 ≥ 30 % *and* (D2 severe *or* D3 bimodal) | ✗ — D1 = 8 % < 30 % |
| **Pass B** | **D1 < 30 % *and* D2 severe *and* at least one task R² > 0.5** | **✓ — D1 = 8 %, D2 severe at 1ep / 3ep, det R² = 0.99 and motion R² = 0.86** |
| Fail | D1 < 30 % *and* D2 < severe *and* D4 stable | ✗ — D2 IS severe |

### Rationale

The directional-conflict story (the headline framing of the original 2026-04-20 spec) does **not** survive Phase 1's signal test: 8 % of cells distinguish from noise, with no bimodal distributions. But the magnitude story is real: the det / motion norm ratio is 8.4× at 1ep, persists above 4× through 18 epochs, and both det and motion exhibit clean monotone growth (R² ≥ 0.86) over training. Map and plan are flat by comparison.

This matches the memory record from the 2026-03-25 debate ("진짜 병목은 magnitude imbalance") empirically rather than rhetorically: now we have a measured trajectory, not a snapshot.

The α-sweep further says the probe-based affinity numbers for motion sources cannot be trusted at face value until the freeze audit lands.

---

## Implications for Phase 2 / 3

Per spec § 5.5 Pass-B branching, Phase 3's desiderata matrix:

- **Drops D1 (conflict signal), D3 (cosine distribution), D7 (pseudo-shared ratio).** None contributed enough signal to justify retention.
- **Retains D2 (magnitude imbalance), D4 (temporal drift), D5 (layer concentration), D6 (sample-level vs batch averaged).** The pivot is from "what direction does the algorithm need to modify?" to "what magnitude / scheduling is the algorithm responsible for?".

Phase 3's gap-analysis table re-orients the algorithm-property axes:
- **P1 magnitude rescaling**: required (D2 severe at 1ep, persists).
- **P5 temporal scheduling**: required (D4 monotone growth on det / motion; CVs ≥ 0.3 on motion-involving pairs).
- **P2 direction modification, P7 bimodal-aware aggregation**: not required by Phase 1 evidence — the algorithm should not waste capacity on these.

The gap analysis pivots toward GradNorm-like and uncertainty-weighting families. PCGrad-style direction-modifying interventions cannot be motivated from this data; the desideratum table will reflect that.

Phase 2 (HiP-AD + VAD adapters): proceeds as scheduled. Confirming whether VAD also exhibits magnitude-dominated, conflict-weak signal is now the load-bearing question for the multi-model claim — not whether VAD has the same conflict patterns as HiP-AD.

---

## Open follow-ups (must address before Phase 2 / 3 narratives)

1. **R6 freeze audit (motion source) — RESOLVED 2026-05-04.** Three diagnostic experiments ruled out the obvious freeze-related causes:

   | Hypothesis | Experiment | Result |
   |---|---|---|
   | (A) fp16 numerical noise | re-ran α-sweep with `fp16: false` (`gradient_analysis_results_expA_fp32/`) | violation rates within ±5 % of fp16 — **rejected** |
   | (B) L1 + cumsum non-smoothness | re-ran with `loss_motion_reg = SmoothL1Loss` (`gradient_analysis_results_expB_smoothl1/`) | violation 0.66 → 0.64 at α = 10⁻⁴ normalized — **rejected** |
   | (C) step-too-large alone | added α = 10⁻⁵ to the sweep (`gradient_analysis_results_expC_alpha1e5/`) | violation drops 0.94 (10⁻³) → 0.66 (10⁻⁴) → **0.61 (10⁻⁵)**. Floor exists. Step size is partial cause, not full cause |

   **Conclusion:** motion is **probe-unfriendly by nature**, not a freeze bug. Supporting evidence:
   - motion gradient norm 0.238 (det/map's 1/8) — signal small relative to noise
   - `*→motion` helpful_ratio = 0.00 across every checkpoint — *no source's step ever decreased motion's loss*, indicating a near-flat / saddle-region loss landscape
   - α reduced 1000× (5×10⁻³ → 10⁻⁵) only halves the violation rate (1.00 → 0.61) — diminishing returns indicate the residual violation is **gradient-direction noise**, not step-size

   **Implication:** motion-source rows in Phase 1's probe affinity matrices and helpful_ratio tables carry inherent unreliability. They should be either
   - **excluded** from Phase 3 desiderata-matrix evidence rows for D5/D6 axes, *or*
   - kept but **explicitly tagged with the caveat "motion's loss landscape is near-flat at the probed checkpoints; affinity values are noise-dominated"**

   The reviewer's CRITICAL/HIGH findings (sample_shuffle bypass, Bonferroni double-count, n_pseudo_shared math) remain fixed in code; only the R6 motion-violation cell carries this forward caveat.

2. **Heavy-tail growth at late epochs.** D3 shows heavy-tail cell count climbing from 14 → 57 across 1ep → 18ep. Distribution shape is still unimodal-near-0 in label, but the kurtosis threshold is being crossed by an increasing number of cells. Worth a follow-up read of `distribution_report.csv` at 18ep to see *which* groups developed heavy tails — they may be the same groups where motion-pair gradients became most informative late in training.

3. **D2 severity drift.** Magnitude imbalance shrinks 8.4× → 4.3×. The Pass B branch was triggered by 1ep (severe), but at 18ep the model has half-resolved its own imbalance through training. Phase 3 must report the evolution explicitly so the desiderata claim "magnitude imbalance is the real bottleneck" is paired with "imbalance is partially self-correcting under standard training" — that nuance changes which P-properties are required (e.g. P5 temporal scheduling becomes more load-bearing because the magnitude problem isn't fixed-state).

4. **det-vs-map mid-decoder norm conflict.** Top conflict_ratio cells in the summary report concentrate on `dec4/dec5_ffn_0_pre_norm` and `dec4/dec5_norm_*` for det vs map (conflict_ratio 0.65–0.75, mean cos −0.025 to −0.039). These are the few cells that *did* survive D1's noise gate (small mean cos but consistent sign across 100 batches). Worth surfacing in Phase 3 even though the global D1 signal is weak — they are the most credible candidate for a layer-localized direction signal.
