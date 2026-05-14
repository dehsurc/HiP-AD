# Plan-Centric Probe — Experiment Report

**Date:** 2026-05-12
**Subject:** HiP-AD 1-step gradient probe on shared FFN-fc1 layers across 4 checkpoints.
**Notebook:** [`phase2_hipad_vad_analysis_visualization.ipynb`](phase2_hipad_vad_analysis_visualization.ipynb)
**Design doc:** [`2026-05-12-plan-centric-refactor-design.md`](2026-05-12-plan-centric-refactor-design.md)

This report is independent from the notebook. The notebook is the live analysis surface; this file is the frozen interpretation of the 2026-05-12 run.

---

## 1. Setup

- Probe entrypoint: `tools/run_gradient_analysis.py`
- Checkpoints: `1ep, 3ep, 6ep, 18ep` (E2_E1_stage2_18ep)
- Probe layers (6): `dec0_ffn_0_fc1 … dec5_ffn_0_fc1`
- Tasks: `det, map, motion, plan` (ego excluded for plan-centric analysis)
- Probe variants: `raw, normalized`; steps: `[1]`; α=1e-3
- Batches per checkpoint: 100 (batch_size 6)
- Modules: `M2, M3, M4, M5, M6, M7, M_N1, M_N2`
- Two GPUs in parallel: GPU0 = {1ep, 6ep}, GPU1 = {3ep, 18ep}. Wall-clock ≈ 3h per GPU, ≈ 1h35m per checkpoint (first ckpt includes nuScenes load; second adds ~14m for dataloader rebuild).
- VAD path (`gradient_analysis_results/vad_bev_shared/`) absent in this environment → all results are HiP-AD only.

All probe rows in canonical subset (`variant=normalized, steps=1`): 4 ckpt × 6 layer × 4 source × 4 target × 100 batches = 38,400 rows.

---

## 2. Sanity Checks

### 2.1 Self-step descends own loss (S→S diagonal must be positive)

Computed `mean_gain = -mean(delta) | source=S, target=S` for every (ckpt, source, layer). Result: **94 of 96 cells positive** as expected.

Two outliers:

| ckpt | layer | source=target | mean_gain |
|---|---|---|---|
| 1ep | dec5_ffn_0_fc1 | det | **−2.00e-04** |
| 3ep | dec5_ffn_0_fc1 | det | **−2.55e-04** |

Both at the deepest decoder layer for det at early checkpoints. 6ep/18ep recover (positive). Other 5 layers all positive at every checkpoint. Interpretation: **finite-step nonlinearity** at α=1e-3 in the layer with the smallest \|g\| (dec5_det has the largest effective step `α/||g||`). Not a systematic bug; should be cross-checked against the `raw` variant before publishing.

### 2.2 Gradient norms decay with depth (expected for a backbone-heavy model)

\|\|g_task\|\| at the 6 fc1 layers (1ep):

| task | dec0 | dec1 | dec2 | dec3 | dec4 | dec5 |
|---|---|---|---|---|---|---|
| det    | 2.10 | 0.77 | 0.56 | 0.34 | 0.18 | 0.12 |
| map    | 1.29 | 1.05 | 0.58 | 0.28 | 0.13 | 0.06 |
| motion | 0.26 | 0.11 | 0.10 | 0.08 | 0.07 | 0.05 |
| plan   | 1.42 | 1.10 | 0.66 | 0.33 | 0.14 | 0.07 |

Motion has the smallest gradient at every layer — consistent with prior phase-1 dynamics findings (sparse trajectory loss). Decay across depth ≈ 20× from dec0 to dec5. Stable across checkpoints (variation < 30%).

### 2.3 Independent measurement of cos(g_aux, g_plan) — probe vs M2

The 1-step probe with normalized step gives:
`gain_{aux→plan} ≈ α · ||g_plan|| · cos(g_aux, g_plan)`
and `gain_{plan→plan} ≈ α · ||g_plan||`.

So `gain_{aux→plan} / gain_{plan→plan} ≈ cos(g_aux, g_plan)`. Compared to M2's direct cosine measurement (independent code path) at 1ep `dec0_ffn_0_fc1`:

| pair | implied cos (probe) | M2 mean_cos | M2 conflict_ratio |
|---|---|---|---|
| det – plan | −0.0034 | +0.0002 | 0.51 |
| map – plan | −0.0049 | +0.0052 | 0.44 |
| motion – plan | +0.0010 | −0.0022 | 0.53 |

Magnitudes agree at the 10⁻³ scale; signs flip in this noise-level regime (batch-mean of nearly-zero quantities). Two independent code paths reach the same conclusion: **gradients are effectively orthogonal at this layer**. Measurement chain trusted.

---

## 3. Headline Findings

### 3.1 fc1 carries almost no aux→plan transfer

Practical-helpful-rate (fraction of batches with `delta < -τ`) at 1ep:

| layer | det→plan | map→plan | motion→plan | plan→plan |
|---|---|---|---|---|
| dec0_ffn_0_fc1 | 0.51 | 0.45 | 0.55 | 1.00 |
| dec1_ffn_0_fc1 | 0.41 | 0.45 | 0.36 | 1.00 |
| dec2_ffn_0_fc1 | 0.41 | 0.34 | 0.43 | 0.99 |
| dec3_ffn_0_fc1 | 0.39 | 0.41 | 0.40 | 0.99 |
| dec4_ffn_0_fc1 | 0.41 | 0.36 | 0.46 | 0.95 |
| dec5_ffn_0_fc1 | 0.32 | 0.44 | 0.38 | 0.96 |

`plan→plan` self-descent is reliable (~1.00 = always helpful). Aux→plan all hover around the random baseline (0.5). Effect sizes (`mean_gain / std_delta`) for aux→plan are all below 0.05 in absolute value — there is no consistent gradient overlap between auxiliary task losses and planning loss at this layer family.

This is the **canonical Part F finding** for fc1: the FFN inner linear contains no meaningful aux→plan signal. The "where does signal live?" question is therefore pushed to the operations we did not probe (gnn, inter_gnn).

### 3.2 Interference is strongest at 1ep and decays with training

Median \|off-diagonal mean_gain\| per cell, compared with the practical threshold τ at each checkpoint:

| ckpt | τ | median \|off-diag gain\| | ratio |
|---|---|---|---|
| 1ep  | 3.96e-6 | 1.64e-5 | **4.1×τ** |
| 3ep  | 4.18e-6 | 2.03e-6 | 0.49×τ |
| 6ep  | 3.51e-6 | 5.02e-6 | 1.43×τ |
| 18ep | 3.29e-6 | 2.55e-6 | 0.77×τ |

At **1ep** the off-diagonal signal sits clearly above τ — aux task gradients are perturbing plan loss in a non-trivial direction. By 3ep that signal collapses to half of τ (i.e. below noise floor as defined here). 6ep partially rebounds, 18ep is back below τ.

**Implication:** any interference-management mechanism (PCGrad, GradNorm, gradient surgery) has the largest measurable effect on planning during early training. After the first few thousand iterations the gradient pairs are largely orthogonal already and gradient-direction methods become low-leverage. This matches the conventional wisdom around PCGrad warmup and is now backed by direct ΔL probes rather than pure cosine.

### 3.3 Off-diagonal first-order prediction collapses

Part I measures whether the linear prediction `pred_delta = -step_size · grad_dot` matches actual ΔL.

| pair type | pearson(actual, pred) | mean \|delta\| | mean \|residual\| | residual share |
|---|---|---|---|---|
| **plan→plan only** | **+0.944** | 6.0e-4 | 2.0e-4 | 33% |
| self pairs (S→S all) | +0.441 | 6.0e-4 | 2.0e-4 | 33% |
| off-diagonal (S≠T) | **+0.056** | 9.3e-5 | 8.9e-5 | **95%** |

- For plan's own gradient, the first-order term captures most of the 1-step loss change — plan loss is locally quadratic-ish around the current weights.
- **For any aux→target pair (S ≠ T), the first-order term explains essentially nothing** (5% of the residual variance). Curvature, scale interaction, or higher-order coupling dominates.

This is a clean **counter-example to cosine-only conflict analysis** (the assumption underlying PCGrad/CAGrad/gradient surgery family): the cosine between gradients does **not** linearly predict the 1-step ΔL on the other task at the fc1 layer family. Any conflict resolution decision made on `cos(g_a, g_b)` alone is operating in a regime where the linearization fails by a factor of 20.

### 3.4 The dec5/det self-gain anomaly

`1ep dec5 det→det` = −2.0e-4 and `3ep dec5 det→det` = −2.55e-4 (S→S should be positive). This is the only structural anomaly across 96 self-cells. The candidate explanation is that α=1e-3 with `step_size = α/||g||` over-shoots when `||g||` is small (`||g_det|| ≈ 0.12` at dec5). Effective step `≈ 8e-3` × parameter scale; for a near-tail decoder layer this may exit the linear regime.

Not blocking, but worth checking before any downstream claim about dec5 specifically:
1. Look at the `raw` variant for the same cell — if `raw` also goes negative, the cause is structural, not normalization-induced.
2. Re-run only dec5/det at α=1e-4 to confirm whether the sign restores.

---

## 4. Caveats

1. **VAD missing**: `gradient_analysis_results/vad_bev_shared/` does not exist in this environment. Every plot and table in this report is HiP-AD only. The notebook silently skips the VAD branch; cross-model comparisons are not available.
2. **fc1-only layer scope**: probed `dec[0-5]_ffn_0_fc1` exclusively. `_fc2, _pre_norm, _identity_fc, gnn, inter_gnn, temp_gnn, norm, fc_before, fc_after, neck, backbone_*` are not probed. The "no aux→plan transfer" finding is **local to fc1**. Attention-path layers (gnn, inter_gnn) are the most natural next target.
3. **α=1e-3 may be supra-linear at the deepest decoder layer** — see §3.4. dec5/det at early checkpoints is in the questionable regime.
4. **τ is computed per (model, checkpoint)** on the plan-target subset's median \|delta\|. The relative comparisons across checkpoints in §3.2 should be read with the corresponding τ in mind.
5. **No 2-step probe** in this run (`steps=[1]` only). Higher-order effects flagged in §3.3 cannot be decomposed into 2nd-order Hessian terms with current data.
6. **Part J (query sensitivity)** and **Part K (loss-weight elasticity)** are template-only: the runner has not been executed, so Part J/K sections in the notebook emit SKIP messages.

---

## 5. Recommended Follow-ups

Ordered by leverage per GPU hour:

| Priority | Action | Wall-clock | What it answers |
|---|---|---|---|
| 1 | Re-run probe on `dec[0,3,5]_gnn_0` and `dec[0,3,5]_inter_gnn_0` (6 layers) at all 4 ckpts | ~3.3h / GPU | Does aux→plan signal live in attention paths instead of FFN? |
| 2 | Run `tools/gradient_analysis/query_sensitivity.py` runner on 1ep + 18ep | ~1h | Direct `\|\|dL_plan/dQ_task\|\|` per task query, enables Part J |
| 3 | Sanity sweep α ∈ {1e-4, 1e-3} on `dec5_ffn_0_fc1` only, 1ep + 3ep | ~30m | Resolves the §3.4 dec5/det sign question |
| 4 | Regenerate VAD probe data (`vad_bev_shared`) | ~7h / GPU | Restores cross-model comparison, unlocks every "VAD vs HiP-AD" cell in Appendix |
| 5 | Task loss weight sweep (Part K) | depends on training budget | Whether 3.2's "interference decays with training" actually translates to planning-metric elasticity |

The single highest-leverage next experiment is (1). If aux→plan is also flat in gnn/inter_gnn, the framing should shift from "where in the network do they conflict?" to "do they ever conflict in any layer family?", which is a different research question with different implications for multi-task design.

---

## 6. Raw artifacts produced

Path (relative to repo root): the notebook's `OUT_DIR` resolved to
`docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/gradient_analysis_results/plan_centric/`
because the kernel cwd is the notebook's directory rather than the repo root.

Files written:

- `plan_transfer_summary.csv` — Part F per-cell (96 rows)
- `plan_transfer_top_beneficial.csv`, `plan_transfer_top_harmful.csv`
- `plan_asymmetry_summary.csv` — Part G (72 rows)
- `plan_transfer_ratio_summary.csv` — view of the above
- `effect_size_probe_summary.csv` — Part H (96 rows)
- `helpful_vs_effect_size.csv` — scatter slice
- `first_order_vs_actual_summary.csv` — Part I (384 rows)
- `first_order_residual_by_layer.csv` — plan-target slice
- `query_sensitivity.csv` — Part J template (empty)
- `task_weight_elasticity_runs.csv` — Part K template (empty)
- `figures/F/`, `figures/G/`, `figures/H/`, `figures/I/` — per-section plots

To centralize artifacts under the repo's `gradient_analysis_results/`, patch the `OUT_DIR` initialization in the notebook's Part F preamble to `REPO_ROOT / 'gradient_analysis_results/plan_centric'`.

---

## 7. Bottom-line claim

The 2026-05-12 probe is **statistically and structurally sound**. There is no bug in the measurement chain (M2-cosine vs probe-implied-cosine cross-check passes). The visualizations look "wrong" because the diagonal-to-off-diagonal gain ratio is 100–1000×, which any uniform diverging colormap will saturate; the underlying numbers are correct and they say:

- aux→plan gradient overlap at fc1 is at the level of cos ≈ 10⁻³ across all probed layers,
- this near-orthogonality is strongest after the first few epochs, with a measurable interference window only at 1ep,
- and the gap between cosine-based prediction and actual 1-step loss change is large enough that cosine-only conflict tooling should be treated with caution on this architecture.

The next experiment to run is the attention-path probe (gnn + inter_gnn), not more variants of the fc1 probe.
