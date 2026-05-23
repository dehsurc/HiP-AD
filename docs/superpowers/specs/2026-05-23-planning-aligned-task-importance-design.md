# Planning-Aligned Task Importance — Design Spec

**Date:** 2026-05-23
**Status:** Approved (brainstorming) → implementation plan next
**Related:** `direction.md` (research direction), `2026-05-02-gradient-dynamics-diagnostic-framework/`, `tools/gradient_analysis/`

---

## 1. Motivation (from `direction.md`)

The end goal is **Planning-Aligned Modular Distillation**: distill module-wise teacher
knowledge (det/map/motion) into the E2E model, but **weight each task's distillation by how
much that task helps *planning*, per scene** — because different scenes need different tasks
(some need object detection, some need map, some both).

`direction.md` proposes estimating task importance **indirectly via gradient alignment**:
> "planning loss와 같은 방향으로 task loss가 간다면 이는 중요한 task로 분류"

This spec defines the analysis that answers **item 3** of `direction.md`'s "1 step probe 분석을
통해 알고싶은 것":

1. ~~task 간 loss 변화율 분석~~ — already covered by `probe.py` (M3) + `plan_centric.build_plan_transfer_summary` (Part F).
2. planning sensitivity 분석 보완 — secondary (see §7); the existing query-based Part J is near-tautological.
3. **planning에게 도움을 주는 task가 무엇인지 어떻게 판단할지** — **primary deliverable of this spec.**

---

## 2. Background: where tasks interact in HiP-AD

Established by reading `projects/mmdet3d_plugin/models/sparse_onedecoder.py` and configs.

### 2.1 Task / query structure (E2 config)
- `task_select  = ["det", "map", "plan", "ego", "motion"]` — five tasks have losses/heads.
- `query_select = ["det", "map", "plan", "ego"]` — only four have **their own queries**
  (instance features). `plan` and `ego` are separate query modalities in E2 (stage-2).
- **`motion` has no query of its own.** It is a head built on top of `det`:
  ```python
  # sparse_onedecoder.py:983-987
  motion_anchor = self.get_motion_anchor(det_cls, det_anchor)
  motion_query  = motion_mode_query + (det_instance_feature + det_anchor_embed).unsqueeze(2)
  motion_cls, motion_reg = self.motion_refine[refine_i](motion_query)
  ```
  → at every shared decoder parameter, `motion`'s gradient flows **only through the `det`
  pathway**. This has consequences for interpreting motion importance (see §6.4).

For the analysis we treat `{det, map, motion}` as the aux tasks competing for "helps planning"
ranking, with `plan` as the target. `ego` is part of the planning side of the structured-mask
(§2.2) and is not ranked as an aux task — distillation targets in `direction.md` are det/map/motion.

### 2.2 `inter_gnn` = directional cross-task attention into planning (via structured mask)

The reference config for this analysis is **`projects/configs/experiments/E2_E1_stage2_18ep_new.py`** (the 18-epoch stage-2 run). Its `inter_graph_model` is `SeparateAttention`, **not** the stage-1 `InteractiveAttention`:

```python
# E2_E1_stage2_18ep_new.py:310
inter_graph_model = dict(
    type="SeparateAttention",
    query_select=query_select,
    separate_list=[["det", "map", "plan", "ego"]],   # ONE group of 4 query modalities
    decouple_list=[False],
    with_distance_attn_mask=True,
    with_structured_mask=True,                        # directional masking, see below
    attn=[dict(type="MultiheadFlashAttention", embed_dims=256, num_heads=8, dropout=0.1)],
)
# task_select  = ["det", "map", "plan", "ego", "motion"]
# query_select = ["det", "map", "plan", "ego"]       # motion has no query (rides on det)
```

Mechanically this is **joint self-attention** over the concatenation `[det, map, plan, ego]` queries — the `key is None` branch in `SeparateAttention.forward` (`separate_attn.py:72-130`). The directional cross-task semantics come from `with_structured_mask=True`, documented in-class (`separate_attn.py:103-106`):

> "planning queries access info from all tasks, but perception (det/map) does NOT attend to plan/ego. Block perception row × planning column with -inf so the symmetric self-attn matrix behaves like the figure's grey-cell Collaborative Attention Map."

Concretely:
- **plan / ego queries → attend to {det, map, plan, ego}** (read from all four).
- **det / map queries → attend to {det, map} only** (plan/ego columns masked out).
- Plus a learned distance-based attn modulation (`with_distance_attn_mask=True`, `distance_tau` is a small per-decoder `Linear(256→8)`; lives outside the inter_gnn group itself).

So `inter_gnn` *is* the directional information channel through which det/map information reaches planning queries. For item 3 — "which task helps planning" — this is the on-target shared parameter, strictly better than `neck` (a generic shared FPN with no task-interaction semantics).

### 2.3 `inter_gnn` structure & parameter count
- Wrapper `SeparateAttention` holds one `MultiheadFlashAttention` (since `separate_list` has
  length 1); `decouple_list=[False]` → no extra projection in the wrapper.
- `MultiheadFlashAttention` wraps local `FlashMHA(embed_dim=256, num_heads=8)`
  (`attention.py:102`), whose parameters are:

  | Parameter | Shape | Count |
  |-----------|-------|-------|
  | `in_proj_weight` (packed Wq/Wk/Wv) | 768 × 256 | 196,608 |
  | `in_proj_bias` | 768 | 768 |
  | `out_proj.weight` | 256 × 256 | 65,536 |
  | `out_proj.bias` | 256 | 256 |
  | **per instance** | | **263,168 (~0.26M)** |

- `FlashAttention` inner module is parameter-free.
- `operation_order` = `single_frame_layer × 1 + temporal_frame_layer × 5` (`num_decoder=6`),
  each frame layer contains exactly one `inter_gnn` op → **6 independent instances**
  `dec0_inter_gnn_0 … dec5_inter_gnn_0`, total **6 × 263,168 = 1,579,008 (~1.58M) params**.
- dtype fp16/cuda. Even though `inter_gnn` is *joint self-attention* over the four query
  modalities (key is None), `FlashMHA` still uses the packed `in_proj_weight` and splits it
  into Wq/Wk/Wv via `_in_projection_packed`. The structured mask shapes which queries
  attend to which (§2.2), not the parameter shape.

---

## 3. Goal & Scope

**Primary (item 3):** For each scene (batch) and each aux task `i ∈ {det, map, motion}`, measure
how much `i` helps planning at the `inter_gnn` layers, using two complementary signals, and
quantify **scene-dependence** (does the most-helpful task vary by scene?) and **agreement**
between the two signals.

**Measurement site:** `inter_gnn` (groups `dec{0..5}_inter_gnn_0`). Default = aggregate across the
six decoders; also report per-decoder.

**Out of scope:** true per-sample (batch_size=1) collection; online distillation weighting (this
spec produces the *evidence/analysis*, not the training-time mechanism).

---

## 4. Data source — targeted re-run (scoped to `inter_gnn`)

The existing v3 outputs do **not** contain `inter_gnn`: base conflict only emitted
`{neck, fc_before, fc_after}` (the adapter's `exact` filter keeps only flat-named groups), and
the v3 probe `target_layers` were `dec*_norm/ffn` only. So `inter_gnn` analysis requires a
re-run.

**Model config (adapter_config) — new file based on E2:** create
`projects/configs/experiments/E2_E1_stage2_18ep_grad_analysis.py` as a copy of
`E2_E1_stage2_18ep_new.py` so the analysis run is isolated from the training config and any
analysis-only tweaks (e.g. disabling augmentations, fixing test pipeline) live there. The
gradient-analysis adapter loads it via `--adapter-config`.

**Gradient-analysis YAML changes** (the run-time config consumed by `run_gradient_analysis.py`):
- `adapter_config`: point to the new file above.
- `layer_conflict.layers`: add `dec0_inter_gnn_0 … dec5_inter_gnn_0`.
- `probe.target_layers`: set to `dec0_inter_gnn_0 … dec5_inter_gnn_0`.
- `--modules M2,M3,M_PI` (M4/M5/M7 full are auto-skipped in per-layer probe mode).

**Produced inputs** (per checkpoint):
- `layer_conflict/conflict_<i>_plan_per_batch.csv` — `cos(g_i, g_plan)` per batch at each inter_gnn group.
- `layer_conflict/conflict_det_motion_per_batch.csv` — `cos(g_motion, g_det)` for the motion collinearity flag (§6.4).
- `probe/probe_per_batch.csv` — rows with `target_task=plan`, `layer=dec*_inter_gnn_0`: `delta`, `grad_dot`, `grad_norm`, `step_size`.

> Checkpoint paths / data location are wired later (data not present in this working copy yet).
> All analysis functions take explicit path arguments.

---

## 5. Module API — `tools/gradient_analysis/planning_importance.py`

Pure functions, testable with toy CSVs (no model env), mirroring `plan_centric.py` conventions.

### Loaders
- `load_plan_alignment(layer_conflict_dir, aux_tasks=("det","map","motion"), inter_gnn_groups=None)`
  → long df `[aux_task, batch_idx, group, cos]`. Reads `conflict_<i>_plan_per_batch.csv`; `plan` is
  always `task_b`, so `cos` is `cos(g_i, g_plan)`. Filters to inter_gnn groups (auto-discovers
  `*_inter_gnn_*` if `None`). Pseudo-shared rows (norm < EPS) → NaN.
- `load_plan_transfer(probe_csv, variant="normalized", steps=1, inter_gnn_layers=None)`
  → long df `[aux_task(=source), batch_idx, layer, delta, gain=-delta, grad_dot, grad_norm]`,
  filtered to `target_task == plan`, the chosen variant/steps, inter_gnn layers.
- `load_task_collinearity(layer_conflict_dir, ref="det")` → `[batch_idx, group, cos_motion_ref]`
  from `conflict_<ref>_motion_per_batch.csv` (used for the motion det-collinearity flag).

### Summaries
- `build_importance_summary(align_df, transfer_df, collinearity_df=None, per_decoder=False)`
  → one row per `aux_task` (× decoder if `per_decoder`): `align_score` (mean cos), `align_pos_rate`
  (P(cos>0)), `transfer_gain` (mean −Δplan), `helpful_rate` (P(gain>0)), bootstrap CIs for each,
  `n`, and for motion `motion_det_collinearity` (mean cos(g_motion,g_det)) +
  `flag_det_collinear` (True if ≥ 0.8).
- `build_scene_winner(metric_df, value_col, by="batch_idx")`
  → `(winner_per_batch[batch_idx, winner_task, top_value, margin], winner_distribution[aux_task,
  win_count, win_frac], normalized_entropy_scalar)`. Entropy ≈ 0 → one task always wins (not
  scene-dependent); ≈ 1 → tasks win equally (strongly scene-dependent). Computed for both the
  alignment metric and the transfer metric.
- `build_alignment_transfer_agreement(align_df, transfer_df)`
  → after averaging each to per-`(batch_idx, aux_task)` scalar (mean over inter_gnn
  layers/groups): `spearman` (rank corr across all (batch,task)), `per_batch_winner_agreement`
  (fraction of batches whose argmax-by-cos == argmax-by-gain), `sign_agreement` (P(sign(cos)==sign(gain))).

### Scope/join handling
- Alignment scope = inter_gnn groups; transfer scope = inter_gnn probe layers. The agreement
  join collapses both to `(batch_idx, aux_task)` by averaging over the six inter_gnn layers.

---

## 6. Methodology details

### 6.1 Alignment `A_i`
`A_i = cos(g_i, g_plan)` on inter_gnn params (from layer_conflict). `direction.md`'s literal
definition: positive = task pulls params the same direction planning wants = helpful. Also
available as `grad_dot` (unnormalized) directly from the probe rows.

### 6.2 Transfer `T_i`
`T_i = -ΔL_plan` after a 1-step virtual update in source `i`'s gradient direction at inter_gnn
(probe `target=plan`). Positive = stepping toward `i` reduces planning loss (causal, but
step-size/curvature dependent).

### 6.3 Scene-dependence (core item-3 evidence)
Per batch, rank `{det, map, motion}` by `A_i` (and separately by `T_i`); the argmax is that
scene's "most planning-helpful task". The **distribution of winners across batches** plus its
**normalized entropy** quantifies the central `direction.md` claim that the important task
varies by scene.

### 6.4 motion handling (det-collinearity)
motion has no query (§2.1); its inter_gnn gradient flows through `det`. We still measure
`A_motion`/`T_motion` like the others, **and** report `cos(g_motion, g_det)` at inter_gnn. If
this collinearity is high (≥ 0.8), `flag_det_collinear=True` signals that motion's "importance"
is largely the det pathway re-weighted by the motion head — interpret accordingly.

### 6.5 Agreement (validates the cheap proxy)
A Planning-Aligned distillation would weight tasks online using the cheap cosine alignment.
§5's agreement metrics check whether cosine alignment actually tracks the causal probe transfer
(rank correlation, per-scene winner agreement, sign agreement). High agreement justifies using
cosine as the online importance signal.

---

## 7. Secondary — item 2 (planning sensitivity, lightweight)

The query-based Part J (`query_sensitivity.py`) is near-tautological: `||∂L_plan/∂Q_task||` is
nonzero only for plan's own queries because the hook captures task queries at `refine[-1]`,
after the task branches split. Without a re-run at a pre-split hook we cannot fix it.

Replacement proxy from the same probe data: **planning layer-sensitivity** = mean
`grad_norm` for `source=plan` per inter_gnn decoder → which inter_gnn layer planning depends on
most. Small addition (`build_planning_layer_sensitivity`), reported in the notebook.

---

## 8. Notebook section (in `phase2_v3_analysis_visualization.ipynb`)

A new "Planning-aligned task importance" section:
- Bars: per-task `align_score` and `transfer_gain` with bootstrap CIs.
- Winner distribution bar + normalized-entropy annotation (alignment and transfer side by side).
- Scatter: per-`(batch,task)` cos vs gain with Spearman annotation.
- Per-batch winner stacked-area across `batch_idx` (visual scene-dependence).
- motion det-collinearity panel.
- (secondary) planning layer-sensitivity bar across the six inter_gnn decoders.

All paths built from a configurable `RESULTS_ROOT` + checkpoint list (placeholders until data is wired).

---

## 9. Pipeline wiring & tests

### Runner
- New module code **`M_PI`** in `run_gradient_analysis.py`: after M3, read the checkpoint's
  `layer_conflict/` + `probe/` and write `planning_importance/{importance_summary,
  scene_winner_alignment, scene_winner_transfer, agreement, planning_layer_sensitivity}.csv`.
- Cross-checkpoint dynamics of the importance scores can reuse the existing M6 pattern (optional, future).

### Tests — `tests/gradient_analysis/test_planning_importance.py`
Toy CSVs (no model env):
- alignment loader maps `conflict_<i>_plan` filename → aux_task and reads `cos` correctly; pseudo-shared → NaN.
- transfer loader filters `target=plan`, computes `gain=-delta` with correct sign.
- `build_scene_winner`: constructed case where two tasks alternate winning → entropy ≈ 1; one task always wins → entropy = 0; `margin` correct.
- `build_alignment_transfer_agreement`: monotonic case → spearman ≈ +1; anti-monotonic → ≈ −1; sign-agreement count correct.
- motion collinearity flag: cos(g_motion,g_det) ≥ 0.8 → `flag_det_collinear=True`.

---

## 10. Open questions / future work
- **Per-scene granularity:** batch_size=12 mixes 12 scenes per batch_idx; the scene-dependence
  story is batch-level. A future bs=1 supplementary run would give true per-scene importance.
- **Distillation use:** turning the validated alignment signal into an online per-scene
  distillation weight is the follow-on project (separate spec).
- **Structured-mask asymmetry interpretation:** plan/ego attend to det/map but not vice versa
  (`with_structured_mask=True`, §2.2). The cosine alignment `cos(g_i, g_plan)` at inter_gnn
  params is still well-defined (gradient flows back through both rows and columns), but the
  *causal* contribution may be dominated by the plan/ego→det/map attention rows. Document this
  asymmetry in the notebook interpretation; a per-row gradient decomposition is a possible
  future refinement.
