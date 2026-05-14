# Plan-Centric Refactor — Design

Date: 2026-05-12
Notebook in scope: `phase2_hipad_vad_analysis_visualization.ipynb`
Primary subject: 1-step supervised probe data already collected at
`gradient_analysis_results/{hipad_layer,vad_bev_shared}/ckpt_*/probe/probe_per_batch.csv`.

## 1. Goal

Restructure the phase-2 analysis from "global gradient conflict" framing to
"does each task loss update actually help planning, and where?". Existing
global conflict / null / distribution / affinity work is preserved but
demoted to `Appendix. Diagnostic background`.

Scope is intentionally bounded:
- Only the standard supervised 1-step probe (`variant="normalized"`, `steps=1`).
- Drop `ego` from main analysis. Diagnostic appendix keeps existing behavior.
- Code only. Probe re-run on HiP-AD checkpoints is the user's responsibility.
- Part I & J also touch experiment code (probe.py, hipad adapter, new runner).
- Part J is HiP-AD only. VAD path emits SKIP message.
- Part K stays template + analysis only; no sweep runner.

## 2. Architecture

```
tools/gradient_analysis/
├── plan_centric.py           [NEW]   compute helpers + summary builders
├── query_sensitivity.py      [NEW]   HiP-AD planner query grad dumper
├── probe.py                  [EDIT]  add grad_dot, step_size; emit on 1-step path
└── adapters/hipad.py         [EDIT]  get_task_queries(model, fwd) -> dict

tests/gradient_analysis/
├── test_plan_centric.py      [NEW]   helpers + summary builders
└── test_probe_grad_dot.py    [NEW]   probe.py regression on new columns

docs/superpowers/specs/2026-05-02-gradient-dynamics-diagnostic-framework/
├── phase2_hipad_vad_analysis_visualization.ipynb   [EDIT]
└── 2026-05-12-plan-centric-refactor-design.md      [THIS]
```

Compute helpers and summary builders live in `plan_centric.py`. Plotting and
narration stay inside the notebook for fast iteration. The notebook becomes
a thin orchestrator that imports builders and calls them once per section.

Output CSV root: `gradient_analysis_results/plan_centric/`. Each builder
writes a single long-form CSV keyed by `(model, checkpoint, ...)`, not
per-model directories. Figures are saved to
`gradient_analysis_results/plan_centric/figures/<part>/`.

## 3. Notebook structure after refactor

```
(0)  Header / concept note               [unchanged + 1 line on demotion]
(1)  Data loading + utilities             [extended: imports + standardize]
Part F  Plan-centric directed transfer    [MAIN]
Part G  Directed asymmetry around plan    [MAIN]
Part H  Effect-size-aware 1-step probe    [MAIN]
Part I  First-order vs actual             [MAIN, SKIP fallback]
Part J  Planning sensitivity / query      [MAIN, HiP-AD only, SKIP fallback]
Part K  Task loss weight elasticity       [MAIN, template only, SKIP fallback]
Appendix.A  HiP-AD diagnostic             [demoted]
Appendix.B  VAD diagnostic                [demoted]
Appendix.C  Task-pair helpful matrix      [demoted, header marked Appendix only]
Appendix.D  Distribution / correlation    [demoted]
Appendix.E  Affinity / heavy-tail / etc.  [demoted]
```

Existing cells in Part A–E keep their bodies. Only their headers are demoted
and the cell order is moved past Part F–K. Section "0. 처음 보는 사람을 위한
개념 정리" is updated with one paragraph noting that plan-centric analysis
is now the main subject.

## 4. Data convention

Source data: long-format `probe_per_batch.csv` with columns
```
batch_idx, source_task, target_task, steps, variant, layer,
grad_norm, baseline_loss, stepped_loss, delta, rel_delta
```

Spec text uses `loss_before` / `loss_after`. `standardize_probe_df` adds
those as aliases of `baseline_loss` / `stepped_loss` and computes:

- `gain = -delta`
- `delta_rel = (loss_after - loss_before) / (|loss_before| + EPS)` if both
  exist; otherwise falls back to existing `rel_delta`.
- `gain_rel = -delta_rel`.

Sign convention: `delta < 0` means target loss decreased; `gain > 0` means
the source step helped the target task.

Canonical subset for all main sections:
```python
base = get_probe_base(probe, variant="normalized", steps=1)
```

Practical threshold `tau` is computed **per (model, checkpoint)** on the
plan-target subset of `delta`:
```python
tau = max(min_value, 0.1 * median(|delta|))
```
The same `tau` is broadcast back via group-transform when filtering
practical-helpful / large-harm rates.

`ego` is excluded from main builders by filtering `source_task` and
`target_task` to `{det, map, motion, plan}`.

## 5. plan_centric.py public surface

```python
EPS = 1e-8

# foundations
def ensure_columns(df, required, df_name) -> bool
def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=0)
def practical_threshold(series, ratio=0.1, min_value=1e-8) -> float
def standardize_probe_df(probe) -> pd.DataFrame
def get_probe_base(probe, variant="normalized", steps=1) -> pd.DataFrame

# Part F
def build_plan_transfer_summary(base) -> pd.DataFrame
def top_beneficial(summary, k=10, by="mean_gain") -> pd.DataFrame
def top_harmful(summary, k=10, by="mean_gain") -> pd.DataFrame

# Part G
def build_asymmetry_summary(base) -> pd.DataFrame
def interpret_asymmetry_row(row) -> str

# Part H
def build_effect_size_summary(base, target="plan") -> pd.DataFrame

# Part I
def detect_first_order_columns(probe) -> dict | None
def build_first_order_residual_summary(base, cols) -> pd.DataFrame

# Part J
def load_or_template_query_sensitivity(path) -> tuple[pd.DataFrame | None, str]
def build_query_sensitivity_summary(df) -> pd.DataFrame

# Part K
def load_or_template_elasticity(path) -> tuple[pd.DataFrame | None, str]
def build_elasticity_summary(df) -> pd.DataFrame
def planning_safe_weight_range(elast_summary, tol=0.01) -> pd.DataFrame
```

## 6. Part F — plan-centric directed transfer

Group keys: `["model","checkpoint","checkpoint_order","layer","source_task"]`
Subset: `target_task == "plan"`, `source_task ∈ {det, map, motion, plan}`.

CSV columns (in order):
```
n, mean_delta, median_delta, p05_delta, p95_delta,
mean_gain, median_gain,
helpful_rate, harmful_rate,
practical_helpful_rate, large_harm_rate,
std_delta, sem_delta,
ci_lo_delta, ci_hi_delta,
ci_lo_gain, ci_hi_gain,
mean_grad_norm,
effect_size,
tau
```

Visualizations (per model, per checkpoint):
1. Heatmap `layer × source_task` of `mean_gain`.
2. Heatmap `layer × source_task` of `practical_helpful_rate`.

Tables:
3. Top 10 by `mean_gain` and by `practical_helpful_rate` (beneficial).
4. Bottom 10 by `mean_gain` and top 10 by `large_harm_rate` (harmful).

Output files:
```
gradient_analysis_results/plan_centric/plan_transfer_summary.csv
gradient_analysis_results/plan_centric/plan_transfer_top_beneficial.csv
gradient_analysis_results/plan_centric/plan_transfer_top_harmful.csv
gradient_analysis_results/plan_centric/figures/F/<model>_<ckpt>_<metric>.png
```

## 7. Part G — directed asymmetry around plan

`aux_task ∈ {det, map, motion}`.

For each (model, checkpoint, layer, aux_task):
```
gain_A_to_plan   = mean(gain | source=aux_task, target=plan)
gain_plan_to_A   = mean(gain | source=plan,     target=aux_task)
self_gain_A      = mean(gain | source=aux_task, target=aux_task)
asymmetry        = gain_A_to_plan - gain_plan_to_A
plan_transfer_ratio = gain_A_to_plan / (|self_gain_A| + EPS)
helpful_A_to_plan   = mean(delta<0 | source=aux_task, target=plan)
helpful_plan_to_A   = mean(delta<0 | source=plan,     target=aux_task)
n_A_to_plan, n_plan_to_A, n_self_A
ci_lo_asym, ci_hi_asym
interpretation  # from interpret_asymmetry_row
```

`interpret_asymmetry_row` rules (use `tau` per (model, ckpt) for the ≈0
band):

| `gain_A_to_plan` | `gain_plan_to_A` | message |
|---|---|---|
| `> tau` | `< -tau` | "A는 planning auxiliary로 유용하지만, planning update는 A representation을 보존하지 않는다." |
| `> tau` | `> tau`  | "A와 planning이 상호 보완적." |
| `< -tau`| `< -tau` | "A와 planning이 해당 layer에서 상호 간섭." |
| `|·| <= tau` | * | "A→plan transfer가 미미. A task update가 planning에 도달하지 않을 가능성." |

Visualizations:
1. Heatmap `layer × aux_task` of `asymmetry`.
2. Heatmap `layer × aux_task` of `plan_transfer_ratio`.
3. Checkpoint trend plot: x=`checkpoint_order`, y=`asymmetry` or
   `plan_transfer_ratio`, one line per `aux_task`, faceted by model.

Output:
```
plan_centric/plan_asymmetry_summary.csv
plan_centric/plan_transfer_ratio_summary.csv     # view of the above
plan_centric/figures/G/<model>_<ckpt>_<metric>.png
plan_centric/figures/G/<model>_<metric>_trend.png
```

## 8. Part H — effect-size-aware 1-step probe

Target = `"plan"` (extendable later). Group keys: same as Part F.

Columns:
```
n,
helpful_rate, practical_helpful_rate, large_harm_rate,
mean_gain, median_gain,
std_delta, effect_size,
ci_lo_gain, ci_hi_gain,
ci_contains_zero,
flag_high_helpful_low_gain,
flag_positive_but_insig,
flag_high_practical_helpful,
flag_high_large_harm
```

Flag thresholds:
```
flag_high_helpful_low_gain    = helpful_rate >= 0.6 and |mean_gain| < tau
flag_positive_but_insig       = mean_gain > 0 and ci_contains_zero
flag_high_practical_helpful   = practical_helpful_rate >= 0.5
flag_high_large_harm          = large_harm_rate >= 0.5
```

Visualizations:
1. Scatter `helpful_rate vs mean_gain` (color = source_task, marker = model);
   label only top-k by `|mean_gain|`.
2. `mean_gain` with bootstrap CI bar/line plot, per model and source_task,
   showing top influential layers.

Output:
```
plan_centric/effect_size_probe_summary.csv
plan_centric/helpful_vs_effect_size.csv     # scatter-friendly slice
plan_centric/figures/H/*.png
```

## 9. Part I — first-order prediction vs actual

### 9.1 Required new probe columns

`probe.py` must emit, on the 1-step path, for each row:

- `grad_dot`: `<g_src, g_tgt>` at the layer's parameter scope.
- `step_size`: the effective scalar applied to `g_src` in the virtual update:
  - `variant="raw"`: `alpha`.
  - `variant="normalized"`: `alpha / max(||g_src||, EPS)`.
- `source_grad_norm`: existing `grad_norm`; renamed in summary only for
  clarity, the CSV column stays `grad_norm`.

`grad_dot` is computed once per `(batch, layer)` for all source/target pairs:
before the source loop, the runner backs up `g_task[task] = ∇L_task` at the
current layer's params for every task. Then inside the source loop,
`grad_dot[s,t] = (g_task[s] * g_task[t]).sum()`. Memory cost is `T` extra
gradient tensors per layer; runtime cost is `T-1` extra backward passes per
layer per batch. No additional forward passes are required (the existing
forward is reused).

### 9.2 detect_first_order_columns

Returns:
```python
{"grad_dot": "grad_dot", "step_size": "step_size"}
```
when both columns exist, else `None`. When `None`, the notebook cell prints
the required schema (per spec) and SKIPs.

### 9.3 Builder

`build_first_order_residual_summary(base, cols)`:
```
pred_delta = -step_size * grad_dot
residual   = delta - pred_delta
```
Group keys: `["model","checkpoint","layer","source_task","target_task"]`.
Per group:
```
n, mean_actual, mean_pred, mean_residual, std_residual,
pearson_actual_pred, spearman_actual_pred, r_squared
```

Also a second pass restricted to `target_task=="plan"` for the residual
heatmap (`layer × source_task`).

Visualizations:
1. Scatter actual vs predicted, all pairs.
2. Residual heatmap (`layer × source_task`, plan target only).
3. Residual distribution histogram, faceted by model.

Output:
```
plan_centric/first_order_vs_actual_summary.csv
plan_centric/first_order_residual_by_layer.csv
plan_centric/figures/I/*.png
```

## 10. Part J — planning sensitivity / query reachability (HiP-AD)

### 10.1 New runner

`tools/gradient_analysis/query_sensitivity.py`:

```python
def run_query_sensitivity(
    collector,
    dataloader,
    num_batches: int,
    out_dir: Path,
    capture_vectors: bool = False,
    capture_delta: bool = False,
    forward_seed: int | None = 42,
    freeze_matching: bool = True,
) -> pd.DataFrame
```

`capture_vectors=True` writes raw per-token `gQ` to a sibling `.pt` file;
`capture_delta=True` records `Q_after - Q_before` after a single virtual
plan-loss step so `directional_usefulness` can be computed downstream. Both
default to False because the .pt sidecar is large.

Per batch:
1. Standard forward, retrieve task query tensors via the adapter
   (`adapter.get_task_queries(model, fwd_artifacts)` → `{task: Tensor}`).
2. Compute `L_plan` via `adapter.split_losses(fwd, "plan")`.
3. `torch.autograd.grad(L_plan, list(queries.values()), retain_graph=False)`.
4. For each task query tensor `Q`, with gradient `gQ`:
   - `query_norm = ||Q||` per token (if shape allows) and overall.
   - `grad_plan_wrt_query_norm = ||gQ||` per token and overall.
   - If `capture_vectors`: save raw `gQ` to .pt next to CSV.

CSV schema written:
```
model, checkpoint, checkpoint_order, batch_idx, scene_token,
layer, task_query_type, query_index, query_norm,
grad_plan_wrt_query_norm
```
Plus optional `grad_plan_wrt_query_vector`, `delta_query_vector`,
`directional_usefulness` (when computable).

### 10.2 HipadAdapter.get_task_queries

Returns a dict mapping `task_query_type ∈ {det, map, motion, plan}` to the
**graph-attached** query tensor used inside the head for that task (no
`.detach()`; the tensor must still be part of the `L_plan` autograd graph so
`autograd.grad(L_plan, [Q])` is non-zero). Concrete attribute paths on the
HiP-AD head are resolved at implementation time. VAD's `get_task_queries`
raises `NotImplementedError`; the runner catches it and emits a SKIP
message that includes the model name.

### 10.3 Notebook side

`load_or_template_query_sensitivity(path)`:
- If file exists: return `(df, "loaded")`.
- Else: write template CSV with the schema above, return `(None, "template")`.

`build_query_sensitivity_summary(df)` groups by
`["model","checkpoint","layer","task_query_type"]`:
```
mean_sensitivity = mean(grad_plan_wrt_query_norm)
median_sensitivity, p05, p95
mean_query_norm  (if column present)
sensitivity_normed = grad_plan_wrt_query_norm / (query_norm + EPS)
share_task        = sum_per_task / sum_total_per_(model, ckpt, layer)
```

Visualizations:
1. Bar plot per `task_query_type` of mean sensitivity.
2. Heatmap `layer × task_query_type`.
3. Checkpoint trend.
4. Token-level top-k table (when `query_index` exists).
5. Directional usefulness scatter (when vectors are present):
   ```
   directional_usefulness = -dot(grad_plan_wrt_query_vector, delta_query_vector)
                            / (||g|| * ||Δ|| + EPS)
   ```

Output:
```
plan_centric/query_sensitivity_summary.csv
plan_centric/query_sensitivity_template.csv      # if input missing
plan_centric/figures/J/*.png
```

## 11. Part K — task loss weight elasticity (template only)

No experiment runner this PR. The notebook checks for
`ELASTICITY_LOG_PATH` (env-overridable, default
`gradient_analysis_results/plan_centric/task_weight_elasticity_runs.csv`).

If missing: write template CSV, emit SKIP message.

If present: `build_elasticity_summary(df)`:
- Identify baseline run (`is_baseline` column, or all `lambda_*` == 1).
- Build long-format with `swept_task`, `lambda_value`, `log_lambda`,
  `task_metric`, `plan_metric`, `relative_task_metric`,
  `relative_plan_metric`.
- Direction handling:
  - plan_l2, collision: lower-is-better.
  - task metric: higher-is-better unless name contains
    `{loss, error, ade, fde}`.
- Per `swept_task`:
  - `elasticity = Δmetric / Δlog_lambda` (sequential finite diff).
- `planning_safe_weight_range`:
  - `plan_degradation = (plan_metric - plan_metric_baseline) / |plan_metric_baseline|`
    (orient so positive = worse for plan_l2/collision).
  - `planning_safe = plan_degradation <= PLAN_DEGRADATION_TOL (default 0.01)`.

Visualizations:
1. `log(lambda_task) vs task_metric`.
2. `log(lambda_task) vs plan_metric`.
3. Pareto: task gain vs plan degradation.
4. Planning-safe weight range table.

Output:
```
plan_centric/task_weight_elasticity_summary.csv
plan_centric/planning_safe_weight_range.csv
plan_centric/task_weight_elasticity_template.csv  # if input missing
plan_centric/figures/K/*.png
```

## 12. Testing strategy

- `tests/gradient_analysis/test_plan_centric.py`:
  - `ensure_columns` returns False + warning when missing.
  - `bootstrap_ci` returns `(nan, nan)` for `n < 5`; brackets the mean of a
    known normal sample.
  - `practical_threshold` returns `min_value` floor when median is 0.
  - `standardize_probe_df`:
    - Aliases `baseline_loss`/`stepped_loss` → `loss_before`/`loss_after`.
    - `gain = -delta` always.
    - Falls back to `rel_delta` when no `loss_before`/`loss_after`.
  - `build_plan_transfer_summary` produces all required columns and
    correct `helpful_rate` / `practical_helpful_rate` on a synthetic fixture.
  - `build_asymmetry_summary` returns the right `asymmetry` sign on a
    constructed fixture and emits the correct interpretation string for each
    of the 4 quadrants.
  - `build_effect_size_summary` raises no errors on empty input and sets
    flags correctly on edge fixtures.
  - `detect_first_order_columns` returns `None` when columns missing and
    `{"grad_dot", "step_size"}` when present.
  - `load_or_template_*` writes a template CSV with the right schema when
    input missing.

- `tests/gradient_analysis/test_probe_grad_dot.py`:
  - On a 2-task synthetic mini-model, run `probe_one_batch` with
    `steps_list=[1]` and check that for `s == t`,
    `grad_dot ≈ ||g_src||²` and `step_size > 0`.
  - For variant="normalized": `step_size == alpha / ||g_src||`.

- HiP-AD query sensitivity runner: smoke test marked
  `@pytest.mark.requires_gpu` and skipped by default; sanity checks the
  output schema only when an env variable enables it. CI does not run it.

## 13. Non-goals (out of scope this PR)

- Re-running probes on existing checkpoints to populate `grad_dot` columns.
- VAD query sensitivity.
- Task-weight elasticity sweep runner.
- Scene-type stratification (will be a follow-up; Part J template already
  carries `scene_token` for forward compatibility).
- Any change to existing Part A–E content beyond cell reordering and one
  header-level relabel to "Appendix".

## 14. Rollout

1. Land `plan_centric.py` + tests.
2. Land `probe.py` + adapter changes + `query_sensitivity.py` + tests.
3. Refactor the notebook in place. After this, re-running the notebook on
   the existing CSVs will produce Parts F–H with real data and Parts I–J–K
   in SKIPPED/template state. Once the user re-runs the probe and the
   query sensitivity runner, Parts I and J auto-activate.
