# MTL Gradient Analysis — Design Spec

**Date:** 2026-04-20
**Project:** HiP-AD (nusc/pcgrad branch)
**Scope:** Redesigning the multi-task gradient analysis pipeline to produce publication-worthy insights about task gradient relationships.

---

## 1. Motivation

Existing gradient analysis in this repo (`tools/analyze_gradient_conflict.py`, `tools/one_step_interference_probe.ipynb`) produces results that are hard to interpret:

- Cosine similarities are near-orthogonal across most task pairs, with no clear signal.
- One-step probe results are inconsistent across batches and task pairs.
- Reports are mean-only summaries; batch-level distributions are collapsed.

Two root causes were identified:

1. **Mixed parameter scopes.** The one-step probe applies a virtual update only to *shared* parameters, even for the source task. When a task's gradient is applied to shared params but not to its own task-specific head, the representation and the head become inconsistent — the source task's own loss can increase, which contaminates every downstream measurement.
2. **Mean-only summaries.** Per-batch variance is large; mean cosine / mean Δloss hides the underlying distribution. Signal that is visible in binned scatter plots disappears when collapsed to a single number.

Additionally, we currently lack a systematic view of gradient magnitude imbalance (GradNorm-style analysis), training-time dynamics of alignment/conflict, and task-pair asymmetry.

## 2. Goals

Produce the following, with publication-ready figures and tables:

- **G1.** Evidence for the hypothesis *"gradient alignment → single-task update benefits other tasks"*, and evidence for the inverse for conflicting pairs.
- **G2.** A task-pair conflict map that motivates gradient surgery (e.g., PCGrad) on specific pairs/layers.
- **G3.** An analysis of gradient magnitude imbalance across tasks, and its effect on shared-parameter updates (raw vs. normalized steps).
- **G4.** Training dynamics of alignment/conflict across four checkpoints.
- **G5.** Asymmetry analysis of task affinity: does A→B help imply B→A helps?

Non-goals (explicitly deferred):

- Single-task vs multi-task retraining comparison (too expensive; deferred to a follow-up).

## 3. Key Design Principles

- **Parameter-scope separation.** Conflict computations (cosine, projection decomposition) operate on **shared parameters only**, grouped into small units (backbone, decoder_ffn, decoder_norm, gnn, inter_gnn, decouple_fc) to avoid averaging-trap. One/two-step probe virtual updates use the **full parameter set reachable from the source task's loss** (shared + that task's task-specific head), so the source task's own loss behaves correctly.
- **Distributions over means.** Every batch-level measurement retains its per-batch values. Correlation, binning, and scatter analyses are first-class outputs, not afterthoughts.
- **Caching gradients once.** Gradient collection is the most expensive step; collected gradients are cached per checkpoint and reused across M2/M3/M5.
- **Configurability.** All quantities (num_batches, num_samples, alpha, bin edges, checkpoints, param groups) are exposed in a YAML config; no hard-coded numbers inside modules.
- **Reproducibility.** Fixed seeds and CUDA deterministic mode.

## 4. Story for Paper

1. Existing MTL gradient analysis conflates parameter scopes and summarizes with means → inconsistent, low-signal results.
2. Separating scopes and retaining batch-level distributions exposes structure.
3. Cosine similarity (on shared params) correlates with actual loss change (from full-param virtual step) — aligned pairs benefit each other, conflicting pairs hurt each other.
4. Gradient magnitude imbalance lets some tasks dominate shared-parameter updates; normalizing gradients produces a more symmetric task-affinity matrix.
5. Conflict pairs motivate gradient surgery; norm imbalance motivates normalization; alignment pairs justify shared-backbone MTL.

## 5. Analysis Modules

### M1. Gradient Collector (shared infrastructure)

For each checkpoint, for each mini-batch, for each task, compute and cache:

- `g_A^shared`: gradient of task A's loss w.r.t. shared parameters, grouped by layer group.
- `g_A^full`: gradient of task A's loss w.r.t. all parameters reachable from task A's loss (shared + task-A-specific head).

Caching format: per-checkpoint `grad_cache/*.pt` files. Loaded on demand by downstream modules.

### M2. Conflict Analysis (improved)

Inputs: `g_A^shared`, `g_B^shared` per batch, per layer group.

Metrics (all per task-pair, per layer group, per batch):

- **Cosine similarity:** `cos(g_A^shared, g_B^shared)`
- **Conflict ratio (CR):** fraction of batches with `cos < 0`
- **Projection decomposition:**
  - Cooperative magnitude: `||proj_{g_B}(g_A)||` if `cos ≥ 0`, else 0
  - Conflicting magnitude: `||proj_{g_B}(g_A)||` if `cos < 0`, else 0

Plots:

- Per-group cosine histogram (all batches).
- Per-group violin plots across task pairs.

### M3. One/Two-Step Probe (fixed)

For each source task A and target task B:

- Apply virtual update `θ' = θ - α · g_A^full` (one-step) or twice (two-step, using a fresh batch for the second step).
- Measure `ΔL_B = L_B(θ') - L_B(θ)` per batch.

Variants:

- **Raw:** use `g_A^full` directly.
- **Normalized:** use `g_A^full / ||g_A^full||`.

Metrics: per-batch `ΔL_B`, relative `ΔL_B / L_B(θ)`, helpful/harmful ratios, 5×5 task-pair matrix (aggregated).

### M4. Correlation & Binning Analysis

Inputs: batch-level `(cos_AB, ΔL_B)` pairs from M2 and M3.

Outputs per task-pair per layer group:

- **Pearson** and **Spearman** correlation between cosine and Δloss.
- **Binning:** cosine similarity bucketed via config edges (default `[-1, -0.3, -0.1, 0.1, 0.3, 1]`). Each bin reports mean Δloss, std, median, helpful ratio, sample count.
- **Plots:** scatter (cos vs Δloss), binned bar chart with 95% CI, per-bin violin.

This is where the "alignment → mutual benefit" hypothesis is empirically tested.

### M5. GradNorm Analysis

Inputs: `||g_A^shared||` per task, per batch, per layer group.

Metrics:

- Per-task gradient norm distribution.
- Task-pair norm ratio matrix (who dominates?).
- **Raw vs normalized step comparison** (uses M3 raw and normalized variants): measure how much more symmetric the 5×5 Δloss matrix becomes under normalization (Frobenius norm of the antisymmetric component).

Plots:

- Per-task norm boxplots per checkpoint.
- Norm-ratio heatmap.
- Side-by-side affinity matrices (raw vs normalized).

### M6. Training Dynamics

Re-run M2, M3, M5 at four checkpoints (1ep, 3ep, 6ep, 18ep).

Plots:

- Per task-pair time series: mean cosine, helpful ratio, norm ratio over epoch.
- Annotations marking transitions (e.g., aligned → conflict crossovers).

### M7. Task Affinity Asymmetry

Input: M3's 5×5 Δloss matrix `M[A,B] = mean(ΔL_B after A update)`.

Decomposition:

- Symmetric part: `(M + M^T) / 2`
- Antisymmetric part: `(M - M^T) / 2`

Outputs:

- Three heatmaps (original, symmetric, antisymmetric).
- Table of top-k asymmetric pairs with interpretation.

## 6. Supplementary Per-Sample Analysis

- Rerun M3 + M4 at **1ep checkpoint only**, with `batch_size=1` on 500 samples.
- Rationale: 1ep is in the early active-learning phase, so per-sample gradients have strong signal; at 18ep gradients are smaller and per-sample noise would dominate.
- Purpose: show that batch-level averaging may hide sample-level heterogeneity (e.g., bimodal alignment/conflict patterns, if present). This is an existence probe for the "mean collapses signal" claim — the finding could be positive (heterogeneity visible) or negative (batch-12 already captures the signal), both of which are informative.
- All sample counts and checkpoint choice are config-configurable.

## 7. Data Flow

```
[4 checkpoints] → M1 Gradient Collector → [shared grad, full grad per batch]
                                              │
                   ┌──────────────┬───────────┴─────────┬──────────┐
                   ↓              ↓                     ↓          ↓
               M2 Conflict    M3 Probe              M5 GradNorm   (cache)
               (shared)       (full, 1&2 step,      (shared)
                              raw/normalized)
                   │              │                     │
                   └──────┬───────┴─────────────────────┘
                          ↓
                   M4 Correlation / Binning
                          │
                   ┌──────┴──────┐
                   ↓             ↓
             M6 Dynamics    M7 Asymmetry
             (repeat on     (from M3 matrix)
             4 ckpts)
```

## 8. File & Directory Structure

```
HiP-AD/
├── tools/gradient_analysis/
│   ├── __init__.py
│   ├── collector.py        # M1
│   ├── conflict.py         # M2 (refactored from analyze_gradient_conflict.py)
│   ├── probe.py            # M3
│   ├── correlation.py      # M4
│   ├── gradnorm.py         # M5
│   ├── dynamics.py         # M6 (orchestrates M2/M3/M5 across checkpoints)
│   ├── asymmetry.py        # M7
│   ├── binning.py          # shared utility
│   └── viz.py              # shared plotting utility
├── tools/run_gradient_analysis.py    # CLI entry point
├── configs/gradient_analysis.yaml    # all tunables
└── gradient_analysis_results/
    ├── ckpt_1ep/
    │   ├── grad_cache/
    │   ├── conflict/
    │   ├── probe/
    │   ├── correlation/
    │   └── gradnorm/
    ├── ckpt_3ep/ …  ckpt_6ep/ …  ckpt_18ep/
    ├── supplementary_1ep_bs1/
    ├── dynamics/
    ├── asymmetry/
    └── summary_report.md
```

## 9. Configuration

```yaml
# configs/gradient_analysis.yaml
tasks: [det, map, motion, ego, plan]

checkpoints:
  1ep:  iter_2344.pth
  3ep:  iter_7032.pth
  6ep:  iter_14064.pth
  18ep: iter_42192.pth
ckpt_root: data/ckpts/E2_E1_stage2_18ep

primary:
  batch_size: 12
  num_batches: 100          # user-adjustable

supplementary:
  enabled: true
  batch_size: 1
  num_samples: 500          # user-adjustable
  checkpoint: 1ep           # user-adjustable

probe:
  alpha: 0.001
  steps: [1, 2]
  variants: [raw, normalized]

binning:
  cosine_bins: [-1.0, -0.3, -0.1, 0.1, 0.3, 1.0]  # user-adjustable

shared_param_groups:
  - backbone
  - decoder_ffn
  - decoder_norm
  - gnn
  - inter_gnn
  - decouple_fc

seed: 42
deterministic: true
```

## 10. CLI Entry Points

```bash
# full run
python tools/run_gradient_analysis.py --config configs/gradient_analysis.yaml --all

# specific modules
python tools/run_gradient_analysis.py --config ... --modules M2,M4

# specific checkpoints
python tools/run_gradient_analysis.py --config ... --checkpoints 1ep,3ep

# toggle supplementary
python tools/run_gradient_analysis.py --config ... --no-supplementary

# gradient collection only (for caching)
python tools/run_gradient_analysis.py --config ... --collect-only

# reuse cached gradients
python tools/run_gradient_analysis.py --config ... --use-cache
```

## 11. Error Handling

**Memory:**

- Gradients moved to CPU immediately after backward (`.detach().cpu()`), sliced to configured param groups.
- OOM fallback: automatic micro-batch accumulation (optional).

**Numerical:**

- Skip batches where `||g|| < 1e-8` with a warning log.
- Skip batches where virtual step produces NaN/Inf `L_B`.
- Empty-bin warnings in M4.

**Checkpoint compatibility:**

- Validate model config hash matches training config at load time.

**Determinism:**

- `torch.manual_seed`, `np.random.seed`, dataloader seed, `torch.use_deterministic_algorithms(True)`.

## 12. Verification & Testing

**Sanity checks (automatic, fail loud):**

- **V1.** `g_A^shared` must equal the shared-param slice of `g_A^full` (tolerance 1e-6).
- **V2.** `L_A(θ - α · g_A^full) ≤ L_A(θ)` for reasonable α — i.e., a source task's own loss must not increase from its own update. This is the assertion that would have caught the original bug.
- **V3.** `cos ∈ [-1, 1]`; `cos(g_A, g_A) = 1` within tolerance.
- **V4.** Bin sample-count sums equal total batch count.
- **V5.** Aligned-bin (`cos ≥ 0.3`) mean Δloss must be ≤ conflicting-bin (`cos ≤ -0.3`) mean Δloss. If violated, either the hypothesis is wrong or there is a bug.

**Unit tests** (`tests/gradient_analysis/`):

- `test_collector.py` — gradient shape / slicing consistency
- `test_conflict.py` — toy 2-task cosine/projection math
- `test_probe.py` — V2 reverse test on a toy model
- `test_binning.py` — bin edges, empty bins
- `test_gradnorm.py` — norm computation, normalized grad has norm 1

**Integration test** (`tests/test_pipeline.py`):

- Mock HiP-AD with a small dummy model; run full pipeline end-to-end on 5 mini-batches; verify directory layout and file contents.

**Smoke test (real model):**

- `iter_2344` + 3 batches + all modules, to catch dataloader/collator/forward issues before a long run.

## 13. Outputs

**Per checkpoint:**

- `grad_cache/*.pt` — cached gradients
- `conflict/cosine_matrix.csv`, `projection.csv`, `plots/*.png`
- `probe/per_batch.csv`, `matrix_{1,2}step_{raw,norm}.csv`, `plots/*.png`
- `correlation/scatter_*.png`, `binned_*.csv`, `correlation_table.csv`
- `gradnorm/per_task_norm.csv`, `norm_ratio_heatmap.png`, `raw_vs_norm_comparison.png`

**Aggregate:**

- `dynamics/timeseries_*.png`, `dynamics_table.csv`
- `asymmetry/{original,symmetric,antisymmetric}.png`, `top_asymmetric_pairs.csv`
- `summary_report.md` — highlights for paper figure/table drafting

## 14. Cost Estimate

- **Primary** (batch=12, 100 batches, 4 ckpts, all modules): ~1–1.5 days on a single GPU.
- **Supplementary** (batch=1, 500 samples, 1ep only, M3+M4): ~6–8 hours additional.
- Total: roughly 1.5–2 days.

## 15. Deferred Items

- Single-task vs multi-task retraining comparison (would require 5× 18-epoch reruns; too expensive).
- Extension to non-shared param conflict analysis (deliberately out of scope; conflict analysis must stay on shared params to avoid averaging trap).
- Additional checkpoints between 6ep and 18ep (can be added later if dynamics plots show interesting mid-training transitions).
