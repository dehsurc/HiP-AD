# Gradient Analysis Pipeline

Multi-task gradient analysis for HiP-AD. Designed to produce publication-ready figures
on task-pair gradient relationships (alignment, conflict, magnitude imbalance,
training dynamics, asymmetry) without falling into the "mean of means" trap.

Full design spec: [`docs/superpowers/specs/2026-04-20-mtl-gradient-analysis-design.md`](../../docs/superpowers/specs/2026-04-20-mtl-gradient-analysis-design.md).

## Modules

| Code | File | What it does |
|------|------|--------------|
| M1 | `collector.py` | Per-task gradient collector (shared + full param scopes) |
| M2 | `conflict.py` | Cosine similarity + projection decomposition over shared params |
| M3 | `probe.py` | One/two-step virtual-update probe over full params (raw + normalized) |
| M4 | `correlation.py` | Pearson/Spearman + binned summaries between cosine and Δloss |
| M5 | `gradnorm.py` | Per-task gradient norm distributions + raw-vs-normalized symmetry |
| M6 | `dynamics.py` | Cross-checkpoint time-series aggregator (consumes M2/M3/M5 outputs) |
| M7 | `asymmetry.py` | Symmetric/antisymmetric decomposition of the affinity matrix |
| — | `binning.py` | Binning utility (used by M4) |
| — | `viz.py` | Shared matplotlib helpers (scatter+bins, heatmap, violin, time-series) |
| — | `summary.py` | Markdown digest of top findings for paper drafting |

CLI entrypoint: `tools/run_gradient_analysis.py`.

## Quick start

All commands assume cwd is repo root (`/home/yongjae/e2e/HiP-AD`) and the conda env
`hipad` is on the path:

```bash
PY=/home/yongjae/miniconda3/envs/hipad/bin/python
```

### Smoke test (≈10 min, 1 checkpoint × 3 batches)

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --smoke \
    --no-supplementary
```

Pick `CUDA_VISIBLE_DEVICES` based on `nvidia-smi --query-gpu=index,memory.free --format=csv`. The pipeline needs ≈25 GB of GPU memory for HiP-AD at batch_size=12.

### Primary run (≈1–1.5 days, 4 checkpoints × 100 batches, all modules)

```bash
mkdir -p gradient_analysis_results
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --no-supplementary \
    > gradient_analysis_results/primary.log 2>&1 &
echo $! > gradient_analysis_results/primary.pid
tail -f gradient_analysis_results/primary.log
```

### Supplementary per-sample (≈6–8 h, 1ep × batch_size=1 × 500 samples, M3+M4 only)

```bash
nohup env CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --checkpoints 1ep \
    --modules M3,M4 \
    > gradient_analysis_results/supplementary.log 2>&1 &
```

### Aggregate dynamics post-hoc

If the per-checkpoint runs already exist, you can re-aggregate dynamics without re-running anything:

```bash
PYTHONPATH=. $PY tools/run_gradient_analysis.py \
    --config configs/gradient_analysis.yaml \
    --modules M6 \
    --no-supplementary
```

## Outputs

`gradient_analysis_results/` (gitignored). Layout per checkpoint:

```
ckpt_<tag>/
├── conflict/
│   ├── conflict_<a>_<b>_per_batch.csv      (M2)
│   ├── conflict_<a>_<b>_summary.csv
│   └── cosine_violin.png
├── probe/
│   ├── probe_per_batch.csv                  (M3)
│   └── probe_matrix_<k>step_<variant>.csv
├── correlation/
│   ├── binned_<a>_<b>_<k>step_<variant>.csv (M4)
│   ├── scatter_<a>_<b>_<k>step_<variant>.png
│   └── correlation_table.csv
├── gradnorm/
│   ├── per_task_norm.csv                    (M5)
│   ├── norm_ratio_heatmap.png
│   ├── affinity_matrix_<k>step_<variant>.png
│   └── raw_vs_norm_symmetry.csv
└── asymmetry/
    ├── original.png                         (M7)
    ├── symmetric.png
    ├── antisymmetric.png
    └── top_asymmetric_pairs.csv
```

Cross-checkpoint:

```
dynamics/                                    (M6)
├── cos_dynamics.csv
├── helpful_dynamics.csv
├── norm_dynamics.csv
├── timeseries_cosine.png
├── timeseries_conflict_ratio.png
├── timeseries_helpful.png
└── timeseries_norm.png

summary_report.md                            (markdown digest)
```

## Tests

Unit tests use toy models — no HiP-AD env needed. Run from repo root:

```bash
PYTHONPATH=. $PY -m pytest tests/gradient_analysis/ -v
```

## Configuration

`configs/gradient_analysis.yaml` exposes everything users typically tune:

- `tasks` — list of task names (default: det, map, motion, ego, plan)
- `checkpoints` — tag → filename mapping under `ckpt_root`
- `primary.batch_size` / `primary.num_batches`
- `supplementary.batch_size` / `supplementary.num_samples` / `supplementary.checkpoint`
- `probe.alpha` / `probe.steps` / `probe.variants`
- `binning.cosine_bins`
- `shared_param_groups` — which decoder operation types count as "shared"
- `device` (default `cuda:0` — override at launch with `CUDA_VISIBLE_DEVICES` rather than editing the YAML)
- `seed` / `deterministic` / `fp16`

## Known issues

These were found during early smoke runs and may resurface:

1. **GPU memory contention** — HiP-AD at `batch_size=12` needs ≈25 GB. If GPU 0 is occupied (other users' processes), use `CUDA_VISIBLE_DEVICES` to pick a freer GPU. Smoke run failed on GPU 0 with only 11.5 GB free.
2. **`Indexing.cu:384 INTERNAL ASSERT FAILED: number of flattened indices did not match number of elements in the value tensor: 480 vs 12`** — observed on the smoke retry. Root cause not yet investigated. Likely related to `torch.use_deterministic_algorithms(True)` interacting with a scatter/gather inside the model. If you hit this, try setting `deterministic: false` in the YAML to confirm and triage.
3. The CLI initially imported `from mmdet3d.models import build_detector` (mmdet3d isn't installed in the `hipad` env). Fixed in commit `54723c0` to use `from mmdet.models import build_detector` (HiP-AD's convention — see `tools/train.py`).
4. The CLI initially didn't honor `cfg.plugin`/`cfg.plugin_dir`, so `SparseDetector` failed to register. Fixed in commit `f054f1a` by adding `_load_plugins(cfg)` to mirror `tools/train.py:90-119`.

## Architectural notes

- **Conflict math** (M2) operates only on the **shared** parameter groups, in **small units** (per decoder op type per layer), to avoid the averaging trap that hides per-layer/pair structure.
- **Probe virtual updates** (M3) operate on the **full** parameter set reachable from the source task's loss (shared + that task's task-specific head), so the source task's own loss correctly decreases. This is the fix to the original "shared-only update" bug; the V2 reverse-sanity test in `tests/gradient_analysis/test_probe.py` enforces it.
- Pure helpers in `collector.py`/`conflict.py`/`probe.py`/etc. have toy-model unit tests; HiP-AD-specific orchestration is exercised end-to-end via the smoke run.
- Gradient cache: per-batch shared gradients are kept in memory across modules within one CLI invocation. Full gradients are not cached to disk (per-batch they can be GB-sized).
