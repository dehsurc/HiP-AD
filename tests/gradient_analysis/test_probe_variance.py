"""Phase 1 #9 — bootstrap variance bands + α-sweep aggregation."""
import numpy as np
import pandas as pd

from tools.gradient_analysis.probe import aggregate_affinity_matrix_with_ci


def test_aggregate_affinity_matrix_with_ci_attaches_bands():
    rng = np.random.default_rng(0)
    rows = []
    for b in range(50):
        for s_task in ("a", "b"):
            for t_task in ("a", "b"):
                # Off-diagonals shifted so cells differ visibly.
                delta = rng.standard_normal() * 0.1 + (0.05 if s_task != t_task else -0.05)
                rows.append({
                    "batch_idx": b,
                    "source_task": s_task, "target_task": t_task,
                    "steps": 1, "variant": "raw", "layer": "_all",
                    "delta": delta,
                })
    df = pd.DataFrame(rows)
    mean_mat, lo_mat, hi_mat = aggregate_affinity_matrix_with_ci(
        df, steps=1, variant="raw", layer="_all", n_resamples=500, seed=0,
    )
    assert mean_mat.shape == (2, 2)
    assert lo_mat.shape == (2, 2)
    assert hi_mat.shape == (2, 2)
    assert (lo_mat.values <= mean_mat.values).all()
    assert (mean_mat.values <= hi_mat.values).all()


def test_aggregate_affinity_matrix_with_ci_handles_nan_cells():
    rows = [{"batch_idx": 0, "source_task": "a", "target_task": "a",
             "steps": 1, "variant": "raw", "layer": "_all", "delta": np.nan}]
    df = pd.DataFrame(rows)
    mean_mat, lo_mat, hi_mat = aggregate_affinity_matrix_with_ci(
        df, steps=1, variant="raw", layer="_all", n_resamples=100, seed=0,
    )
    # Single all-NaN cell → mean/lo/hi all NaN, no crash.
    assert np.isnan(mean_mat.iloc[0, 0])
    assert np.isnan(lo_mat.iloc[0, 0])


def test_run_alpha_sweep_requires_factory_or_loader():
    """run_alpha_sweep without either dl_factory or dataloader must raise."""
    import pytest
    from tools.gradient_analysis.probe import run_alpha_sweep
    with pytest.raises(ValueError, match="dl_factory|dataloader"):
        run_alpha_sweep(collector=None, num_batches=1, alphas=[1e-4],
                        sources=["a"], out_dir="/tmp/_swp_test")
