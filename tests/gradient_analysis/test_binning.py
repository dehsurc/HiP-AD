"""Tests for binning utility."""
import numpy as np
import pandas as pd
import pytest

from tools.gradient_analysis.binning import bin_by_edges, summarize_bins


def test_bin_by_edges_assigns_integers():
    x = np.array([-0.5, -0.2, 0.0, 0.2, 0.5])
    edges = np.array([-1.0, -0.3, -0.1, 0.1, 0.3, 1.0])
    bins = bin_by_edges(x, edges)
    # Expect bin indices 0..4 for values falling in each of 5 bins
    assert bins.tolist() == [0, 1, 2, 3, 4]


def test_bin_by_edges_handles_boundary_values():
    x = np.array([-1.0, 1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    bins = bin_by_edges(x, edges)
    # Left edge inclusive, right edge inclusive for last bin
    assert bins[0] == 0
    assert bins[1] == 1


def test_summarize_bins_reports_count_and_mean():
    x = np.array([-0.5, -0.4, 0.2, 0.25, 0.8])
    y = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    edges = np.array([-1.0, 0.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert list(df["bin_idx"]) == [0, 1]
    assert df.loc[df["bin_idx"] == 0, "count"].iloc[0] == 2
    assert df.loc[df["bin_idx"] == 0, "delta_mean"].iloc[0] == pytest.approx(2.0)
    assert df.loc[df["bin_idx"] == 1, "count"].iloc[0] == 3
    assert df.loc[df["bin_idx"] == 1, "delta_mean"].iloc[0] == pytest.approx((10 + 20 + 100) / 3)


def test_summarize_bins_reports_helpful_ratio():
    x = np.array([-0.5, -0.4])
    y = np.array([-1.0, 2.0])  # one helpful (negative delta), one harmful
    edges = np.array([-1.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert df["helpful_ratio"].iloc[0] == pytest.approx(0.5)


def test_summarize_bins_emits_empty_bins_as_rows_with_zero_count():
    x = np.array([0.5])
    y = np.array([1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    df = summarize_bins(x, y, edges, stat_label="delta")
    assert len(df) == 2
    empty = df[df["bin_idx"] == 0]
    assert empty["count"].iloc[0] == 0
