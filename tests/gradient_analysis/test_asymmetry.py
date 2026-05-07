import numpy as np
import pytest

from tools.gradient_analysis.asymmetry import decompose_matrix, top_asymmetric_pairs
from tools.gradient_analysis.gradnorm import antisymmetric_frobenius


def test_decompose_matrix_satisfies_identity():
    M = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    S, A = decompose_matrix(M)
    assert np.allclose(S + A, M)
    assert np.allclose(S, S.T)
    assert np.allclose(A, -A.T)


def test_top_asymmetric_pairs_ranked_by_absolute_antisym():
    M = np.array([[0.0, 1.0, 0.1], [-1.0, 0.0, 0.2], [-0.1, -0.2, 0.0]])
    labels = ["a", "b", "c"]
    ranked = top_asymmetric_pairs(M, labels, k=2)
    assert ranked[0]["pair"] == ("a", "b")


def test_top_asymmetric_pairs_handles_nan_cells():
    """An NaN row (e.g. ego under fp16) must not push NaN entries to the top
    of the ranking — finite pairs should still be reachable."""
    M = np.array([
        [0.0, 1.0, np.nan],
        [-0.5, 0.0, 2.0],
        [np.nan, -1.5, 0.0],
    ])
    ranked = top_asymmetric_pairs(M, ["a", "b", "c"], k=3)
    # First two ranked entries must be finite.
    assert np.isfinite(ranked[0]["abs_antisym"])
    assert np.isfinite(ranked[1]["abs_antisym"])
    # The (a, c) pair has NaN component → trails behind in NaN-aware sort.
    last = ranked[-1]
    if last["pair"] == ("a", "c"):
        assert not np.isfinite(last["abs_antisym"])


def test_antisymmetric_frobenius_skips_nan_pairs():
    M = np.array([
        [0.0, 1.0, np.nan],
        [-0.5, 0.0, 2.0],
        [np.nan, -1.5, 0.0],
    ])
    val = antisymmetric_frobenius(M)
    assert np.isfinite(val)
    # Only (a,b)<->(b,a) and (b,c)<->(c,b) contribute.
    # (1.0 - (-0.5))/2 = 0.75 ; (2.0 - (-1.5))/2 = 1.75
    expected = float(np.sqrt(2 * (0.75 ** 2 + 1.75 ** 2)))
    assert abs(val - expected) < 1e-6


def test_antisymmetric_frobenius_all_nan_returns_nan():
    M = np.full((3, 3), np.nan)
    val = antisymmetric_frobenius(M)
    assert np.isnan(val)


def test_antisymmetric_frobenius_zero_on_symmetric_matrix():
    M = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    assert abs(antisymmetric_frobenius(M)) < 1e-12
