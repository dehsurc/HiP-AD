import numpy as np
import pytest

from tools.gradient_analysis.asymmetry import decompose_matrix, top_asymmetric_pairs


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
