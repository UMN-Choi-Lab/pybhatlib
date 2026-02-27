"""Tests for LDLT decomposition."""

from __future__ import annotations

import numpy as np

from pybhatlib.vecup import ldlt_decompose, ldlt_rank1_update


class TestLDLTDecompose:
    def test_3x3_pd(self, pd_3x3):
        L, D = ldlt_decompose(pd_3x3)
        recon = L @ np.diag(D) @ L.T
        np.testing.assert_allclose(recon, pd_3x3, atol=1e-10)

    def test_unit_diagonal(self, pd_3x3):
        L, D = ldlt_decompose(pd_3x3)
        np.testing.assert_allclose(np.diag(L), np.ones(3), atol=1e-10)

    def test_lower_triangular(self, pd_3x3):
        L, D = ldlt_decompose(pd_3x3)
        assert L[0, 1] == 0.0
        assert L[0, 2] == 0.0
        assert L[1, 2] == 0.0

    def test_positive_D(self, pd_3x3):
        _, D = ldlt_decompose(pd_3x3)
        assert np.all(D > 0)

    def test_2x2(self):
        A = np.array([[4.0, 2.0], [2.0, 5.0]])
        L, D = ldlt_decompose(A)
        recon = L @ np.diag(D) @ L.T
        np.testing.assert_allclose(recon, A, atol=1e-10)

    def test_1x1(self):
        A = np.array([[3.0]])
        L, D = ldlt_decompose(A)
        np.testing.assert_allclose(L, [[1.0]])
        np.testing.assert_allclose(D, [3.0])


class TestLDLTRank1Update:
    def test_preserves_factorization(self, pd_3x3):
        L, D = ldlt_decompose(pd_3x3)
        v = np.array([1.0, 0.5, 0.3])
        alpha = 1.0

        L_new, D_new = ldlt_rank1_update(L, D, v, alpha)

        # Check: L_new @ diag(D_new) @ L_new.T == A + alpha * v @ v.T
        expected = pd_3x3 + alpha * np.outer(v, v)
        recon = L_new @ np.diag(D_new) @ L_new.T
        np.testing.assert_allclose(recon, expected, atol=1e-8)

    def test_negative_alpha(self, pd_3x3):
        L, D = ldlt_decompose(pd_3x3)
        v = np.array([0.1, 0.1, 0.1])
        alpha = -0.5

        L_new, D_new = ldlt_rank1_update(L, D, v, alpha)
        expected = pd_3x3 + alpha * np.outer(v, v)
        recon = L_new @ np.diag(D_new) @ L_new.T
        np.testing.assert_allclose(recon, expected, atol=1e-8)
