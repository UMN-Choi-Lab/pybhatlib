"""Tests for LDLT rank-2 update."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.backend import get_backend
from pybhatlib.vecup._ldlt import ldlt_decompose, ldlt_rank1_update, ldlt_rank2_update


@pytest.fixture
def xp():
    return get_backend("numpy")


class TestLDLTRank2Update:
    def test_rank2_equals_two_rank1(self, xp):
        """Rank-2 update should equal two successive rank-1 updates."""
        A = np.array([[4.0, 2.0, 1.0], [2.0, 5.0, 3.0], [1.0, 3.0, 6.0]])
        u = np.array([1.0, 0.5, -0.3])
        v = np.array([0.2, -0.8, 0.6])
        alpha = 1.0

        L, D = ldlt_decompose(xp.array(A), xp=xp)

        # Two rank-1 updates
        L1, D1 = ldlt_rank1_update(L, D, xp.array(u), alpha, xp=xp)
        L_two, D_two = ldlt_rank1_update(L1, D1, xp.array(v), alpha, xp=xp)

        # One rank-2 update
        L_r2, D_r2 = ldlt_rank2_update(L, D, xp.array(u), xp.array(v), alpha, xp=xp)

        np.testing.assert_allclose(np.asarray(L_r2), np.asarray(L_two), atol=1e-10)
        np.testing.assert_allclose(np.asarray(D_r2), np.asarray(D_two), atol=1e-10)

    def test_rank2_recovers_matrix(self, xp):
        """Verify L*D*L^T = A + alpha*(u*u^T + v*v^T)."""
        A = np.array([[3.0, 1.0], [1.0, 4.0]])
        u = np.array([1.0, 2.0])
        v = np.array([0.5, -1.0])
        alpha = 0.5

        L, D = ldlt_decompose(xp.array(A), xp=xp)
        L_new, D_new = ldlt_rank2_update(L, D, xp.array(u), xp.array(v), alpha, xp=xp)

        L_np = np.asarray(L_new)
        D_np = np.asarray(D_new)

        # Reconstruct matrix
        A_new = L_np @ np.diag(D_np) @ L_np.T

        # Expected
        A_expected = A + alpha * (np.outer(u, u) + np.outer(v, v))

        np.testing.assert_allclose(A_new, A_expected, atol=1e-10)

    def test_rank2_with_negative_alpha(self, xp):
        """Rank-2 update with negative scaling."""
        A = np.array([[5.0, 1.0, 0.5], [1.0, 4.0, 0.3], [0.5, 0.3, 3.0]])
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.3, 0.1, 0.2])
        alpha = -0.5

        L, D = ldlt_decompose(xp.array(A), xp=xp)
        L_new, D_new = ldlt_rank2_update(L, D, xp.array(u), xp.array(v), alpha, xp=xp)

        L_np = np.asarray(L_new)
        D_np = np.asarray(D_new)
        A_new = L_np @ np.diag(D_np) @ L_np.T

        A_expected = A + alpha * (np.outer(u, u) + np.outer(v, v))
        np.testing.assert_allclose(A_new, A_expected, atol=1e-10)
