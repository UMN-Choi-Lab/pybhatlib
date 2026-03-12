"""Tests for GGE ordering and TG/TGBME MVNCD methods."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn import gge_ordering, mvncd


@pytest.fixture
def xp():
    return get_backend("numpy")


# ---------------------------------------------------------------------------
# GGE Ordering tests
# ---------------------------------------------------------------------------

class TestGGEOrdering:
    """Tests for the GGE adaptive ordering algorithm."""

    def test_returns_valid_permutation_k3(self):
        """GGE ordering returns a valid permutation for K=3."""
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        perm = gge_ordering(a, sigma)
        assert len(perm) == 3
        assert set(perm.tolist()) == {0, 1, 2}

    def test_returns_valid_permutation_k5(self):
        """GGE ordering returns a valid permutation for K=5."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((5, 5))
        sigma = A @ A.T / 5 + np.eye(5)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.5, 0.8, 0.3, 1.0, 0.6])
        perm = gge_ordering(a, sigma)
        assert len(perm) == 5
        assert set(perm.tolist()) == {0, 1, 2, 3, 4}

    def test_single_variable(self):
        """GGE ordering for K=1 returns [0]."""
        perm = gge_ordering(np.array([1.0]), np.array([[1.0]]))
        np.testing.assert_array_equal(perm, [0])

    def test_two_variables(self):
        """GGE ordering for K=2 returns valid permutation."""
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([2.0, 0.1])
        perm = gge_ordering(a, sigma)
        assert len(perm) == 2
        assert set(perm.tolist()) == {0, 1}
        # Variable with smaller a/sd should come first
        assert perm[0] == 1  # a[1]/sd[1] = 0.1 < a[0]/sd[0] = 2.0

    def test_identity_covariance(self):
        """With identity covariance, GGE ordering matches ascending a."""
        a = np.array([3.0, 1.0, 2.0, 0.5])
        sigma = np.eye(4)
        perm = gge_ordering(a, sigma)
        # With identity cov, GGE should produce ascending order of a
        # (since sd = 1 for all, z = a)
        assert perm[0] == 3  # a[3] = 0.5 is smallest

    def test_permutation_dtype(self):
        """GGE ordering returns integer indices."""
        sigma = np.eye(3)
        a = np.array([1.0, 2.0, 3.0])
        perm = gge_ordering(a, sigma)
        assert perm.dtype in (np.intp, np.int64, np.int32)


# ---------------------------------------------------------------------------
# TG method tests
# ---------------------------------------------------------------------------

class TestTGMethod:
    """Tests for the TG MVNCD method."""

    def test_bivariate_exact(self, xp):
        """TG reduces to exact bivariate CDF for K=2."""
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([1.0, 1.0])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(2), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=0.01)

    def test_trivariate_accuracy(self, xp):
        """TG accuracy for K=3 against scipy reference."""
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        # TG is less accurate than ME but should be within 10%
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_k4_accuracy(self, xp):
        """TG accuracy for K=4 against scipy reference."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((4, 4))
        sigma = A @ A.T / 4 + np.eye(4)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.8, 0.5, 1.0, 0.3])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(4), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_k5_accuracy(self, xp):
        """TG accuracy for K=5 against scipy reference."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((5, 5))
        sigma = A @ A.T / 5 + np.eye(5)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.5, 0.8, 0.3, 1.0, 0.6])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(5), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_independent_variables(self, xp):
        """TG with independent variables (diagonal covariance)."""
        sigma = np.eye(3)
        a = np.array([0.0, 0.0, 0.0])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        # P = 0.5^3 = 0.125
        np.testing.assert_allclose(p, 0.125, atol=0.02)

    def test_positive_result(self, xp):
        """TG always returns a non-negative probability."""
        sigma = np.array([[1.0, 0.9, 0.5], [0.9, 1.0, 0.4], [0.5, 0.4, 1.0]])
        a = np.array([-2.0, -1.5, -1.0])
        p = mvncd(xp.array(a), xp.array(sigma), method="tg", xp=xp)
        assert p >= 0.0
        assert p <= 1.0


# ---------------------------------------------------------------------------
# TGBME method tests
# ---------------------------------------------------------------------------

class TestTGBMEMethod:
    """Tests for the TGBME MVNCD method."""

    def test_bivariate_exact(self, xp):
        """TGBME reduces to exact bivariate CDF for K=2."""
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([1.0, 1.0])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(2), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=0.01)

    def test_trivariate_accuracy(self, xp):
        """TGBME accuracy for K=3 against scipy reference."""
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        # TGBME should be reasonably accurate
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_k4_accuracy(self, xp):
        """TGBME accuracy for K=4 against scipy reference."""
        rng = np.random.default_rng(99)
        A = rng.standard_normal((4, 4))
        sigma = A @ A.T / 4 + np.eye(4)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.8, 0.5, 1.0, 0.3])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(4), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_k5_accuracy(self, xp):
        """TGBME accuracy for K=5 against scipy reference."""
        rng = np.random.default_rng(123)
        A = rng.standard_normal((5, 5))
        sigma = A @ A.T / 5 + np.eye(5)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.5, 0.8, 0.3, 1.0, 0.6])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(5), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_odd_k_last_univariate(self, xp):
        """TGBME handles odd K (last variable via univariate conditioning)."""
        sigma = np.array([
            [1.0, 0.3, 0.1, 0.2, 0.05],
            [0.3, 1.0, 0.4, 0.1, 0.15],
            [0.1, 0.4, 1.0, 0.3, 0.10],
            [0.2, 0.1, 0.3, 1.0, 0.20],
            [0.05, 0.15, 0.10, 0.20, 1.0],
        ])
        a = np.array([1.0, 0.5, 0.8, 0.3, 0.6])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(5), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)

    def test_independent_variables(self, xp):
        """TGBME with independent variables."""
        sigma = np.eye(4)
        a = np.array([0.0, 0.0, 0.0, 0.0])
        p = mvncd(xp.array(a), xp.array(sigma), method="tgbme", xp=xp)
        # P = 0.5^4 = 0.0625
        np.testing.assert_allclose(p, 0.0625, atol=0.01)


# ---------------------------------------------------------------------------
# Combined method comparison tests
# ---------------------------------------------------------------------------

class TestAllMethodsIncludingNew:
    """Compare all methods (including TG and TGBME) against scipy."""

    @pytest.mark.parametrize(
        "method",
        ["me", "ovus", "bme", "tvbs", "ovbs", "tg", "tgbme"],
    )
    def test_bivariate_all_methods(self, xp, method):
        """All methods should reduce to exact bivariate CDF for K=2."""
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([1.0, 1.0])
        p = mvncd(xp.array(a), xp.array(sigma), method=method, xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(2), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=0.02)

    @pytest.mark.parametrize(
        "method",
        ["me", "ovus", "bme", "tvbs", "ovbs", "tg", "tgbme"],
    )
    def test_trivariate_all_methods(self, xp, method):
        """All methods for K=3 within reasonable accuracy."""
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p = mvncd(xp.array(a), xp.array(sigma), method=method, xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        np.testing.assert_allclose(p, p_ref, rtol=0.10)


# ---------------------------------------------------------------------------
# Existing methods still work
# ---------------------------------------------------------------------------

class TestExistingMethodsStillWork:
    """Verify existing methods are not broken by the additions."""

    @pytest.fixture
    def pentavariate_setup(self):
        rng = np.random.default_rng(123)
        A = rng.standard_normal((5, 5))
        sigma = A @ A.T / 5 + np.eye(5)
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)
        a = np.array([0.5, 0.8, 0.3, 1.0, 0.6])
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(5), cov=sigma)
        return a, sigma, p_ref

    @pytest.mark.parametrize(
        "method,rtol",
        [("me", 0.05), ("ovus", 0.05), ("bme", 0.03), ("tvbs", 0.01), ("ovbs", 0.03)],
    )
    def test_existing_methods_k5(self, xp, pentavariate_setup, method, rtol):
        """Existing methods still produce correct results."""
        a, sigma, p_ref = pentavariate_setup
        p = mvncd(xp.array(a), xp.array(sigma), method=method, xp=xp)
        np.testing.assert_allclose(p, p_ref, rtol=rtol)
