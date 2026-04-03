"""Finite-difference verification of analytic OVUS gradient (_mvncd_grad_ovus.py).

Tests the backward/adjoint analytic gradient of the OVUS-approximated MVNCD
against corrected central finite differences.

OVUS = ME + bivariate screening. For K=1 and K=2, the OVUS gradient delegates
to the exact analytic formulas (same as ME). For K>=3, the OVUS-specific
forward pass is differentiated via reverse-mode AD.

Uses well-separated |a_k / sqrt(sigma_kk)| values to prevent variable
reordering changes under FD perturbation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pybhatlib.gradmvn._mvncd import mvncd
from pybhatlib.gradmvn._mvncd_grad_ovus import mvncd_grad_ovus_analytic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fd_grad_a(a, sigma, eps=1e-7):
    """Central FD for d(prob)/d(a_k) using OVUS forward pass."""
    K = len(a)
    grad = np.zeros(K)
    for k in range(K):
        a_plus = a.copy()
        a_plus[k] += eps
        a_minus = a.copy()
        a_minus[k] -= eps
        prob_plus = mvncd(a_plus, sigma, method="ovus")
        prob_minus = mvncd(a_minus, sigma, method="ovus")
        grad[k] = (prob_plus - prob_minus) / (2.0 * eps)
    return grad


def _fd_grad_sigma(a, sigma, eps=1e-7):
    """Corrected central FD for d(prob)/d(sigma_vech) using OVUS forward pass.

    For off-diagonal (i<j): perturbs both sigma[i,j] and sigma[j,i].
    For diagonal (i==j): perturbs sigma[i,i] once only.
    """
    K = len(a)
    n_vech = K * (K + 1) // 2
    grad = np.zeros(n_vech)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            sigma_plus = sigma.copy()
            sigma_minus = sigma.copy()
            sigma_plus[i, j] += eps
            sigma_minus[i, j] -= eps
            if i != j:
                sigma_plus[j, i] += eps
                sigma_minus[j, i] -= eps
            prob_plus = mvncd(a, sigma_plus, method="ovus")
            prob_minus = mvncd(a, sigma_minus, method="ovus")
            grad[idx] = (prob_plus - prob_minus) / (2.0 * eps)
            idx += 1
    return grad


# ---------------------------------------------------------------------------
# Test K=1 and K=2 (delegates to exact formulas, same as ME)
# ---------------------------------------------------------------------------


class TestOvusGradK1:
    """K=1: P = Phi(a/sd), delegates to exact formula."""

    def test_k1(self):
        a = np.array([0.5])
        sigma = np.array([[2.0]])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)

        sd = np.sqrt(2.0)
        w = 0.5 / sd
        expected_prob = norm.cdf(w)
        assert abs(prob - expected_prob) < 1e-14


class TestOvusGradK2:
    """K=2: exact BVN gradient, same as ME."""

    def test_k2_vs_fd(self):
        a = np.array([0.8, 1.2])
        sigma = np.array([[1.5, 0.3 * np.sqrt(1.5 * 2.0)],
                          [0.3 * np.sqrt(1.5 * 2.0), 2.0]])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-10
        np.testing.assert_allclose(ga, fd_ga, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-8, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test K=3: OVUS-specific gradient (bivariate screening)
# ---------------------------------------------------------------------------


class TestOvusGradK3:
    """K=3 OVUS gradient verified against finite differences."""

    def test_k3_identity(self):
        a = np.array([0.5, 1.0, 1.5])
        sigma = np.eye(3)
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 0.1
        np.testing.assert_allclose(ga, fd_ga, atol=1e-8, rtol=1e-4)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-8, rtol=1e-4)

    def test_k3_moderate_corr(self):
        """Well-separated standardized values with moderate correlations."""
        a = np.array([0.3, 0.9, 1.8])
        sigma = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.5, 0.2],
            [0.1, 0.2, 2.0],
        ])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-7, rtol=1e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-7, rtol=1e-3)

    def test_k3_negative_corr(self):
        """Negative correlations."""
        a = np.array([0.5, 1.0, 1.5])
        sigma = np.array([
            [1.0, -0.3, 0.1],
            [-0.3, 1.0, -0.2],
            [0.1, -0.2, 1.0],
        ])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-7, rtol=1e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-7, rtol=1e-3)

    def test_k3_nonunit_variance(self):
        """Non-unit diagonal variances."""
        a = np.array([1.0, 1.5, 2.0])
        sigma = np.diag([2.0, 3.0, 4.0])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 0.1
        np.testing.assert_allclose(ga, fd_ga, atol=1e-8, rtol=1e-4)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-8, rtol=1e-4)


# ---------------------------------------------------------------------------
# Test K=4
# ---------------------------------------------------------------------------


class TestOvusGradK4:
    """K=4 OVUS gradient verified against finite differences."""

    def test_k4(self):
        a = np.array([0.2, 0.6, 1.1, 1.8])
        sigma = np.array([
            [1.0, 0.2, 0.1, 0.05],
            [0.2, 1.3, 0.15, 0.1],
            [0.1, 0.15, 1.6, 0.2],
            [0.05, 0.1, 0.2, 2.0],
        ])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=2e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=2e-3)

    def test_k4_strong_corr(self):
        """Stronger correlations."""
        a = np.array([0.3, 0.7, 1.2, 2.0])
        sigma = np.array([
            [1.0, 0.5, 0.3, 0.1],
            [0.5, 1.5, 0.4, 0.2],
            [0.3, 0.4, 2.0, 0.5],
            [0.1, 0.2, 0.5, 2.5],
        ])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=2e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test K=5
# ---------------------------------------------------------------------------


class TestOvusGradK5:
    """K=5 OVUS gradient verified against finite differences."""

    def test_k5(self):
        a = np.array([0.2, 0.5, 0.9, 1.3, 1.8])
        sigma = np.array([
            [1.0, 0.2, 0.1, 0.05, 0.02],
            [0.2, 1.2, 0.15, 0.1, 0.05],
            [0.1, 0.15, 1.4, 0.12, 0.08],
            [0.05, 0.1, 0.12, 1.6, 0.15],
            [0.02, 0.05, 0.08, 0.15, 2.0],
        ])
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=3e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=3e-3)

    def test_k5_random_pd(self):
        """Random positive-definite covariance."""
        rng = np.random.default_rng(42)
        a = np.sort(rng.uniform(0.2, 2.0, 5))
        L = np.eye(5) + 0.2 * rng.standard_normal((5, 5))
        sigma = L @ L.T / 5

        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=3e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=3e-3)


# ---------------------------------------------------------------------------
# Probability consistency with OVUS forward pass
# ---------------------------------------------------------------------------


class TestOvusGradProbConsistency:
    """Probability from analytic gradient matches mvncd(method='ovus')."""

    @pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
    def test_prob_matches_mvncd(self, K):
        rng = np.random.default_rng(42 + K)
        a = np.sort(rng.uniform(0.2, 2.0, K))
        L = np.eye(K) + 0.2 * rng.standard_normal((K, K))
        sigma = L @ L.T / K

        prob_analytic, _, _ = mvncd_grad_ovus_analytic(a, sigma)
        prob_mvncd = mvncd(a, sigma, method="ovus")

        np.testing.assert_allclose(prob_analytic, prob_mvncd, atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestOvusGradEdgeCases:
    """Edge cases: K=0, extreme values."""

    def test_k0(self):
        prob, ga, gs = mvncd_grad_ovus_analytic(np.array([]), np.zeros((0, 0)))
        assert prob == 1.0
        assert len(ga) == 0
        assert len(gs) == 0

    def test_large_negative_a(self):
        """Very small probability region — should return zeros gracefully."""
        a = np.array([-5.0, -5.0, -5.0])
        sigma = np.eye(3)
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        assert prob >= 0.0
        assert np.all(np.isfinite(ga))
        assert np.all(np.isfinite(gs))

    def test_extreme_truncation(self):
        """Extreme truncation where prob -> 0."""
        a = np.array([-30.0, -30.0, -30.0])
        sigma = np.eye(3)
        prob, ga, gs = mvncd_grad_ovus_analytic(a, sigma)
        assert prob == 0.0
        np.testing.assert_array_equal(ga, np.zeros(3))
        np.testing.assert_array_equal(gs, np.zeros(6))
