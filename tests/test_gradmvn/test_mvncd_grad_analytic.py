"""Finite-difference verification of analytic ME gradient (_mvncd_grad_analytic.py).

Tests the backward/adjoint analytic gradient against:
- Exact formulas for K=1 and K=2
- Corrected finite differences for K>=3

Uses well-separated |a_k / sqrt(sigma_kk)| values to prevent variable
reordering changes under FD perturbation (which would cause discontinuous
probability jumps).

NOTE: The existing _grad_sigma_numerical in _mvncd_grad.py has a diagonal
perturbation bug (doubles eps for i==j). This test uses a corrected FD helper.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pybhatlib.gradmvn._mvncd import mvncd
from pybhatlib.gradmvn._mvncd_grad_analytic import mvncd_grad_me_analytic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fd_grad_a(a, sigma, eps=1e-6):
    """Central FD for d(prob)/d(a_k)."""
    K = len(a)
    grad = np.zeros(K)
    for k in range(K):
        a_plus = a.copy()
        a_plus[k] += eps
        a_minus = a.copy()
        a_minus[k] -= eps
        prob_plus = mvncd(a_plus, sigma, method="me")
        prob_minus = mvncd(a_minus, sigma, method="me")
        grad[k] = (prob_plus - prob_minus) / (2.0 * eps)
    return grad


def _fd_grad_sigma(a, sigma, eps=1e-6):
    """Corrected central FD for d(prob)/d(sigma_vech).

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
            prob_plus = mvncd(a, sigma_plus, method="me")
            prob_minus = mvncd(a, sigma_minus, method="me")
            grad[idx] = (prob_plus - prob_minus) / (2.0 * eps)
            idx += 1
    return grad


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestMvncdGradAnalyticK0:
    """K=0 edge case: P=1, no gradients."""

    def test_k0(self):
        prob, ga, gs = mvncd_grad_me_analytic(np.array([]), np.zeros((0, 0)))
        assert prob == 1.0
        assert len(ga) == 0
        assert len(gs) == 0


class TestMvncdGradAnalyticK1:
    """K=1: P = Phi(a/sd), exact analytic gradient."""

    @pytest.mark.parametrize("a_val, sig_val", [
        (0.5, 1.0),
        (1.0, 2.0),
        (-1.0, 0.5),
        (2.0, 3.0),
    ])
    def test_k1_exact(self, a_val, sig_val):
        a = np.array([a_val])
        sigma = np.array([[sig_val]])
        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)

        sd = np.sqrt(sig_val)
        w = a_val / sd
        expected_prob = norm.cdf(w)
        expected_ga = norm.pdf(w) / sd
        expected_gs = norm.pdf(w) * (-w / (2.0 * sig_val))

        assert abs(prob - expected_prob) < 1e-14
        np.testing.assert_allclose(ga, [expected_ga], atol=1e-14)
        np.testing.assert_allclose(gs, [expected_gs], atol=1e-14)


class TestMvncdGradAnalyticK2:
    """K=2: exact bivariate normal CDF gradient."""

    @pytest.mark.parametrize("rho", [0.3, 0.7, -0.5, 0.0])
    def test_k2_vs_fd(self, rho):
        a = np.array([0.8, 1.2])
        sigma = np.array([[1.5, rho * np.sqrt(1.5 * 2.0)],
                          [rho * np.sqrt(1.5 * 2.0), 2.0]])

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)

        # Compare against corrected FD
        fd_ga = _fd_grad_a(a, sigma, eps=1e-7)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-7)

        assert prob > 1e-10
        np.testing.assert_allclose(ga, fd_ga, atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-8, rtol=1e-5)


class TestMvncdGradAnalyticK3Identity:
    """K=3 with identity covariance: P = prod Phi(a_k), exact verification.

    With identity covariance, ME is exact: P = Phi(a1)*Phi(a2)*Phi(a3).
    grad_a has exact closed-form; grad_sigma verified against FD.

    Note: off-diagonal d(P)/d(sigma_{ij}) is NOT zero even at identity —
    it measures sensitivity to introducing correlation (e.g., d(P)/d(sigma_{01})
    = phi(a_0)*phi(a_1)*Phi(a_2) at identity covariance).
    """

    def test_k3_identity_prob_and_grad_a(self):
        a = np.array([0.3, 0.8, 1.5])
        sigma = np.eye(3)

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)

        Phi = norm.cdf(a)
        phi = norm.pdf(a)
        expected_prob = np.prod(Phi)

        expected_ga = np.array([
            phi[0] * Phi[1] * Phi[2],
            phi[1] * Phi[0] * Phi[2],
            phi[2] * Phi[0] * Phi[1],
        ])

        assert abs(prob - expected_prob) < 1e-12
        np.testing.assert_allclose(ga, expected_ga, atol=1e-10, rtol=1e-8)

    def test_k3_identity_grad_sigma_vs_fd(self):
        a = np.array([0.3, 0.8, 1.5])
        sigma = np.eye(3)

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        np.testing.assert_allclose(gs, fd_gs, atol=1e-7, rtol=1e-3)


class TestMvncdGradAnalyticK3Correlated:
    """K=3 with moderate correlations, verified against corrected FD."""

    def test_k3_moderate_corr(self):
        # Well-separated standardized values
        a = np.array([0.3, 0.9, 1.8])
        sigma = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.5, 0.2],
            [0.1, 0.2, 2.0],
        ])

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma, eps=1e-6)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-7, rtol=1e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-7, rtol=1e-3)

    def test_k3_negative_corr(self):
        a = np.array([0.5, 1.0, 1.5])
        sigma = np.array([
            [1.0, -0.3, 0.1],
            [-0.3, 1.0, -0.2],
            [0.1, -0.2, 1.0],
        ])

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma, eps=1e-6)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-7, rtol=1e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-7, rtol=1e-3)


class TestMvncdGradAnalyticK4:
    """K=4 with moderate correlations, verified against corrected FD."""

    def test_k4(self):
        a = np.array([0.2, 0.6, 1.1, 1.8])
        sigma = np.array([
            [1.0, 0.2, 0.1, 0.05],
            [0.2, 1.3, 0.15, 0.1],
            [0.1, 0.15, 1.6, 0.2],
            [0.05, 0.1, 0.2, 2.0],
        ])

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma, eps=1e-6)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=2e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=2e-3)


class TestMvncdGradAnalyticK5:
    """K=5 with moderate correlations, verified against corrected FD."""

    def test_k5(self):
        a = np.array([0.2, 0.5, 0.9, 1.3, 1.8])
        sigma = np.array([
            [1.0, 0.2, 0.1, 0.05, 0.02],
            [0.2, 1.2, 0.15, 0.1, 0.05],
            [0.1, 0.15, 1.4, 0.12, 0.08],
            [0.05, 0.1, 0.12, 1.6, 0.15],
            [0.02, 0.05, 0.08, 0.15, 2.0],
        ])

        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        fd_ga = _fd_grad_a(a, sigma, eps=1e-6)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        assert prob > 1e-6
        np.testing.assert_allclose(ga, fd_ga, atol=1e-6, rtol=3e-3)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-6, rtol=3e-3)


class TestMvncdGradAnalyticProbConsistency:
    """Probability from analytic gradient matches mvncd(method='me')."""

    @pytest.mark.parametrize("K", [1, 2, 3, 4, 5])
    def test_prob_matches_mvncd(self, K):
        rng = np.random.default_rng(42 + K)
        a = np.sort(rng.uniform(0.2, 2.0, K))  # sorted for stable ordering
        # Generate PD covariance via random factor model
        L = np.eye(K) + 0.2 * rng.standard_normal((K, K))
        sigma = L @ L.T / K

        prob_analytic, _, _ = mvncd_grad_me_analytic(a, sigma)
        prob_mvncd = mvncd(a, sigma, method="me")

        np.testing.assert_allclose(prob_analytic, prob_mvncd, atol=1e-12, rtol=1e-10)


class TestMvncdGradAnalyticEdgeCases:
    """Edge cases: very small probabilities, high truncation."""

    def test_large_negative_a(self):
        """Very small probability region — should return zeros gracefully."""
        a = np.array([-5.0, -5.0, -5.0])
        sigma = np.eye(3)
        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        # prob ~ Phi(-5)^3 ~ 2.87e-7^3 ~ 2.36e-20 — still computable
        assert prob >= 0.0
        # Gradients should be finite
        assert np.all(np.isfinite(ga))
        assert np.all(np.isfinite(gs))

    def test_extreme_truncation(self):
        """Extreme truncation where prob -> 0."""
        a = np.array([-30.0, -30.0, -30.0])
        sigma = np.eye(3)
        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)
        assert prob == 0.0
        np.testing.assert_array_equal(ga, np.zeros(3))
        np.testing.assert_array_equal(gs, np.zeros(6))

    def test_nonunit_variance(self):
        """Non-unit diagonal variances."""
        a = np.array([1.0, 1.5, 2.0])
        sigma = np.diag([2.0, 3.0, 4.0])
        prob, ga, gs = mvncd_grad_me_analytic(a, sigma)

        fd_ga = _fd_grad_a(a, sigma, eps=1e-6)
        fd_gs = _fd_grad_sigma(a, sigma, eps=1e-6)

        assert prob > 0.1
        np.testing.assert_allclose(ga, fd_ga, atol=1e-8, rtol=1e-4)
        np.testing.assert_allclose(gs, fd_gs, atol=1e-8, rtol=1e-4)
