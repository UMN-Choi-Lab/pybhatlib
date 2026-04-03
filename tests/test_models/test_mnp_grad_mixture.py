"""Finite-difference verification of mixture-of-normals MNP analytic gradient.

Tests the analytic gradient chain for mixture models (nseg > 1) against
central finite differences for:
- nseg=2, IID kernel, no random coefficients
- nseg=2, flexible covariance kernel, no random coefficients
- nseg=2, IID kernel, diagonal random coefficients
- nseg=2, IID kernel, full random coefficients
- nseg=2, flexible covariance kernel, full random coefficients
- nseg=3, IID kernel, no random coefficients
- NLL consistency with mnp_loglik (method=me)
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.models.mnp._mnp_control import MNPControl
from pybhatlib.models.mnp._mnp_grad_analytic import mnp_analytic_gradient
from pybhatlib.models.mnp._mnp_loglik import mnp_loglik, count_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(N: int, I: int, n_vars: int, seed: int = 42):
    """Generate synthetic MNP data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, I, n_vars))
    beta_true = rng.standard_normal(n_vars) * 0.5
    V = X @ beta_true
    noise = rng.standard_normal((N, I))
    utility = V + noise
    y = utility.argmax(axis=1)
    avail = np.ones((N, I), dtype=np.float64)
    return X, y, avail


def _fd_gradient(
    theta, X, y, avail, n_alts, n_beta, control, ranvar_indices=None, eps=1e-6,
):
    """Central finite-difference gradient of negative mean log-likelihood."""
    n_params = len(theta)
    grad = np.zeros(n_params, dtype=np.float64)
    for k in range(n_params):
        theta_plus = theta.copy()
        theta_plus[k] += eps
        theta_minus = theta.copy()
        theta_minus[k] -= eps
        control_me = MNPControl(
            iid=control.iid,
            mix=control.mix,
            heteronly=control.heteronly,
            randdiag=control.randdiag,
            nseg=control.nseg,
            method="me",
        )
        nll_plus = mnp_loglik(
            theta_plus, X, y, avail, n_alts, n_beta, control_me, ranvar_indices,
        )
        nll_minus = mnp_loglik(
            theta_minus, X, y, avail, n_alts, n_beta, control_me, ranvar_indices,
        )
        grad[k] = (nll_plus - nll_minus) / (2.0 * eps)
    return grad


# ---------------------------------------------------------------------------
# Test: NLL consistency for mixture models
# ---------------------------------------------------------------------------


class TestMixtureNLLConsistency:
    """Analytic NLL matches mnp_loglik with method='me' for mixture models."""

    def test_nseg2_iid_nll(self):
        """nseg=2, IID: NLL from analytic matches mnp_loglik."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # Layout: beta_1(2) + seg_params(1) + beta_2(2) = 5
        n_params = count_params(n_vars, I, MNPControl(iid=True, nseg=2))
        assert n_params == 5
        theta = np.array([0.3, -0.2, 0.5, 0.1, -0.3])
        control = MNPControl(iid=True, nseg=2, method="me")

        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)

    def test_nseg2_flexible_nll(self):
        """nseg=2, flexible covariance: NLL from analytic matches mnp_loglik."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # Layout: beta_1(2) + lambda(3) + seg_params(1) + beta_2(2) = 8
        control = MNPControl(iid=False, heteronly=False, nseg=2, method="me")
        n_params = count_params(n_vars, I, control)
        assert n_params == 8
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0, 0.5, 0.1, -0.3])

        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)

    def test_nseg2_mix_diag_nll(self):
        """nseg=2, IID + diagonal random coeff: NLL consistency."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0]
        # Layout: beta_1(2) + omega_1(1) + seg_params(1) + beta_2(2) + omega_2(1) = 7
        control = MNPControl(iid=True, mix=True, randdiag=True, nseg=2, method="me")
        n_params = count_params(n_vars, I, control, ranvar_indices)
        assert n_params == 7
        theta = np.array([0.3, -0.2, -0.5, 0.5, 0.1, -0.3, -0.4])

        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test: nseg=2, IID kernel, no random coefficients
# ---------------------------------------------------------------------------


class TestMixtureIIDGradient:
    """Mixture (nseg=2) with IID kernel, no random coefficients."""

    def test_3alt_2vars(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # Layout: beta_1(2) + seg_params(1) + beta_2(2) = 5
        theta = np.array([0.3, -0.2, 0.5, 0.1, -0.3])
        control = MNPControl(iid=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_4alt_3vars(self):
        N, I, n_vars = 12, 4, 3
        X, y, avail = _make_data(N, I, n_vars, seed=77)
        # Layout: beta_1(3) + seg_params(1) + beta_2(3) = 7
        theta = np.array([0.3, -0.2, 0.1, 0.5, 0.4, -0.1, 0.2])
        control = MNPControl(iid=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_different_segment_params(self):
        """Test with extreme segment weight (one segment dominates)."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=99)
        # Large positive seg_param means segment 2 dominates
        theta = np.array([0.3, -0.2, 2.0, 0.1, -0.3])
        control = MNPControl(iid=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: nseg=2, flexible covariance, no random coefficients
# ---------------------------------------------------------------------------


class TestMixtureFlexibleGradient:
    """Mixture (nseg=2) with flexible covariance kernel."""

    def test_3alt_heteronly(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # Layout: beta_1(2) + scale(2) + seg_params(1) + beta_2(2) = 7
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.5, 0.1, -0.3])
        control = MNPControl(iid=False, heteronly=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_3alt_full_cov(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # Layout: beta_1(2) + scale(2) + corr(1) + seg_params(1) + beta_2(2) = 8
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0, 0.5, 0.1, -0.3])
        control = MNPControl(iid=False, heteronly=False, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_4alt_full_cov(self):
        N, I, n_vars = 12, 4, 2
        X, y, avail = _make_data(N, I, n_vars, seed=77)
        # Layout: beta_1(2) + scale(3) + corr(3) + seg_params(1) + beta_2(2) = 11
        theta = np.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0, 0.5, 0.4, -0.1])
        control = MNPControl(iid=False, heteronly=False, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: nseg=2, IID kernel, diagonal random coefficients
# ---------------------------------------------------------------------------


class TestMixtureMixedDiagGradient:
    """Mixture (nseg=2) with IID kernel and diagonal random coefficients."""

    def test_1_ranvar(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0]
        # Layout: beta_1(2) + omega_1(1) + seg_params(1) + beta_2(2) + omega_2(1) = 7
        theta = np.array([0.3, -0.2, -0.5, 0.5, 0.1, -0.3, -0.4])
        control = MNPControl(iid=True, mix=True, randdiag=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_2_ranvar(self):
        N, I, n_vars = 10, 3, 3
        X, y, avail = _make_data(N, I, n_vars, seed=88)
        ranvar_indices = [0, 1]
        # Layout: beta_1(3) + omega_1(2) + seg_params(1) + beta_2(3) + omega_2(2) = 11
        theta = np.array([0.3, -0.2, 0.1, -0.3, -0.4, 0.5, 0.4, -0.1, 0.2, -0.2, -0.5])
        control = MNPControl(iid=True, mix=True, randdiag=True, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: nseg=2, IID kernel, full random coefficients
# ---------------------------------------------------------------------------


class TestMixtureMixedFullGradient:
    """Mixture (nseg=2) with IID kernel and full Cholesky random coefficients."""

    def test_2_ranvar_full(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0, 1]
        # Layout: beta_1(2) + omega_1(3) + seg_params(1) + beta_2(2) + omega_2(3) = 11
        theta = np.array([0.3, -0.2, 0.5, 0.1, 0.4, 0.5, 0.1, -0.3, 0.4, 0.2, 0.3])
        control = MNPControl(iid=True, mix=True, randdiag=False, nseg=2, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: nseg=2, flexible covariance, full random coefficients
# ---------------------------------------------------------------------------


class TestMixtureFlexibleMixedGradient:
    """Mixture (nseg=2) with flexible kernel and full random coefficients."""

    def test_3alt_full_cov_full_omega(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=33)
        ranvar_indices = [0, 1]
        # Layout: beta_1(2) + scale(2) + corr(1) + omega_1(3) + seg_params(1) +
        #         beta_2(2) + omega_2(3) = 14
        control = MNPControl(
            iid=False, heteronly=False, mix=True, randdiag=False, nseg=2, method="me",
        )
        n_params = count_params(n_vars, I, control, ranvar_indices)
        assert n_params == 14
        theta = np.array([
            0.3, -0.2,         # beta_1
            0.1, 0.2, 0.0,     # scale + corr
            0.5, 0.1, 0.4,     # omega_1
            0.5,                # seg_param
            0.1, -0.3,         # beta_2
            0.4, 0.2, 0.3,     # omega_2
        ])

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: nseg=3, IID kernel, no random coefficients
# ---------------------------------------------------------------------------


class TestMixture3SegGradient:
    """Mixture with 3 segments."""

    def test_nseg3_iid(self):
        N, I, n_vars = 15, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=55)
        # Layout: beta_1(2) + seg_params(2) + beta_2(2) + beta_3(2) = 8
        theta = np.array([0.3, -0.2, 0.5, -0.3, 0.1, -0.3, 0.4, 0.2])
        control = MNPControl(iid=True, nseg=3, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: mnp_loglik integration for mixture models
# ---------------------------------------------------------------------------


class TestMixtureLoglikIntegration:
    """Verify mnp_loglik dispatches to analytic gradient for mixture models."""

    def test_loglik_returns_analytic_grad_for_mixture(self):
        """mnp_loglik(return_gradient=True, method='me', nseg=2) uses analytic gradient."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2, 0.5, 0.1, -0.3])
        control = MNPControl(iid=True, nseg=2, method="me", analytic_grad=True)

        nll_loglik, grad_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        nll_analytic, grad_analytic = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )

        np.testing.assert_allclose(nll_loglik, nll_analytic, atol=1e-14)
        np.testing.assert_allclose(grad_loglik, grad_analytic, atol=1e-14)

    def test_loglik_mixture_analytic_grad_false_uses_numerical(self):
        """analytic_grad=False forces numerical gradient for mixture models."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2, 0.5, 0.1, -0.3])
        control = MNPControl(
            iid=True, nseg=2, method="me", analytic_grad=False,
        )

        nll, grad = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        assert nll > 0
        assert len(grad) == 5


# ---------------------------------------------------------------------------
# Test: OVUS method for mixture
# ---------------------------------------------------------------------------


class TestMixtureOVUSGradient:
    """Mixture analytic gradient with OVUS method."""

    def test_nseg2_iid_ovus(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2, 0.5, 0.1, -0.3])
        control = MNPControl(iid=True, nseg=2, method="ovus")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)
