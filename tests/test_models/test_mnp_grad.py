"""Finite-difference verification of MNP analytic gradient (_mnp_grad_analytic.py).

Tests the analytic gradient chain (MVNCD ME → MNP parameters) against central
finite differences for:
- IID MNP (beta only)
- Heteroscedastic MNP (beta + scale params)
- Full covariance MNP (beta + scale + correlation params)
- Mixed MNP with diagonal random coefficients
- Mixed MNP with full random coefficients
- NLL consistency with mnp_loglik (method=me)
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.models.mnp._mnp_control import MNPControl
from pybhatlib.models.mnp._mnp_grad_analytic import mnp_analytic_gradient
from pybhatlib.models.mnp._mnp_loglik import mnp_loglik


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(N: int, I: int, n_vars: int, seed: int = 42):
    """Generate synthetic MNP data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, I, n_vars))
    # Generate choices based on simple utility model
    beta_true = rng.standard_normal(n_vars) * 0.5
    V = X @ beta_true
    # Add Gumbel noise (approximate probit-like choices)
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
        # Use mnp_loglik with method="me" so probability matches analytic
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
# Test: NLL consistency
# ---------------------------------------------------------------------------


class TestNLLConsistency:
    """Analytic NLL matches mnp_loglik with method='me'."""

    def test_iid_nll(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me")
        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)

    def test_full_cov_nll(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # beta(2) + scale(2) + corr(1) = 5 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0])
        control = MNPControl(iid=False, heteronly=False, method="me")
        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)

    def test_mixed_nll(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0]
        # beta(2) + omega(1 diag) = 3 params
        theta = np.array([0.3, -0.2, -0.5])
        control = MNPControl(iid=True, mix=True, randdiag=True, method="me")
        nll_analytic, _ = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        nll_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        np.testing.assert_allclose(nll_analytic, nll_loglik, atol=1e-12, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test: IID MNP gradient
# ---------------------------------------------------------------------------


class TestIIDGradient:
    """IID MNP: only beta gradient, no covariance parameters."""

    def test_small_3alt(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_4alt_3vars(self):
        N, I, n_vars = 15, 4, 3
        X, y, avail = _make_data(N, I, n_vars, seed=123)
        theta = np.array([0.5, -0.3, 0.1])
        control = MNPControl(iid=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_5alt(self):
        N, I, n_vars = 10, 5, 2
        X, y, avail = _make_data(N, I, n_vars, seed=99)
        theta = np.array([0.4, -0.1])
        control = MNPControl(iid=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: Heteroscedastic MNP gradient
# ---------------------------------------------------------------------------


class TestHeteroscedasticGradient:
    """Heteroscedastic MNP: beta + scale parameters."""

    def test_3alt(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        dim_lambda = I - 1  # 2
        # beta(2) + scale(2) = 4 params
        theta = np.array([0.3, -0.2, 0.1, 0.2])
        control = MNPControl(iid=False, heteronly=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_4alt(self):
        N, I, n_vars = 12, 4, 2
        X, y, avail = _make_data(N, I, n_vars, seed=77)
        # beta(2) + scale(3) = 5 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, -0.1])
        control = MNPControl(iid=False, heteronly=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: Full covariance MNP gradient
# ---------------------------------------------------------------------------


class TestFullCovarianceGradient:
    """Full covariance MNP: beta + scale + correlation parameters."""

    def test_3alt(self):
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        dim_lambda = I - 1  # 2
        n_scale = dim_lambda  # 2
        n_corr = dim_lambda * (dim_lambda - 1) // 2  # 1
        # beta(2) + scale(2) + corr(1) = 5 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0])
        control = MNPControl(iid=False, heteronly=False, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_4alt(self):
        N, I, n_vars = 12, 4, 2
        X, y, avail = _make_data(N, I, n_vars, seed=77)
        dim_lambda = I - 1  # 3
        n_scale = 3
        n_corr = 3
        # beta(2) + scale(3) + corr(3) = 8 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0])
        control = MNPControl(iid=False, heteronly=False, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)

    def test_nonzero_corr(self):
        """Full covariance with nonzero initial correlation angles."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=55)
        # beta(2) + scale(2) + corr(1) = 5 params, nonzero corr angle
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.5])
        control = MNPControl(iid=False, heteronly=False, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Mixed MNP with diagonal Omega
# ---------------------------------------------------------------------------


class TestMixedDiagonalGradient:
    """Mixed MNP with diagonal random coefficients."""

    def test_1_ranvar_iid(self):
        """One random coefficient, IID kernel errors."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0]
        # beta(2) + omega(1 diag) = 3 params
        theta = np.array([0.3, -0.2, -0.5])
        control = MNPControl(iid=True, mix=True, randdiag=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_2_ranvar_heteronly(self):
        """Two random coefficients, heteroscedastic kernel."""
        N, I, n_vars = 10, 3, 3
        X, y, avail = _make_data(N, I, n_vars, seed=88)
        ranvar_indices = [0, 1]
        # beta(3) + scale(2) + omega(2 diag) = 7 params
        theta = np.array([0.3, -0.2, 0.1, 0.1, 0.2, -0.3, -0.4])
        control = MNPControl(
            iid=False, heteronly=True, mix=True, randdiag=True, method="me",
        )

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Mixed MNP with full Omega
# ---------------------------------------------------------------------------


class TestMixedFullGradient:
    """Mixed MNP with full (lower triangular) Omega parameterization."""

    def test_2_ranvar_full(self):
        """Two random coefficients, full Cholesky, IID kernel."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        ranvar_indices = [0, 1]
        # beta(2) + omega(3 lower-tri) = 5 params
        # L = [[L00, 0], [L10, L11]]
        theta = np.array([0.3, -0.2, 0.5, 0.1, 0.4])
        control = MNPControl(iid=True, mix=True, randdiag=False, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)

    def test_2_ranvar_full_cov_kernel(self):
        """Two random coefficients, full Cholesky, full covariance kernel."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=33)
        ranvar_indices = [0, 1]
        # beta(2) + scale(2) + corr(1) + omega(3) = 8 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0, 0.5, 0.1, 0.4])
        control = MNPControl(
            iid=False, heteronly=False, mix=True, randdiag=False, method="me",
        )

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )
        fd_grad = _fd_gradient(
            theta, X, y, avail, I, n_vars, control, ranvar_indices,
        )

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-4, rtol=2e-3)


# ---------------------------------------------------------------------------
# Test: Variable availability
# ---------------------------------------------------------------------------


class TestAvailabilityGradient:
    """Gradient correctness with some alternatives unavailable."""

    def test_some_unavailable(self):
        N, I, n_vars = 8, 4, 2
        X, y, avail = _make_data(N, I, n_vars, seed=44)
        # Make alt 2 unavailable for first 3 obs (ensure chosen != 2 for them)
        for q in range(3):
            if y[q] == 2:
                y[q] = 0
            avail[q, 2] = 0.0

        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: 2 alternatives, single observation."""

    def test_2_alts(self):
        """Binary probit: only 1D MVNCD, gradient should be exact."""
        N, I, n_vars = 10, 2, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me")

        nll, grad = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )
        fd_grad = _fd_gradient(theta, X, y, avail, I, n_vars, control)

        assert nll > 0
        np.testing.assert_allclose(grad, fd_grad, atol=1e-8, rtol=1e-6)

    def test_nseg_raises(self):
        """nseg > 1 should raise NotImplementedError."""
        N, I, n_vars = 5, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, nseg=2, method="me")

        with pytest.raises(NotImplementedError, match="nseg > 1"):
            mnp_analytic_gradient(
                theta, X, y, avail, I, n_vars, control,
            )


# ---------------------------------------------------------------------------
# Test: mnp_loglik integration (Phase D2)
# ---------------------------------------------------------------------------


class TestMnpLoglikIntegration:
    """Verify mnp_loglik dispatches to analytic gradient correctly."""

    def test_loglik_returns_analytic_grad_for_me(self):
        """mnp_loglik(return_gradient=True, method='me') uses analytic gradient."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me", analytic_grad=True)

        nll_loglik, grad_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        nll_analytic, grad_analytic = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )

        np.testing.assert_allclose(nll_loglik, nll_analytic, atol=1e-14)
        np.testing.assert_allclose(grad_loglik, grad_analytic, atol=1e-14)

    def test_loglik_falls_back_to_numerical_for_ovus(self):
        """mnp_loglik(method='ovus') uses numerical gradient even with analytic_grad=True."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="ovus", analytic_grad=True)

        nll, grad = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        # Should still work (numerical FD)
        assert nll > 0
        assert len(grad) == n_vars

    def test_loglik_analytic_grad_false_uses_numerical(self):
        """analytic_grad=False forces numerical gradient."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        theta = np.array([0.3, -0.2])
        control = MNPControl(iid=True, method="me", analytic_grad=False)

        nll, grad = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        # Numerical gradient: NLL matches but grad may differ slightly from analytic
        assert nll > 0
        assert len(grad) == n_vars

    def test_loglik_full_cov_integration(self):
        """Full covariance gradient via mnp_loglik matches direct analytic call."""
        N, I, n_vars = 10, 3, 2
        X, y, avail = _make_data(N, I, n_vars)
        # beta(2) + scale(2) + corr(1) = 5 params
        theta = np.array([0.3, -0.2, 0.1, 0.2, 0.0])
        control = MNPControl(iid=False, heteronly=False, method="me")

        nll_loglik, grad_loglik = mnp_loglik(
            theta, X, y, avail, I, n_vars, control, return_gradient=True,
        )
        nll_analytic, grad_analytic = mnp_analytic_gradient(
            theta, X, y, avail, I, n_vars, control,
        )

        np.testing.assert_allclose(nll_loglik, nll_analytic, atol=1e-14)
        np.testing.assert_allclose(grad_loglik, grad_analytic, atol=1e-14)


class TestBFGSConvergence:
    """BFGS convergence with analytic vs numerical gradient."""

    @pytest.mark.slow
    def test_iid_bfgs_convergence(self):
        """IID MNP: BFGS with analytic gradient converges to similar optimum."""
        from scipy.optimize import minimize

        N, I, n_vars = 50, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=42)
        theta0 = np.zeros(n_vars)

        # Analytic gradient
        control_me = MNPControl(iid=True, method="me", analytic_grad=True)

        def obj_analytic(th):
            nll, grad = mnp_loglik(
                th, X, y, avail, I, n_vars, control_me,
                return_gradient=True,
            )
            return nll, grad

        res_analytic = minimize(
            obj_analytic, theta0, method="BFGS", jac=True,
            options={"maxiter": 100, "gtol": 1e-5},
        )

        # Numerical gradient
        control_num = MNPControl(iid=True, method="me", analytic_grad=False)

        def obj_numerical(th):
            nll, grad = mnp_loglik(
                th, X, y, avail, I, n_vars, control_num,
                return_gradient=True,
            )
            return nll, grad

        res_numerical = minimize(
            obj_numerical, theta0, method="BFGS", jac=True,
            options={"maxiter": 100, "gtol": 1e-5},
        )

        # Both should converge to similar optimum
        np.testing.assert_allclose(
            res_analytic.fun, res_numerical.fun, atol=1e-4, rtol=1e-3,
        )
        np.testing.assert_allclose(
            res_analytic.x, res_numerical.x, atol=1e-2, rtol=1e-2,
        )

    @pytest.mark.slow
    def test_full_cov_bfgs_convergence(self):
        """Full covariance MNP: BFGS with analytic gradient converges."""
        from scipy.optimize import minimize

        N, I, n_vars = 50, 3, 2
        X, y, avail = _make_data(N, I, n_vars, seed=42)
        # beta(2) + scale(2) + corr(1) = 5 params
        theta0 = np.zeros(5)

        control = MNPControl(iid=False, heteronly=False, method="me")

        def obj(th):
            return mnp_loglik(
                th, X, y, avail, I, n_vars, control,
                return_gradient=True,
            )

        res = minimize(
            obj, theta0, method="BFGS", jac=True,
            options={"maxiter": 100, "gtol": 1e-4},
        )

        assert res.fun < mnp_loglik(theta0, X, y, avail, I, n_vars, control)
        assert res.success or res.fun < 1.2  # reasonable NLL for 3-alt
