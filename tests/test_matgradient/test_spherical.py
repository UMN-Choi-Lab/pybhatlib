"""Tests for spherical parameterization (Pinheiro & Bates / BHATLIB)."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.matgradient._spherical import (
    grad_corr_theta,
    theta_to_corr,
)


class TestThetaToCorr:
    """Verify theta_to_corr produces valid PD correlation matrices."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_unit_diagonal(self, K: int):
        rng = np.random.default_rng(42)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = theta_to_corr(theta, K)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-14)

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_symmetric(self, K: int):
        rng = np.random.default_rng(123)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = theta_to_corr(theta, K)
        np.testing.assert_allclose(corr, corr.T, atol=1e-14)

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_positive_definite(self, K: int):
        rng = np.random.default_rng(999)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = theta_to_corr(theta, K)
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() > 0, f"Not PD: min eigval = {eigvals.min()}"

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_correlations_in_range(self, K: int):
        rng = np.random.default_rng(77)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 2.0
        corr = theta_to_corr(theta, K)
        for i in range(K):
            for j in range(K):
                if i != j:
                    assert -1.0 <= corr[i, j] <= 1.0, (
                        f"corr[{i},{j}] = {corr[i,j]} out of range"
                    )

    def test_zero_theta_gives_identity(self):
        """theta=0 → logistic(0)=pi/2 → cos(pi/2)=0 → L=I → corr=I."""
        for K in [2, 3, 4, 5]:
            n_params = K * (K - 1) // 2
            theta = np.zeros(n_params)
            corr = theta_to_corr(theta, K)
            np.testing.assert_allclose(corr, np.eye(K), atol=1e-14)

    def test_constrained_vs_unconstrained(self):
        """Constrained and unconstrained should agree when angles match."""
        K = 4
        n_params = K * (K - 1) // 2
        rng = np.random.default_rng(88)
        theta_u = rng.standard_normal(n_params)
        # Map to constrained manually
        theta_c = np.pi / (1.0 + np.exp(-theta_u))
        corr_u = theta_to_corr(theta_u, K, constrained=False)
        corr_c = theta_to_corr(theta_c, K, constrained=True)
        np.testing.assert_allclose(corr_u, corr_c, atol=1e-14)

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_random_inputs_always_pd(self, K: int):
        """Fuzz test: random inputs always produce PD correlation."""
        rng = np.random.default_rng(314)
        n_params = K * (K - 1) // 2
        for _ in range(20):
            theta = rng.standard_normal(n_params) * 3.0
            corr = theta_to_corr(theta, K)
            eigvals = np.linalg.eigvalsh(corr)
            assert eigvals.min() > -1e-10


class TestGradCorrTheta:
    """Verify analytic Jacobian against finite differences."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_jacobian_vs_fd_unconstrained(self, K: int):
        """Analytic Jacobian matches FD for unconstrained parameters."""
        rng = np.random.default_rng(42)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 0.5

        jac_analytic = grad_corr_theta(theta, K, constrained=False)
        jac_fd = _fd_jacobian(theta, K, constrained=False)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"K={K}: analytic Jacobian differs from FD (unconstrained)",
        )

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_jacobian_vs_fd_constrained(self, K: int):
        """Analytic Jacobian matches FD for constrained angles in (0, pi)."""
        rng = np.random.default_rng(123)
        n_params = K * (K - 1) // 2
        theta = rng.uniform(0.2, np.pi - 0.2, size=n_params)

        jac_analytic = grad_corr_theta(theta, K, constrained=True)
        jac_fd = _fd_jacobian(theta, K, constrained=True)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"K={K}: analytic Jacobian differs from FD (constrained)",
        )

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_jacobian_vs_fd_large(self, K: int):
        """Analytic Jacobian matches FD at larger parameter values."""
        rng = np.random.default_rng(456)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 2.0

        jac_analytic = grad_corr_theta(theta, K, constrained=False)
        jac_fd = _fd_jacobian(theta, K, constrained=False)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"K={K}: analytic Jacobian differs from FD (large theta)",
        )

    def test_jacobian_vs_fd_zero(self):
        """Analytic Jacobian matches FD at theta=0."""
        for K in [2, 3, 4]:
            n_params = K * (K - 1) // 2
            theta = np.zeros(n_params)

            jac_analytic = grad_corr_theta(theta, K, constrained=False)
            jac_fd = _fd_jacobian(theta, K, constrained=False)

            np.testing.assert_allclose(
                jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
                err_msg=f"K={K}: Jacobian at theta=0",
            )

    def test_jacobian_shape(self):
        """Verify Jacobian dimensions."""
        for K in [2, 3, 4, 5]:
            n_theta = K * (K - 1) // 2
            n_upper = K * (K + 1) // 2
            theta = np.zeros(n_theta)
            jac = grad_corr_theta(theta, K)
            assert jac.shape == (n_theta, n_upper)

    def test_diagonal_entries_zero_gradient(self):
        """Diagonal of corr is always 1, so its gradient should be zero."""
        K = 4
        n_params = K * (K - 1) // 2
        rng = np.random.default_rng(55)
        theta = rng.standard_normal(n_params)

        jac = grad_corr_theta(theta, K)

        # Identify diagonal indices in corr_upper vector
        diag_indices = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    diag_indices.append(idx)
                idx += 1

        for d_idx in diag_indices:
            np.testing.assert_allclose(
                jac[:, d_idx], 0.0, atol=1e-10,
                err_msg=f"Diagonal entry {d_idx} has non-zero gradient",
            )

    def test_constrained_unconstrained_chain_rule(self):
        """Unconstrained Jacobian equals constrained × logistic derivative."""
        K = 3
        n_params = K * (K - 1) // 2
        rng = np.random.default_rng(77)
        theta_u = rng.standard_normal(n_params) * 0.5

        jac_u = grad_corr_theta(theta_u, K, constrained=False)

        # Compute constrained Jacobian at the mapped angle
        theta_c = np.pi / (1.0 + np.exp(-theta_u))
        jac_c = grad_corr_theta(theta_c, K, constrained=True)

        # Chain rule: jac_u = diag(logistic_deriv) @ jac_c
        sig = 1.0 / (1.0 + np.exp(-theta_u))
        logistic_deriv = np.pi * sig * (1.0 - sig)
        jac_chained = np.diag(logistic_deriv) @ jac_c

        np.testing.assert_allclose(
            jac_u, jac_chained, atol=1e-12,
            err_msg="Chain rule consistency failed",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corr_to_upper_vec(corr: np.ndarray, K: int) -> np.ndarray:
    """Extract upper-triangular elements row-by-row."""
    n = K * (K + 1) // 2
    vec = np.zeros(n, dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            vec[idx] = corr[i, j]
            idx += 1
    return vec


def _fd_jacobian(
    theta: np.ndarray, K: int, *, constrained: bool = False, eps: float = 1e-7,
) -> np.ndarray:
    """Compute Jacobian via central finite differences."""
    n_theta = K * (K - 1) // 2
    n_upper = K * (K + 1) // 2
    jac = np.zeros((n_theta, n_upper), dtype=np.float64)

    for p in range(n_theta):
        theta_p = theta.copy()
        theta_p[p] += eps
        corr_p = theta_to_corr(theta_p, K, constrained=constrained)
        vec_p = _corr_to_upper_vec(corr_p, K)

        theta_m = theta.copy()
        theta_m[p] -= eps
        corr_m = theta_to_corr(theta_m, K, constrained=constrained)
        vec_m = _corr_to_upper_vec(corr_m, K)

        jac[p] = (vec_p - vec_m) / (2.0 * eps)

    return jac
