"""Tests for radial parameterization (Van Oest 2019)."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.matgradient._radial import (
    grad_radial_theta,
    radial_to_corr,
)


class TestRadialToCorr:
    """Verify radial_to_corr produces valid PD correlation matrices."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_unit_diagonal(self, K: int):
        rng = np.random.default_rng(42)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = radial_to_corr(theta, K)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-14)

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_symmetric(self, K: int):
        rng = np.random.default_rng(123)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = radial_to_corr(theta, K)
        np.testing.assert_allclose(corr, corr.T, atol=1e-14)

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6])
    def test_positive_definite(self, K: int):
        rng = np.random.default_rng(999)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params)
        corr = radial_to_corr(theta, K)
        eigvals = np.linalg.eigvalsh(corr)
        assert eigvals.min() > 0, f"Not PD: min eigval = {eigvals.min()}"

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_correlations_in_range(self, K: int):
        rng = np.random.default_rng(77)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 2.0  # larger values
        corr = radial_to_corr(theta, K)
        # Off-diagonal elements must be in [-1, 1]
        for i in range(K):
            for j in range(K):
                if i != j:
                    assert -1.0 <= corr[i, j] <= 1.0, (
                        f"corr[{i},{j}] = {corr[i,j]} out of range"
                    )

    def test_zero_theta_gives_identity(self):
        """theta=0 → tanh(0)=0 → L=I → corr=I."""
        for K in [2, 3, 4, 5]:
            n_params = K * (K - 1) // 2
            theta = np.zeros(n_params)
            corr = radial_to_corr(theta, K)
            np.testing.assert_allclose(corr, np.eye(K), atol=1e-14)

    def test_large_theta_high_correlation(self):
        """Large theta → tanh → ±1 → high correlations."""
        K = 3
        theta = np.array([5.0, 0.0, 0.0])  # large first param
        corr = radial_to_corr(theta, K)
        # corr[0,1] should be close to tanh(5)^2 ~ 1 (since L[0,1]=tanh(5))
        assert abs(corr[0, 1]) > 0.99

    def test_pd_with_extreme_params(self):
        """PD holds even for extreme parameter values."""
        for K in [3, 4, 5]:
            n_params = K * (K - 1) // 2
            # Very large parameters
            theta = np.ones(n_params) * 10.0
            corr = radial_to_corr(theta, K)
            eigvals = np.linalg.eigvalsh(corr)
            assert eigvals.min() > -1e-10, f"K={K}: min eigval = {eigvals.min()}"
            # Very negative parameters
            theta = -np.ones(n_params) * 10.0
            corr = radial_to_corr(theta, K)
            eigvals = np.linalg.eigvalsh(corr)
            assert eigvals.min() > -1e-10, f"K={K}: min eigval = {eigvals.min()}"

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_random_inputs_always_pd(self, K: int):
        """Fuzz test: random inputs always produce PD correlation."""
        rng = np.random.default_rng(314)
        n_params = K * (K - 1) // 2
        for _ in range(20):
            theta = rng.standard_normal(n_params) * 3.0
            corr = radial_to_corr(theta, K)
            eigvals = np.linalg.eigvalsh(corr)
            assert eigvals.min() > -1e-10


class TestGradRadialTheta:
    """Verify analytic Jacobian against finite differences."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_jacobian_vs_fd_moderate(self, K: int):
        """Analytic Jacobian matches FD at moderate parameter values."""
        rng = np.random.default_rng(42)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 0.5

        jac_analytic = grad_radial_theta(theta, K)
        jac_fd = _fd_jacobian(theta, K)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"K={K}: analytic Jacobian differs from FD",
        )

    @pytest.mark.parametrize("K", [2, 3, 4, 5])
    def test_jacobian_vs_fd_large(self, K: int):
        """Analytic Jacobian matches FD at larger parameter values."""
        rng = np.random.default_rng(123)
        n_params = K * (K - 1) // 2
        theta = rng.standard_normal(n_params) * 2.0

        jac_analytic = grad_radial_theta(theta, K)
        jac_fd = _fd_jacobian(theta, K)

        np.testing.assert_allclose(
            jac_analytic, jac_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"K={K}: analytic Jacobian differs from FD (large theta)",
        )

    def test_jacobian_vs_fd_zero(self):
        """Analytic Jacobian matches FD at theta=0."""
        for K in [2, 3, 4]:
            n_params = K * (K - 1) // 2
            theta = np.zeros(n_params)

            jac_analytic = grad_radial_theta(theta, K)
            jac_fd = _fd_jacobian(theta, K)

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
            jac = grad_radial_theta(theta, K)
            assert jac.shape == (n_theta, n_upper)

    def test_diagonal_entries_zero_gradient(self):
        """Diagonal of corr is always 1, so its gradient should be zero."""
        K = 4
        n_params = K * (K - 1) // 2
        rng = np.random.default_rng(55)
        theta = rng.standard_normal(n_params)

        jac = grad_radial_theta(theta, K)

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
    theta: np.ndarray, K: int, eps: float = 1e-7,
) -> np.ndarray:
    """Compute Jacobian via central finite differences."""
    n_theta = K * (K - 1) // 2
    n_upper = K * (K + 1) // 2
    jac = np.zeros((n_theta, n_upper), dtype=np.float64)

    for p in range(n_theta):
        theta_p = theta.copy()
        theta_p[p] += eps
        corr_p = radial_to_corr(theta_p, K)
        vec_p = _corr_to_upper_vec(corr_p, K)

        theta_m = theta.copy()
        theta_m[p] -= eps
        corr_m = radial_to_corr(theta_m, K)
        vec_m = _corr_to_upper_vec(corr_m, K)

        jac[p] = (vec_p - vec_m) / (2.0 * eps)

    return jac
