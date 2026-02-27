"""Tests for gradcovcor."""

from __future__ import annotations

import numpy as np

from pybhatlib.matgradient import gradcovcor


class TestGradcovcor:
    def test_output_shapes_3x3(self, cov_3x3):
        result = gradcovcor(cov_3x3)
        K = 3
        n_cov = K * (K + 1) // 2  # 6
        n_corr = K * (K - 1) // 2  # 3
        assert result.glitomega.shape == (K, n_cov)
        assert result.gomegastar.shape == (n_corr, n_cov)

    def test_numerical_gradient_omega(self, cov_3x3):
        """Verify glitomega against numerical differentiation."""
        K = 3
        omega_diag = np.sqrt(np.diag(cov_3x3))
        corr = np.diag(1 / omega_diag) @ cov_3x3 @ np.diag(1 / omega_diag)

        result = gradcovcor(cov_3x3)

        eps = 1e-6
        for k in range(K):
            omega_plus = omega_diag.copy()
            omega_plus[k] += eps
            cov_plus = np.diag(omega_plus) @ corr @ np.diag(omega_plus)

            omega_minus = omega_diag.copy()
            omega_minus[k] -= eps
            cov_minus = np.diag(omega_minus) @ corr @ np.diag(omega_minus)

            # Extract upper triangular elements
            from pybhatlib.vecup import vecdup
            v_plus = vecdup(cov_plus)
            v_minus = vecdup(cov_minus)
            num_grad = (v_plus - v_minus) / (2 * eps)

            np.testing.assert_allclose(result.glitomega[k], num_grad, atol=1e-4)

    def test_correlation_matrix_input(self):
        """If input is already a correlation matrix, glitomega should be all 1s."""
        corr = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]])
        result = gradcovcor(corr)
        np.testing.assert_allclose(result.glitomega, np.ones_like(result.glitomega))
