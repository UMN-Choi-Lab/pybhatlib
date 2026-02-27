"""Tests for bivariate truncated normal moments."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pybhatlib.gradmvn._bivariate_trunc import (
    truncated_bivariate_cov,
    truncated_bivariate_mean,
)


class TestTruncatedBivariateMean:
    def test_independent_reduces_to_univariate(self):
        """For independent variables, truncated mean should match univariate."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = np.array([1.0, 1.0])

        E_trunc = truncated_bivariate_mean(mu, sigma, a)

        # For independent: E[X1|X1<=a1] = mu1 - phi(a1)/Phi(a1)
        expected_1 = -norm.pdf(1.0) / norm.cdf(1.0)
        expected_2 = -norm.pdf(1.0) / norm.cdf(1.0)

        np.testing.assert_allclose(E_trunc[0], expected_1, atol=0.02)
        np.testing.assert_allclose(E_trunc[1], expected_2, atol=0.02)

    def test_truncated_mean_less_than_limit(self):
        """Truncated mean should be less than the truncation limit."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([0.5, 0.5])

        E_trunc = truncated_bivariate_mean(mu, sigma, a)

        assert E_trunc[0] < a[0]
        assert E_trunc[1] < a[1]

    def test_symmetric_case(self):
        """For symmetric setup, both truncated means should be equal."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        a = np.array([1.0, 1.0])

        E_trunc = truncated_bivariate_mean(mu, sigma, a)
        np.testing.assert_allclose(E_trunc[0], E_trunc[1], atol=1e-10)

    def test_large_limit_mean_near_original(self):
        """With very large limits, truncated mean should approach original mean."""
        mu = np.array([1.0, -0.5])
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        a = np.array([10.0, 10.0])

        E_trunc = truncated_bivariate_mean(mu, sigma, a)
        np.testing.assert_allclose(E_trunc, mu, atol=0.01)


class TestTruncatedBivariateMeanMC:
    """Monte Carlo validation of truncated bivariate mean (Property 1)."""

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, -0.5])
    def test_mean_against_mc(self, rho):
        """Truncated bivariate mean should match MC rejection sampling."""
        rng = np.random.default_rng(42)
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, rho], [rho, 1.0]])
        a = np.array([0.5, 0.8])

        # MC rejection sampling
        n_samples = 200_000
        samples = rng.multivariate_normal(mu, sigma, size=n_samples)
        mask = np.all(samples <= a, axis=1)
        mc_mean = samples[mask].mean(axis=0)

        analytic_mean = truncated_bivariate_mean(mu, sigma, a)
        np.testing.assert_allclose(analytic_mean, mc_mean, atol=0.03)


class TestTruncatedBivariateCovMC:
    """Monte Carlo validation of truncated bivariate covariance (Property 2)."""

    @pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, -0.5])
    def test_cov_against_mc(self, rho):
        """Truncated bivariate cov should match MC rejection sampling."""
        rng = np.random.default_rng(42)
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, rho], [rho, 1.0]])
        a = np.array([0.5, 0.8])

        # MC rejection sampling
        n_samples = 200_000
        samples = rng.multivariate_normal(mu, sigma, size=n_samples)
        mask = np.all(samples <= a, axis=1)
        accepted = samples[mask]
        mc_mean = accepted.mean(axis=0)
        mc_cov = np.cov(accepted.T)

        analytic_cov = truncated_bivariate_cov(mu, sigma, a)
        # 3% tolerance for diagonal, 5% absolute for off-diagonal
        np.testing.assert_allclose(
            analytic_cov[0, 0], mc_cov[0, 0], rtol=0.03, atol=0.01
        )
        np.testing.assert_allclose(
            analytic_cov[1, 1], mc_cov[1, 1], rtol=0.03, atol=0.01
        )
        np.testing.assert_allclose(
            analytic_cov[0, 1], mc_cov[0, 1], atol=0.02
        )


class TestTruncatedBivariateCov:
    def test_truncated_variance_less_than_original(self):
        """Truncation should reduce variance."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        a = np.array([1.0, 1.0])

        Cov_trunc = truncated_bivariate_cov(mu, sigma, a)

        # Diagonal elements should be less than original
        assert Cov_trunc[0, 0] < sigma[0, 0]
        assert Cov_trunc[1, 1] < sigma[1, 1]

    def test_large_limit_cov_near_original(self):
        """With very large limits, truncated cov should approach original."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])
        a = np.array([10.0, 10.0])

        Cov_trunc = truncated_bivariate_cov(mu, sigma, a)
        np.testing.assert_allclose(Cov_trunc, sigma, atol=0.05)

    def test_positive_definite(self):
        """Truncated covariance should be positive semi-definite."""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([0.5, 0.5])

        Cov_trunc = truncated_bivariate_cov(mu, sigma, a)
        eigvals = np.linalg.eigvalsh(Cov_trunc)
        assert np.all(eigvals >= -1e-10)
