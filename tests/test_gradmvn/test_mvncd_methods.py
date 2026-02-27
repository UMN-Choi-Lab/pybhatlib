"""Tests for all MVNCD methods (OVUS, TVBS, BME, OVBS, SSJ)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn import mvncd


@pytest.fixture
def xp():
    return get_backend("numpy")


class TestAllMethodsBivariate:
    """All methods should reduce to exact bivariate CDF for K=2."""

    @pytest.mark.parametrize("method", ["me", "ovus", "bme", "tvbs", "ovbs", "ssj"])
    def test_bivariate_independent(self, xp, method):
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = np.array([0.0, 0.0])
        kwargs = {"method": method, "xp": xp}
        if method == "ssj":
            kwargs["n_draws"] = 5000
            kwargs["seed"] = 42
        p = mvncd(xp.array(a), xp.array(sigma), **kwargs)
        np.testing.assert_allclose(p, 0.25, atol=0.02)

    @pytest.mark.parametrize("method", ["me", "ovus", "bme", "tvbs", "ovbs", "ssj"])
    def test_bivariate_correlated(self, xp, method):
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([1.0, 1.0])
        kwargs = {"method": method, "xp": xp}
        if method == "ssj":
            kwargs["n_draws"] = 5000
            kwargs["seed"] = 42
        p = mvncd(xp.array(a), xp.array(sigma), **kwargs)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(2), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=0.02)


class TestTrivariate:
    """K=3: compare each method against scipy reference."""

    @pytest.fixture
    def trivariate_setup(self):
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        return a, sigma, p_ref

    @pytest.mark.parametrize(
        "method,rtol",
        [("me", 0.03), ("ovus", 0.02), ("bme", 0.01), ("tvbs", 0.01), ("ovbs", 0.01)],
    )
    def test_against_scipy(self, xp, trivariate_setup, method, rtol):
        a, sigma, p_ref = trivariate_setup
        p = mvncd(xp.array(a), xp.array(sigma), method=method, xp=xp)
        np.testing.assert_allclose(p, p_ref, rtol=rtol)

    def test_ssj_against_scipy(self, xp, trivariate_setup):
        a, sigma, p_ref = trivariate_setup
        p = mvncd(
            xp.array(a), xp.array(sigma),
            method="ssj", n_draws=10000, seed=42, xp=xp,
        )
        np.testing.assert_allclose(p, p_ref, rtol=0.05)


class TestPentavariate:
    """K=5: compare methods against SSJ with large n_draws as reference."""

    @pytest.fixture
    def pentavariate_setup(self, xp):
        rng = np.random.default_rng(123)
        # Generate a valid positive-definite correlation matrix
        A = rng.standard_normal((5, 5))
        sigma = A @ A.T / 5 + np.eye(5)
        # Make it a correlation matrix
        d = np.sqrt(np.diag(sigma))
        sigma = sigma / np.outer(d, d)
        np.fill_diagonal(sigma, 1.0)

        a = np.array([0.5, 0.8, 0.3, 1.0, 0.6])

        # Reference: scipy
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(5), cov=sigma)
        return a, sigma, p_ref

    @pytest.mark.parametrize(
        "method,rtol",
        [("me", 0.05), ("ovus", 0.05), ("bme", 0.03), ("tvbs", 0.01), ("ovbs", 0.03)],
    )
    def test_against_scipy_k5(self, xp, pentavariate_setup, method, rtol):
        a, sigma, p_ref = pentavariate_setup
        p = mvncd(xp.array(a), xp.array(sigma), method=method, xp=xp)
        np.testing.assert_allclose(p, p_ref, rtol=rtol)

    def test_ssj_k5(self, xp, pentavariate_setup):
        a, sigma, p_ref = pentavariate_setup
        p = mvncd(
            xp.array(a), xp.array(sigma),
            method="ssj", n_draws=20000, seed=42, xp=xp,
        )
        np.testing.assert_allclose(p, p_ref, rtol=0.05)


class TestMVNCDDefaultMethod:
    """Verify default method is now 'ovus'."""

    def test_default_is_ovus(self, xp):
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        # Both should give same result since default is ovus
        p_default = mvncd(xp.array(a), xp.array(sigma), xp=xp)
        p_ovus = mvncd(xp.array(a), xp.array(sigma), method="ovus", xp=xp)
        np.testing.assert_allclose(p_default, p_ovus, atol=1e-4)


class TestInvalidMethod:
    def test_unknown_method_raises(self, xp):
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = np.array([0.0, 0.0])
        # K=2 is handled before dispatch, so use K=3
        sigma3 = np.eye(3)
        a3 = np.zeros(3)
        with pytest.raises(ValueError, match="Unknown MVNCD method"):
            mvncd(xp.array(a3), xp.array(sigma3), method="bad_method", xp=xp)
