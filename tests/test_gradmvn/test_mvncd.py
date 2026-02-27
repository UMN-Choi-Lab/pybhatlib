"""Tests for MVNCD function."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn import mvncd


@pytest.fixture
def xp():
    return get_backend("numpy")


class TestMVNCDUnivariate:
    def test_standard_normal_at_zero(self, xp):
        p = mvncd(xp.array([0.0]), xp.array([[1.0]]), xp=xp)
        np.testing.assert_allclose(p, 0.5, atol=1e-6)

    def test_standard_normal_at_inf(self, xp):
        p = mvncd(xp.array([10.0]), xp.array([[1.0]]), xp=xp)
        np.testing.assert_allclose(p, 1.0, atol=1e-6)

    def test_standard_normal_at_neg_inf(self, xp):
        p = mvncd(xp.array([-10.0]), xp.array([[1.0]]), xp=xp)
        assert p < 1e-10

    def test_scaled_variance(self, xp):
        # P(X <= 2) where X ~ N(0, 4) = P(Z <= 1) where Z ~ N(0,1)
        p = mvncd(xp.array([2.0]), xp.array([[4.0]]), xp=xp)
        np.testing.assert_allclose(p, norm.cdf(1.0), atol=1e-6)


class TestMVNCDBivariate:
    def test_independent(self, xp):
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        a = np.array([0.0, 0.0])
        p = mvncd(xp.array(a), xp.array(sigma), xp=xp)
        np.testing.assert_allclose(p, 0.25, atol=1e-3)

    def test_correlated(self, xp):
        sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
        a = np.array([1.0, 1.0])
        p = mvncd(xp.array(a), xp.array(sigma), xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(2), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=1e-3)


class TestMVNCDTrivariate:
    def test_against_scipy(self, xp):
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p_me = mvncd(xp.array(a), xp.array(sigma), method="me", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        # ME approximation should be within 5% for K=3
        np.testing.assert_allclose(p_me, p_ref, rtol=0.05)

    def test_scipy_method(self, xp):
        sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
        a = np.array([1.0, 0.5, 0.8])
        p = mvncd(xp.array(a), xp.array(sigma), method="scipy", xp=xp)
        p_ref = multivariate_normal.cdf(a, mean=np.zeros(3), cov=sigma)
        np.testing.assert_allclose(p, p_ref, atol=1e-4)
