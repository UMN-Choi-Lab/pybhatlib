"""Tests for rectangular MVNCD (mvncd_rect)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn import mvncd, mvncd_rect


@pytest.fixture
def xp():
    return get_backend("numpy")


class TestMVNCDRectUnivariate:
    def test_rect_univariate(self, xp):
        # P(-1 <= X <= 1) for X ~ N(0,1) = Phi(1) - Phi(-1)
        lower = xp.array([-1.0])
        upper = xp.array([1.0])
        sigma = xp.array([[1.0]])
        p = mvncd_rect(lower, upper, sigma, method="me", xp=xp)
        expected = norm.cdf(1.0) - norm.cdf(-1.0)
        np.testing.assert_allclose(p, expected, atol=1e-4)

    def test_neg_inf_lower_reduces_to_mvncd(self, xp):
        # P(-inf <= X <= a) = P(X <= a) = standard MVNCD
        lower = xp.array([-np.inf])
        upper = xp.array([1.5])
        sigma = xp.array([[1.0]])
        p_rect = mvncd_rect(lower, upper, sigma, method="me", xp=xp)
        p_std = mvncd(upper, sigma, method="me", xp=xp)
        np.testing.assert_allclose(p_rect, p_std, atol=1e-6)


class TestMVNCDRectBivariate:
    def test_rect_bivariate_independent(self, xp):
        # P(-1 <= X1 <= 1, -1 <= X2 <= 1) for independent X1, X2
        lower = xp.array([-1.0, -1.0])
        upper = xp.array([1.0, 1.0])
        sigma = xp.array([[1.0, 0.0], [0.0, 1.0]])
        p = mvncd_rect(lower, upper, sigma, method="me", xp=xp)
        expected = (norm.cdf(1.0) - norm.cdf(-1.0)) ** 2
        np.testing.assert_allclose(p, expected, atol=0.01)

    def test_rect_bivariate_correlated(self, xp):
        lower = np.array([-0.5, -0.5])
        upper = np.array([1.0, 1.0])
        sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

        p = mvncd_rect(xp.array(lower), xp.array(upper), xp.array(sigma),
                        method="scipy", xp=xp)

        # Reference from scipy
        p_upper = multivariate_normal.cdf(upper, mean=np.zeros(2), cov=sigma)
        p_lower_both = multivariate_normal.cdf(lower, mean=np.zeros(2), cov=sigma)
        p_mixed1 = multivariate_normal.cdf(
            [upper[0], lower[1]], mean=np.zeros(2), cov=sigma
        )
        p_mixed2 = multivariate_normal.cdf(
            [lower[0], upper[1]], mean=np.zeros(2), cov=sigma
        )
        p_ref = p_upper - p_mixed1 - p_mixed2 + p_lower_both

        np.testing.assert_allclose(p, p_ref, atol=0.01)

    def test_neg_inf_lower_bivariate(self, xp):
        lower = xp.array([-np.inf, -np.inf])
        upper = xp.array([1.0, 0.5])
        sigma = xp.array([[1.0, 0.3], [0.3, 1.0]])
        p_rect = mvncd_rect(lower, upper, sigma, method="scipy", xp=xp)
        p_std = mvncd(upper, sigma, method="scipy", xp=xp)
        np.testing.assert_allclose(p_rect, p_std, atol=1e-4)


class TestMVNCDRectTrivariate:
    def test_rect_trivariate(self, xp):
        sigma = np.array([[1.0, 0.2, 0.1], [0.2, 1.0, 0.3], [0.1, 0.3, 1.0]])
        lower = np.array([-0.5, -0.5, -0.5])
        upper = np.array([1.0, 1.0, 1.0])
        p = mvncd_rect(
            xp.array(lower), xp.array(upper), xp.array(sigma),
            method="scipy", xp=xp,
        )
        # Must be positive and less than 1
        assert 0.0 < p < 1.0

    def test_partial_neg_inf(self, xp):
        sigma = np.eye(3)
        lower = np.array([-np.inf, -1.0, -np.inf])
        upper = np.array([1.0, 1.0, 0.5])
        p = mvncd_rect(
            xp.array(lower), xp.array(upper), xp.array(sigma),
            method="scipy", xp=xp,
        )
        # = P(X1<=1) * P(-1<=X2<=1) * P(X3<=0.5)
        expected = norm.cdf(1.0) * (norm.cdf(1.0) - norm.cdf(-1.0)) * norm.cdf(0.5)
        np.testing.assert_allclose(p, expected, atol=0.01)
