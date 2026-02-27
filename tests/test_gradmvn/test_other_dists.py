"""Tests for additional distributions."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from pybhatlib.gradmvn._other_dists import (
    gumbel_cdf,
    gumbel_pdf,
    logistic_cdf,
    logistic_pdf,
    reverse_gumbel_cdf,
    skew_normal_cdf,
    skew_normal_pdf,
)


class TestLogistic:
    def test_cdf_at_zero(self):
        np.testing.assert_allclose(logistic_cdf(0.0), 0.5, atol=1e-10)

    def test_cdf_monotone(self):
        assert logistic_cdf(-1.0) < logistic_cdf(0.0) < logistic_cdf(1.0)

    def test_pdf_symmetric(self):
        np.testing.assert_allclose(logistic_pdf(1.0), logistic_pdf(-1.0), atol=1e-10)

    def test_pdf_max_at_zero(self):
        assert logistic_pdf(0.0) > logistic_pdf(1.0)


class TestGumbel:
    def test_cdf_at_zero(self):
        np.testing.assert_allclose(gumbel_cdf(0.0), np.exp(-1), atol=1e-10)

    def test_cdf_large_x(self):
        assert gumbel_cdf(100.0) > 0.999

    def test_pdf_positive(self):
        assert gumbel_pdf(0.0) > 0

    def test_reverse_gumbel(self):
        np.testing.assert_allclose(reverse_gumbel_cdf(0.0), 1 - np.exp(-1), atol=1e-10)


class TestSkewNormal:
    def test_alpha_zero_is_normal(self):
        """With alpha=0, skew-normal should be twice the half-normal = standard normal."""
        x_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
        for x in x_vals:
            sn = skew_normal_pdf(x, alpha=0.0)
            n = norm.pdf(x)
            np.testing.assert_allclose(sn, n, atol=1e-10)

    def test_cdf_alpha_zero(self):
        """With alpha=0, CDF should match standard normal."""
        np.testing.assert_allclose(skew_normal_cdf(0.0, alpha=0.0), 0.5, atol=1e-6)
        np.testing.assert_allclose(skew_normal_cdf(1.96, alpha=0.0), norm.cdf(1.96), atol=1e-4)
