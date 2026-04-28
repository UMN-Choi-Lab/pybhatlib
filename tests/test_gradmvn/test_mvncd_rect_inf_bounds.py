"""Regression + correctness tests for mvncd_rect with +inf upper bounds.

Bug reference
-------------
mvncd_rect (and mvncd) return NaN / wrong values when an upper bound element
is +inf and Sigma has off-diagonal mass.  The root cause is in the ME
sequential-conditioning core: when w_h = +inf, the truncated variance
formula evaluates 0.0 * inf = NaN under IEEE 754.

Fix: collapse +inf dimensions before running any approximation method
(see pybhatlib/gradmvn/_mvncd.py, _mvncd_drop_inf_dims helper).

GAUSS reference: UTAcode_0402/gradients mvn.src, proc cdrectmvnanl (line 1189),
which calls rectcombs (vecup.src line 1481) + cdfmvnanalytic.  GAUSS's own
cdfn / cdfmvnanalytic handle +inf by returning 1, which has the same effect
as collapsing the unbounded dimension.

Cross-branch note
-----------------
PR #8 (feat/mnp-002-se-options, MORP analytic gradients) has a temporary
workaround _rect_prob_finite_only in
src/pybhatlib/models/morp/_morp_loglik.py that should be removed once this
PR *and* PR #8 both land on main.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import multivariate_normal

from pybhatlib.gradmvn import mvncd, mvncd_rect


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sigma3_offdiag():
    """3x3 covariance matrix with off-diagonal mass."""
    return np.array(
        [[1.0, 0.4, 0.2],
         [0.4, 1.0, 0.3],
         [0.2, 0.3, 1.0]],
        dtype=np.float64,
    )


def _scipy_rect(lower, upper, sigma):
    """Exact rectangular probability via inclusion-exclusion + scipy."""
    K = len(lower)
    prob = 0.0
    for s in range(1 << K):
        c = np.zeros(K)
        sign = 1
        for i in range(K):
            if s & (1 << i):
                c[i] = lower[i]
                sign *= -1
            else:
                c[i] = upper[i]
        # skip vertices where any bound is ±inf (probability contribution = 0)
        if not np.all(np.isfinite(c)):
            continue
        prob += sign * float(
            multivariate_normal.cdf(c, mean=np.zeros(K), cov=sigma)
        )
    return prob


# ---------------------------------------------------------------------------
# Test 1 — golden values for all-finite inputs (regression anchor)
#
# These values are captured from the *current* (pre-fix) implementation and
# serve as a regression guard: after the fix, all-finite inputs must not
# change by more than 1e-10.
# ---------------------------------------------------------------------------

GOLDEN_LOWER = np.array([-0.5, -1.0, 0.0], dtype=np.float64)
GOLDEN_UPPER = np.array([1.5, 0.8, 2.0], dtype=np.float64)
# Captured pre-fix; scipy exact reference = 0.19738330215997826
GOLDEN_OVUS = 0.19737287773823692
GOLDEN_ME = 0.19733971548014204


class TestAllFiniteUnchanged:
    """Case 1 — golden values must be preserved after fix."""

    def test_mvncd_rect_finite_inputs_ovus(self, sigma3_offdiag):
        p = mvncd_rect(GOLDEN_LOWER, GOLDEN_UPPER, sigma3_offdiag, method="ovus")
        assert np.isfinite(p), "Result must be finite for all-finite inputs"
        np.testing.assert_allclose(
            p, GOLDEN_OVUS, atol=1e-10,
            err_msg="mvncd_rect/ovus changed on all-finite inputs",
        )

    def test_mvncd_rect_finite_inputs_me(self, sigma3_offdiag):
        p = mvncd_rect(GOLDEN_LOWER, GOLDEN_UPPER, sigma3_offdiag, method="me")
        assert np.isfinite(p), "Result must be finite for all-finite inputs"
        np.testing.assert_allclose(
            p, GOLDEN_ME, atol=1e-10,
            err_msg="mvncd_rect/me changed on all-finite inputs",
        )


# ---------------------------------------------------------------------------
# Test 2 — IID Sigma, upper = +inf on one dim
# ---------------------------------------------------------------------------

class TestInfUpperIID:
    """Case 2 — Sigma=identity, upper[k]=+inf reduces to 1-D Phi."""

    @pytest.mark.parametrize("k_inf", [0, 1, 2])
    def test_mvncd_rect_inf_upper_iid_returns_marginal(self, k_inf):
        """P(X <= upper) with Sigma=I, upper[k]=+inf, others finite.

        lower = [-inf,...,-inf] so this reduces to:
        P(X_{j1} <= a_{j1}, ..., X_{j_{K-1}} <= a_{j_{K-1}}) independent
        = product of Phi(a_ji).
        """
        a_finite = np.array([1.5, 0.3, -0.5], dtype=np.float64)
        sigma_id = np.eye(3, dtype=np.float64)

        upper = a_finite.copy()
        upper[k_inf] = np.inf
        lower = np.full(3, -np.inf)

        p = mvncd_rect(lower, upper, sigma_id, method="ovus")
        assert np.isfinite(p), f"NaN/inf returned for IID k_inf={k_inf}"
        assert 0.0 < p <= 1.0

        # expected: product of Phi(a_j) for j != k_inf
        other = [j for j in range(3) if j != k_inf]
        expected = float(np.prod([ndtr(a_finite[j]) for j in other]))
        np.testing.assert_allclose(
            p, expected, atol=1e-8,
            err_msg=f"IID inf-upper mismatch for k_inf={k_inf}",
        )


# ---------------------------------------------------------------------------
# Test 3 — off-diagonal Sigma, upper[k]=+inf → finite result
# ---------------------------------------------------------------------------

class TestInfUpperFullCovFinite:
    """Case 3 — off-diagonal Sigma, upper[k]=+inf must return finite value."""

    @pytest.mark.parametrize("method", ["me", "ovus", "bme", "tvbs"])
    @pytest.mark.parametrize("k_inf", [0, 1, 2])
    def test_mvncd_rect_inf_upper_full_cov_returns_finite(
        self, sigma3_offdiag, method, k_inf
    ):
        lower = np.array([-0.5, -1.0, 0.0], dtype=np.float64)
        upper = np.array([1.5, 0.8, 2.0], dtype=np.float64)
        upper[k_inf] = np.inf

        p = mvncd_rect(lower, upper, sigma3_offdiag, method=method)
        assert np.isfinite(p), (
            f"NaN/inf from mvncd_rect(method={method}, k_inf={k_inf}): got {p}"
        )
        assert p > 0.0, f"Negative/zero result: {p}"
        assert p <= 1.0, f"Probability > 1: {p}"


# ---------------------------------------------------------------------------
# Test 4 — off-diagonal Sigma, upper[k]=+inf matches (K-1)-D marginal
# ---------------------------------------------------------------------------

class TestInfUpperMatchesMarginal:
    """Case 4 — dropping the +inf dim gives the correct marginal probability.

    Setup: lower = [-inf, -inf, -inf], upper = [a0, a1, +inf]
    Expected: P(X0 <= a0, X1 <= a1) under sigma[:2, :2].
    """

    @pytest.mark.parametrize("method", ["me", "ovus", "bme", "tvbs"])
    def test_mvncd_rect_inf_upper_full_cov_matches_marginal(
        self, sigma3_offdiag, method
    ):
        lower = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
        upper = np.array([1.5, 0.8, np.inf], dtype=np.float64)

        p = mvncd_rect(lower, upper, sigma3_offdiag, method=method)
        assert np.isfinite(p), f"NaN from mvncd_rect(method={method}): got {p}"

        # Reference: marginal P(X0<=1.5, X1<=0.8) under sigma[0:2, 0:2]
        sigma_sub = sigma3_offdiag[:2, :2]
        expected = float(
            multivariate_normal.cdf([1.5, 0.8], mean=np.zeros(2), cov=sigma_sub)
        )
        np.testing.assert_allclose(
            p, expected, atol=1e-8,
            err_msg=f"Marginal mismatch for method={method}",
        )

    @pytest.mark.parametrize("k_inf", [0, 1, 2])
    def test_mvncd_rect_single_inf_dim_matches_2d_marginal(
        self, sigma3_offdiag, k_inf
    ):
        """P(X0<=a0,X1<=a1,X2<=a2) with a[k_inf]=+inf = 2D marginal."""
        a = np.array([1.5, 0.8, 2.0], dtype=np.float64)
        a[k_inf] = np.inf
        lower = np.full(3, -np.inf)

        p = mvncd_rect(lower, a, sigma3_offdiag, method="ovus")
        assert np.isfinite(p), f"NaN for k_inf={k_inf}"

        finite_idx = [j for j in range(3) if j != k_inf]
        sigma_sub = sigma3_offdiag[np.ix_(finite_idx, finite_idx)]
        a_sub = a[finite_idx]
        expected = float(
            multivariate_normal.cdf(a_sub, mean=np.zeros(2), cov=sigma_sub)
        )
        np.testing.assert_allclose(p, expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 5 — lower = -inf (existing behavior, sanity check)
# ---------------------------------------------------------------------------

class TestNegInfLower:
    """Case 5 — lower=-inf should already work (regression / sanity)."""

    def test_mvncd_rect_neg_inf_lower_all_finite_upper(self, sigma3_offdiag):
        lower = np.full(3, -np.inf)
        upper = np.array([1.5, 0.8, 2.0], dtype=np.float64)
        p = mvncd_rect(lower, upper, sigma3_offdiag, method="ovus")
        assert np.isfinite(p), f"NaN for all-neginf lower: {p}"
        assert 0.0 < p <= 1.0

    def test_mvncd_rect_partial_neg_inf_lower(self, sigma3_offdiag):
        lower = np.array([-np.inf, -1.0, -np.inf], dtype=np.float64)
        upper = np.array([1.5, 0.8, 2.0], dtype=np.float64)
        p = mvncd_rect(lower, upper, sigma3_offdiag, method="ovus")
        assert np.isfinite(p), f"NaN for partial neginf lower: {p}"
        assert 0.0 < p <= 1.0


# ---------------------------------------------------------------------------
# Tests for mvncd (direct function) with +inf
# ---------------------------------------------------------------------------

class TestMvncdDirectInfBounds:
    """mvncd called directly with a[k]=+inf must return the correct marginal."""

    def test_mvncd_inf_bound_iid(self):
        """P(X0<=1.0, X1<=+inf) for IID sigma = Phi(1.0)."""
        sigma = np.eye(2)
        a = np.array([1.0, np.inf])
        p = mvncd(a, sigma, method="ovus")
        assert np.isfinite(p)
        np.testing.assert_allclose(p, ndtr(1.0), atol=1e-10)

    @pytest.mark.parametrize("method", ["me", "ovus", "bme", "tvbs"])
    def test_mvncd_inf_bound_full_cov(self, sigma3_offdiag, method):
        """P(X0<=1.5, X1<=0.8, X2<=+inf) = P(X0<=1.5, X1<=0.8) marginal."""
        a = np.array([1.5, 0.8, np.inf])
        p = mvncd(a, sigma3_offdiag, method=method)
        assert np.isfinite(p), f"NaN for method={method}: {p}"

        sigma_sub = sigma3_offdiag[:2, :2]
        expected = float(
            multivariate_normal.cdf([1.5, 0.8], mean=np.zeros(2), cov=sigma_sub)
        )
        np.testing.assert_allclose(p, expected, atol=1e-8)

    def test_mvncd_all_inf_returns_one(self, sigma3_offdiag):
        """P(X <= +inf) = 1 for all methods."""
        a = np.full(3, np.inf)
        for method in ["me", "ovus", "bme", "tvbs"]:
            p = mvncd(a, sigma3_offdiag, method=method)
            np.testing.assert_allclose(
                p, 1.0, atol=1e-10,
                err_msg=f"All-inf bound should return 1, got {p} for {method}",
            )
