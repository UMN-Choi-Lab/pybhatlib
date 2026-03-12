"""Finite-difference verification of atomic gradient helpers (_trunc_grads.py).

Tests every function at multiple parameter regimes:
- moderate truncation (w ≈ 0..1)
- extreme truncation (w ≈ -2..3)
- high and low correlation (|ρ| ∈ {0, 0.3, 0.8, -0.5})
- non-unit variance and non-zero mean

NOTE: scipy.stats.multivariate_normal.cdf has only ~2.6e-6 absolute precision
for the bivariate case, making finite differences with eps=1e-6 unreliable.
All BVN-based FD tests use _bvn_cdf_precise() which achieves ~15-digit precision
via scipy.integrate.quad.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from pybhatlib.gradmvn._bivariate_trunc import (
    truncated_bivariate_cov,
    truncated_bivariate_mean,
)
from pybhatlib.gradmvn._trunc_grads import (
    grad_bivariate_normal_trunc,
    grad_cdf_bvn,
    grad_cdf_bvn_by_cdfn,
    grad_noncdfbvn,
    grad_noncdfbvn_by_cdfn,
    grad_noncdfn,
    grad_univariate_normal_trunc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bvn_cdf_precise(w1: float, w2: float, rho: float) -> float:
    """High-precision BVN CDF via adaptive Gauss-Kronrod quadrature.

    BVN(w1,w2,ρ) = ∫_{-∞}^{w1} φ(t)·Φ((w2 - ρt)/√(1-ρ²)) dt

    Achieves ~15 digits of precision, vs ~6 digits from
    scipy.stats.multivariate_normal.cdf.
    """
    if abs(rho) < 1e-15:
        return float(norm.cdf(w1) * norm.cdf(w2))

    rhotilde = np.sqrt(1.0 - rho**2)

    def integrand(t):
        return norm.pdf(t) * norm.cdf((w2 - rho * t) / rhotilde)

    result, _ = quad(integrand, -np.inf, w1, limit=100)
    return result


def _fd_scalar(f, x, idx, eps=1e-6):
    """Central FD for scalar output f(x) w.r.t. x[idx]."""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[idx] += eps
    x_minus[idx] -= eps
    return (f(x_plus) - f(x_minus)) / (2.0 * eps)


def _fd_vec(f, x, idx, eps=1e-6):
    """Central FD for vector output f(x) w.r.t. x[idx]."""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[idx] += eps
    x_minus[idx] -= eps
    return (f(x_plus) - f(x_minus)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# 1. grad_noncdfn
# ---------------------------------------------------------------------------

class TestGradNonCdfN:
    """FD verification of ∂Φ((x-μ)/σ)/∂(μ, σ², x)."""

    @pytest.mark.parametrize("mu, sig2, x", [
        (0.0, 1.0, 0.5),     # moderate
        (0.0, 1.0, -2.0),    # left tail
        (0.0, 1.0, 3.0),     # right tail
        (1.5, 2.0, 2.0),     # non-standard
        (-1.0, 0.5, -0.5),   # small variance
    ])
    def test_fd(self, mu, sig2, x):
        eps = 1e-6
        g_mu, g_sig2, g_x = grad_noncdfn(mu, sig2, x)

        def f_mu(m):
            return norm.cdf((x - m) / np.sqrt(sig2))
        def f_sig2(s2):
            return norm.cdf((x - mu) / np.sqrt(s2))
        def f_x(xx):
            return norm.cdf((xx - mu) / np.sqrt(sig2))

        fd_mu = (f_mu(mu + eps) - f_mu(mu - eps)) / (2 * eps)
        fd_sig2 = (f_sig2(sig2 + eps) - f_sig2(sig2 - eps)) / (2 * eps)
        fd_x = (f_x(x + eps) - f_x(x - eps)) / (2 * eps)

        np.testing.assert_allclose(g_mu, fd_mu, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(g_sig2, fd_sig2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(g_x, fd_x, atol=1e-5, rtol=1e-4)

    def test_symmetry(self):
        """g_x should equal -g_mu."""
        g_mu, _, g_x = grad_noncdfn(0.0, 1.0, 1.0)
        np.testing.assert_allclose(g_x, -g_mu, atol=1e-14)


# ---------------------------------------------------------------------------
# 2. grad_cdf_bvn
# ---------------------------------------------------------------------------

class TestGradCdfBvn:
    """FD verification of ∂BVN(w₁,w₂,ρ)/∂(w₁,w₂,ρ)."""

    @pytest.mark.parametrize("w1, w2, rho", [
        (0.5, 0.8, 0.3),     # moderate
        (-1.0, -0.5, 0.0),   # independent, left region
        (2.0, 1.5, 0.8),     # high correlation
        (0.0, 0.0, -0.5),    # negative correlation
        (-2.0, 3.0, 0.6),    # asymmetric tails
    ])
    def test_fd(self, w1, w2, rho):
        eps = 1e-6
        gw1, gw2, grho = grad_cdf_bvn(w1, w2, rho)

        fd_w1 = (_bvn_cdf_precise(w1 + eps, w2, rho)
                 - _bvn_cdf_precise(w1 - eps, w2, rho)) / (2 * eps)
        fd_w2 = (_bvn_cdf_precise(w1, w2 + eps, rho)
                 - _bvn_cdf_precise(w1, w2 - eps, rho)) / (2 * eps)
        fd_rho = (_bvn_cdf_precise(w1, w2, rho + eps)
                  - _bvn_cdf_precise(w1, w2, rho - eps)) / (2 * eps)

        np.testing.assert_allclose(gw1, fd_w1, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(gw2, fd_w2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(grho, fd_rho, atol=1e-5, rtol=1e-4)

    def test_symmetry(self):
        """For w1=w2 and rho=0, gw1 should equal gw2."""
        gw1, gw2, _ = grad_cdf_bvn(1.0, 1.0, 0.0)
        np.testing.assert_allclose(gw1, gw2, atol=1e-10)


# ---------------------------------------------------------------------------
# 3. grad_cdf_bvn_by_cdfn
# ---------------------------------------------------------------------------

class TestGradCdfBvnByCdfn:
    """FD verification of ∂[BVN(w₁,w₂,ρ)/Φ(w₁)]/∂(w₁,w₂,ρ)."""

    @pytest.mark.parametrize("w1, w2, rho", [
        (0.5, 0.8, 0.3),
        (1.0, -0.5, 0.0),
        (2.0, 1.5, 0.8),
        (0.5, 0.5, -0.5),
        (-0.5, 1.0, 0.6),
    ])
    def test_fd(self, w1, w2, rho):
        eps = 1e-6

        def ratio(ww1, ww2, rr):
            return _bvn_cdf_precise(ww1, ww2, rr) / norm.cdf(ww1)

        gw1, gw2, grho = grad_cdf_bvn_by_cdfn(w1, w2, rho)

        fd_w1 = (ratio(w1 + eps, w2, rho) - ratio(w1 - eps, w2, rho)) / (2 * eps)
        fd_w2 = (ratio(w1, w2 + eps, rho) - ratio(w1, w2 - eps, rho)) / (2 * eps)
        fd_rho = (ratio(w1, w2, rho + eps) - ratio(w1, w2, rho - eps)) / (2 * eps)

        np.testing.assert_allclose(gw1, fd_w1, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(gw2, fd_w2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(grho, fd_rho, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# 4. grad_univariate_normal_trunc
# ---------------------------------------------------------------------------

def _univ_trunc_mean(mu, sig2, w):
    """Forward: E[Z|Z≤w] for Z~N(μ,σ²)."""
    sig = np.sqrt(sig2)
    z = (w - mu) / sig
    lam = -norm.pdf(z) / max(norm.cdf(z), 1e-300)
    return mu + sig * lam


def _univ_trunc_var(mu, sig2, w):
    """Forward: Var[Z|Z≤w] for Z~N(μ,σ²)."""
    sig = np.sqrt(sig2)
    z = (w - mu) / sig
    Phi_z = max(norm.cdf(z), 1e-300)
    lam = -norm.pdf(z) / Phi_z
    return sig2 * (1.0 + lam * (z - lam))


class TestGradUnivariateNormalTrunc:
    """FD verification of ∂(μ̃, σ̃²)/∂(μ, σ², w)."""

    @pytest.mark.parametrize("mu, sig2, w", [
        (0.0, 1.0, 0.5),     # moderate truncation
        (0.0, 1.0, -1.0),    # tight truncation
        (0.0, 1.0, 3.0),     # loose truncation
        (1.5, 2.0, 2.0),     # non-standard
        (-1.0, 0.5, 0.0),    # small variance
    ])
    def test_dmutrunc_fd(self, mu, sig2, w):
        """FD verification of ∂μ̃/∂(μ, σ², w)."""
        eps = 1e-6
        dmutrunc, _ = grad_univariate_normal_trunc(mu, sig2, w)

        fd_mu = (_univ_trunc_mean(mu + eps, sig2, w)
                 - _univ_trunc_mean(mu - eps, sig2, w)) / (2 * eps)
        fd_sig2 = (_univ_trunc_mean(mu, sig2 + eps, w)
                   - _univ_trunc_mean(mu, sig2 - eps, w)) / (2 * eps)
        fd_w = (_univ_trunc_mean(mu, sig2, w + eps)
                - _univ_trunc_mean(mu, sig2, w - eps)) / (2 * eps)

        np.testing.assert_allclose(dmutrunc[0], fd_mu, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(dmutrunc[1], fd_sig2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(dmutrunc[2], fd_w, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("mu, sig2, w", [
        (0.0, 1.0, 0.5),
        (0.0, 1.0, -1.0),
        (0.0, 1.0, 3.0),
        (1.5, 2.0, 2.0),
        (-1.0, 0.5, 0.0),
    ])
    def test_dsigtrunc_fd(self, mu, sig2, w):
        """FD verification of ∂σ̃²/∂(μ, σ², w)."""
        eps = 1e-6
        _, dsigtrunc = grad_univariate_normal_trunc(mu, sig2, w)

        fd_mu = (_univ_trunc_var(mu + eps, sig2, w)
                 - _univ_trunc_var(mu - eps, sig2, w)) / (2 * eps)
        fd_sig2 = (_univ_trunc_var(mu, sig2 + eps, w)
                   - _univ_trunc_var(mu, sig2 - eps, w)) / (2 * eps)
        fd_w = (_univ_trunc_var(mu, sig2, w + eps)
                - _univ_trunc_var(mu, sig2, w - eps)) / (2 * eps)

        np.testing.assert_allclose(dsigtrunc[0], fd_mu, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(dsigtrunc[1], fd_sig2, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(dsigtrunc[2], fd_w, atol=1e-5, rtol=1e-4)

    def test_dmutrunc_dmu_near_one_at_loose(self):
        """With very loose truncation, ∂μ̃/∂μ ≈ 1."""
        dmutrunc, _ = grad_univariate_normal_trunc(0.0, 1.0, 10.0)
        np.testing.assert_allclose(dmutrunc[0], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 5. grad_bivariate_normal_trunc
# ---------------------------------------------------------------------------

def _biv_mean_from_params(params):
    """Compute truncated bivariate mean from flat param vector.

    params = [a1, a2, sig11, sig12, sig22, w1, w2]
    """
    mu = params[:2]
    cov = np.array([[params[2], params[3]], [params[3], params[4]]])
    trpoint = params[5:7]
    return truncated_bivariate_mean(mu, cov, trpoint)


def _biv_cov_upper_from_params(params):
    """Compute upper-tri of truncated bivariate cov from flat param vector.

    Returns [Sigma11, Sigma12, Sigma22].
    """
    mu = params[:2]
    cov = np.array([[params[2], params[3]], [params[3], params[4]]])
    trpoint = params[5:7]
    C = truncated_bivariate_cov(mu, cov, trpoint)
    return np.array([C[0, 0], C[0, 1], C[1, 1]])


class TestGradBivariateNormalTrunc:
    """FD verification of bivariate truncated moment gradients.

    Uses monkeypatch to replace bivariate_normal_cdf with the precise
    quad-based version in both _bivariate_trunc (forward functions) and
    _trunc_grads (analytic gradient). This ensures both sides of the
    FD comparison use ~15-digit BVN precision.
    """

    @pytest.fixture(autouse=True)
    def _patch_bvn(self, monkeypatch):
        """Patch bivariate_normal_cdf in both modules with precise version."""
        import pybhatlib.gradmvn._bivariate_trunc as _bt
        import pybhatlib.gradmvn._trunc_grads as _tg
        monkeypatch.setattr(_bt, "bivariate_normal_cdf", _bvn_cdf_precise)
        monkeypatch.setattr(_tg, "bivariate_normal_cdf", _bvn_cdf_precise)

    @pytest.mark.parametrize("mu, cov_upper, trpoint", [
        # moderate truncation, unit variance, ρ=0.3
        ([0.0, 0.0], [1.0, 0.3, 1.0], [0.5, 0.8]),
        # tight truncation
        ([0.0, 0.0], [1.0, 0.3, 1.0], [-0.5, -0.3]),
        # high correlation
        ([0.0, 0.0], [1.0, 0.8, 1.0], [1.0, 1.0]),
        # negative correlation
        ([0.0, 0.0], [1.0, -0.5, 1.0], [0.5, 0.5]),
        # non-unit variance and non-zero mean
        ([1.0, -0.5], [2.0, 0.5, 1.5], [2.0, 0.5]),
    ])
    def test_dmuderiv_fd(self, mu, cov_upper, trpoint):
        """FD verification of the mean Jacobian (dmuderiv)."""
        mu_arr = np.array(mu)
        cov_mat = np.array([[cov_upper[0], cov_upper[1]],
                            [cov_upper[1], cov_upper[2]]])
        tp_arr = np.array(trpoint)

        dmuderiv, _ = grad_bivariate_normal_trunc(mu_arr, cov_mat, tp_arr)

        # Pack into flat vector for FD
        params = np.array([mu[0], mu[1], cov_upper[0], cov_upper[1],
                           cov_upper[2], trpoint[0], trpoint[1]])

        eps = 1e-6
        for i in range(7):
            fd = _fd_vec(_biv_mean_from_params, params, i, eps)
            np.testing.assert_allclose(
                dmuderiv[i, :], fd, atol=5e-4, rtol=5e-3,
                err_msg=f"dmuderiv[{i},:] mismatch for input {i}"
            )

    @pytest.mark.parametrize("mu, cov_upper, trpoint", [
        ([0.0, 0.0], [1.0, 0.3, 1.0], [0.5, 0.8]),
        ([0.0, 0.0], [1.0, 0.3, 1.0], [-0.5, -0.3]),
        ([0.0, 0.0], [1.0, 0.8, 1.0], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, -0.5, 1.0], [0.5, 0.5]),
        ([1.0, -0.5], [2.0, 0.5, 1.5], [2.0, 0.5]),
    ])
    def test_domgderiv_fd(self, mu, cov_upper, trpoint):
        """FD verification of the covariance Jacobian (domgderiv)."""
        mu_arr = np.array(mu)
        cov_mat = np.array([[cov_upper[0], cov_upper[1]],
                            [cov_upper[1], cov_upper[2]]])
        tp_arr = np.array(trpoint)

        _, domgderiv = grad_bivariate_normal_trunc(mu_arr, cov_mat, tp_arr)

        params = np.array([mu[0], mu[1], cov_upper[0], cov_upper[1],
                           cov_upper[2], trpoint[0], trpoint[1]])

        eps = 1e-6
        for i in range(7):
            fd = _fd_vec(_biv_cov_upper_from_params, params, i, eps)
            np.testing.assert_allclose(
                domgderiv[i, :], fd, atol=5e-4, rtol=5e-3,
                err_msg=f"domgderiv[{i},:] mismatch for input {i}"
            )


# ---------------------------------------------------------------------------
# 6. grad_noncdfbvn
# ---------------------------------------------------------------------------

class TestGradNonCdfBvn:
    """FD verification of ∂BVN/∂(μ, Σ, x) non-standard."""

    @pytest.mark.parametrize("mu, cov_upper, x", [
        ([0.0, 0.0], [1.0, 0.3, 1.0], [0.5, 0.8]),
        ([1.0, -0.5], [2.0, 0.5, 1.5], [2.0, 0.5]),
        ([0.0, 0.0], [1.0, 0.8, 1.0], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, -0.5, 1.0], [-0.5, 0.5]),
    ])
    def test_fd(self, mu, cov_upper, x):
        mu_arr = np.array(mu, dtype=np.float64)
        cov_mat = np.array([[cov_upper[0], cov_upper[1]],
                            [cov_upper[1], cov_upper[2]]], dtype=np.float64)
        x_arr = np.array(x, dtype=np.float64)

        gmu, gcov, gx = grad_noncdfbvn(mu_arr, cov_mat, x_arr)

        eps = 1e-6

        def bvn_cdf(mm, cc, xx):
            return _bvn_cdf_precise(
                (xx[0] - mm[0]) / np.sqrt(cc[0, 0]),
                (xx[1] - mm[1]) / np.sqrt(cc[1, 1]),
                cc[0, 1] / np.sqrt(cc[0, 0] * cc[1, 1]),
            )

        # ∂/∂μ
        for i in range(2):
            m_p = mu_arr.copy(); m_p[i] += eps
            m_m = mu_arr.copy(); m_m[i] -= eps
            fd = (bvn_cdf(m_p, cov_mat, x_arr) - bvn_cdf(m_m, cov_mat, x_arr)) / (2 * eps)
            np.testing.assert_allclose(gmu[i], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gmu[{i}]")

        # ∂/∂x
        for i in range(2):
            x_p = x_arr.copy(); x_p[i] += eps
            x_m = x_arr.copy(); x_m[i] -= eps
            fd = (bvn_cdf(mu_arr, cov_mat, x_p) - bvn_cdf(mu_arr, cov_mat, x_m)) / (2 * eps)
            np.testing.assert_allclose(gx[i], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gx[{i}]")

        # ∂/∂Σ (upper-tri: Σ₁₁, Σ₁₂, Σ₂₂)
        cov_indices = [(0, 0), (0, 1), (1, 1)]
        for k, (r, c) in enumerate(cov_indices):
            c_p = cov_mat.copy(); c_p[r, c] += eps; c_p[c, r] = c_p[r, c]
            c_m = cov_mat.copy(); c_m[r, c] -= eps; c_m[c, r] = c_m[r, c]
            fd = (bvn_cdf(mu_arr, c_p, x_arr) - bvn_cdf(mu_arr, c_m, x_arr)) / (2 * eps)
            np.testing.assert_allclose(gcov[k], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gcov[{k}] ({r},{c})")


# ---------------------------------------------------------------------------
# 7. grad_noncdfbvn_by_cdfn
# ---------------------------------------------------------------------------

class TestGradNonCdfBvnByCdfn:
    """FD verification of ∂[BVN/Φ]/∂(μ, Σ, x) non-standard."""

    @pytest.mark.parametrize("mu, cov_upper, x", [
        ([0.0, 0.0], [1.0, 0.3, 1.0], [0.5, 0.8]),
        ([1.0, -0.5], [2.0, 0.5, 1.5], [2.0, 0.5]),
        ([0.0, 0.0], [1.0, 0.8, 1.0], [1.0, 1.0]),
        ([0.0, 0.0], [1.0, -0.5, 1.0], [0.5, 0.5]),
    ])
    def test_fd(self, mu, cov_upper, x):
        mu_arr = np.array(mu, dtype=np.float64)
        cov_mat = np.array([[cov_upper[0], cov_upper[1]],
                            [cov_upper[1], cov_upper[2]]], dtype=np.float64)
        x_arr = np.array(x, dtype=np.float64)

        gmu, gcov, gx = grad_noncdfbvn_by_cdfn(mu_arr, cov_mat, x_arr)

        eps = 1e-6

        def ratio(mm, cc, xx):
            w1 = (xx[0] - mm[0]) / np.sqrt(cc[0, 0])
            w2 = (xx[1] - mm[1]) / np.sqrt(cc[1, 1])
            rho = cc[0, 1] / np.sqrt(cc[0, 0] * cc[1, 1])
            return _bvn_cdf_precise(w1, w2, rho) / norm.cdf(w1)

        # ∂/∂μ
        for i in range(2):
            m_p = mu_arr.copy(); m_p[i] += eps
            m_m = mu_arr.copy(); m_m[i] -= eps
            fd = (ratio(m_p, cov_mat, x_arr) - ratio(m_m, cov_mat, x_arr)) / (2 * eps)
            np.testing.assert_allclose(gmu[i], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gmu[{i}]")

        # ∂/∂x
        for i in range(2):
            x_p = x_arr.copy(); x_p[i] += eps
            x_m = x_arr.copy(); x_m[i] -= eps
            fd = (ratio(mu_arr, cov_mat, x_p) - ratio(mu_arr, cov_mat, x_m)) / (2 * eps)
            np.testing.assert_allclose(gx[i], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gx[{i}]")

        # ∂/∂Σ
        cov_indices = [(0, 0), (0, 1), (1, 1)]
        for k, (r, c) in enumerate(cov_indices):
            c_p = cov_mat.copy(); c_p[r, c] += eps; c_p[c, r] = c_p[r, c]
            c_m = cov_mat.copy(); c_m[r, c] -= eps; c_m[c, r] = c_m[r, c]
            fd = (ratio(mu_arr, c_p, x_arr) - ratio(mu_arr, c_m, x_arr)) / (2 * eps)
            np.testing.assert_allclose(gcov[k], fd, atol=1e-5, rtol=1e-3,
                                       err_msg=f"gcov[{k}] ({r},{c})")
