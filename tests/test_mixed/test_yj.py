"""Tests for the Yeo-Johnson transform family and Gauss-Hermite quadrature.

Reference values are ground truth from live GAUSS BHATLIB (Bhat).
Gradients are checked against central finite differences.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.utils._quadrature import SUPPORTED_ORDERS, gauss_hermite
from pybhatlib.vecup._yj import (
    gradmeanyj,
    gradstandyjinvnonpgiven,
    gyjinvnonp,
    gyjnonp,
    meanyj,
    standyjinvnonpgiven,
    yjinvnonp,
    yjnonp,
)

SQRT_PI = np.sqrt(np.pi)


# --------------------------------------------------------------------------
# (a) Gauss-Hermite weight-sum == sqrt(pi)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("intord", SUPPORTED_ORDERS)
def test_gauss_hermite_weight_sum(intord):
    nodes, weights = gauss_hermite(intord)
    assert nodes.shape == (intord,)
    assert weights.shape == (intord,)
    assert weights.sum() == pytest.approx(SQRT_PI, abs=1e-12)
    # symmetric about zero
    assert np.max(np.abs(np.sort(nodes) + np.sort(nodes)[::-1])) < 1e-10


def test_gauss_hermite_bad_order():
    with pytest.raises(ValueError):
        gauss_hermite(7)


# --------------------------------------------------------------------------
# (b) meanyj matches the three GAUSS reference values
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lam, mu_ref, sig_ref",
    [
        (0.5, 0.1938973358, 1.128623976),
        (1.2, -0.07143979697, 1.017956989),
    ],
)
def test_meanyj_reference(lam, mu_ref, sig_ref):
    mu, sig = meanyj(lam, 20)
    assert mu[0] == pytest.approx(mu_ref, abs=1e-7)
    assert sig[0] == pytest.approx(sig_ref, abs=1e-7)


# --------------------------------------------------------------------------
# (c) meanyj(1, 20) == (0, 1)  -- lambda == 1 is the identity transform
# --------------------------------------------------------------------------
def test_meanyj_identity():
    mu, sig = meanyj(1.0, 20)
    assert mu[0] == pytest.approx(0.0, abs=1e-12)
    assert sig[0] == pytest.approx(1.0, abs=1e-12)


def test_standyjinvnonpgiven_identity():
    # lambda == 1 => standardized inverse YJ returns x unchanged.
    x = np.array([[-1.3, -0.4, 0.0, 0.7, 2.1]])
    mu, sig = meanyj(1.0, 20)
    F = standyjinvnonpgiven(1.0, mu, sig, x)
    assert np.max(np.abs(F - x)) < 1e-10


def test_yj_forward_inverse_roundtrip():
    # yjnonp is the forward transform; yjinvnonp (mu=0, w=I) is its inverse.
    lam = np.array([0.6, 1.3])
    x = np.array([0.8, -1.1])
    y = yjnonp(lam, x)
    back = yjinvnonp(np.zeros(2), np.eye(2), lam, y)
    assert np.max(np.abs(back - x)) < 1e-10


# --------------------------------------------------------------------------
# (d) analytic gradients vs central finite differences
# --------------------------------------------------------------------------
def _central_diff(f, x0, h=1e-6):
    x0 = np.asarray(x0, dtype=np.float64)
    grad = np.zeros_like(x0)
    for i in range(x0.size):
        xp_ = x0.copy()
        xm_ = x0.copy()
        xp_.flat[i] += h
        xm_.flat[i] -= h
        grad.flat[i] = (f(xp_) - f(xm_)) / (2.0 * h)
    return grad


def test_gyjnonp_gradients():
    lam = np.array([0.6, 1.3, 0.9])
    x = np.array([0.8, -1.1, 0.3])
    F, glam, gx = gyjnonp(lam, x)

    # diagonal Jacobians
    assert np.allclose(glam, np.diag(np.diag(glam)))
    assert np.allclose(gx, np.diag(np.diag(gx)))

    dlam = _central_diff(lambda L: yjnonp(L, x).sum(), lam)
    dx = _central_diff(lambda X: yjnonp(lam, X).sum(), x)
    assert np.allclose(np.diag(glam), dlam, atol=1e-6)
    assert np.allclose(np.diag(gx), dx, atol=1e-6)


def test_gyjinvnonp_gradients():
    mu = np.array([0.2, -0.5, 0.1])
    wdiag = np.diag([1.1, 0.9, 1.3])
    lam = np.array([0.6, 1.3, 0.9])
    x = np.array([0.4, -0.7, 1.2])
    F, gmu, gwdiag, glam, gx = gyjinvnonp(mu, wdiag, lam, x)

    def f_mu(M):
        return yjinvnonp(M, wdiag, lam, x)

    def f_w(wv):
        return yjinvnonp(mu, np.diag(wv), lam, x)

    def f_lam(L):
        return yjinvnonp(mu, wdiag, L, x)

    def f_x(X):
        return yjinvnonp(mu, wdiag, lam, X)

    # component-wise (diagonal) checks
    w = np.diag(wdiag)
    for i in range(3):
        assert gmu[i, i] == pytest.approx(
            _central_diff(lambda M: f_mu(M)[i], mu)[i], abs=1e-6
        )
        assert gwdiag[i, i] == pytest.approx(
            _central_diff(lambda wv: f_w(wv)[i], w)[i], abs=1e-6
        )
        assert glam[i, i] == pytest.approx(
            _central_diff(lambda L: f_lam(L)[i], lam)[i], abs=1e-6
        )
        assert gx[i, i] == pytest.approx(
            _central_diff(lambda X: f_x(X)[i], x)[i], abs=1e-6
        )


def test_gradmeanyj_gradients():
    lam = np.array([0.6, 1.3, 0.9])
    mu, sig, gmuyj, gsigyj = gradmeanyj(lam, 20)

    # values consistent with meanyj
    mu0, sig0 = meanyj(lam, 20)
    assert np.allclose(mu, mu0)
    assert np.allclose(sig, sig0)

    # meanyj returns K-vectors; the diagonal of the FD Jacobian is d(.)_i/dlam_i
    assert np.allclose(gmuyj, np.diag(_fd_jac(lambda L: meanyj(L, 20)[0], lam)),
                       atol=1e-5)
    assert np.allclose(gsigyj, np.diag(_fd_jac(lambda L: meanyj(L, 20)[1], lam)),
                       atol=1e-5)


def _fd_jac(f, x0, h=1e-6):
    x0 = np.asarray(x0, dtype=np.float64)
    f0 = np.asarray(f(x0))
    jac = np.zeros((f0.size, x0.size))
    for j in range(x0.size):
        xp_ = x0.copy()
        xm_ = x0.copy()
        xp_[j] += h
        xm_[j] -= h
        jac[:, j] = (np.asarray(f(xp_)) - np.asarray(f(xm_))) / (2.0 * h)
    return jac


def test_gradstandyjinvnonpgiven_gradients():
    lam = np.array([0.6, 1.3])
    mu, sig, gmuyj, gsigyj = gradmeanyj(lam, 20)
    x = np.array([[0.4, -0.7, 1.2], [-0.9, 0.5, 0.2]])  # (K=2, Q=3)

    F, glam, gx = gradstandyjinvnonpgiven(lam, mu, sig, gmuyj, gsigyj, x)

    # value consistency with standyjinvnonpgiven
    F_ref = standyjinvnonpgiven(lam, mu, sig, x)
    assert np.allclose(F, F_ref)

    # d F / d lambda : lambda enters via yjinvnonp AND via mu(lam), sig(lam).
    def f_full(L):
        m, s = meanyj(L, 20)
        return standyjinvnonpgiven(L, m, s, x)

    for q in range(x.shape[1]):
        for k in range(x.shape[0]):
            dlam_k = _central_diff(
                lambda L: f_full(L)[k, q], lam
            )[k]
            assert glam[k, q] == pytest.approx(dlam_k, abs=1e-5)

            dx_k = _central_diff(
                lambda X: standyjinvnonpgiven(lam, mu, sig, X)[k, q],
                x,
            )
            # only x[k,q] affects F[k,q]
            assert gx[k, q] == pytest.approx(dx_k.reshape(x.shape)[k, q], abs=1e-5)
