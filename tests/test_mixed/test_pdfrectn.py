"""Tests for pdfrectn / gradpdfrectn (rectangle MVN prob + analytic gradient).

Gate 1 (GAUSS reference): ``pdfrectn`` values match GAUSS ``pdfrectn`` to 1e-7
for a dim-2 case across six index configurations (plain rectangle, one-sided,
one-sided complement, mixed one-sided, and two equality/"given" configs).

Gate 2 (finite differences): ``gradpdfrectn`` matches central FD of ``pdfrectn``
w.r.t. mu / cova / xg / xlow / xup to 1e-5.

The GAUSS reference values were generated with ``pdfrectn_ref.gss`` (library
maxlik, bhatlib; _method="OVUS"; _covarr=1) run under GAUSS 26.1.1 via
``tgauss -b``.  See the module for the exact fixed inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.gradmvn import mvncd_rect
from pybhatlib.gradmvn._pdfrectn import gradpdfrectn, pdfrectn

# --- fixed inputs (must match pdfrectn_ref.gss) --------------------------------
MU = np.array([0.1, -0.2])
COVA = np.array([[1.5, 0.4], [0.4, 0.8]])
XG = np.array([0.5, 0.3])
XLOW = np.array([-0.5, -0.7])
XUP = np.array([0.9, 1.1])
SEED = 28989898.0

# name -> (indxone, indxcomp, indxeq, GAUSS pdfrectn value)
CASES = {
    "rect": ((0, 0), (0, 0), (0, 0), 0.288624176595915),
    "onesided_upper_d0": ((1, 0), (0, 0), (0, 0), 0.375557864170913),
    "onesided_comp_d0": ((1, 0), (1, 0), (0, 0), 0.26331684185486),
    "both_onesided_mixed": ((1, 1), (0, 1), (0, 0), 0.131403411065249),
    "equality_d0_rect_d1": ((0, 0), (0, 0), (1, 0), 0.213381767990146),
    "equality_d0_onesided_d1": ((0, 1), (0, 0), (1, 0), 0.210512171732361),
}


def _idx(triple):
    io, ic, ie = triple
    return np.array(io), np.array(ic), np.array(ie)


@pytest.mark.parametrize("name", list(CASES))
def test_pdfrectn_gauss_reference(name):
    io, ic, ie, ref = (*_idx(CASES[name][:3]), CASES[name][3])
    P, _s = pdfrectn(MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie)
    assert abs(P - ref) < 1e-7, f"{name}: {P} vs GAUSS {ref}"


def test_plain_rectangle_matches_mvncd_rect():
    """sumc(indxeq)==0 plain-rectangle path reuses mvncd_rect."""
    io = np.array([0, 0])
    ic = np.array([0, 0])
    ie = np.array([0, 0])
    P, _s = pdfrectn(MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie)
    ref = float(mvncd_rect(XLOW - MU, XUP - MU, COVA))
    assert abs(P - ref) < 1e-12


@pytest.mark.parametrize("method", ["me", "ovus", "tvbs", "bme", "ovbs", "ssj", "scipy"])
def test_pdfrectn_methods_return_finite_probabilities(method):
    io, ic, ie = _idx(CASES["rect"][:3])
    probability, _ = pdfrectn(
        MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie, method=method
    )

    assert np.isfinite(probability)
    assert 0.0 <= probability <= 1.0


@pytest.mark.parametrize("name", list(CASES))
def test_gradpdfrectn_returns_consistent_value(name):
    io, ic, ie = _idx(CASES[name][:3])
    Pv, _ = pdfrectn(MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie)
    Pg = gradpdfrectn(MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie)[0]
    assert abs(Pv - Pg) < 1e-12


def _fd_grad(io, ic, ie, mu, cova, xg, xlow, xup, eps=1e-6):
    k = len(mu)

    def val(m, c, g, lo, up):
        return pdfrectn(m, c, g, lo, up, SEED, io, ic, ie)[0]

    gmu = np.zeros(k)
    gxg = np.zeros(k)
    gx1 = np.zeros(k)
    gx2 = np.zeros(k)
    for i in range(k):
        e = np.zeros(k)
        e[i] = eps
        gmu[i] = (val(mu + e, cova, xg, xlow, xup) - val(mu - e, cova, xg, xlow, xup)) / (2 * eps)
        gxg[i] = (val(mu, cova, xg + e, xlow, xup) - val(mu, cova, xg - e, xlow, xup)) / (2 * eps)
        gx1[i] = (val(mu, cova, xg, xlow + e, xup) - val(mu, cova, xg, xlow - e, xup)) / (2 * eps)
        gx2[i] = (val(mu, cova, xg, xlow, xup + e) - val(mu, cova, xg, xlow, xup - e)) / (2 * eps)
    pairs = [(i, j) for i in range(k) for j in range(i, k)]
    gcov = np.zeros(len(pairs))
    for c, (p, q) in enumerate(pairs):
        dc = np.zeros((k, k))
        dc[p, q] += eps
        if p != q:
            dc[q, p] += eps
        gcov[c] = (val(mu, cova + dc, xg, xlow, xup) - val(mu, cova - dc, xg, xlow, xup)) / (2 * eps)
    return gmu, gcov, gxg, gx1, gx2


@pytest.mark.parametrize("name", list(CASES))
def test_gradpdfrectn_fd(name):
    io, ic, ie = _idx(CASES[name][:3])
    _P, gmu, gcov, gxg, gx1, gx2, _s = gradpdfrectn(
        MU, COVA, XG, XLOW, XUP, SEED, io, ic, ie
    )
    fmu, fcov, fxg, fx1, fx2 = _fd_grad(io, ic, ie, MU, COVA, XG, XLOW, XUP)
    np.testing.assert_allclose(gmu, fmu, atol=1e-5, err_msg=f"{name} gmu")
    np.testing.assert_allclose(gcov, fcov, atol=1e-5, err_msg=f"{name} gcov")
    np.testing.assert_allclose(gxg, fxg, atol=1e-5, err_msg=f"{name} gxg")
    np.testing.assert_allclose(gx1, fx1, atol=1e-5, err_msg=f"{name} gx1")
    np.testing.assert_allclose(gx2, fx2, atol=1e-5, err_msg=f"{name} gx2")


def test_gradpdfrectn_fd_random_correlated():
    """FD gate on a second, correlated 3-D configuration with an equality dim."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((3, 3))
    cova = A @ A.T + 0.5 * np.eye(3)
    mu = rng.standard_normal(3) * 0.3
    xg = rng.standard_normal(3) * 0.4
    xlow = -np.abs(rng.standard_normal(3)) - 0.2
    xup = np.abs(rng.standard_normal(3)) + 0.2
    # dim0 equality, dim1 rectangle, dim2 one-sided complement
    io = np.array([0, 0, 1])
    ic = np.array([0, 0, 1])
    ie = np.array([1, 0, 0])
    _P, gmu, gcov, gxg, gx1, gx2, _s = gradpdfrectn(mu, cova, xg, xlow, xup, SEED, io, ic, ie)
    fmu, fcov, fxg, fx1, fx2 = _fd_grad(io, ic, ie, mu, cova, xg, xlow, xup)
    np.testing.assert_allclose(gmu, fmu, atol=1e-5)
    np.testing.assert_allclose(gcov, fcov, atol=1e-5)
    np.testing.assert_allclose(gxg, fxg, atol=1e-5)
    np.testing.assert_allclose(gx1, fx1, atol=1e-5)
    np.testing.assert_allclose(gx2, fx2, atol=1e-5)
