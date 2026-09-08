"""Tests for the conditional-Gaussian copula port (``pybhatlib.mixed._copula``).

Two independent validation gates, both required:

(1) **Live-GAUSS reference** -- values generated with GAUSS 26.1.1 (``tgauss``)
    for a concrete random-coefficient <-> kernel case ``nrndcoef=2, nc=3``
    (``nrndtot=4``): a valid unit-diagonal 4x4 correlation ``omegastar``, an
    i.i.d. draw ``e`` / marginal draw ``g``, ``indxrand=[1,1,0,0]`` and
    ``Y=eye(2)``.  The Python port must reproduce ``condition``, ``gcondnewcov``,
    ``gcondnewmean`` and ``gcondspecialnewmean`` to ~1e-9.

(2) **Finite-difference gates** -- ``gcondnewcov`` vs central-FD of
    ``condition``'s ``COVB`` w.r.t. the correlation off-diagonals;
    ``gcondnewmean`` vs central-FD of ``condition``'s ``B`` w.r.t. ``mu`` and the
    correlation; ``gcondspecialnewmean``'s ``gmu`` vs central-FD of the special
    ``B`` w.r.t. ``mu`` (the special ``gX``/``gW`` are structured Cholesky
    partials, validated against GAUSS rather than a plain total-derivative FD).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._copula import (
    condition,
    gcondnewcov,
    gcondnewmean,
    gcondspecialnewmean,
)

# ---------------------------------------------------------------------------
# Fixed inputs (identical to the GAUSS reference script)
# ---------------------------------------------------------------------------
OMEGASTAR = np.array(
    [
        [1.0, 0.3, 0.2, 0.1],
        [0.3, 1.0, 0.25, 0.15],
        [0.2, 0.25, 1.0, 0.35],
        [0.1, 0.15, 0.35, 1.0],
    ]
)
INDXRAND = np.array([1, 1, 0, 0])
Y2 = np.eye(2)
G_MARG = np.array([0.5, -0.3])            # marginal draw for condition / gcondnewcov
G_FULL = np.array([0.5, -0.3, 0.0, 0.0])  # full-length g for gcondnewmean
EVEC = np.array([0.5, -0.3])              # i.i.d. draw for gcondspecialnewmean
MU0 = np.zeros(4)
MU2 = np.array([0.4, -0.2, 0.1, 0.3])

TOL = 1e-9
FD_TOL = 1e-5

# ---------------------------------------------------------------------------
# Live-GAUSS 26.1.1 reference values (mu = 0)
# ---------------------------------------------------------------------------
GA_condition_B = np.array([[0.00604395604395606], [-0.00934065934065932]])
GA_condition_COVB = np.array(
    [[0.92032967032967, 0.304945054945055],
     [0.304945054945055, 0.974175824175824]]
)
GA_gcondnewcov_gY = np.array(
    [[1.84065934065934, 0.304945054945055, 0.0],
     [0.0, 0.304945054945055, 1.94835164835165]]
)
GA_gcondnewcov_gX = np.array(
    [[0.0573602221953871, 0.0307330032604758, 0.0159401038521918],
     [-0.274725274725275, -0.0604395604395605, 0.0],
     [0.0, -0.137362637362637, -0.120879120879121],
     [-0.417582417582418, -0.131868131868132, 0.0],
     [0.0, -0.208791208791209, -0.263736263736264],
     [0.0, 1.0, 0.0]]
)
GA_gcondnewmean_gY = np.array(
    [[0.00604395604395606, 0.0], [0.0, -0.00934065934065932]]
)
GA_gcondnewmean_gmu = np.array(
    [[-0.137362637362637, -0.0604395604395605],
     [-0.208791208791209, -0.131868131868132],
     [1.0, 0.0],
     [0.0, 1.0]]
)
GA_gcondnewmean_gX = np.array(
    [[-0.0674435454655235, -0.0556092259388963],
     [0.648351648351648, 0.0],
     [0.0, 0.648351648351648],
     [-0.494505494505494, 0.0],
     [0.0, -0.494505494505494],
     [0.0, 0.0]]
)
GA_gcondnewmean_gg = np.array(
    [[0.137362637362637, 0.0604395604395605],
     [0.208791208791209, 0.131868131868132],
     [0.0, 0.0],
     [0.0, 0.0]]
)
GA_special_B = np.array([[0.0402477643068507], [0.0122617458780109]])
GA_special_gY = np.array(
    [[0.0402477643068507, 0.0], [0.0, 0.0122617458780109]]
)
GA_special_gmu = GA_gcondnewmean_gmu.copy()
GA_special_gW = np.array(
    [[0.0, 0.0],
     [0.0, 0.0],
     [0.0402477643068506, 0.0],
     [0.0, 0.0122617458780109]]
)
GA_special_gX = np.array(
    [[0.0, 0.0],
     [0.5, 0.0],
     [0.0, 0.5],
     [-0.3, 0.0],
     [0.0, -0.3],
     [0.0, 0.0]]
)

# ---------------------------------------------------------------------------
# Live-GAUSS 26.1.1 reference values (mu = MU2, non-zero)
# ---------------------------------------------------------------------------
GA_m2_gcondnewmean_gmu = GA_gcondnewmean_gmu.copy()
GA_m2_gcondnewmean_gX = np.array(
    [[-0.0102040816326531, -0.0102040816326531],
     [0.142857142857143, 0.0],
     [0.0, 0.142857142857143],
     [-0.142857142857143, 0.0],
     [0.0, -0.142857142857143],
     [0.0, 0.0]]
)
GA_m2_special_B = np.array([[0.127060951120037], [0.314459548075813]])
GA_m2_special_gmu = GA_gcondnewmean_gmu.copy()
GA_m2_special_gW = np.array(
    [[0.0549450549450549, 0.0241758241758242],
     [-0.0417582417582418, -0.0263736263736264],
     [0.0270609511200375, 0.0],
     [0.0, 0.0144595480758131]]
)
GA_m2_special_gX = np.array(
    [[0.105542808839512, 0.0666586161091656],
     [0.1, 0.0],
     [0.0, 0.1],
     [0.0354511477510139, 0.0],
     [0.0, 0.0354511477510139],
     [0.0, 0.0]]
)


# ---------------------------------------------------------------------------
# (1) GAUSS reference matches
# ---------------------------------------------------------------------------
def test_condition_matches_gauss():
    B, COVB = condition(Y2, MU0, OMEGASTAR, G_MARG, INDXRAND)
    assert np.allclose(B, GA_condition_B, atol=TOL, rtol=0)
    assert np.allclose(COVB, GA_condition_COVB, atol=TOL, rtol=0)
    assert B.shape == (2, 1)
    assert COVB.shape == (2, 2)


def test_gcondnewcov_matches_gauss():
    gY, gX = gcondnewcov(Y2, OMEGASTAR, INDXRAND, cholesky=False, condcov=False)
    assert np.allclose(gY, GA_gcondnewcov_gY, atol=TOL, rtol=0)
    assert np.allclose(gX, GA_gcondnewcov_gX, atol=TOL, rtol=0)
    assert gY.shape == (2, 3)
    assert gX.shape == (6, 3)


def test_gcondnewmean_matches_gauss():
    gY, gmu, gX, gg = gcondnewmean(
        Y2, MU0, OMEGASTAR, G_FULL, INDXRAND, cholesky=False, condcov=False
    )
    assert np.allclose(gY, GA_gcondnewmean_gY, atol=TOL, rtol=0)
    assert np.allclose(gmu, GA_gcondnewmean_gmu, atol=TOL, rtol=0)
    assert np.allclose(gX, GA_gcondnewmean_gX, atol=TOL, rtol=0)
    assert np.allclose(gg, GA_gcondnewmean_gg, atol=TOL, rtol=0)
    assert gmu.shape == (4, 2)
    assert gX.shape == (6, 2)
    assert gg.shape == (4, 2)


def test_gcondnewmean_matches_gauss_nonzero_mu():
    _, gmu, gX, _ = gcondnewmean(
        Y2, MU2, OMEGASTAR, G_FULL, INDXRAND, cholesky=False, condcov=False
    )
    assert np.allclose(gmu, GA_m2_gcondnewmean_gmu, atol=TOL, rtol=0)
    assert np.allclose(gX, GA_m2_gcondnewmean_gX, atol=TOL, rtol=0)


def test_gcondspecialnewmean_matches_gauss():
    B, gY, gmu, gW, gX = gcondspecialnewmean(
        Y2, MU0, OMEGASTAR, EVEC, INDXRAND, cholesky=False, condcov=False
    )
    assert np.allclose(B, GA_special_B, atol=TOL, rtol=0)
    assert np.allclose(gY, GA_special_gY, atol=TOL, rtol=0)
    assert np.allclose(gmu, GA_special_gmu, atol=TOL, rtol=0)
    assert np.allclose(gW, GA_special_gW, atol=TOL, rtol=0)
    assert np.allclose(gX, GA_special_gX, atol=TOL, rtol=0)
    assert B.shape == (2, 1)
    assert gW.shape == (4, 2)
    assert gX.shape == (6, 2)


def test_gcondspecialnewmean_matches_gauss_nonzero_mu():
    B, _, gmu, gW, gX = gcondspecialnewmean(
        Y2, MU2, OMEGASTAR, EVEC, INDXRAND, cholesky=False, condcov=False
    )
    assert np.allclose(B, GA_m2_special_B, atol=TOL, rtol=0)
    assert np.allclose(gmu, GA_m2_special_gmu, atol=TOL, rtol=0)
    assert np.allclose(gW, GA_m2_special_gW, atol=TOL, rtol=0)
    assert np.allclose(gX, GA_m2_special_gX, atol=TOL, rtol=0)


# ---------------------------------------------------------------------------
# (2) Finite-difference gates
# ---------------------------------------------------------------------------
def _offdiag_pairs(K):
    return [(p, q) for p in range(K) for q in range(p + 1, K)]


def _perturb_corr(X, p, q, h):
    Xp = X.copy()
    Xp[p, q] += h
    Xp[q, p] += h
    return Xp


def _vecdup(mat):
    K = mat.shape[0]
    return np.array([mat[i, j] for i in range(K) for j in range(i, K)])


def test_gcondnewcov_fd_wrt_correlation():
    """gcondnewcov.gX vs central-FD of condition's COVB w.r.t. correlation."""
    _, gX = gcondnewcov(Y2, OMEGASTAR, INDXRAND)
    h = 1e-6
    K = 4
    fd = np.zeros_like(gX)
    for r, (p, q) in enumerate(_offdiag_pairs(K)):
        _, Cp = condition(Y2, MU0, _perturb_corr(OMEGASTAR, p, q, h),
                          G_MARG, INDXRAND)
        _, Cm = condition(Y2, MU0, _perturb_corr(OMEGASTAR, p, q, -h),
                          G_MARG, INDXRAND)
        fd[r, :] = _vecdup((Cp - Cm) / (2 * h))
    assert np.allclose(gX, fd, atol=FD_TOL, rtol=0)


def test_gcondnewmean_fd_wrt_mu():
    """gcondnewmean.gmu vs central-FD of condition's B w.r.t. mu."""
    _, gmu, _, _ = gcondnewmean(Y2, MU2, OMEGASTAR, G_FULL, INDXRAND)
    h = 1e-6
    fd = np.zeros_like(gmu)
    for j in range(4):
        mup = MU2.copy(); mup[j] += h
        mum = MU2.copy(); mum[j] -= h
        Bp, _ = condition(Y2, mup, OMEGASTAR, G_MARG, INDXRAND)
        Bm, _ = condition(Y2, mum, OMEGASTAR, G_MARG, INDXRAND)
        fd[j, :] = ((Bp - Bm) / (2 * h)).ravel()
    assert np.allclose(gmu, fd, atol=FD_TOL, rtol=0)


def test_gcondnewmean_fd_wrt_correlation():
    """gcondnewmean.gX vs central-FD of condition's B w.r.t. correlation."""
    _, _, gX, _ = gcondnewmean(Y2, MU2, OMEGASTAR, G_FULL, INDXRAND)
    h = 1e-6
    fd = np.zeros_like(gX)
    for r, (p, q) in enumerate(_offdiag_pairs(4)):
        Bp, _ = condition(Y2, MU2, _perturb_corr(OMEGASTAR, p, q, h),
                          G_MARG, INDXRAND)
        Bm, _ = condition(Y2, MU2, _perturb_corr(OMEGASTAR, p, q, -h),
                          G_MARG, INDXRAND)
        fd[r, :] = ((Bp - Bm) / (2 * h)).ravel()
    assert np.allclose(gX, fd, atol=FD_TOL, rtol=0)


def _special_B(Y, mu, X, e, indxmarg):
    B, *_ = gcondspecialnewmean(Y, mu, X, e, indxmarg)
    return B


def test_gcondspecialnewmean_fd_wrt_mu():
    """special.gmu vs central-FD of the special B w.r.t. mu."""
    _, _, gmu, _, _ = gcondspecialnewmean(Y2, MU2, OMEGASTAR, EVEC, INDXRAND)
    h = 1e-6
    fd = np.zeros_like(gmu)
    for j in range(4):
        mup = MU2.copy(); mup[j] += h
        mum = MU2.copy(); mum[j] -= h
        Bp = _special_B(Y2, mup, OMEGASTAR, EVEC, INDXRAND)
        Bm = _special_B(Y2, mum, OMEGASTAR, EVEC, INDXRAND)
        fd[j, :] = ((Bp - Bm) / (2 * h)).ravel()
    assert np.allclose(gmu, fd, atol=FD_TOL, rtol=0)


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_cholesky_flag_not_implemented():
    with pytest.raises(NotImplementedError):
        gcondnewcov(Y2, OMEGASTAR, INDXRAND, cholesky=True)
    with pytest.raises(NotImplementedError):
        gcondnewmean(Y2, MU0, OMEGASTAR, G_FULL, INDXRAND, cholesky=True)
    with pytest.raises(NotImplementedError):
        gcondspecialnewmean(Y2, MU0, OMEGASTAR, EVEC, INDXRAND, cholesky=True)
