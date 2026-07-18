"""Yeo-Johnson (YJ) transformation family and its inverse.

Verbatim port of the GAUSS BHATLIB procedures in ``vecup.src``:

- :func:`yjnonp`      (line 2296) -- forward YJ transform.
- :func:`gyjnonp`     (line 2335) -- gradient of the forward transform.
- :func:`yjinvnonp`   (line 2550) -- inverse YJ transform (normal -> non-normal).
- :func:`gyjinvnonp`  (line 2618) -- gradient of the inverse transform.
- :func:`meanyj`      (line 2758) -- mean/std of standardized inverse YJ via
  Gauss-Hermite quadrature.
- :func:`gradmeanyj`  (line 2801) -- gradient of the mean/std w.r.t. lambda.
- :func:`standyjinvnonpgiven`     (line 2893) -- standardized inverse YJ given
  precomputed (mu, sig).
- :func:`gradstandyjinvnonpgiven` (line 2964) -- gradient of the standardized
  inverse YJ given precomputed (mu, sig, gmuyj, gsigyj).

Conventions (BHATLIB):

- The transform is applied univariate-by-univariate, so the Jacobians w.r.t.
  ``lambda``, ``mu``, ``wdiag`` and ``x`` are all **diagonal** ``(K, K)``
  matrices.
- ``lambda`` (``lamnonp``) is the *non-parameterized* YJ parameter, valid in
  ``(0, 2)``; ``lambda == 1`` is the identity transform.
- ``x`` for the inverse routines is a *standardized* normal abscissa; it is
  de-standardized (``wdiag * x + mu``) before the inverse map.

All public functions accept an optional ``xp`` backend kwarg for parity with
the rest of pybhatlib; the numerical work is done in NumPy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.utils._quadrature import gauss_hermite


def _as_1d(a: object) -> NDArray[np.float64]:
    """Coerce a scalar / array-like to a 1-D float64 array."""
    return np.atleast_1d(np.asarray(a, dtype=np.float64)).ravel()


def yjnonp(lam: object, x: object, *, xp=None) -> NDArray[np.float64]:
    """Forward Yeo-Johnson transform (non-parameterized lambda).

    GAUSS ``yjnonp`` (line 2296).  For each element::

        lam1 = lam        if x >= 0 else (2 - lam)
        g    = sign(x) * (((|x| + 1) ** lam1 - 1) / lam1)

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s), each in ``(0, 2)``.
    x : array-like, shape (K,)
        Abscissae to transform.
    xp : backend, optional
        Array backend (numerical work uses NumPy).

    Returns
    -------
    g : ndarray, shape (K,)
        Transformed values.
    """
    if xp is None:
        xp = array_namespace(x, lam)
    lam = _as_1d(lam)
    x = _as_1d(x)
    x1 = (x >= 0).astype(np.float64)
    x2 = 2.0 * x1 - 1.0
    x3 = np.abs(x)
    lam1 = (2.0 - lam) * (1.0 - x1) + lam * x1
    g = x2 * (((x3 + 1.0) ** lam1 - 1.0) / lam1)
    return g


def gyjnonp(
    lam: object, x: object, *, xp=None
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Gradient of the forward YJ transform w.r.t. lambda and x.

    GAUSS ``gyjnonp`` (line 2335, 3 outputs).

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s).
    x : array-like, shape (K,)
        Abscissae.
    xp : backend, optional
        Array backend.

    Returns
    -------
    F : ndarray, shape (K,)
        Transformed values (same as :func:`yjnonp`).
    glam : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dlam``.
    gx : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dx``.
    """
    if xp is None:
        xp = array_namespace(x, lam)
    lam = _as_1d(lam)
    x = _as_1d(x)
    e1 = x.shape[0]
    x1 = (x >= 0).astype(np.float64)
    x2 = 2.0 * x1 - 1.0
    x3 = np.abs(x)
    x4 = x3 + 1.0
    lam1 = (2.0 - lam) * (1.0 - x1) + lam * x1
    g = x2 * (((x3 + 1.0) ** lam1 - 1.0) / lam1)
    g1 = np.abs(g)
    glam_diag = ((x4 ** lam1) * np.log(x4) - g1) / lam1
    gx_diag = x4 ** (x2 * (lam - 1.0))
    glam = np.diag(glam_diag)
    gx = np.diag(gx_diag)
    return g, glam, gx


def yjinvnonp(
    mu: object, wdiag: object, lamnonp: object, x: object, *, xp=None
) -> NDArray[np.float64]:
    """Inverse Yeo-Johnson transform (normal -> non-normal).

    GAUSS ``yjinvnonp`` (line 2550).  The standardized normal abscissa ``x`` is
    de-standardized to ``newx = diag(wdiag) * x + mu`` and then mapped::

        newx <  0 : zf = 1 - (1 - (2 - lam) * newx) ** (1 / (2 - lam))
        newx >= 0 : zf = (1 + lam * newx) ** (1 / lam) - 1

    Each branch is evaluated only where its base is valid (matching the GAUSS
    ``missrv(...,0)`` masking) so negative fractional-power bases never produce
    NaNs.

    Parameters
    ----------
    mu : array-like, shape (K,)
        Means used to de-standardize ``x``.
    wdiag : array-like, shape (K, K)
        Diagonal matrix of standard deviations (square roots of variances).
    lamnonp : array-like, shape (K,)
        Non-parameterized YJ parameter(s), each in ``(0, 2)``.
    x : array-like, shape (K,) or (K, Q)
        Standardized normal abscissae (``K`` variates, ``Q`` observations).
    xp : backend, optional
        Array backend.

    Returns
    -------
    zf : ndarray, shape (K,) or (K, Q)
        Non-normal (inverse-transformed) values, matching the shape of ``x``.
    """
    if xp is None:
        xp = array_namespace(x, mu, lamnonp)
    lam = _as_1d(lamnonp)
    mu = _as_1d(mu)
    w = np.diag(np.asarray(wdiag, dtype=np.float64))

    x_arr = np.asarray(x, dtype=np.float64)
    squeeze = x_arr.ndim == 1
    if squeeze:
        x_arr = x_arr[:, None]  # (K, 1)

    lam_c = lam[:, None]
    newx = w[:, None] * x_arr + mu[:, None]  # (K, Q)
    mask_neg = newx < 0.0

    # newx < 0 branch: base = 1 - (2 - lam) * newx  (positive where mask_neg).
    base_neg = 1.0 - (2.0 - lam_c) * newx
    safe_base_neg = np.where(mask_neg, base_neg, 1.0)
    zl0 = 1.0 - safe_base_neg ** (1.0 / (2.0 - lam_c))

    # newx >= 0 branch: base = 1 + lam * newx  (positive where ~mask_neg).
    base_pos = 1.0 + lam_c * newx
    safe_base_pos = np.where(mask_neg, 1.0, base_pos)
    znew = safe_base_pos ** (1.0 / lam_c) - 1.0

    zf = np.where(mask_neg, zl0, znew)
    if squeeze:
        zf = zf[:, 0]
    return zf


def _gyjinvnonp_diag(
    mu: NDArray[np.float64],
    w: NDArray[np.float64],
    lam: NDArray[np.float64],
    x_v: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Diagonal-vector core of :func:`gyjinvnonp` (no dense ``np.diag`` builds).

    All inputs are 1-D ``(K,)`` arrays (``w`` is the diagonal of the std-dev
    matrix, i.e. ``np.diag(wdiag)``).  Returns the *diagonals* of the five
    Jacobians as ``(K,)`` vectors instead of materializing ``(K, K)`` diagonal
    matrices.  The elementwise math is byte-identical to :func:`gyjinvnonp`;
    the hot internal callers use this to avoid building diagonal matrices only
    to immediately extract their diagonals back out.
    """
    e1 = mu.shape[0]
    newx = w * x_v + mu  # (K,)
    z = np.zeros(e1, dtype=np.float64)
    gx_diag = np.zeros(e1, dtype=np.float64)
    gw = np.zeros(e1, dtype=np.float64)
    gmu_diag = np.zeros(e1, dtype=np.float64)
    glam_diag = np.zeros(e1, dtype=np.float64)

    for i in range(e1):
        newxelem = newx[i]
        if newxelem < 0.0:
            z1 = 1.0 - (2.0 - lam[i]) * newxelem
            z2 = 1.0 / (2.0 - lam[i])
            y = z1 ** z2
            z[i] = 1.0 - y
            y1 = (z2 / z1) * newxelem + (z2 ** 2) * np.log(z1)
            glam_diag[i] = -y1 * y
            gx_diag[i] = w[i] * (y / z1)
            gw[i] = x_v[i] * (y / z1)
            gmu_diag[i] = y / z1
        else:
            z1 = 1.0 + lam[i] * newxelem
            z2 = 1.0 / lam[i]
            y = z1 ** z2
            z[i] = y - 1.0
            y1 = (z2 / z1) * newxelem - (z2 ** 2) * np.log(z1)
            glam_diag[i] = y1 * y
            gx_diag[i] = w[i] * (y / z1)
            gw[i] = x_v[i] * (y / z1)
            gmu_diag[i] = y / z1

    return z, gmu_diag, gw, glam_diag, gx_diag


def gyjinvnonp(
    mu: object, wdiag: object, lamnonp: object, x: object, *, xp=None
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Gradient of the inverse YJ transform (one observation, ``Q = 1``).

    GAUSS ``gyjinvnonp`` (line 2618, 5 outputs).  All Jacobians are diagonal
    because the transform is univariate.

    Parameters
    ----------
    mu : array-like, shape (K,)
        Means used to de-standardize ``x``.
    wdiag : array-like, shape (K, K)
        Diagonal matrix of standard deviations.
    lamnonp : array-like, shape (K,)
        Non-parameterized YJ parameter(s).
    x : array-like, shape (K,)
        A single standardized normal abscissa per variate.
    xp : backend, optional
        Array backend.

    Returns
    -------
    F : ndarray, shape (K,)
        Inverse-transformed values.
    gmu : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dmu``.
    gwdiag : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dwdiag``.
    glamnonp : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dlamnonp``.
    gx : ndarray, shape (K, K)
        Diagonal Jacobian ``dF/dx``.
    """
    if xp is None:
        xp = array_namespace(x, mu, lamnonp)
    lam = _as_1d(lamnonp)
    mu = _as_1d(mu)
    x_v = _as_1d(x)
    w = np.diag(np.asarray(wdiag, dtype=np.float64))

    z, gmu_diag, gw, glam_diag, gx_diag = _gyjinvnonp_diag(mu, w, lam, x_v)

    gmu = np.diag(gmu_diag)
    gwdiag = np.diag(gw)
    glamnonp = np.diag(glam_diag)
    gx = np.diag(gx_diag)
    return z, gmu, gwdiag, glamnonp, gx


def meanyj(
    lam: object, intord: int, *, xp=None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and standard deviation of the standardized inverse YJ transform.

    GAUSS ``meanyj`` (line 2758) via Gauss-Hermite quadrature::

        newx = sqrt(2) * e            (mu = 0, w = 1)
        F    = yjinvnonp(0, I, lam, newx)
        mu   = (1 / sqrt(pi)) * sum(w * F)
        m2   = (1 / sqrt(pi)) * sum(w * F^2)
        sig  = sqrt(m2 - mu^2)

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s), each in ``(0, 2)``.
    intord : int
        Gauss-Hermite integration order (see :func:`gauss_hermite`).
    xp : backend, optional
        Array backend.

    Returns
    -------
    mu : ndarray, shape (K,)
        Mean of the (unstandardized) inverse YJ variate.
    sig : ndarray, shape (K,)
        Standard deviation of the inverse YJ variate.
    """
    if xp is None:
        xp = array_namespace(lam)
    lam = _as_1d(lam)
    e1 = lam.shape[0]
    e, w = gauss_hermite(intord)
    e2 = np.sqrt(2.0) * e  # (N,)
    e3 = np.tile(e2, (e1, 1))  # (K, N): each row is e2
    F = yjinvnonp(np.zeros(e1), np.eye(e1), lam, e3)  # (K, N)
    mu = (1.0 / np.sqrt(np.pi)) * np.sum(F * w, axis=1)
    m2 = (1.0 / np.sqrt(np.pi)) * np.sum(F ** 2 * w, axis=1)
    sig = np.sqrt(m2 - mu ** 2)
    return mu, sig


def gradmeanyj(
    lam: object, intord: int, *, xp=None
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Gradient of the inverse-YJ mean/std w.r.t. lambda.

    GAUSS ``gradmeanyj`` (line 2801, 4 outputs) using :func:`gyjinvnonp` over
    the Gauss-Hermite nodes.

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s).
    intord : int
        Gauss-Hermite integration order.
    xp : backend, optional
        Array backend.

    Returns
    -------
    mu : ndarray, shape (K,)
        Mean (same as :func:`meanyj`).
    sig : ndarray, shape (K,)
        Standard deviation (same as :func:`meanyj`).
    gmuyj : ndarray, shape (K,)
        Gradient ``dmu/dlam``.
    gsigyj : ndarray, shape (K,)
        Gradient ``dsig/dlam``.
    """
    if xp is None:
        xp = array_namespace(lam)
    lam = _as_1d(lam)
    e1 = lam.shape[0]
    e, w = gauss_hermite(intord)
    e2 = np.sqrt(2.0) * e
    e3 = np.tile(e2, (e1, 1))  # (K, N)
    n = e3.shape[1]

    F = np.zeros((e1, n), dtype=np.float64)
    glam = np.zeros((e1, n), dtype=np.float64)
    glam1 = np.zeros((e1, n), dtype=np.float64)
    zeros_e1 = np.zeros(e1, dtype=np.float64)
    ones_e1 = np.ones(e1, dtype=np.float64)
    for i in range(n):
        Ftemp, _gmu, _gw, d, _gx = _gyjinvnonp_diag(
            zeros_e1, ones_e1, lam, e3[:, i]
        )
        F[:, i] = Ftemp
        glam[:, i] = d
        glam1[:, i] = (2.0 * Ftemp) * d

    inv_sqrt_pi = 1.0 / np.sqrt(np.pi)
    F2 = inv_sqrt_pi * np.sum(F * w, axis=1)  # mu
    dF2 = inv_sqrt_pi * np.sum(glam * w, axis=1)  # dmu/dlam
    F4 = inv_sqrt_pi * np.sum(F ** 2 * w, axis=1)
    F5 = np.sqrt(F4 - F2 ** 2)  # sig
    dF4 = inv_sqrt_pi * np.sum(glam1 * w, axis=1) - 2.0 * F2 * dF2
    gsigyj = (1.0 / F5) * (dF4 / 2.0)
    return F2, F5, dF2, gsigyj


def standyjinvnonpgiven(
    lam: object, mu: object, sig: object, x: object, *, xp=None
) -> NDArray[np.float64]:
    """Standardized inverse YJ transform given precomputed mean/std.

    GAUSS ``standyjinvnonpgiven`` (line 2893)::

        F1 = yjinvnonp(0, I, lam, x)
        F  = (F1 - mu) / sig

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s).
    mu : array-like, shape (K,)
        Mean of the inverse YJ variate (from :func:`meanyj`).
    sig : array-like, shape (K,)
        Standard deviation of the inverse YJ variate (from :func:`meanyj`).
    x : array-like, shape (K,) or (K, Q)
        Standardized normal abscissae.
    xp : backend, optional
        Array backend.

    Returns
    -------
    F : ndarray, shape (K,) or (K, Q)
        Standardized inverse-transformed values.
    """
    if xp is None:
        xp = array_namespace(x, lam, mu, sig)
    lam = _as_1d(lam)
    e1 = lam.shape[0]
    mu = _as_1d(mu)
    sig = _as_1d(sig)
    F1 = yjinvnonp(np.zeros(e1), np.eye(e1), lam, x)
    if F1.ndim == 2:
        return (F1 - mu[:, None]) / sig[:, None]
    return (F1 - mu) / sig


def gradstandyjinvnonpgiven(
    lam: object,
    mu: object,
    sig: object,
    gmuyj: object,
    gsigyj: object,
    x: object,
    *,
    xp=None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Gradient of the standardized inverse YJ transform given mean/std/grads.

    GAUSS ``gradstandyjinvnonpgiven`` (line 2964, 3 outputs)::

        F1        = yjinvnonp(0, I, lam, x[:, i])
        glam[:,i] = ((dF1/dlam - gmuyj) * sig - (F1 - mu) * gsigyj) / sig^2
        gx[:,i]   = (dF1/dx) / sig
        F         = (F1 - mu) / sig

    Parameters
    ----------
    lam : array-like, shape (K,)
        YJ transformation parameter(s).
    mu : array-like, shape (K,)
        Mean of the inverse YJ variate (from :func:`meanyj`).
    sig : array-like, shape (K,)
        Standard deviation of the inverse YJ variate (from :func:`meanyj`).
    gmuyj : array-like, shape (K,)
        Gradient ``dmu/dlam`` (from :func:`gradmeanyj`).
    gsigyj : array-like, shape (K,)
        Gradient ``dsig/dlam`` (from :func:`gradmeanyj`).
    x : array-like, shape (K,) or (K, Q)
        Standardized normal abscissae.
    xp : backend, optional
        Array backend.

    Returns
    -------
    F : ndarray, shape (K, Q)
        Standardized inverse-transformed values.
    glam : ndarray, shape (K, Q)
        Gradient of ``F`` w.r.t. lambda.
    gx : ndarray, shape (K, Q)
        Gradient of ``F`` w.r.t. ``x``.
    """
    if xp is None:
        xp = array_namespace(x, lam, mu, sig)
    lam = _as_1d(lam)
    mu = _as_1d(mu)
    sig = _as_1d(sig)
    gmuyj = _as_1d(gmuyj)
    gsigyj = _as_1d(gsigyj)

    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr[:, None]
    e1, e2 = x_arr.shape

    F = np.zeros((e1, e2), dtype=np.float64)
    glam = np.zeros((e1, e2), dtype=np.float64)
    gx = np.zeros((e1, e2), dtype=np.float64)
    zeros_e1 = np.zeros(e1, dtype=np.float64)
    ones_e1 = np.ones(e1, dtype=np.float64)
    sig2 = sig ** 2
    for i in range(e2):
        Ftemp, _gmu, _gw, g1, gxtemp = _gyjinvnonp_diag(
            zeros_e1, ones_e1, lam, x_arr[:, i]
        )
        F[:, i] = Ftemp
        glam[:, i] = ((g1 - gmuyj) * sig - (Ftemp - mu) * gsigyj) / sig2
        gx[:, i] = gxtemp / sig

    Ffin = (F - mu[:, None]) / sig[:, None]
    return Ffin, glam, gx
