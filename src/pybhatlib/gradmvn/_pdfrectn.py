"""Rectangle MVN probability with extreme-category routing and its gradient.

Port of GAUSS ``pdfrectn`` / ``gradpdfrectn`` (``gradients mvn.src`` lines 136,
254).  These generalise the plain rectangle MVN CDF (:func:`pdfrectn` reduces to
:func:`pybhatlib.gradmvn.mvncd_rect` when there are no equality dimensions) by
adding three per-dimension routing indicators:

Index conventions (all length ``K``, entries in ``{0, 1}``)
----------------------------------------------------------
``indxeq``
    ``1`` marks an **equality / "given"** dimension: the density is evaluated at
    the abscissa ``xg[i]`` (the variable is observed exactly, contributing a
    density factor rather than an integral).  ``0`` marks a dimension that is
    integrated (one-sided or rectangular).
``indxone``
    Among integrated dimensions (``indxeq[i] == 0``), ``1`` marks a **one-sided
    / orthant** integral (a single open bound at ``xg[i]``); ``0`` marks a
    **rectangular** integral over ``[xlow[i], xup[i]]``.  For equality
    dimensions ``indxone`` is immaterial.
``indxcomp``
    For a one-sided dimension, ``0`` means truncation from above,
    ``(-inf, xg[i]]``; ``1`` means truncation from below (complement),
    ``[xg[i], +inf)``.  Must be ``0`` for rectangular and equality dimensions.

So for dimension ``i`` the integration region is

* ``indxeq[i] == 1``                       -> equality at ``xg[i]``  (density)
* ``indxone[i] == 1, indxcomp[i] == 0``    -> ``(-inf, xg[i]]``
* ``indxone[i] == 1, indxcomp[i] == 1``    -> ``[xg[i], +inf)``
* ``indxone[i] == 0`` (and ``indxeq==0``)  -> ``[xlow[i], xup[i]]``

``pdfrectn`` returns ``P``, the joint density-times-probability of that region,
and ``s`` (an SSJ seed, passed through unchanged for the analytic methods used
here).  ``gradpdfrectn`` returns the analytic gradient of ``P`` with respect to
``mu``, ``cova`` (covariance elements, upper-triangular ``vech``), ``xg``,
``xlow`` and ``xup``.

Only the covariance parameterisation (GAUSS ``_covarr = 1``) is implemented; the
covariance gradient ``gcov`` is the length-``K(K+1)/2`` row-based upper-triangular
``vech`` of ``dP/dcova`` (off-diagonal entries account for both ``sigma_ij`` and
``sigma_ji``, matching a symmetric finite-difference perturbation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.gradmvn._mvncd_grad_analytic import mvncd_grad_me_analytic

_LOG_2PI = float(np.log(2.0 * np.pi))
_NEG_INF = float("-inf")
_POS_INF = float("inf")


# ---------------------------------------------------------------------------
# vech helpers (row-based upper triangle, incl. diagonal)
# ---------------------------------------------------------------------------
def _upper_pairs(k: int) -> list[tuple[int, int]]:
    """Row-based upper-triangular index pairs ``(i, j)`` with ``i <= j``."""
    return [(i, j) for i in range(k) for j in range(i, k)]


def _full_to_vech(gfull: NDArray, k: int) -> NDArray:
    """Convert a symmetric per-entry gradient matrix to ``vech`` form.

    ``gfull`` satisfies ``dP = <gfull, dSigma>_F`` for symmetric ``dSigma``.
    The returned vector holds ``dP/dsigma_ij`` for the free upper-triangular
    parameters: diagonal entries are ``gfull_ii`` and off-diagonal entries are
    ``2 * gfull_ij`` (both matrix cells move together).
    """
    out = np.empty(k * (k + 1) // 2, dtype=np.float64)
    for c, (i, j) in enumerate(_upper_pairs(k)):
        out[c] = gfull[i, i] if i == j else 2.0 * gfull[i, j]
    return out


def _vech_to_full(gvech: NDArray, k: int) -> NDArray:
    """Inverse of :func:`_full_to_vech` for an analytic ``vech`` gradient.

    ``gvech`` is a ``grad_sigma`` from :func:`mvncd_grad_me_analytic` whose
    off-diagonal entries already account for both ``sigma_ij`` and
    ``sigma_ji``.  The reconstructed symmetric matrix ``gfull`` therefore uses
    ``gfull_ii = gvech_ii`` and ``gfull_ij = gvech_ij / 2``.
    """
    gfull = np.zeros((k, k), dtype=np.float64)
    for c, (i, j) in enumerate(_upper_pairs(k)):
        if i == j:
            gfull[i, i] = gvech[c]
        else:
            gfull[i, j] = 0.5 * gvech[c]
            gfull[j, i] = 0.5 * gvech[c]
    return gfull


# ---------------------------------------------------------------------------
# bound construction (cdorrectmvn region -> lower/upper for mvncd_rect)
# ---------------------------------------------------------------------------
def _build_bounds(
    mu: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    indxone: NDArray,
    indxcomp: NDArray,
) -> tuple[NDArray, NDArray]:
    """Return centred ``(lower - mu, upper - mu)`` for the rectangle region.

    One-sided dimensions map to a half-open interval with ``+/- inf`` on the
    open side; :func:`mvncd_rect` handles the infinite bounds by
    inclusion-exclusion, so no correlation sign-flip (GAUSS ``indxcov``) is
    needed for the value.
    """
    k = len(mu)
    lower = np.empty(k, dtype=np.float64)
    upper = np.empty(k, dtype=np.float64)
    for i in range(k):
        if indxone[i]:
            if indxcomp[i]:
                lower[i] = xg[i]
                upper[i] = np.inf
            else:
                lower[i] = -np.inf
                upper[i] = xg[i]
        else:
            lower[i] = xlow[i]
            upper[i] = xup[i]
    return lower - mu, upper - mu


# ---------------------------------------------------------------------------
# rectangle/orthant probability (cdorrectmvn) — value and gradient
# ---------------------------------------------------------------------------
def _rect_prob(
    mu: NDArray,
    cova: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    indxone: NDArray,
    indxcomp: NDArray,
) -> float:
    """Value of the combined orthant/rectangle integral (GAUSS ``cdorrectmvn``).

    Reuses :func:`pybhatlib.gradmvn.mvncd_rect` for the plain-rectangle machinery.
    """
    lower_c, upper_c = _build_bounds(mu, xg, xlow, xup, indxone, indxcomp)
    return float(mvncd_rect(lower_c, upper_c, cova))


def _rect_prob_analytic(
    mu: NDArray,
    cova: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    indxone: NDArray,
    indxcomp: NDArray,
) -> float:
    """Value of the combined orthant/rectangle integral via the analytic BVN.

    Value-only twin of :func:`_rect_prob_grad`: mirrors the same
    ``2^K``-corner inclusion-exclusion but computes only the probability,
    routing each finite corner CDF through :func:`mvncd_grad_me_analytic`
    (the *same* analytic method the gradient path already uses) instead of the
    scipy frozen-distribution value path in :func:`mvncd_rect`.  For ``K <= 2``
    the analytic BVN/univariate CDF is exact, so the returned value matches the
    scipy :func:`_rect_prob` to machine precision while skipping the per-call
    frozen-dist construction + ``eigh`` + ``apply_along_axis``.

    Only meaningful (and only used) for ``K <= 2`` corners; for ``K >= 3`` the
    caller keeps :func:`_rect_prob` (OVUS) to avoid substituting the ME
    approximation on the value path.
    """
    k = len(mu)
    cova = np.asarray(cova, dtype=np.float64)
    lower_c, upper_c = _build_bounds(mu, xg, xlow, xup, indxone, indxcomp)

    P = 0.0
    for mask in range(1 << k):
        sign = 1
        skip = False
        val = np.empty(k, dtype=np.float64)
        for i in range(k):
            bit = (mask >> i) & 1
            if bit:
                sign = -sign
                val[i] = lower_c[i]
            else:
                val[i] = upper_c[i]
            if val[i] == _NEG_INF:
                skip = True
                break
        if skip:
            continue

        # marginalise +inf dims (they drop out of the CDF)
        keep = [i for i in range(k) if val[i] != _POS_INF]
        m = len(keep)
        if m == 0:
            P += sign * 1.0
            continue

        sub_a = val[keep]
        sub_sigma = cova[np.ix_(keep, keep)]
        p_v, _ga, _gs = mvncd_grad_me_analytic(sub_a, sub_sigma)
        P += sign * float(p_v)

    return P


def _rect_prob_grad(
    mu: NDArray,
    cova: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    indxone: NDArray,
    indxcomp: NDArray,
) -> tuple[float, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """Analytic gradient of the combined orthant/rectangle integral.

    Uses the inclusion-exclusion identity
    ``P = sum_s (-1)^{|s|} Phi(c_s; Sigma)`` over the ``2^K`` corners and chains
    the per-corner analytic MVN-CDF gradient (:func:`mvncd_grad_me_analytic`)
    back to the physical bounds/mean/covariance.

    Returns
    -------
    P : float
    gmu : ndarray, shape (K,)          dP/dmu
    gxg : ndarray, shape (K,)          dP/dxg   (nonzero only on one-sided dims)
    gxlow : ndarray, shape (K,)        dP/dxlow (nonzero only on rectangle dims)
    gxup : ndarray, shape (K,)         dP/dxup  (nonzero only on rectangle dims)
    gfull : ndarray, shape (K, K)      symmetric, dP = <gfull, dSigma>_F
    """
    k = len(mu)
    cova = np.asarray(cova, dtype=np.float64)

    # centred upper/lower bound per dim; +/-inf on open sides
    lower_c, upper_c = _build_bounds(mu, xg, xlow, xup, indxone, indxcomp)

    P = 0.0
    gmu = np.zeros(k, dtype=np.float64)
    gxg = np.zeros(k, dtype=np.float64)
    gxlow = np.zeros(k, dtype=np.float64)
    gxup = np.zeros(k, dtype=np.float64)
    gfull = np.zeros((k, k), dtype=np.float64)

    for mask in range(1 << k):
        sign = 1
        skip = False
        val = np.empty(k, dtype=np.float64)
        use_lower = np.empty(k, dtype=bool)
        for i in range(k):
            bit = (mask >> i) & 1
            use_lower[i] = bool(bit)
            if bit:
                sign = -sign
                val[i] = lower_c[i]
            else:
                val[i] = upper_c[i]
            if val[i] == _NEG_INF:
                skip = True
                break
        if skip:
            continue

        # marginalise +inf dims (they drop out of the CDF)
        keep = [i for i in range(k) if val[i] != _POS_INF]
        m = len(keep)
        if m == 0:
            P += sign * 1.0
            continue

        sub_a = val[keep]
        sub_sigma = cova[np.ix_(keep, keep)]
        p_v, ga_v, gsig_v = mvncd_grad_me_analytic(sub_a, sub_sigma)
        P += sign * float(p_v)

        # threshold / mean gradients
        for jj, i in enumerate(keep):
            contrib = sign * float(ga_v[jj])
            gmu[i] -= contrib
            if use_lower[i]:
                if indxone[i]:  # one-sided complement lower bound = xg
                    gxg[i] += contrib
                else:  # rectangle lower bound = xlow
                    gxlow[i] += contrib
            else:
                if indxone[i]:  # one-sided (comp=0) upper bound = xg
                    gxg[i] += contrib
                else:  # rectangle upper bound = xup
                    gxup[i] += contrib

        # covariance gradient: scatter reduced vech into full matrix
        gsub_full = _vech_to_full(np.asarray(gsig_v, dtype=np.float64), m)
        for aa in range(m):
            for bb in range(m):
                gfull[keep[aa], keep[bb]] += sign * gsub_full[aa, bb]

    return P, gmu, gxg, gxlow, gxup, gfull


# ---------------------------------------------------------------------------
# MVN density and its gradient (GAUSS pdfmvnanl / gradpdfmvnanl, covarr=1)
# ---------------------------------------------------------------------------
def _mvn_density_grad(
    x: NDArray, sigma: NDArray
) -> tuple[float, NDArray, NDArray]:
    """Zero-mean MVN density at ``x`` and its analytic gradient.

    Returns ``(D, gx, gfull)`` with ``gx = dD/dx`` and ``gfull`` symmetric such
    that ``dD = <gfull, dSigma>_F``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64)
    k = len(x)
    inv_s = np.linalg.inv(sigma)
    sign_det, logdet = np.linalg.slogdet(sigma)
    a = inv_s @ x
    quad = float(x @ a)
    logd = -0.5 * (k * _LOG_2PI + logdet) - 0.5 * quad
    D = float(np.exp(logd))
    gx = -D * a
    gfull = 0.5 * D * (np.outer(a, a) - inv_s)
    return D, gx, gfull


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def _prep(*arrs, xp):
    out = []
    for a in arrs:
        if xp is not None:
            a = xp.to_numpy(a)
        out.append(np.asarray(a, dtype=np.float64).ravel())
    return out


def pdfrectn(
    mu: NDArray,
    cova: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    s: float,
    indxone: NDArray,
    indxcomp: NDArray,
    indxeq: NDArray,
    *,
    xp=None,
) -> tuple[float, float]:
    """Rectangle MVN probability with extreme-category / equality routing.

    Port of GAUSS ``pdfrectn``.  See the module docstring for the meaning of
    ``indxone`` / ``indxcomp`` / ``indxeq``.

    Parameters
    ----------
    mu : ndarray, shape (K,)
        Mean vector.
    cova : ndarray, shape (K, K)
        Covariance matrix.
    xg : ndarray, shape (K,)
        Abscissae for equality dimensions and one-sided open bounds.
    xlow, xup : ndarray, shape (K,)
        Lower / upper limits for rectangular dimensions.
    s : float
        SSJ seed (returned unchanged for the analytic methods used here).
    indxone, indxcomp, indxeq : ndarray, shape (K,)
        Routing indicators (0/1); see module docstring.
    xp : backend, optional
        Array backend used only to coerce inputs; computation is NumPy.

    Returns
    -------
    P : float
        Density-times-probability of the routed region.
    s : float
        The seed, unchanged.
    """
    if xp is None:
        xp = array_namespace(mu, cova)
    mu, xg, xlow, xup = _prep(mu, xg, xlow, xup, xp=xp)
    cova = np.asarray(xp.to_numpy(cova) if xp is not None else cova, dtype=np.float64)
    indxone, indxcomp, indxeq = (
        np.asarray(v).ravel().astype(int) for v in (indxone, indxcomp, indxeq)
    )
    k = len(mu)

    if int(indxeq.sum()) == 0:
        # For K <= 2 the per-corner CDF is an exact analytic BVN/univariate; use
        # the same analytic method the gradient path uses (skipping scipy's
        # frozen-dist value path). For K >= 3 keep OVUS (mvncd_rect) so the value
        # path is not silently swapped to the coarser ME approximation.
        if k <= 2:
            P = _rect_prob_analytic(mu, cova, xg, xlow, xup, indxone, indxcomp)
        else:
            P = _rect_prob(mu, cova, xg, xlow, xup, indxone, indxcomp)
        return P, float(s)

    # --- equality dimensions present: condition on the "given" block ---
    indx1 = ((-1.0) ** indxcomp)
    indxcov = ((-1.0) ** (indxcomp[:, None] + indxcomp[None, :]))
    newxg = indx1 * (xg - mu)
    newx1 = xlow - mu
    newx2 = xup - mu
    cov = indxcov * cova

    indeq = np.nonzero(indxeq == 1)[0]
    indneq = np.nonzero(indxeq == 0)[0]
    newxg1 = newxg[indeq]
    newxg2 = newxg[indneq]
    newx12 = newx1[indneq]
    newx22 = newx2[indneq]
    X11 = cov[np.ix_(indeq, indeq)]
    X12 = cov[np.ix_(indneq, indeq)]
    X22 = cov[np.ix_(indneq, indneq)]
    invX11 = np.linalg.inv(X11)
    mu2condcov = X12 @ invX11 @ newxg1
    X22condcov = X22 - X12 @ invX11 @ X12.T
    newindxone = indxone[indneq]
    newindxcomp = np.zeros(len(indneq), dtype=int)

    D, _gx, _gf = _mvn_density_grad(newxg1, X11)
    Q = _rect_prob(
        mu2condcov, X22condcov, newxg2, newx12, newx22, newindxone, newindxcomp
    )
    return float(D * Q), float(s)


def gradpdfrectn(
    mu: NDArray,
    cova: NDArray,
    xg: NDArray,
    xlow: NDArray,
    xup: NDArray,
    s: float,
    indxone: NDArray,
    indxcomp: NDArray,
    indxeq: NDArray,
    *,
    xp=None,
) -> tuple[float, NDArray, NDArray, NDArray, NDArray, NDArray, float]:
    """Analytic gradient of :func:`pdfrectn` (GAUSS ``gradpdfrectn``, ``_covarr=1``).

    Returns
    -------
    P : float
        The :func:`pdfrectn` value.
    gmu : ndarray, shape (K,)
        ``dP/dmu``.
    gcov : ndarray, shape (K(K+1)/2,)
        ``dP/dcova`` in row-based upper-triangular ``vech`` order (off-diagonal
        entries account for both ``sigma_ij`` and ``sigma_ji``).
    gxg : ndarray, shape (K,)
        ``dP/dxg`` (nonzero on equality and one-sided dimensions).
    gx1 : ndarray, shape (K,)
        ``dP/dxlow`` (nonzero on rectangular dimensions).
    gx2 : ndarray, shape (K,)
        ``dP/dxup`` (nonzero on rectangular dimensions).
    s : float
        The seed, unchanged.
    """
    if xp is None:
        xp = array_namespace(mu, cova)
    mu, xg, xlow, xup = _prep(mu, xg, xlow, xup, xp=xp)
    cova = np.asarray(xp.to_numpy(cova) if xp is not None else cova, dtype=np.float64)
    indxone, indxcomp, indxeq = (
        np.asarray(v).ravel().astype(int) for v in (indxone, indxcomp, indxeq)
    )
    k = len(mu)
    upper_pairs = _upper_pairs(k)

    if int(indxeq.sum()) == 0:
        P, gmu, gxg, gxlow, gxup, gfull = _rect_prob_grad(
            mu, cova, xg, xlow, xup, indxone, indxcomp
        )
        gcov = _full_to_vech(gfull, k)
        return P, gmu, gcov, gxg, gxlow, gxup, float(s)

    # --- equality dimensions present: forward-mode analytic chain ---
    indx1 = ((-1.0) ** indxcomp)
    indxcov = ((-1.0) ** (indxcomp[:, None] + indxcomp[None, :]))
    newxg = indx1 * (xg - mu)
    newx1 = xlow - mu
    newx2 = xup - mu
    cov = indxcov * cova

    indeq = np.nonzero(indxeq == 1)[0]
    indneq = np.nonzero(indxeq == 0)[0]
    newxg1 = newxg[indeq]
    newxg2 = newxg[indneq]
    newx12 = newx1[indneq]
    newx22 = newx2[indneq]
    X11 = cov[np.ix_(indeq, indeq)]
    X12 = cov[np.ix_(indneq, indeq)]
    X22 = cov[np.ix_(indneq, indneq)]
    invX11 = np.linalg.inv(X11)
    mu2condcov = X12 @ invX11 @ newxg1
    X22condcov = X22 - X12 @ invX11 @ X12.T
    newindxone = indxone[indneq]
    newindxcomp = np.zeros(len(indneq), dtype=int)

    # direct partials (evaluated once)
    D, gD_x, GD_full = _mvn_density_grad(newxg1, X11)
    Q, gQ_mu2, gQ_xg2, gQ_x12, gQ_x22, GQ_full = _rect_prob_grad(
        mu2condcov, X22condcov, newxg2, newx12, newx22, newindxone, newindxcomp
    )
    P = D * Q

    def directional(dmu, dcova, dxg, dxlow, dxup) -> float:
        dcov = indxcov * dcova
        dnewxg = indx1 * (dxg - dmu)
        dnewx1 = dxlow - dmu
        dnewx2 = dxup - dmu
        dnewxg1 = dnewxg[indeq]
        dnewxg2 = dnewxg[indneq]
        dnewx12 = dnewx1[indneq]
        dnewx22 = dnewx2[indneq]
        dX11 = dcov[np.ix_(indeq, indeq)]
        dX12 = dcov[np.ix_(indneq, indeq)]
        dX22 = dcov[np.ix_(indneq, indneq)]
        dinvX11 = -invX11 @ dX11 @ invX11
        dmu2 = (
            dX12 @ invX11 @ newxg1
            + X12 @ dinvX11 @ newxg1
            + X12 @ invX11 @ dnewxg1
        )
        dX22cond = dX22 - (
            dX12 @ invX11 @ X12.T
            + X12 @ dinvX11 @ X12.T
            + X12 @ invX11 @ dX12.T
        )
        dD = float(gD_x @ dnewxg1) + float(np.sum(GD_full * dX11))
        dQ = (
            float(gQ_mu2 @ dmu2)
            + float(np.sum(GQ_full * dX22cond))
            + float(gQ_xg2 @ dnewxg2)
            + float(gQ_x12 @ dnewx12)
            + float(gQ_x22 @ dnewx22)
        )
        return dD * Q + D * dQ

    z_mu = np.zeros(k)
    z_cov = np.zeros((k, k))
    gmu = np.zeros(k, dtype=np.float64)
    gxg = np.zeros(k, dtype=np.float64)
    gxlow = np.zeros(k, dtype=np.float64)
    gxup = np.zeros(k, dtype=np.float64)
    for i in range(k):
        e = np.zeros(k)
        e[i] = 1.0
        gmu[i] = directional(e, z_cov, z_mu, z_mu, z_mu)
        gxg[i] = directional(z_mu, z_cov, e, z_mu, z_mu)
        gxlow[i] = directional(z_mu, z_cov, z_mu, e, z_mu)
        gxup[i] = directional(z_mu, z_cov, z_mu, z_mu, e)

    gcov = np.zeros(len(upper_pairs), dtype=np.float64)
    for c, (p, q) in enumerate(upper_pairs):
        dcova = np.zeros((k, k))
        dcova[p, q] = 1.0
        if p != q:
            dcova[q, p] = 1.0
        gcov[c] = directional(z_mu, dcova, z_mu, z_mu, z_mu)

    return float(P), gmu, gcov, gxg, gxlow, gxup, float(s)
