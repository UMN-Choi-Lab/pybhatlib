"""Radial parameterization for positive-definite correlation matrices.

Implements the Van Oest (2019) radial parameterization used in BHATLIB.
Maps unconstrained parameters to (-1,1) via tanh, then builds the Cholesky
factor through accumulative sine products.  Each row of the upper-triangular
Cholesky factor L has unit norm, so L^T @ L is automatically a valid
correlation matrix.

The Cholesky structure is identical to the spherical parameterization
(Pinheiro & Bates), but the mapping to (-1,1) via tanh is numerically
simpler than cos/sin of angles in [0, pi].

References
----------
Van Oest, R. (2019). A new coefficient of interrater agreement: The
    challenge of highly unequal category proportions. Psychological
    Methods, 24(4), 439-451.

GAUSS reference: ``newcholparm`` (line 3752), ``gnewcholparm`` (line 4048).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def radial_to_corr(theta: NDArray, K: int) -> NDArray:
    """Convert unconstrained parameters to PD correlation matrix.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained parameters (mapped to (-1,1) via tanh).
    K : int
        Dimension of the correlation matrix.

    Returns
    -------
    corr : ndarray, shape (K, K)
        Positive-definite correlation matrix with unit diagonal.
    """
    theta = np.asarray(theta, dtype=np.float64)
    c = np.tanh(theta)
    s = np.sqrt(1.0 - c**2)  # = sech(theta)

    L = _build_radial_cholesky(c, s, K)
    corr = L.T @ L

    # Ensure exact unit diagonal
    np.fill_diagonal(corr, 1.0)
    return corr


def _build_radial_cholesky(c: NDArray, s: NDArray, K: int) -> NDArray:
    """Build upper-triangular Cholesky factor from radial parameters.

    Each row of L has unit norm, guaranteeing L^T @ L has unit diagonal.

    Parameters
    ----------
    c : ndarray, shape (K*(K-1)//2,)
        tanh(theta) values in (-1, 1).
    s : ndarray, shape (K*(K-1)//2,)
        sqrt(1-c^2) values in [0, 1].
    K : int
        Matrix dimension.

    Returns
    -------
    L : ndarray, shape (K, K)
        Upper-triangular Cholesky factor.
    """
    # Expand to K×K upper-triangular matrices with ones on diagonal
    Scos = np.eye(K, dtype=np.float64)
    Ssin = np.eye(K, dtype=np.float64)

    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            Scos[i, j] = c[idx]
            Ssin[i, j] = s[idx]
            idx += 1

    # Accumulative sine products (GAUSS newcholparm loop)
    for r in range(1, K):
        Scos[r:K, r:K] *= Ssin[r - 1, r:K]

    return Scos


def _param_index(p: int, q: int, K: int) -> int:
    """Map (p, q) with p < q to flat index in theta vector."""
    return p * K - p * (p + 1) // 2 + (q - p - 1)


def _corr_upper_index(a: int, b: int, K: int) -> int:
    """Map (a, b) with a <= b to flat index in upper-tri correlation vector."""
    return a * K - a * (a - 1) // 2 + (b - a)


def grad_radial_theta(theta: NDArray, K: int) -> NDArray:
    """Analytic Jacobian d(corr_upper)/d(theta) for radial parameterization.

    Derives the gradient by computing dL/dtheta for each parameter (which
    affects only one column of L), then chains through corr = L^T @ L.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained parameters.
    K : int
        Dimension of the correlation matrix.

    Returns
    -------
    jac : ndarray, shape (K*(K-1)//2, K*(K+1)//2)
        Jacobian: jac[p_idx, c_idx] = d(corr_upper[c_idx]) / d(theta[p_idx]).
    """
    n_theta = K * (K - 1) // 2
    n_corr_upper = K * (K + 1) // 2
    theta = np.asarray(theta, dtype=np.float64)

    c = np.tanh(theta)
    s = np.sqrt(1.0 - c**2)

    # Build Cholesky factor L (upper-triangular)
    L = _build_radial_cholesky(c, s, K)

    # Store c, s in matrix form for lookup
    c_mat = np.zeros((K, K), dtype=np.float64)
    s_mat = np.ones((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            c_mat[i, j] = c[idx]
            s_mat[i, j] = s[idx]
            idx += 1

    jac = np.zeros((n_theta, n_corr_upper), dtype=np.float64)

    for p in range(K):
        for q in range(p + 1, K):
            theta_idx = _param_index(p, q, K)

            # --- Compute dL[:,q] for parameter theta_{pq} ---
            dL_q = np.zeros(K, dtype=np.float64)

            # Direct term: dL[p,q]/dtheta = s_{pq}^2 * prod_{m<p} s_{mq}
            prod_s = 1.0
            for m in range(p):
                prod_s *= s_mat[m, q]
            dL_q[p] = s_mat[p, q] ** 2 * prod_s

            # Sine product terms: dL[i,q]/dtheta = -c_{pq} * L[i,q]
            for i in range(p + 1, q + 1):
                dL_q[i] = -c_mat[p, q] * L[i, q]

            # --- Chain through corr = L^T @ L ---
            # Only entries in row q or column q of corr are affected.

            # corr[a,q] for a < q:  dcorr = sum_k L[k,a] * dL_q[k]
            for a in range(q):
                val = 0.0
                for k in range(a + 1):  # L[k,a] is non-zero only for k <= a
                    val += L[k, a] * dL_q[k]
                jac[theta_idx, _corr_upper_index(a, q, K)] = val

            # corr[q,q] (diagonal): dcorr = 2 * sum_k L[k,q] * dL_q[k]
            jac[theta_idx, _corr_upper_index(q, q, K)] = (
                2.0 * np.dot(L[:q + 1, q], dL_q[:q + 1])
            )

            # corr[q,b] for b > q:  dcorr = sum_k dL_q[k] * L[k,b]
            for b in range(q + 1, K):
                val = 0.0
                for k in range(q + 1):  # dL_q[k] is non-zero for k=p..q
                    val += dL_q[k] * L[k, b]
                jac[theta_idx, _corr_upper_index(q, b, K)] = val

    return jac


# ---------------------------------------------------------------------------
# Scaled radial (newcholparm) parameterization
#
# GAUSS reference: ``newcholparmscaled`` (matgradient.src line 1319),
# ``revnewcholparmscaled`` (line 1568), ``gnewcholparmcorscaled`` (line 1665),
# with helpers ``newcholparm`` (line 1286), ``revnewcholparm`` (line 1535),
# ``revtrmin1to1`` (vecup.src line 3113).
#
# The scaled radial map differs from ``radial_to_corr`` above in two ways:
#
# 1. The off-diagonal Cholesky element uses the GAUSS ``newcholparm`` mapping
#    ``c = (exp(theta/scal) - 1) / (exp(theta/scal) + 1) = tanh(theta/(2*scal))``
#    rather than ``tanh(theta)`` used by :func:`radial_to_corr`.  The two agree
#    up to a rescaling of ``theta`` (``scal = 1/2`` recovers ``radial_to_corr``),
#    but ``newcholparmscaled`` is a verbatim port of the GAUSS convention so its
#    Cholesky factor plugs directly into GAUSS estimates.
# 2. A positive scalar ``scal`` widens/narrows the effective step of ``theta``.
#
# These functions RETURN THE CHOLESKY FACTOR ``L`` (upper triangular, so that
# ``L.T @ L`` is a unit-diagonal correlation matrix), matching GAUSS.
# ---------------------------------------------------------------------------


def _newchol_cs(theta: NDArray, scal: float) -> tuple[NDArray, NDArray]:
    """Return (c, s) for the GAUSS newcholparm/newcholparmscaled mapping.

    ``c = (exp(theta/scal) - 1) / (exp(theta/scal) + 1)`` and
    ``s = sqrt(1 - c**2)``.
    """
    temp = np.asarray(theta, dtype=np.float64) / scal
    # Algebraically identical to (exp(temp) - 1) / (exp(temp) + 1), but
    # stable for the large unconstrained trial steps an optimizer may take.
    # Keep the result inside the open interval so downstream Cholesky calls do
    # not receive an exactly singular correlation matrix after saturation.
    limit = 1.0 - 1e-12
    c = np.clip(np.tanh(temp / 2.0), -limit, limit)
    s = np.sqrt(1.0 - c**2)
    return c, s


def newcholparmscaled(theta: NDArray, scal: float = 1.0, *, xp=None) -> NDArray:
    """Cholesky factor of a unit-diagonal correlation matrix (scaled radial).

    Verbatim port of GAUSS ``newcholparmscaled`` (matgradient.src line 1319),
    which is ``newcholparm(theta / scal)``.  Each off-diagonal Cholesky element
    is ``c = (exp(theta/scal) - 1) / (exp(theta/scal) + 1)`` and the rows of the
    returned upper-triangular ``L`` have unit norm, so ``L.T @ L`` is a
    positive-definite correlation matrix with unit diagonal for any real
    ``theta``.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained radial parameters (span the real line), in row-based
        upper-triangular (vech, off-diagonal) order ``{S12, S13, ..., S23, ...}``.
    scal : float, default 1.0
        Positive scale applied as ``theta / scal`` before the radial map.
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    L : ndarray, shape (K, K)
        Upper-triangular Cholesky factor with unit-norm rows.  ``L.T @ L`` is a
        unit-diagonal correlation matrix.
    """
    if scal <= 0.0:
        raise ValueError(f"scal must be positive, got {scal}")
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.shape[0]
    # K*(K-1)/2 = n  ->  K = (1 + sqrt(1 + 8n)) / 2
    K = int((1.0 + np.sqrt(1.0 + 8.0 * n)) / 2.0)
    if K * (K - 1) // 2 != n:
        raise ValueError(f"theta length {n} is not K*(K-1)/2 for integer K")
    c, s = _newchol_cs(theta, scal)
    return _build_radial_cholesky(c, s, K)


def _revtrmin1to1(x: NDArray) -> NDArray:
    """Inverse of the (-1, 1) radial map: ``ln((1 + x) / (1 - x))``.

    GAUSS ``revtrmin1to1`` (vecup.src line 3113).  This inverts
    ``c = (exp(u) - 1) / (exp(u) + 1)`` giving ``u = ln((1 + c) / (1 - c))``.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.log((1.0 + x) / (1.0 - x))


def revnewcholparmscaled(L: NDArray, scal: float = 1.0, *, xp=None) -> NDArray:
    """Recover scaled radial parameters from a correlation Cholesky factor.

    Verbatim port of GAUSS ``revnewcholparmscaled`` (matgradient.src line 1568),
    which returns ``scal * revnewcholparm(L)``.  Exact inverse of
    :func:`newcholparmscaled`: given the upper-triangular Cholesky factor ``L``
    of a positive-definite correlation matrix (``L[0, 0] == 1``), recover the
    unconstrained parameters ``theta`` such that
    ``newcholparmscaled(theta, scal) == L``.

    Parameters
    ----------
    L : ndarray, shape (K, K)
        Upper-triangular Cholesky factor of a correlation matrix
        (``L.T @ L`` has unit diagonal, ``L[0, 0] == 1``).
    scal : float, default 1.0
        Positive scale (must match the value used in :func:`newcholparmscaled`).
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    theta : ndarray, shape (K*(K-1)//2,)
        Recovered radial parameters in row-based off-diagonal order.
    """
    if scal <= 0.0:
        raise ValueError(f"scal must be positive, got {scal}")
    L = np.asarray(L, dtype=np.float64)
    K = L.shape[0]
    n = K * (K - 1) // 2
    sstar = np.zeros(n, dtype=np.float64)
    if n == 0:
        return sstar

    # First row off-diagonals map directly.
    sstar[0:K - 1] = _revtrmin1to1(L[0, 1:K])

    # Block offsets: ks1 = [0, K-1, K-2, ..., 1]; ks2 = cumsum(ks1).
    ks1 = np.concatenate(([0], np.arange(K - 1, 0, -1)))
    ks2 = np.cumsum(ks1)

    Ltemp = L.copy()
    Ltemp[1:, :] = 0.0  # zero rows 1..K-1 (row 0 retained; later rows filled)

    temp = np.ones(K - 1, dtype=np.float64)
    for r in range(1, K - 1):  # python row index r (GAUSS row i = r + 1)
        temp = temp[1:]  # drop first element
        temp = temp * np.sqrt(1.0 - Ltemp[r - 1, r + 1:K] ** 2)
        Ltemp[r, r + 1:K] = L[r, r + 1:K] / temp
        sstar[ks2[r]:ks2[r + 1]] = _revtrmin1to1(Ltemp[r, r + 1:K])

    return scal * sstar


def gnewcholparmcorscaled(
    capomega: NDArray, scal: float = 1.0, *, xp=None
) -> tuple[NDArray, NDArray]:
    """Gradient of a correlation matrix w.r.t. scaled radial Cholesky params.

    Port of GAUSS ``gnewcholparmcorscaled`` (matgradient.src line 1665).  Given a
    correlation matrix ``capomega`` whose Cholesky factor is parameterized by
    :func:`newcholparmscaled`, return the gradients of the off-diagonal
    correlation elements with respect to (i) the unconstrained radial parameters
    ``theta`` and (ii) the scale ``scal``.

    The correlation off-diagonal elements are taken in row-based upper-triangular
    (vech, off-diagonal) order, identical to the ordering of ``theta``.

    Parameters
    ----------
    capomega : ndarray, shape (K, K)
        Positive-definite correlation matrix with unit diagonal (K > 2).
    scal : float, default 1.0
        Positive scale used in the parameterization.
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    grad1 : ndarray, shape (K*(K-1)//2, K*(K-1)//2)
        ``grad1[p, j] = d(corr_offdiag[j]) / d(theta[p])``.
    grad2 : ndarray, shape (K*(K-1)//2,)
        ``grad2[j] = d(corr_offdiag[j]) / d(scal)``.

    Notes
    -----
    The GAUSS routine composes ``gcholeskycov`` with the (scaled)
    ``gnewcholparm`` gradient.  This port instead chains the correlation
    gradient directly through ``corr = L.T @ L`` in the scaled radial mapping;
    the two are mathematically identical (verified against central differences).
    The ``scal`` gradient uses the GAUSS identity
    ``grad2[j] = -sum_p grad1[p, j] * (theta[p] / scal)``.
    """
    if scal <= 0.0:
        raise ValueError(f"scal must be positive, got {scal}")
    capomega = np.asarray(capomega, dtype=np.float64)
    K = capomega.shape[0]
    n_theta = K * (K - 1) // 2

    # Upper-triangular Cholesky (U.T @ U = capomega) and its radial params.
    L_lower = np.linalg.cholesky(capomega)
    U = L_lower.T
    theta = revnewcholparmscaled(U, scal)
    c, s = _newchol_cs(theta, scal)

    # Rebuild L from (c, s) to stay consistent with the parameterization.
    L = _build_radial_cholesky(c, s, K)

    # Store c, s in matrix form for lookup.
    c_mat = np.zeros((K, K), dtype=np.float64)
    s_mat = np.ones((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            c_mat[i, j] = c[idx]
            s_mat[i, j] = s[idx]
            idx += 1

    grad1 = np.zeros((n_theta, n_theta), dtype=np.float64)
    inv_2scal = 1.0 / (2.0 * scal)

    for p in range(K):
        for q in range(p + 1, K):
            theta_idx = _param_index(p, q, K)

            # d(L[:, q]) / d(theta_{pq}) for the scaled newcholparm mapping.
            # Direct term: dc/dtheta = s^2 / (2 scal); sine chain contributes
            # -c/(2 scal) * L[i, q] (the 1/s^2 cancels the s^2 in dc/dtheta).
            dL_q = np.zeros(K, dtype=np.float64)
            prod_s = 1.0
            for m in range(p):
                prod_s *= s_mat[m, q]
            dL_q[p] = s_mat[p, q] ** 2 * prod_s * inv_2scal
            for i in range(p + 1, q + 1):
                dL_q[i] = -c_mat[p, q] * L[i, q] * inv_2scal

            # Chain through corr = L.T @ L; only row/col q of corr are affected.
            for a in range(q):  # corr[a, q], a < q
                val = 0.0
                for k in range(a + 1):
                    val += L[k, a] * dL_q[k]
                grad1[theta_idx, _param_index(a, q, K)] = val
            for b in range(q + 1, K):  # corr[q, b], b > q
                val = 0.0
                for k in range(q + 1):
                    val += dL_q[k] * L[k, b]
                grad1[theta_idx, _param_index(q, b, K)] = val

    # d(corr)/d(scal): grad2[j] = -sum_p grad1[p, j] * (theta[p] / scal)
    grad2 = -(theta / scal) @ grad1

    return grad1, grad2
