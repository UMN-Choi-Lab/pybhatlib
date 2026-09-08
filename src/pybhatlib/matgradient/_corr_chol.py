"""Cholesky-of-correlation gradients from BHATLIB's Matgradient.src.

Ports two GAUSS procedures used when a multivariate-normal conditional mean is
formed from a Cholesky factor of a correlation (or covariance) matrix:

- ``gcholeskycor`` (matgradient.src line 875) -- gradient of the correlation
  matrix elements with respect to the (free, off-diagonal) elements of its
  Cholesky factor.
- ``ggradchol`` (matgradient.src line 978) -- gradient of ``B = A @ L.T @ e``
  with respect to the unique elements of the Cholesky factor ``L``.

All vectorization is row-based upper-triangular (vech): correlation off-diagonal
elements and Cholesky elements are ordered ``{(0,1), (0,2), ..., (1,2), ...}``.

The small helpers ``matndupdiagonefull`` and ``vecndup`` already live in
``pybhatlib.vecup`` and are imported here.

GAUSS reference: ``gcholeskycor``, ``ggradchol`` (Matgradient.src).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace

# GAUSS ``matndupdiagonefull`` and ``vecndup`` already exist in vecup (the
# former under the Python name ``matdupdiagonefull``); import and re-expose them.
from pybhatlib.vecup._vec_ops import matdupdiagonefull as matndupdiagonefull
from pybhatlib.vecup._vec_ops import vecndup

__all__ = ["gcholeskycor", "ggradchol", "matndupdiagonefull", "vecndup"]


def _upper_chol(capomega: NDArray) -> NDArray:
    """Upper-triangular Cholesky ``U`` with ``U.T @ U == capomega`` (GAUSS chol)."""
    return np.linalg.cholesky(np.asarray(capomega, dtype=np.float64)).T


def _offdiag_index(p: int, q: int, K: int) -> int:
    """Flat index of upper off-diagonal element (p, q), p < q, row-based order."""
    return p * K - p * (p + 1) // 2 + (q - p - 1)


def gcholeskycor(capomega: NDArray, *, xp=None) -> NDArray:
    """Gradient of a correlation matrix w.r.t. its Cholesky factor elements.

    Port of GAUSS ``gcholeskycor`` (matgradient.src line 875).  For a
    correlation matrix ``capomega`` with upper-triangular Cholesky factor ``U``
    (``U.T @ U == capomega``, ``U[0, 0] == 1``, off-diagonal entries ``S`` free
    and diagonal entries ``U[i, i] = sqrt(1 - sum_{k<i} U[k, i]**2)`` determined),
    return the Jacobian of the off-diagonal correlation elements with respect to
    the off-diagonal Cholesky elements.

    Parameters
    ----------
    capomega : ndarray, shape (K, K)
        Positive-definite correlation matrix with unit diagonal.
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    gcholcor : ndarray, shape (K*(K-1)//2, K*(K-1)//2)
        ``gcholcor[i, j] = d(rho[j]) / d(S[i])``, where ``S`` are the
        off-diagonal Cholesky elements and ``rho`` the off-diagonal correlation
        elements, both in row-based upper-triangular order.  For ``K <= 2`` a
        1x1 array of ones is returned (Cholesky has no coupling to resolve),
        matching GAUSS.

    Notes
    -----
    This matches the direction of the GAUSS procedure, whose format block lists
    rows indexed by ``dS`` (Cholesky) and columns by ``drho`` (correlation),
    i.e. ``d(corr)/d(chol)``.  The inverse map ``d(chol)/d(corr)`` is the matrix
    inverse of the returned Jacobian.
    """
    if xp is None:
        xp = array_namespace(capomega)
    capomega = np.asarray(capomega, dtype=np.float64)
    K = capomega.shape[0]
    n = K * (K - 1) // 2

    if K <= 2:
        return np.ones((1, 1), dtype=np.float64)

    U = _upper_chol(capomega)  # upper triangular, U.T @ U = capomega

    # d(U[k, a]) / d(S_{pq}); S_{pq} = U[p, q] with p < q.
    #   off-diagonal (k < a): 1 if (k, a) == (p, q) else 0
    #   diagonal   (k == a):  -U[p, a] / U[a, a] if a == q and p < a, else 0
    gcholcor = np.zeros((n, n), dtype=np.float64)

    for p in range(K):
        for q in range(p + 1, K):
            s_idx = _offdiag_index(p, q, K)

            # dU/dS_{pq}: only U[p, q] (=1) and the diagonal U[q, q] depend on it.
            dU = np.zeros((K, K), dtype=np.float64)
            dU[p, q] = 1.0
            dU[q, q] = -U[p, q] / U[q, q]

            # corr = U.T @ U ; d(corr[a, b]) = sum_k dU[k, a] U[k, b] + U[k, a] dU[k, b]
            for a in range(K):
                for b in range(a + 1, K):
                    val = 0.0
                    for k in range(K):
                        val += dU[k, a] * U[k, b] + U[k, a] * dU[k, b]
                    gcholcor[s_idx, _offdiag_index(a, b, K)] = val

    return gcholcor


def ggradchol(
    A: NDArray,
    L: NDArray,
    e: NDArray,
    *,
    cholcov: bool = True,
    diagL: bool = False,
    xp=None,
) -> NDArray:
    """Gradient of ``B = A @ L.T @ e`` w.r.t. the unique elements of ``L``.

    Port of GAUSS ``ggradchol`` (matgradient.src line 978).  Given an ``M x K``
    matrix ``A``, an upper-triangular ``K x K`` Cholesky factor ``L`` and a
    ``K``-vector ``e``, form ``B = A @ L.T @ e`` (an ``M``-vector) and return the
    gradient of each element of ``B`` with respect to the free elements of ``L``.

    Parameters
    ----------
    A : ndarray, shape (M, K)
        Left multiplier.
    L : ndarray, shape (K, K)
        Upper-triangular Cholesky factor (``B`` uses ``L.T``).  When
        ``cholcov=False`` (``L`` is the Cholesky of a correlation matrix) the
        diagonal is a determined function of the off-diagonal entries.
    e : ndarray, shape (K,) or (K, 1)
        Right vector.
    cholcov : bool, default True
        If True, ``L`` is the Cholesky of a covariance matrix and its upper
        elements (including diagonal, ``K*(K+1)//2`` of them) are free.  If
        False, ``L`` is the Cholesky of a correlation matrix and only the
        off-diagonal upper elements (``K*(K-1)//2``) are free.
    diagL : bool, default False
        If True, ``L`` is diagonal (covariance matrix is diagonal); only the
        ``K`` diagonal elements are free.  Applies to ``cholcov=True``.
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    gchol : ndarray, shape (n_params, M)
        ``gchol[i, m] = d(B[m]) / d(L_element[i])`` with rows in row-based
        upper-triangular order.  ``n_params`` is ``K*(K+1)//2`` for a covariance
        Cholesky (``K`` if ``diagL``), or ``K*(K-1)//2`` for a correlation
        Cholesky.
    """
    if xp is None:
        xp = array_namespace(A)
    A = np.asarray(A, dtype=np.float64)
    L = np.asarray(L, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    K = e.shape[0]
    M = A.shape[0]

    # B[m] = sum_{i,j} A[m,i] L[j,i] e[j]  ->  dB[m]/dL[j,i] = A[m,i] e[j].
    if cholcov and diagL:
        gchol = np.zeros((K, M), dtype=np.float64)
        for k in range(K):
            gchol[k, :] = A[:, k] * e[k]
        return gchol

    if cholcov:
        n = K * (K + 1) // 2
        gchol = np.zeros((n, M), dtype=np.float64)
        idx = 0
        for j in range(K):
            for i in range(j, K):  # upper element (j, i), j <= i
                gchol[idx, :] = A[:, i] * e[j]
                idx += 1
        return gchol

    # Correlation Cholesky: only off-diagonal free; diagonals determined via
    # L[i,i] = sqrt(1 - sum_{k<i} L[k,i]**2), dL[i,i]/dS_{pi} = -L[p,i]/L[i,i].
    n = K * (K - 1) // 2
    if n == 0:
        return np.ones((1, M), dtype=np.float64)
    diag = np.diag(L)
    gchol = np.zeros((n, M), dtype=np.float64)
    idx = 0
    for p in range(K):
        for q in range(p + 1, K):  # off-diagonal element S_{pq} = L[p, q]
            # direct: dB[m]/dL[p,q] = A[m,q] e[p]
            # diagonal chain at i=q: dB[m]/dL[q,q] * dL[q,q]/dS_{pq}
            #                      = A[m,q] e[q] * (-L[p,q]/L[q,q])
            gchol[idx, :] = A[:, q] * (e[p] - e[q] * L[p, q] / diag[q])
            idx += 1
    return gchol
