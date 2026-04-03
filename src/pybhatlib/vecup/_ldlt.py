"""LDLT decomposition and rank-1 update.

These are critical for the Bhat (2018) MVNCD analytic approximation,
which uses LDLT factorization with efficient rank-1 updates.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend

# ---------------------------------------------------------------------------
# Numba availability check
# ---------------------------------------------------------------------------
try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ---------------------------------------------------------------------------
# Numba JIT implementations (pure NumPy, no xp backend)
# ---------------------------------------------------------------------------
if HAS_NUMBA:
    @numba.njit(cache=True)
    def _ldlt_decompose_jit(A):
        """JIT-compiled LDLT decomposition. A = L D L^T.

        Parameters
        ----------
        A : ndarray, shape (n, n), float64

        Returns
        -------
        L : ndarray, shape (n, n), float64
        D : ndarray, shape (n,), float64
        """
        n = A.shape[0]
        L = np.eye(n)
        D = np.zeros(n)

        for j in range(n):
            s = 0.0
            for k in range(j):
                s += L[j, k] * L[j, k] * D[k]
            D[j] = A[j, j] - s

            if abs(D[j]) < 1e-15:
                D[j] = 1e-15

            for i in range(j + 1, n):
                s = 0.0
                for k in range(j):
                    s += L[i, k] * L[j, k] * D[k]
                L[i, j] = (A[i, j] - s) / D[j]

        return L, D

    @numba.njit(cache=True)
    def _ldlt_rank1_update_jit(L, D, v, alpha):
        """JIT-compiled rank-1 update of LDLT factorization.

        Computes LDLT of A + alpha * v @ v.T given existing LDLT of A.

        Parameters
        ----------
        L : ndarray, shape (n, n), float64
        D : ndarray, shape (n,), float64
        v : ndarray, shape (n,), float64
        alpha : float

        Returns
        -------
        L_new : ndarray, shape (n, n), float64
        D_new : ndarray, shape (n,), float64
        """
        n = L.shape[0]
        L_new = L.copy()
        D_new = D.copy()

        p = v.copy()
        beta = alpha

        for j in range(n):
            D_old_j = D_new[j]
            D_new[j] = D_old_j + beta * p[j] * p[j]

            if abs(D_new[j]) < 1e-15:
                D_new[j] = 1e-15

            gamma = beta * p[j] / D_new[j]
            beta = beta * D_old_j / D_new[j]

            for i in range(j + 1, n):
                p[i] -= p[j] * L_new[i, j]
                L_new[i, j] += gamma * p[i]

        return L_new, D_new

    @numba.njit(cache=True)
    def _ldlt_rank2_update_jit(L, D, u, v, alpha):
        """JIT-compiled rank-2 update via two successive rank-1 updates.

        Parameters
        ----------
        L : ndarray, shape (n, n), float64
        D : ndarray, shape (n,), float64
        u : ndarray, shape (n,), float64
        v : ndarray, shape (n,), float64
        alpha : float

        Returns
        -------
        L_new : ndarray, shape (n, n), float64
        D_new : ndarray, shape (n,), float64
        """
        L_new, D_new = _ldlt_rank1_update_jit(L, D, u, alpha)
        L_new, D_new = _ldlt_rank1_update_jit(L_new, D_new, v, alpha)
        return L_new, D_new


# ---------------------------------------------------------------------------
# Pure Python implementations (fallback when Numba unavailable or PyTorch)
# ---------------------------------------------------------------------------
def _ldlt_decompose_python(A, xp):
    """Pure Python LDLT decomposition with backend support."""
    A = xp.array(A, dtype=xp.float64)
    n = A.shape[0]

    L = xp.eye(n, dtype=xp.float64)
    D = xp.zeros((n,), dtype=xp.float64)

    A_work = xp.copy(A)

    for j in range(n):
        s = 0.0
        for k in range(j):
            s += L[j, k] ** 2 * D[k]
        D[j] = A_work[j, j] - s

        if abs(D[j]) < 1e-15:
            D[j] = 1e-15

        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k] * D[k]
            L[i, j] = (A_work[i, j] - s) / D[j]

    return L, D


def _ldlt_rank1_update_python(L, D, v, alpha, xp):
    """Pure Python rank-1 LDLT update with backend support."""
    n = L.shape[0]
    L_new = xp.copy(L)
    D_new = xp.copy(D)

    p = xp.array(v, dtype=xp.float64).copy()
    beta = alpha

    for j in range(n):
        D_old_j = D_new[j]
        D_new[j] = D_old_j + beta * p[j] ** 2

        if abs(D_new[j]) < 1e-15:
            D_new[j] = 1e-15

        gamma = beta * p[j] / D_new[j]
        beta = beta * D_old_j / D_new[j]

        for i in range(j + 1, n):
            p[i] -= p[j] * L_new[i, j]
            L_new[i, j] += gamma * p[i]

    return L_new, D_new


# ---------------------------------------------------------------------------
# Public API (dispatches to JIT or Python based on backend)
# ---------------------------------------------------------------------------
def _is_numpy_backend(xp) -> bool:
    """Check if the backend is NumPy (not PyTorch)."""
    return type(xp).__name__ == "NumpyBackend"


def ldlt_decompose(A: NDArray, *, xp=None) -> tuple[NDArray, NDArray]:
    """Compute A = L D L^T decomposition.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive-definite matrix.
    xp : backend, optional

    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular matrix with unit diagonal.
    D : ndarray, shape (n,)
        Diagonal elements of D.
    """
    if xp is None:
        xp = array_namespace(A)

    if HAS_NUMBA and _is_numpy_backend(xp):
        A_np = np.asarray(A, dtype=np.float64)
        return _ldlt_decompose_jit(A_np)

    return _ldlt_decompose_python(A, xp)


def ldlt_rank1_update(
    L: NDArray,
    D: NDArray,
    v: NDArray,
    alpha: float = 1.0,
    *,
    xp=None,
) -> tuple[NDArray, NDArray]:
    """Rank-1 update of LDLT factorization.

    Computes the LDLT factorization of A + alpha * v @ v.T
    given the existing LDLT factorization of A, in O(n^2) time.

    Parameters
    ----------
    L : ndarray, shape (n, n)
        Lower triangular factor (unit diagonal).
    D : ndarray, shape (n,)
        Diagonal factor.
    v : ndarray, shape (n,)
        Update vector.
    alpha : float
        Scaling factor for the rank-1 update.
    xp : backend, optional

    Returns
    -------
    L_new : ndarray, shape (n, n)
        Updated lower triangular factor.
    D_new : ndarray, shape (n,)
        Updated diagonal factor.
    """
    if xp is None:
        xp = array_namespace(L, D, v)

    if HAS_NUMBA and _is_numpy_backend(xp):
        L_np = np.asarray(L, dtype=np.float64)
        D_np = np.asarray(D, dtype=np.float64)
        v_np = np.asarray(v, dtype=np.float64)
        return _ldlt_rank1_update_jit(L_np, D_np, v_np, float(alpha))

    return _ldlt_rank1_update_python(L, D, v, alpha, xp)


def ldlt_rank2_update(
    L: NDArray,
    D: NDArray,
    u: NDArray,
    v: NDArray,
    alpha: float = 1.0,
    *,
    xp=None,
) -> tuple[NDArray, NDArray]:
    """Rank-2 update of LDLT factorization.

    Computes the LDLT factorization of A + alpha * (u @ u.T + v @ v.T)
    by two successive rank-1 updates.

    Parameters
    ----------
    L : ndarray, shape (n, n)
        Lower triangular factor (unit diagonal).
    D : ndarray, shape (n,)
        Diagonal factor.
    u : ndarray, shape (n,)
        First update vector.
    v : ndarray, shape (n,)
        Second update vector.
    alpha : float
        Scaling factor for both rank-1 updates.
    xp : backend, optional

    Returns
    -------
    L_new : ndarray, shape (n, n)
        Updated lower triangular factor.
    D_new : ndarray, shape (n,)
        Updated diagonal factor.
    """
    L_new, D_new = ldlt_rank1_update(L, D, u, alpha, xp=xp)
    L_new, D_new = ldlt_rank1_update(L_new, D_new, v, alpha, xp=xp)
    return L_new, D_new
