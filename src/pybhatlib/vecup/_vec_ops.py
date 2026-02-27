"""Core vector/matrix operations from BHATLIB's Vecup.src.

All operations follow BHATLIB's row-based arrangement convention:
vectorization proceeds row by row, and for symmetric matrices only
upper triangular elements (including diagonal) are stored.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def vecdup(r: NDArray, *, xp=None) -> NDArray:
    """Extract upper triangular elements (including diagonal) row-by-row.

    Parameters
    ----------
    r : ndarray, shape (K, K)
        Square symmetric matrix.
    xp : backend, optional

    Returns
    -------
    w : ndarray, shape (K*(K+1)//2,)
        Upper triangular elements in row-based order.

    Examples
    --------
    >>> vecdup(np.array([[1,2,3],[2,4,5],[3,5,6]]))
    array([1., 2., 3., 4., 5., 6.])
    """
    if xp is None:
        xp = array_namespace(r)
    r = xp.array(r, dtype=xp.float64)
    K = r.shape[0]
    n = K * (K + 1) // 2
    w = xp.zeros((n,), dtype=xp.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            w[idx] = r[i, j]
            idx += 1
    return w


def vecndup(r: NDArray, *, xp=None) -> NDArray:
    """Extract upper triangular elements (excluding diagonal) row-by-row.

    Parameters
    ----------
    r : ndarray, shape (K, K)
        Square symmetric matrix.
    xp : backend, optional

    Returns
    -------
    w : ndarray, shape (K*(K-1)//2,)
        Off-diagonal upper triangular elements in row-based order.

    Examples
    --------
    >>> vecndup(np.array([[1,2,3],[2,4,5],[3,5,6]]))
    array([2., 3., 5.])
    """
    if xp is None:
        xp = array_namespace(r)
    r = xp.array(r, dtype=xp.float64)
    K = r.shape[0]
    n = K * (K - 1) // 2
    w = xp.zeros((n,), dtype=xp.float64)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            w[idx] = r[i, j]
            idx += 1
    return w


def matdupfull(r: NDArray, *, xp=None) -> NDArray:
    """Expand vector of upper diagonal elements into full symmetric matrix.

    Inverse of vecdup.

    Parameters
    ----------
    r : ndarray, shape (K,) where K = P*(P+1)//2
        Upper triangular elements in row-based order.
    xp : backend, optional

    Returns
    -------
    w : ndarray, shape (P, P)
        Full symmetric matrix.

    Examples
    --------
    >>> matdupfull(np.array([1,2,3,4,5,6]))
    array([[1., 2., 3.],
           [2., 4., 5.],
           [3., 5., 6.]])
    """
    if xp is None:
        xp = array_namespace(r)
    r = xp.array(r, dtype=xp.float64)
    K = len(r)
    # Solve P*(P+1)/2 = K for P: P = (-1 + sqrt(1 + 8K)) / 2
    P = int((-1.0 + np.sqrt(1.0 + 8.0 * K)) / 2.0)
    assert P * (P + 1) // 2 == K, f"Input length {K} is not a triangular number"

    w = xp.zeros((P, P), dtype=xp.float64)
    idx = 0
    for i in range(P):
        for j in range(i, P):
            w[i, j] = r[idx]
            w[j, i] = r[idx]
            idx += 1
    return w


def matdupdiagonefull(r: NDArray, *, xp=None) -> NDArray:
    """Convert vector to symmetric matrix with unit diagonal.

    Parameters
    ----------
    r : ndarray, shape (K,) where K = P*(P-1)//2
        Off-diagonal upper triangular elements in row-based order.
    xp : backend, optional

    Returns
    -------
    w : ndarray, shape (P, P)
        Symmetric matrix with 1s on diagonal.

    Examples
    --------
    >>> matdupdiagonefull(np.array([0.6, 0.5, 0.5]))
    array([[1. , 0.6, 0.5],
           [0.6, 1. , 0.5],
           [0.5, 0.5, 1. ]])
    """
    if xp is None:
        xp = array_namespace(r)
    r = xp.array(r, dtype=xp.float64)
    K = len(r)
    # Solve P*(P-1)/2 = K for P: P = (1 + sqrt(1 + 8K)) / 2
    P = int((1.0 + np.sqrt(1.0 + 8.0 * K)) / 2.0)
    assert P * (P - 1) // 2 == K, f"Input length {K} is not valid"

    w = xp.eye(P, dtype=xp.float64)
    idx = 0
    for i in range(P):
        for j in range(i + 1, P):
            w[i, j] = r[idx]
            w[j, i] = r[idx]
            idx += 1
    return w


def vecsymmetry(r: NDArray, *, xp=None) -> NDArray:
    """Produce position pattern matrix from symmetric matrix.

    Each row of the output corresponds to one upper-triangular element
    of the input. That row has 1s in the positions of the full (row-vectorized)
    matrix that map to that symmetric element.

    Parameters
    ----------
    r : ndarray, shape (K, K)
        Square symmetric matrix.
    xp : backend, optional

    Returns
    -------
    S : ndarray, shape (K*(K+1)//2, K*K)
        Position pattern matrix.
    """
    if xp is None:
        xp = array_namespace(r)
    K = r.shape[0]
    n_upper = K * (K + 1) // 2
    n_full = K * K

    S = xp.zeros((n_upper, n_full), dtype=xp.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            # Position in row-vectorized full matrix: i*K + j
            S[idx, i * K + j] = 1.0
            if i != j:
                # Also the symmetric position: j*K + i
                S[idx, j * K + i] = 1.0
            idx += 1
    return S
