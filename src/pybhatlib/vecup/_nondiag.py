"""Extract non-diagonal elements from a matrix."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace


def nondiag(r: NDArray, *, xp=None) -> NDArray:
    """Extract non-diagonal elements of a matrix row-by-row.

    Parameters
    ----------
    r : ndarray, shape (K, K)
        Square matrix.
    xp : backend, optional

    Returns
    -------
    w : ndarray, shape (K*(K-1),)
        Non-diagonal elements in row-based order.

    Examples
    --------
    >>> nondiag(np.array([[1,2,3],[4,5,6],[7,8,9]]))
    array([2., 3., 4., 6., 7., 8.])
    """
    if xp is None:
        xp = array_namespace(r)
    r = xp.array(r, dtype=xp.float64)
    K = r.shape[0]
    n = K * (K - 1)
    w = xp.zeros((n,), dtype=xp.float64)
    idx = 0
    for i in range(K):
        for j in range(K):
            if i != j:
                w[idx] = r[i, j]
                idx += 1
    return w
