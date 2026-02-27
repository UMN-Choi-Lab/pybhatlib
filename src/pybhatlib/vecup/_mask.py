"""Mask matrix operations for mixed models."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def build_mask_matrix(
    n_outcomes: int,
    fixed_indices: list[int] | None = None,
    *,
    xp=None,
) -> NDArray:
    """Build mask/selection matrix for mixed models.

    Creates a selection matrix that picks specific rows/columns,
    useful for constructing covariance structures in mixed data models.

    Parameters
    ----------
    n_outcomes : int
        Total number of outcomes/variables.
    fixed_indices : list of int or None
        Indices of outcomes that are fixed (not random). If None, all free.
    xp : backend, optional

    Returns
    -------
    mask : ndarray, shape (n_free, n_outcomes)
        Selection matrix where n_free = n_outcomes - len(fixed_indices).
    """
    if xp is None:
        xp = get_backend("numpy")

    if fixed_indices is None or len(fixed_indices) == 0:
        return xp.eye(n_outcomes, dtype=xp.float64)

    free_indices = [i for i in range(n_outcomes) if i not in fixed_indices]
    n_free = len(free_indices)

    mask = xp.zeros((n_free, n_outcomes), dtype=xp.float64)
    for k, idx in enumerate(free_indices):
        mask[k, idx] = 1.0

    return mask


def apply_mask(
    matrix: NDArray,
    mask: NDArray,
    *,
    xp=None,
) -> NDArray:
    """Apply mask to extract submatrix: result = mask @ matrix @ mask.T.

    Parameters
    ----------
    matrix : ndarray, shape (n, n)
        Full matrix.
    mask : ndarray, shape (m, n)
        Selection mask.
    xp : backend, optional

    Returns
    -------
    result : ndarray, shape (m, m)
        Selected submatrix.
    """
    if xp is None:
        xp = array_namespace(matrix, mask)
    return xp.matmul(xp.matmul(mask, matrix), xp.transpose(mask))
