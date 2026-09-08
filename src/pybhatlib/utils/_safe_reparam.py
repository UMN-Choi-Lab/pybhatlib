"""Numerically safe primitives for optimizer reparameterizations.

The transforms are algebraically identical to their GAUSS counterparts in the
identifiable region.  Only saturated optimizer tails and invalid direct-entry
correlation matrices are modified.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.utils._logistic import (
    cdlogit,
    d_lam_d_lamnew,
    gradlogitmod,
    gradpdlogit,
    lam_from_lamnew,
    logitmod,
    pdlogit,
)

_EXP_ARG_MAX = 700.0
_CORR_LIMIT = 1.0 - 1e-12
_CHOL_JITTER0 = 1e-10

# Canonical stable softmax used by reparameterizations.
softmax = logitmod


def safe_exp(x: ArrayLike, *, xp=None):
    """Exponentiate after capping the non-identifiable positive tail.

    Values at or below 700 are passed to the backend exponential unchanged.
    This keeps the transform exact throughout any meaningful estimation region
    while guaranteeing a finite result for arbitrarily large positive inputs.
    """
    if xp is None:
        xp = array_namespace(x)
    values = xp.array(x, dtype=xp.float64)
    return xp.exp(xp.minimum(values, _EXP_ARG_MAX))


def corr_from_angle(theta: ArrayLike, scal: float, *, xp=None):
    """Map unconstrained angles to correlations strictly inside ``(-1, 1)``."""
    if scal <= 0.0 or not np.isfinite(scal):
        raise ValueError("scal must be finite and positive")
    if xp is None:
        xp = array_namespace(theta)
    values = xp.array(theta, dtype=xp.float64) / scal
    decay = xp.exp(-xp.abs(values))
    magnitude = (1.0 - decay) / (1.0 + decay)
    corr = xp.where(values >= 0.0, magnitude, -magnitude)
    return xp.clip(corr, -_CORR_LIMIT, _CORR_LIMIT)


def safe_cholesky(
    matrix: ArrayLike, *, jitter0: float | None = None
) -> tuple[NDArray[np.float64], float]:
    """Return a lower Cholesky factor, escalating diagonal jitter on failure.

    The first retry scales the default jitter by the mean diagonal.  Subsequent
    retries use absolute levels from ``1e-8`` through ``1e2``.  A successful
    unmodified factorization reports exactly zero jitter.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values")
    array = 0.5 * (array + array.T)
    try:
        return np.linalg.cholesky(array), 0.0
    except np.linalg.LinAlgError as original_error:
        n = array.shape[0]
        base = _CHOL_JITTER0 if jitter0 is None else float(jitter0)
        if base <= 0.0 or not np.isfinite(base):
            raise ValueError("jitter0 must be finite and positive") from original_error
        scale = abs(float(np.trace(array))) / n
        first = base * max(scale, 1.0)
        levels = [first]
        levels.extend(10.0**power for power in range(-8, 3, 2))
        identity = np.eye(n, dtype=np.float64)
        for jitter in dict.fromkeys(levels):
            try:
                return np.linalg.cholesky(array + jitter * identity), float(jitter)
            except np.linalg.LinAlgError:
                continue
        raise np.linalg.LinAlgError(
            "matrix is not positive definite after diagonal-jitter escalation"
        ) from original_error


def nearest_pd_correlation(
    matrix: ArrayLike, *, eigenvalue_floor: float = _CHOL_JITTER0
) -> NDArray[np.float64]:
    """Return a positive-definite unit-diagonal approximation to ``matrix``.

    A valid positive-definite correlation matrix is returned byte-for-byte,
    preserving parity at reference estimates.  Otherwise the symmetric matrix
    is eigen-clipped and rescaled to unit diagonal.
    """
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values")
    if eigenvalue_floor <= 0.0 or not np.isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be finite and positive")
    if np.array_equal(array, array.T) and np.array_equal(
        np.diag(array), np.ones(array.shape[0])
    ):
        try:
            np.linalg.cholesky(array)
            return array.copy()
        except np.linalg.LinAlgError:
            pass

    symmetric = 0.5 * (array + array.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = (eigenvectors * np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    diagonal_scale = np.sqrt(np.maximum(np.diag(repaired), eigenvalue_floor))
    repaired = repaired / np.outer(diagonal_scale, diagonal_scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)

    # Rescaling can lose a few ulps at the boundary.  A second spectral floor
    # keeps the documented strict-PD contract without affecting valid inputs.
    minimum = float(np.linalg.eigvalsh(repaired).min())
    if minimum <= 0.0:
        repaired += (eigenvalue_floor - minimum) * np.eye(repaired.shape[0])
        diagonal_scale = np.sqrt(np.diag(repaired))
        repaired /= np.outer(diagonal_scale, diagonal_scale)
        np.fill_diagonal(repaired, 1.0)
    return repaired


__all__ = [
    "_CHOL_JITTER0",
    "_CORR_LIMIT",
    "_EXP_ARG_MAX",
    "cdlogit",
    "corr_from_angle",
    "d_lam_d_lamnew",
    "gradlogitmod",
    "gradpdlogit",
    "lam_from_lamnew",
    "nearest_pd_correlation",
    "pdlogit",
    "safe_cholesky",
    "safe_exp",
    "softmax",
]
