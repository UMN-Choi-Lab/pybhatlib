"""Gaussian per-observation log-likelihood and analytical derivatives.

The parameter vector is ``theta = [beta_1, ..., beta_K, sigma]``: the
regression coefficients followed by the residual standard deviation, exactly
the layout of ``LRResults.params`` (and of the MDCEV scale convention).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_design(X: ArrayLike, n_beta: int) -> NDArray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_beta or not np.isfinite(X).all():
        raise ValueError("X must be a finite (n_obs, n_beta) design matrix")
    return X


def _inputs(
    theta: ArrayLike, X: ArrayLike, y: ArrayLike
) -> tuple[NDArray, NDArray, float]:
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1 or len(theta) < 2 or not np.isfinite(theta).all():
        raise ValueError("theta must be a finite vector [beta..., sigma]")
    sigma = float(theta[-1])
    if sigma <= 0:
        raise ValueError("sigma (last element of theta) must be positive")
    X = _validate_design(X, len(theta) - 1)
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector with one value per observation")
    return X, y - X @ theta[:-1], sigma


def lr_loglik(theta: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray:
    """Gaussian log-likelihood contributions.

    Parameters
    ----------
    theta : array_like, shape (K + 1,)
        Regression coefficients followed by the residual standard deviation
        ``sigma`` (positive).
    X : array_like, shape (N, K)
        Design matrix (include a ``uno`` column for the constant).
    y : array_like, shape (N,)
        Continuous outcome.

    Returns
    -------
    ll : ndarray, shape (N,)
        Per-observation log-likelihood.
    """
    _, residuals, sigma = _inputs(theta, X, y)
    return -0.5 * (np.log(2 * np.pi) + 2 * np.log(sigma) + (residuals / sigma) ** 2)


def lr_gradient(theta: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray:
    """Per-observation scores with respect to ``theta``.

    Parameters
    ----------
    theta, X, y
        As in :func:`lr_loglik`.

    Returns
    -------
    scores : ndarray, shape (N, K + 1)
        ``X * residual / sigma**2`` for the coefficients and
        ``(residual**2 / sigma**2 - 1) / sigma`` for ``sigma``, row by row.
    """
    X, residuals, sigma = _inputs(theta, X, y)
    return np.column_stack([
        X * (residuals / sigma**2)[:, None],
        (residuals**2 / sigma**2 - 1) / sigma,
    ])


def lr_hessian(theta: ArrayLike, X: ArrayLike, y: ArrayLike) -> NDArray:
    """Total log-likelihood Hessian with respect to ``theta``.

    Parameters
    ----------
    theta, X, y
        As in :func:`lr_loglik`.

    Returns
    -------
    hess : ndarray, shape (K + 1, K + 1)
        ``-(X'X) / sigma**2`` in the coefficient block,
        ``-2 X'r / sigma**3`` off the block (zero at the least-squares
        solution), and ``(N - 3 r'r / sigma**2) / sigma**2`` for ``sigma``.
    """
    X, residuals, sigma = _inputs(theta, X, y)
    k = X.shape[1]
    hess = np.empty((k + 1, k + 1))
    hess[:k, :k] = -(X.T @ X) / sigma**2
    hess[:k, k] = hess[k, :k] = -2 * (X.T @ residuals) / sigma**3
    hess[k, k] = (len(residuals) - 3 * (residuals @ residuals) / sigma**2) / sigma**2
    return hess
