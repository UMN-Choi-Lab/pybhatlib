"""Gaussian per-observation log-likelihood and analytical beta derivatives.

The residual variance ``sigma2`` is a separate scalar held fixed in these
derivatives; :class:`LRModel` concentrates it out (``sigma2 = SSE / N``).
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
    beta: ArrayLike, X: ArrayLike, y: ArrayLike, sigma2: float
) -> tuple[NDArray, NDArray]:
    beta = np.asarray(beta, dtype=float)
    if beta.ndim != 1 or not np.isfinite(beta).all():
        raise ValueError("beta must be a finite one-dimensional vector")
    X = _validate_design(X, len(beta))
    y = np.asarray(y, dtype=float)
    if y.shape != (len(X),) or not np.isfinite(y).all():
        raise ValueError("y must be a finite vector with one value per observation")
    if np.ndim(sigma2) != 0 or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be a positive finite scalar")
    return X, y - X @ beta


def lr_loglik(beta: ArrayLike, X: ArrayLike, y: ArrayLike, sigma2: float) -> NDArray:
    """Gaussian log-likelihood contributions.

    Parameters
    ----------
    beta : array_like, shape (K,)
        Regression coefficients.
    X : array_like, shape (N, K)
        Design matrix (include a ``uno`` column for the constant).
    y : array_like, shape (N,)
        Continuous outcome.
    sigma2 : float
        Residual variance (positive scalar).

    Returns
    -------
    ll : ndarray, shape (N,)
        Per-observation log-likelihood.
    """
    _, residuals = _inputs(beta, X, y, sigma2)
    return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + residuals**2 / sigma2)


def lr_gradient(beta: ArrayLike, X: ArrayLike, y: ArrayLike, sigma2: float) -> NDArray:
    """Per-observation scores with respect to ``beta``.

    Parameters
    ----------
    beta, X, y, sigma2
        As in :func:`lr_loglik`.

    Returns
    -------
    scores : ndarray, shape (N, K)
        ``X * residual / sigma2`` row by row.
    """
    X, residuals = _inputs(beta, X, y, sigma2)
    return X * (residuals / sigma2)[:, None]


def lr_hessian(beta: ArrayLike, X: ArrayLike, y: ArrayLike, sigma2: float) -> NDArray:
    """Total log-likelihood Hessian with respect to ``beta``.

    Parameters
    ----------
    beta, X, y, sigma2
        As in :func:`lr_loglik`.

    Returns
    -------
    hess : ndarray, shape (K, K)
        ``-(X'X) / sigma2``.
    """
    X, _ = _inputs(beta, X, y, sigma2)
    return -(X.T @ X) / sigma2
