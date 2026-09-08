"""Gaussian per-observation likelihood and analytical beta derivatives.

The residual variance is a separate scalar, held fixed in these derivatives.
"""

import numpy as np


def _validate_design(X, n_beta):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_beta or not np.isfinite(X).all():
        raise ValueError("X must be a finite (n_obs, n_beta) design matrix")
    return X


def _inputs(beta, X, y, sigma2):
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


def lr_loglik(beta, X, y, sigma2):
    """Return Gaussian log-likelihood contributions, shape (N,)."""
    _, residuals = _inputs(beta, X, y, sigma2)
    return -0.5 * (np.log(2 * np.pi) + np.log(sigma2) + residuals**2 / sigma2)


def lr_gradient(beta, X, y, sigma2):
    """Return per-observation beta scores, shape (N, K)."""
    X, residuals = _inputs(beta, X, y, sigma2)
    return X * (residuals / sigma2)[:, None]


def lr_hessian(beta, X, y, sigma2):
    """Return the total beta log-likelihood Hessian, shape (K, K)."""
    X, _ = _inputs(beta, X, y, sigma2)
    return -(X.T @ X) / sigma2
