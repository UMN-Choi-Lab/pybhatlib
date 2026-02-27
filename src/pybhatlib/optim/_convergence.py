"""Convergence diagnostics and standard error computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def check_convergence(grad: NDArray, tol: float = 1e-5) -> bool:
    """Check if gradient norm is below tolerance."""
    return float(np.linalg.norm(grad)) < tol


def compute_standard_errors(hess_inv: NDArray, n_obs: int) -> NDArray:
    """Compute standard errors from inverse Hessian.

    SE_i = sqrt(H^{-1}_{ii} / N)

    Parameters
    ----------
    hess_inv : ndarray, shape (p, p)
        Inverse Hessian matrix.
    n_obs : int
        Number of observations.

    Returns
    -------
    se : ndarray, shape (p,)
        Standard errors.
    """
    diag = np.diag(hess_inv)
    return np.sqrt(np.abs(diag) / n_obs)


def compute_robust_standard_errors(
    hess_inv: NDArray, grad_contributions: NDArray
) -> NDArray:
    """Compute sandwich (robust) standard errors.

    V = H^{-1} B H^{-1} where B = sum(g_i @ g_i.T)

    Parameters
    ----------
    hess_inv : ndarray, shape (p, p)
        Inverse Hessian.
    grad_contributions : ndarray, shape (N, p)
        Per-observation gradient vectors.

    Returns
    -------
    se : ndarray, shape (p,)
        Robust standard errors.
    """
    B = grad_contributions.T @ grad_contributions
    V = hess_inv @ B @ hess_inv
    return np.sqrt(np.abs(np.diag(V)))
