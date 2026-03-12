"""Gradients of the MVNCD function.

Computes gradients of the multivariate normal CDF with respect to the upper
integration limits and covariance matrix elements (vech form).

Supports two modes:
- ``analytic=True`` (default): Uses backward/adjoint differentiation for the
  ME method, giving exact gradients of the ME approximation in O(K^3) time.
- ``analytic=False``: Falls back to central finite differences (2K + K*(K+1)
  MVNCD evaluations per call).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend
from pybhatlib.gradmvn._mvncd import mvncd
from pybhatlib.gradmvn._mvncd_grad_analytic import mvncd_grad_me_analytic


@dataclass
class MVNCDGradResult:
    """Result of MVNCD gradient computation.

    Attributes
    ----------
    prob : float
        MVNCD probability value.
    grad_a : ndarray, shape (K,)
        Gradient w.r.t. upper integration limits.
    grad_sigma : ndarray, shape (K*(K+1)//2,)
        Gradient w.r.t. upper-triangular covariance elements (row-based).
    """

    prob: float
    grad_a: NDArray
    grad_sigma: NDArray


def mvncd_grad(
    a: NDArray,
    sigma: NDArray,
    *,
    method: str = "me",
    analytic: bool = True,
    xp=None,
) -> MVNCDGradResult:
    """Compute MVNCD and its gradients.

    Parameters
    ----------
    a : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix.
    method : str
        Approximation method for MVNCD.
    analytic : bool
        If True (default) and method is 'me', use backward/adjoint analytic
        gradients. Falls back to numerical FD otherwise.
    xp : backend, optional

    Returns
    -------
    result : MVNCDGradResult
        Contains probability and gradients.
    """
    if xp is None:
        xp = array_namespace(a, sigma)

    a_np = np.asarray(xp.to_numpy(a), dtype=np.float64).ravel()
    sigma_np = np.asarray(xp.to_numpy(sigma), dtype=np.float64)

    # Dispatch to analytic gradient when available
    if analytic and method == "me":
        prob, grad_a, grad_sigma = mvncd_grad_me_analytic(a_np, sigma_np)
        return MVNCDGradResult(
            prob=prob,
            grad_a=xp.array(grad_a),
            grad_sigma=xp.array(grad_sigma),
        )

    # Numerical fallback
    prob = mvncd(xp.array(a_np), xp.array(sigma_np), method=method, xp=xp)
    grad_a = _grad_a_numerical(a_np, sigma_np, prob, method, xp)
    grad_sigma = _grad_sigma_numerical(a_np, sigma_np, prob, method, xp)

    return MVNCDGradResult(
        prob=prob,
        grad_a=xp.array(grad_a),
        grad_sigma=xp.array(grad_sigma),
    )


def _grad_a_numerical(
    a: np.ndarray,
    sigma: np.ndarray,
    prob0: float,
    method: str,
    xp,
) -> np.ndarray:
    """Numerical gradient of MVNCD w.r.t. integration limits."""
    K = len(a)

    if K == 1:
        # Analytic: dPhi(a/sd)/da = phi(a/sd) / sd
        sd = np.sqrt(sigma[0, 0])
        return np.array([norm.pdf(a[0] / sd) / sd])

    eps = 1e-6
    grad = np.zeros(K, dtype=np.float64)
    for i in range(K):
        a_plus = a.copy()
        a_plus[i] += eps
        prob_plus = mvncd(xp.array(a_plus), xp.array(sigma), method=method, xp=xp)
        a_minus = a.copy()
        a_minus[i] -= eps
        prob_minus = mvncd(xp.array(a_minus), xp.array(sigma), method=method, xp=xp)
        grad[i] = (prob_plus - prob_minus) / (2.0 * eps)

    return grad


def _grad_sigma_numerical(
    a: np.ndarray,
    sigma: np.ndarray,
    prob0: float,
    method: str,
    xp,
) -> np.ndarray:
    """Numerical gradient of MVNCD w.r.t. upper-triangular covariance elements."""
    K = len(a)
    n_sigma = K * (K + 1) // 2
    eps = 1e-6
    grad = np.zeros(n_sigma, dtype=np.float64)

    idx = 0
    for i in range(K):
        for j in range(i, K):
            sigma_plus = sigma.copy()
            sigma_minus = sigma.copy()
            sigma_plus[i, j] += eps
            sigma_minus[i, j] -= eps
            if i != j:
                sigma_plus[j, i] += eps
                sigma_minus[j, i] -= eps
            prob_plus = mvncd(xp.array(a), xp.array(sigma_plus), method=method, xp=xp)
            prob_minus = mvncd(xp.array(a), xp.array(sigma_minus), method=method, xp=xp)

            grad[idx] = (prob_plus - prob_minus) / (2.0 * eps)
            idx += 1

    return grad
