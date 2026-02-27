"""Truncated multivariate normal moments computation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend


def truncated_normal_mean_var(
    mu: float,
    sigma: float,
    lower: float = -np.inf,
    upper: float = np.inf,
    *,
    xp=None,
) -> tuple[float, float]:
    """Compute mean and variance of univariate truncated normal.

    Parameters
    ----------
    mu : float
        Mean of untruncated distribution.
    sigma : float
        Standard deviation of untruncated distribution.
    lower, upper : float
        Truncation bounds.

    Returns
    -------
    mu_trunc : float
        Mean of truncated distribution.
    var_trunc : float
        Variance of truncated distribution.
    """
    if sigma <= 0:
        return mu, 0.0

    alpha = (lower - mu) / sigma if np.isfinite(lower) else -np.inf
    beta = (upper - mu) / sigma if np.isfinite(upper) else np.inf

    Phi_alpha = norm.cdf(alpha)
    Phi_beta = norm.cdf(beta)
    Z = Phi_beta - Phi_alpha

    if Z < 1e-300:
        return 0.5 * (lower + upper) if np.isfinite(lower) and np.isfinite(upper) else mu, 0.0

    phi_alpha = norm.pdf(alpha) if np.isfinite(alpha) else 0.0
    phi_beta = norm.pdf(beta) if np.isfinite(beta) else 0.0

    # Truncated mean
    mu_trunc = mu + sigma * (phi_alpha - phi_beta) / Z

    # Truncated variance
    t1 = (alpha * phi_alpha - beta * phi_beta) / Z if np.isfinite(alpha) else (-beta * phi_beta) / Z
    t2 = ((phi_alpha - phi_beta) / Z) ** 2
    var_trunc = sigma**2 * (1.0 + t1 - t2)
    var_trunc = max(var_trunc, 0.0)

    return float(mu_trunc), float(var_trunc)


def truncated_mvn_moments(
    mu: NDArray,
    sigma: NDArray,
    lower: NDArray,
    upper: NDArray,
    *,
    xp=None,
) -> tuple[NDArray, NDArray]:
    """Compute mean and covariance of truncated multivariate normal.

    Uses a sequential conditioning approach: iteratively compute truncated
    moments of each variable conditioned on the previous variables.

    Parameters
    ----------
    mu : ndarray, shape (K,)
        Mean of untruncated distribution.
    sigma : ndarray, shape (K, K)
        Covariance of untruncated distribution.
    lower, upper : ndarray, shape (K,)
        Truncation bounds (-inf/inf allowed).

    Returns
    -------
    mu_trunc : ndarray, shape (K,)
        Mean of truncated distribution.
    sigma_trunc : ndarray, shape (K, K)
        Covariance of truncated distribution (approximate).
    """
    if xp is None:
        xp = get_backend("numpy")

    mu_np = np.asarray(mu, dtype=np.float64)
    sigma_np = np.asarray(sigma, dtype=np.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    K = len(mu_np)

    if K == 1:
        m, v = truncated_normal_mean_var(
            mu_np[0], np.sqrt(sigma_np[0, 0]), lower_np[0], upper_np[0]
        )
        return np.array([m]), np.array([[v]])

    # Sequential conditioning approximation
    mu_trunc = np.zeros(K, dtype=np.float64)
    sigma_trunc = np.zeros((K, K), dtype=np.float64)

    # Working mean and covariance (updated as we condition)
    mu_cond = mu_np.copy()
    sigma_cond = sigma_np.copy()

    for k in range(K):
        sd_k = np.sqrt(max(sigma_cond[k, k], 1e-15))

        # Truncated mean and variance for variable k
        m_k, v_k = truncated_normal_mean_var(
            mu_cond[k], sd_k, lower_np[k], upper_np[k]
        )
        mu_trunc[k] = m_k
        sigma_trunc[k, k] = v_k

        if k < K - 1:
            # Update conditional distribution for remaining variables
            delta = m_k - mu_cond[k]  # shift from truncation

            for j in range(k + 1, K):
                # Regression coefficient
                reg_coef = sigma_cond[j, k] / sigma_cond[k, k] if sigma_cond[k, k] > 1e-15 else 0.0

                # Update conditional mean
                mu_cond[j] += reg_coef * delta

                # Update conditional variance (approximate)
                var_reduction = reg_coef**2 * (sigma_cond[k, k] - v_k)
                sigma_cond[j, j] = max(sigma_cond[j, j] - var_reduction, 1e-15)

                # Cross-covariance
                sigma_trunc[k, j] = reg_coef * v_k
                sigma_trunc[j, k] = sigma_trunc[k, j]

                # Update cross-covariances among remaining variables
                for l in range(j + 1, K):
                    reg_l = sigma_cond[l, k] / sigma_cond[k, k] if sigma_cond[k, k] > 1e-15 else 0.0
                    sigma_cond[j, l] -= reg_coef * reg_l * (sigma_cond[k, k] - v_k)
                    sigma_cond[l, j] = sigma_cond[j, l]

    return mu_trunc, sigma_trunc
