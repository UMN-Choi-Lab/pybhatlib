"""Truncated multivariate normal density and CDF.

Implements truncated (both-end) density and CDF computations using
combinatorial methods as described in BHATLIB's Gradmvn.src.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend
from pybhatlib.gradmvn._mvncd import mvncd


def truncated_normal_pdf(
    x: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    lower: float = -np.inf,
    upper: float = np.inf,
    *,
    xp=None,
) -> float:
    """Univariate truncated normal density.

    Parameters
    ----------
    x : float
        Evaluation point.
    mu, sigma : float
        Mean and standard deviation of the untruncated distribution.
    lower, upper : float
        Truncation bounds.

    Returns
    -------
    pdf : float
        Truncated normal density at x.
    """
    if x < lower or x > upper:
        return 0.0

    z = (x - mu) / sigma
    z_lo = (lower - mu) / sigma if np.isfinite(lower) else -np.inf
    z_hi = (upper - mu) / sigma if np.isfinite(upper) else np.inf

    phi_z = norm.pdf(z)
    Phi_hi = norm.cdf(z_hi)
    Phi_lo = norm.cdf(z_lo)
    denom = Phi_hi - Phi_lo

    if denom < 1e-300:
        return 0.0

    return float(phi_z / (sigma * denom))


def truncated_normal_cdf(
    x: float,
    mu: float = 0.0,
    sigma: float = 1.0,
    lower: float = -np.inf,
    upper: float = np.inf,
    *,
    xp=None,
) -> float:
    """Univariate truncated normal CDF.

    Parameters
    ----------
    x : float
        Evaluation point.
    mu, sigma : float
        Mean and standard deviation of the untruncated distribution.
    lower, upper : float
        Truncation bounds.

    Returns
    -------
    cdf : float
        P(X <= x) under truncation.
    """
    if x <= lower:
        return 0.0
    if x >= upper:
        return 1.0

    z = (x - mu) / sigma
    z_lo = (lower - mu) / sigma if np.isfinite(lower) else -np.inf
    z_hi = (upper - mu) / sigma if np.isfinite(upper) else np.inf

    Phi_z = norm.cdf(z)
    Phi_lo = norm.cdf(z_lo)
    Phi_hi = norm.cdf(z_hi)
    denom = Phi_hi - Phi_lo

    if denom < 1e-300:
        return 0.0

    return float((Phi_z - Phi_lo) / denom)


def truncated_mvn_pdf(
    x: NDArray,
    mu: NDArray,
    sigma: NDArray,
    lower: NDArray,
    upper: NDArray,
    *,
    xp=None,
) -> float:
    """Multivariate truncated normal density.

    f_T(x) = f(x) / P(lower <= X <= upper)

    where f(x) is the untruncated MVN density.

    Parameters
    ----------
    x : ndarray, shape (K,)
        Evaluation point.
    mu : ndarray, shape (K,)
        Mean vector.
    sigma : ndarray, shape (K, K)
        Covariance matrix.
    lower, upper : ndarray, shape (K,)
        Truncation bounds (-inf/inf allowed).

    Returns
    -------
    pdf : float
        Truncated MVN density at x.
    """
    if xp is None:
        xp = get_backend("numpy")

    x_np = np.asarray(x, dtype=np.float64)
    mu_np = np.asarray(mu, dtype=np.float64)
    sigma_np = np.asarray(sigma, dtype=np.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    K = len(x_np)

    # Check if x is within bounds
    for k in range(K):
        if x_np[k] < lower_np[k] or x_np[k] > upper_np[k]:
            return 0.0

    # Untruncated density
    from scipy.stats import multivariate_normal

    f_x = multivariate_normal.pdf(x_np, mean=mu_np, cov=sigma_np)

    # Normalizing constant: P(lower <= X <= upper)
    norm_const = _mvn_rect_prob(mu_np, sigma_np, lower_np, upper_np, xp=xp)

    if norm_const < 1e-300:
        return 0.0

    return float(f_x / norm_const)


def truncated_mvn_cdf(
    x: NDArray,
    mu: NDArray,
    sigma: NDArray,
    lower: NDArray,
    upper: NDArray,
    *,
    xp=None,
) -> float:
    """Multivariate truncated normal CDF.

    P_T(X <= x) = P(lower <= X <= min(x, upper)) / P(lower <= X <= upper)

    Parameters
    ----------
    x : ndarray, shape (K,)
        Upper evaluation limits.
    mu : ndarray, shape (K,)
        Mean vector.
    sigma : ndarray, shape (K, K)
        Covariance matrix.
    lower, upper : ndarray, shape (K,)
        Truncation bounds.

    Returns
    -------
    cdf : float
        P_T(X <= x)
    """
    if xp is None:
        xp = get_backend("numpy")

    x_np = np.asarray(x, dtype=np.float64)
    mu_np = np.asarray(mu, dtype=np.float64)
    sigma_np = np.asarray(sigma, dtype=np.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)

    # Clamp x to upper bounds
    x_clamped = np.minimum(x_np, upper_np)

    # Numerator: P(lower <= X <= x_clamped)
    numer = _mvn_rect_prob(mu_np, sigma_np, lower_np, x_clamped, xp=xp)

    # Denominator: P(lower <= X <= upper)
    denom = _mvn_rect_prob(mu_np, sigma_np, lower_np, upper_np, xp=xp)

    if denom < 1e-300:
        return 0.0

    return float(max(0.0, min(1.0, numer / denom)))


def _mvn_rect_prob(
    mu: np.ndarray,
    sigma: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    xp=None,
) -> float:
    """P(lower <= X <= upper) for X ~ MVN(mu, sigma).

    Uses inclusion-exclusion with MVNCD.
    P(a <= X <= b) = sum over S in {0,1}^K of (-1)^|S| * Phi(c_S)
    where c_S[k] = b[k] if S[k]=0, a[k] if S[k]=1.
    """
    if xp is None:
        xp = get_backend("numpy")

    K = len(mu)

    # For K=1, direct computation
    if K == 1:
        sd = np.sqrt(sigma[0, 0])
        p_upper = norm.cdf((upper[0] - mu[0]) / sd) if np.isfinite(upper[0]) else 1.0
        p_lower = norm.cdf((lower[0] - mu[0]) / sd) if np.isfinite(lower[0]) else 0.0
        return max(0.0, p_upper - p_lower)

    # Check for infinite bounds
    finite_lower = np.isfinite(lower)
    n_finite_lower = np.sum(finite_lower)

    if n_finite_lower == 0:
        # All lower bounds are -inf: just compute P(X <= upper)
        return mvncd(
            xp.array(upper - mu), xp.array(sigma), method="me", xp=xp
        )

    # Inclusion-exclusion over finite lower bounds
    prob = 0.0
    finite_idx = np.where(finite_lower)[0]

    for mask in range(1 << len(finite_idx)):
        c = upper.copy()
        sign = 1
        for bit, idx in enumerate(finite_idx):
            if mask & (1 << bit):
                c[idx] = lower[idx]
                sign *= -1
        p = mvncd(xp.array(c - mu), xp.array(sigma), method="me", xp=xp)
        prob += sign * p

    return max(0.0, prob)
