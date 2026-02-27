"""Partial cumulative normal distribution functions.

Computes mixed point/interval probabilities where some variates are
evaluated at specific points and others are integrated over ranges,
with optional truncation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend
from pybhatlib.gradmvn._mvncd import mvncd


def partial_mvn_cdf(
    points: NDArray | None,
    lower: NDArray,
    upper: NDArray,
    mu: NDArray,
    sigma: NDArray,
    point_indices: list[int] | None = None,
    range_indices: list[int] | None = None,
    *,
    xp=None,
) -> float:
    """Compute mixed point/interval probability for MVN distribution.

    For X ~ MVN(mu, sigma), computes:
    P(X[point_indices] = points, lower[range_indices] <= X[range_indices] <= upper[range_indices])

    The point evaluations are density evaluations, and the range evaluations
    are CDF evaluations.

    Parameters
    ----------
    points : ndarray or None
        Values at which point-evaluated variables are fixed.
    lower, upper : ndarray, shape (K,)
        Integration bounds for range-evaluated variables (-inf/inf OK).
    mu : ndarray, shape (K,)
        Mean vector.
    sigma : ndarray, shape (K, K)
        Covariance matrix.
    point_indices : list of int or None
        Indices of point-evaluated variables.
    range_indices : list of int or None
        Indices of range-evaluated variables.
    xp : backend, optional

    Returns
    -------
    prob : float
        Mixed probability value.
    """
    if xp is None:
        xp = get_backend("numpy")

    mu_np = np.asarray(mu, dtype=np.float64)
    sigma_np = np.asarray(sigma, dtype=np.float64)
    lower_np = np.asarray(lower, dtype=np.float64)
    upper_np = np.asarray(upper, dtype=np.float64)
    K = len(mu_np)

    if point_indices is None or len(point_indices) == 0:
        # Pure range evaluation
        from pybhatlib.gradmvn._truncated import _mvn_rect_prob
        return _mvn_rect_prob(mu_np, sigma_np, lower_np, upper_np, xp=xp)

    points_np = np.asarray(points, dtype=np.float64)

    if range_indices is None:
        range_indices = [i for i in range(K) if i not in point_indices]

    if len(range_indices) == 0:
        # Pure point evaluation (density)
        from scipy.stats import multivariate_normal
        return float(multivariate_normal.pdf(points_np, mean=mu_np, cov=sigma_np))

    # Mixed: condition range variables on point variables
    # X = [X_p, X_r] where p = point indices, r = range indices
    p_idx = np.array(point_indices)
    r_idx = np.array(range_indices)

    mu_p = mu_np[p_idx]
    mu_r = mu_np[r_idx]
    sigma_pp = sigma_np[np.ix_(p_idx, p_idx)]
    sigma_rr = sigma_np[np.ix_(r_idx, r_idx)]
    sigma_rp = sigma_np[np.ix_(r_idx, p_idx)]

    # Density of point variables
    from scipy.stats import multivariate_normal

    if len(p_idx) == 1:
        f_p = norm.pdf(points_np[0], loc=mu_p[0], scale=np.sqrt(sigma_pp[0, 0]))
    else:
        f_p = multivariate_normal.pdf(points_np, mean=mu_p, cov=sigma_pp)

    # Conditional distribution of range variables given point variables
    sigma_pp_inv = np.linalg.solve(sigma_pp, np.eye(len(p_idx)))
    mu_r_cond = mu_r + sigma_rp @ sigma_pp_inv @ (points_np - mu_p)
    sigma_r_cond = sigma_rr - sigma_rp @ sigma_pp_inv @ sigma_rp.T

    # Symmetrize
    sigma_r_cond = 0.5 * (sigma_r_cond + sigma_r_cond.T)

    # CDF of range variables under conditional distribution
    lower_r = lower_np[r_idx]
    upper_r = upper_np[r_idx]

    from pybhatlib.gradmvn._truncated import _mvn_rect_prob

    p_range = _mvn_rect_prob(mu_r_cond, sigma_r_cond, lower_r, upper_r, xp=xp)

    return float(f_p * p_range)
