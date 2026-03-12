"""GGE (Gibson-Geman-Escobar) adaptive ordering for MVNCD.

Implements the adaptive reordering algorithm that selects the most
constrained variable at each step, improving the accuracy of sequential
conditioning approximations.

Reference:
    Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation
    of the MVNCD Function. Transportation Research Part B, 109: 238-256.

GAUSS reference: ordering() function in mnp_lat_TRAVELMODE.gss, line 2039.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def gge_ordering(a: NDArray, sigma: NDArray) -> NDArray:
    """Return permutation indices via GGE adaptive ordering.

    At each step, compute the standardized deviations
    z_i = (a_i - mu_trunc) / sigma_trunc for remaining variables after
    truncation, and reorder by ascending z_i (most constrained variable first).

    Parameters
    ----------
    a : ndarray, shape (m,)
        Upper integration limits.
    sigma : ndarray, shape (m, m)
        Covariance (or correlation) matrix.

    Returns
    -------
    perm : ndarray, shape (m,)
        0-based permutation indices.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64)
    m = len(a)

    if m <= 1:
        return np.arange(m, dtype=np.intp)

    # Step 1: Initial ordering by ascending a / sqrt(diag(sigma))
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-30))
    z_init = a / sd
    order = np.argsort(z_init)
    perm = order.copy()

    # Reorder a and sigma according to initial permutation
    a_work = a[perm].copy()
    sig_work = sigma[np.ix_(perm, perm)].copy()

    # Compute truncated moments and adaptively reorder for steps h=1..m-2
    # We maintain mu_cond (conditional mean) and sig_cond (conditional cov)
    mu_cond = np.zeros(m, dtype=np.float64)

    for h in range(m - 2):
        n = m - h  # remaining dimension

        if h == 0:
            # Already ordered by initial ascending z; proceed with truncation
            pass

        # --- multrunc: truncate variable 0 of the current block ---
        # Cholesky of current conditional covariance
        sig_block = sig_work[:n, :n]

        # Ensure symmetry and positive-definiteness
        sig_block = 0.5 * (sig_block + sig_block.T)
        np.fill_diagonal(sig_block, np.maximum(np.diag(sig_block), 1e-15))

        try:
            R = np.linalg.cholesky(sig_block)  # lower triangular
        except np.linalg.LinAlgError:
            # Regularize if not PD
            sig_block += np.eye(n) * 1e-10
            R = np.linalg.cholesky(sig_block)

        # Standardized truncation point for variable 0
        lam = (a_work[h] - mu_cond[0]) / R[0, 0]
        phi_lam = norm.pdf(lam)
        Phi_lam = max(norm.cdf(lam), 1e-300)
        mu_tilde = -phi_lam / Phi_lam  # E[Z | Z <= lam] for std normal

        # Truncated variance factor: Var[Z | Z <= lam] = 1 + mu_tilde*(lam + mu_tilde)
        # Note: mu_tilde < 0 always, and Var = 1 + mu_tilde*(lam - mu_tilde) would be
        # the formula with lambda = mu_tilde. Using GAUSS convention:
        # sigtilde2 = 1 + mutilde*(lam - mutilde) ... wait, GAUSS code says:
        #   sigtilde2 = 1+(mutilde*(lam-mutilde))
        # But mu_tilde = -phi/Phi < 0, so (lam - mu_tilde) = lam + |mu_tilde|.
        # This matches: Var[Z|Z<=w] = 1 + lambda*(w - lambda) where lambda = E[Z|Z<=w].
        sig_tilde2 = 1.0 + mu_tilde * (lam - mu_tilde)
        sig_tilde2 = max(sig_tilde2, 1e-15)

        # Update conditional mean and covariance via Cholesky factor
        # GAUSS: munew = mutilde | zeros(m-1,1)
        #         signew2 = (sigtilde2|zeros(m-1,1)) ~ (zeros(1,m-1)|eye(m-1))
        #         mu = mu + R' * munew
        #         sig = R' * signew2 * R
        munew = np.zeros(n, dtype=np.float64)
        munew[0] = mu_tilde

        signew2 = np.eye(n, dtype=np.float64)
        signew2[0, 0] = sig_tilde2

        # mu_updated = mu + R^T * munew (GAUSS uses rr' which is upper = R^T)
        mu_updated = mu_cond[:n] + R.T @ munew
        # sig_updated = R^T * signew2 * R
        sig_updated = R.T @ signew2 @ R

        # Strip first variable: take indices [1:n]
        mu_cond_new = mu_updated[1:]
        sig_cond_new = sig_updated[1:, 1:]

        # Remaining a values are a_work[h+1:]
        a_remain = a_work[h + 1:]
        perm_remain = perm[h + 1:].copy()

        # Compute standardized deviations for remaining variables
        sd_remain = np.sqrt(np.maximum(np.diag(sig_cond_new), 1e-30))
        z_remain = (a_remain - mu_cond_new) / sd_remain

        # Reorder remaining variables by ascending z
        sub_order = np.argsort(z_remain)

        # Apply reordering to perm, a_work, mu_cond, sig_cond
        perm[h + 1:] = perm_remain[sub_order]
        a_work[h + 1:] = a_remain[sub_order]
        mu_cond_new = mu_cond_new[sub_order]
        sig_cond_new = sig_cond_new[np.ix_(sub_order, sub_order)]

        # Store back for next iteration
        n_new = n - 1
        mu_cond = np.zeros(n_new, dtype=np.float64)
        mu_cond[:n_new] = mu_cond_new
        sig_work = np.zeros((m, m), dtype=np.float64)
        sig_work[:n_new, :n_new] = sig_cond_new

    return perm.astype(np.intp)
