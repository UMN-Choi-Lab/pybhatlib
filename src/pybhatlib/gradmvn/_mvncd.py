"""Multivariate Normal CDF approximation methods.

Implements multiple analytic approximations to the Multivariate Normal
Cumulative Distribution (MVNCD) function from:

    Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation
    of the MVNCD Function. Transportation Research Part B, 109: 238-256.

Methods:
    ME   — Sequential univariate conditioning (Section 2.3.1, O(K^2))
    OVUS — ME + bivariate screening (Section 2.3.1.1, BHATLIB default)
    OVBS — OVUS + trivariate screening (Section 2.3.1.2, O(K^3))
    BME  — Bivariate sequential conditioning (Section 2.3.2.1, O(K^2))
    TVBS — BME + quadrivariate screening (Section 2.3.2.2, O(K^2))
    SSJ  — Switzer-Solow-Joe QMC simulation (exact in limit)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal as _scipy_mvn
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend


def mvncd(
    a: NDArray,
    sigma: NDArray,
    *,
    method: str = "ovus",
    n_draws: int = 1000,
    seed: int | None = None,
    xp=None,
) -> float:
    """Compute P(X_1 <= a_1, ..., X_K <= a_K) for X ~ MVN(0, Sigma).

    Parameters
    ----------
    a : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix (must be symmetric positive-definite).
    method : str
        Approximation method:
        - "ovus": Bhat (2018) ME + bivariate screening (default, recommended)
        - "me": Bhat (2018) ME sequential conditioning
        - "bme": Bivariate sequential conditioning
        - "tvbs": BME + quadrivariate screening (most accurate analytic)
        - "ovbs": OVUS + trivariate screening
        - "ssj": QMC simulation-based (exact in limit)
        - "scipy": scipy.stats.multivariate_normal.cdf
    n_draws : int
        Number of QMC draws for SSJ method.
    seed : int or None
        Random seed for SSJ method.
    xp : backend, optional

    Returns
    -------
    prob : float
        Probability P(X <= a).
    """
    if xp is None:
        xp = array_namespace(a, sigma)

    a_np = np.asarray(xp.to_numpy(a), dtype=np.float64).ravel()
    sigma_np = np.asarray(xp.to_numpy(sigma), dtype=np.float64)
    K = len(a_np)

    if K == 0:
        return 1.0

    if K == 1:
        return float(
            _scipy_mvn.cdf(a_np[0], mean=0.0, cov=float(sigma_np[0, 0]))
        )

    if method == "scipy":
        return _mvncd_scipy(a_np, sigma_np)

    if K == 2:
        # All methods reduce to exact bivariate CDF for K=2
        return _bvn_cdf(a_np, sigma_np)

    if method == "me":
        return _mvncd_me(a_np, sigma_np)
    elif method == "ovus":
        return _mvncd_ovus(a_np, sigma_np)
    elif method == "bme":
        return _mvncd_bme(a_np, sigma_np)
    elif method == "tvbs":
        return _mvncd_tvbs(a_np, sigma_np)
    elif method == "ovbs":
        return _mvncd_ovbs(a_np, sigma_np)
    elif method == "ssj":
        from pybhatlib.gradmvn._mvncd_ssj import _mvncd_ssj
        return _mvncd_ssj(a_np, sigma_np, n_draws=n_draws, seed=seed)
    else:
        raise ValueError(f"Unknown MVNCD method: {method!r}")


def mvncd_rect(
    lower: NDArray,
    upper: NDArray,
    sigma: NDArray,
    *,
    method: str = "ovus",
    xp=None,
) -> float:
    """Compute P(lower <= X <= upper) for X ~ MVN(0, Sigma).

    Uses the inclusion-exclusion identity:
    P(a <= X <= b) = sum_{s in {0,1}^K} (-1)^{|s|} * P(X <= c_s)
    where c_s[i] = b[i] if s[i]=0 else a[i].

    Parameters
    ----------
    lower : ndarray, shape (K,)
        Lower integration limits. Use -np.inf for no lower bound.
    upper : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix.
    method : str
        MVNCD method for each vertex evaluation.
    xp : backend, optional

    Returns
    -------
    prob : float
        Probability P(lower <= X <= upper).
    """
    if xp is None:
        xp = array_namespace(lower, upper, sigma)

    lower_np = np.asarray(xp.to_numpy(lower), dtype=np.float64).ravel()
    upper_np = np.asarray(xp.to_numpy(upper), dtype=np.float64).ravel()
    sigma_np = np.asarray(xp.to_numpy(sigma), dtype=np.float64)
    K = len(lower_np)

    if K == 0:
        return 1.0

    # Check for -inf lower bounds: if all are -inf, this is just standard MVNCD
    all_neg_inf = np.all(np.isneginf(lower_np))
    if all_neg_inf:
        return mvncd(xp.array(upper_np), xp.array(sigma_np), method=method, xp=xp)

    # Inclusion-exclusion over 2^K vertices
    prob = 0.0
    for s in range(1 << K):
        c = np.zeros(K, dtype=np.float64)
        sign = 1
        for i in range(K):
            if s & (1 << i):
                c[i] = lower_np[i]
                sign *= -1
            else:
                c[i] = upper_np[i]

        # Skip if any limit is -inf (contribution is 0)
        if np.any(np.isneginf(c)):
            continue

        p = mvncd(xp.array(c), xp.array(sigma_np), method=method, xp=xp)
        prob += sign * p

    return max(0.0, min(1.0, prob))


def mvncd_batch(
    a: NDArray,
    sigma: NDArray,
    *,
    method: str = "ovus",
    xp=None,
) -> NDArray:
    """Vectorized MVNCD for N observations.

    Parameters
    ----------
    a : ndarray, shape (N, K)
        Upper integration limits for each observation.
    sigma : ndarray, shape (K, K) or (N, K, K)
        Common or per-observation covariance matrix.
    method : str
        Approximation method.
    xp : backend, optional

    Returns
    -------
    probs : ndarray, shape (N,)
        Probabilities for each observation.
    """
    if xp is None:
        xp = array_namespace(a, sigma)

    a_np = np.asarray(xp.to_numpy(a), dtype=np.float64)

    if a_np.ndim == 1:
        return xp.array([mvncd(a_np, sigma, method=method, xp=xp)])

    N = a_np.shape[0]
    sigma_np = np.asarray(xp.to_numpy(sigma), dtype=np.float64)

    probs = np.zeros(N, dtype=np.float64)
    for i in range(N):
        if sigma_np.ndim == 3:
            sig_i = sigma_np[i]
        else:
            sig_i = sigma_np
        probs[i] = mvncd(
            xp.array(a_np[i]), xp.array(sig_i), method=method, xp=xp
        )

    return xp.array(probs)


# ---------------------------------------------------------------------------
# Internal method implementations
# ---------------------------------------------------------------------------

def _mvncd_scipy(a: np.ndarray, sigma: np.ndarray) -> float:
    """Exact MVNCD using scipy (reliable for K <= ~5)."""
    try:
        result = _scipy_mvn.cdf(a, mean=np.zeros(len(a)), cov=sigma)
        return max(0.0, min(1.0, float(result)))
    except Exception:
        return _mvncd_me(a, sigma)


def _bvn_cdf(a: np.ndarray, sigma: np.ndarray) -> float:
    """Bivariate normal CDF (exact)."""
    sd1 = np.sqrt(max(sigma[0, 0], 1e-30))
    sd2 = np.sqrt(max(sigma[1, 1], 1e-30))
    rho = sigma[0, 1] / (sd1 * sd2) if sd1 > 1e-15 and sd2 > 1e-15 else 0.0
    rho = max(-0.9999, min(0.9999, rho))

    from pybhatlib.gradmvn._univariate import bivariate_normal_cdf

    return bivariate_normal_cdf(a[0] / sd1, a[1] / sd2, rho)


def _standardize(sigma: np.ndarray):
    """Extract standard deviations and correlation matrix from covariance."""
    sd = np.sqrt(np.diag(sigma))
    sd = np.maximum(sd, 1e-15)
    D_inv = np.diag(1.0 / sd)
    corr = D_inv @ sigma @ D_inv
    np.fill_diagonal(corr, 1.0)
    return sd, corr


def _reorder_by_limits(a: np.ndarray, sigma: np.ndarray):
    """Reorder variables by standardized limits (smallest first) for stability."""
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-30))
    z = a / sd
    order = np.argsort(z)
    return a[order], sigma[np.ix_(order, order)], order


# ---------------------------------------------------------------------------
# ME: Sequential univariate conditioning (Bhat 2018, Section 2.3.1)
# ---------------------------------------------------------------------------

def _mvncd_me(a: np.ndarray, sigma: np.ndarray) -> float:
    """Bhat (2018) ME analytic approximation to MVNCD."""
    K = len(a)

    if K == 1:
        sd = np.sqrt(sigma[0, 0])
        return float(norm.cdf(a[0] / sd))

    if K == 2:
        return _bvn_cdf(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)
    prob = _sequential_conditioning(a_ord, sigma_ord)
    return max(0.0, min(1.0, prob))


def _sequential_conditioning(a: np.ndarray, sigma: np.ndarray) -> float:
    """Sequential conditioning approach for MVNCD (ME method core)."""
    K = len(a)

    sd = np.sqrt(sigma[0, 0])
    if sd < 1e-15:
        return 0.0

    prob = float(norm.cdf(a[0] / sd))
    if prob < 1e-300 or K == 1:
        return max(0.0, prob)

    mu_cond = np.zeros(K)
    sigma_cond = sigma.copy()

    for k in range(1, K):
        sigma_11 = sigma_cond[:k, :k]
        sigma_12 = sigma_cond[:k, k]
        sigma_22 = sigma_cond[k, k]

        if np.linalg.det(sigma_11) < 1e-30:
            sigma_11 = sigma_11 + np.eye(k) * 1e-10

        sigma_11_inv = np.linalg.solve(sigma_11, np.eye(k))
        cond_var = sigma_22 - sigma_12 @ sigma_11_inv @ sigma_12
        cond_var = max(cond_var, 1e-15)
        cond_sd = np.sqrt(cond_var)

        trunc_means = np.zeros(k)
        for j in range(k):
            sd_j = np.sqrt(max(sigma_cond[j, j], 1e-30))
            alpha_j = (a[j] - mu_cond[j]) / sd_j
            phi_alpha = norm.pdf(alpha_j)
            Phi_alpha = norm.cdf(alpha_j)
            if Phi_alpha > 1e-300:
                trunc_means[j] = mu_cond[j] - sd_j * phi_alpha / Phi_alpha
            else:
                trunc_means[j] = a[j]

        cond_mean = mu_cond[k] + sigma_12 @ sigma_11_inv @ (trunc_means - mu_cond[:k])

        z_k = (a[k] - cond_mean) / cond_sd
        p_k = float(norm.cdf(z_k))
        prob *= max(1e-300, p_k)

        if prob < 1e-300:
            return 0.0

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# OVUS: ME + bivariate screening (Bhat 2018, Section 2.3.1.1)
# ---------------------------------------------------------------------------

def _mvncd_ovus(a: np.ndarray, sigma: np.ndarray) -> float:
    """OVUS: Sequential conditioning with bivariate screening.

    At each step k, instead of just using the univariate conditional CDF,
    screen the remaining variables to find the one most correlated with X_k,
    and use the bivariate CDF for the pair for improved accuracy.
    """
    from pybhatlib.gradmvn._univariate import bivariate_normal_cdf

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    sd = np.sqrt(np.maximum(np.diag(sigma_ord), 1e-30))
    prob = 1.0

    mu = np.zeros(K)
    cov = sigma_ord.copy()

    processed = np.zeros(K, dtype=bool)

    for step in range(K):
        # Find next unprocessed variable with smallest standardized limit
        remaining = np.where(~processed)[0]
        if len(remaining) == 0:
            break

        # Select variable with smallest (a - mu)/sd
        z_vals = np.array([
            (a_ord[i] - mu[i]) / np.sqrt(max(cov[i, i], 1e-30))
            for i in remaining
        ])
        k = remaining[np.argmin(z_vals)]
        processed[k] = True

        sd_k = np.sqrt(max(cov[k, k], 1e-30))
        z_k = (a_ord[k] - mu[k]) / sd_k

        remaining_after = np.where(~processed)[0]

        if len(remaining_after) == 0:
            # Last variable: univariate
            p_k = float(norm.cdf(z_k))
            prob *= max(1e-300, p_k)
        else:
            # Screen for most correlated remaining variable
            best_j = None
            best_abs_corr = -1.0
            for j in remaining_after:
                sd_j = np.sqrt(max(cov[j, j], 1e-30))
                corr_kj = cov[k, j] / (sd_k * sd_j)
                if abs(corr_kj) > best_abs_corr:
                    best_abs_corr = abs(corr_kj)
                    best_j = j

            if best_j is not None and best_abs_corr > 0.01:
                # Use bivariate CDF for (k, best_j)
                sd_j = np.sqrt(max(cov[best_j, best_j], 1e-30))
                z_j = (a_ord[best_j] - mu[best_j]) / sd_j
                rho_kj = cov[k, best_j] / (sd_k * sd_j)
                rho_kj = max(-0.9999, min(0.9999, rho_kj))

                bvn = bivariate_normal_cdf(z_k, z_j, rho_kj)
                p_marginal_j = float(norm.cdf(z_j))

                if p_marginal_j > 1e-300:
                    # P(k | screening) = BVN(k,j) / P(j)
                    p_k = bvn / p_marginal_j
                else:
                    p_k = float(norm.cdf(z_k))
            else:
                p_k = float(norm.cdf(z_k))

            prob *= max(1e-300, p_k)

        if prob < 1e-300:
            return 0.0

        # Update conditional moments for remaining variables
        for j in np.where(~processed)[0]:
            sd_k_sq = max(cov[k, k], 1e-30)
            # Conditional mean update using truncated moment
            phi_k = norm.pdf(z_k)
            Phi_k = norm.cdf(z_k)
            if Phi_k > 1e-300:
                E_trunc_k = mu[k] - np.sqrt(sd_k_sq) * phi_k / Phi_k
            else:
                E_trunc_k = a_ord[k]

            mu[j] += cov[k, j] / sd_k_sq * (E_trunc_k - mu[k])

            # Conditional variance update (Schur complement-like)
            # Also account for variance reduction from truncation
            V_trunc_k = sd_k_sq * (1.0 - z_k * phi_k / max(Phi_k, 1e-300)
                                    - (phi_k / max(Phi_k, 1e-300))**2)
            V_trunc_k = max(V_trunc_k, 1e-15)

            for j2 in np.where(~processed)[0]:
                cov[j, j2] -= cov[k, j] * cov[k, j2] / sd_k_sq
                # Adjust for truncation variance
                cov[j, j2] += cov[k, j] * cov[k, j2] / sd_k_sq**2 * (V_trunc_k - sd_k_sq)

        # Zero out row/col k
        cov[k, :] = 0.0
        cov[:, k] = 0.0

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# BME: Bivariate sequential conditioning (Bhat 2018, Section 2.3.2.1)
# ---------------------------------------------------------------------------

def _mvncd_bme(a: np.ndarray, sigma: np.ndarray) -> float:
    """BME: Process variables in pairs with bivariate conditioning.

    Instead of conditioning one variable at a time (ME), process pairs
    (1,2), (3,4), ... using bivariate CDFs and bivariate truncated moments.
    """
    from pybhatlib.gradmvn._univariate import bivariate_normal_cdf
    from pybhatlib.gradmvn._bivariate_trunc import (
        truncated_bivariate_cov,
        truncated_bivariate_mean,
    )

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    prob = 1.0
    mu = np.zeros(K)
    cov = sigma_ord.copy()

    k = 0
    while k < K:
        if k + 1 < K:
            # Process pair (k, k+1)
            sd_k = np.sqrt(max(cov[k, k], 1e-30))
            sd_k1 = np.sqrt(max(cov[k + 1, k + 1], 1e-30))

            z_k = (a_ord[k] - mu[k]) / sd_k
            z_k1 = (a_ord[k + 1] - mu[k + 1]) / sd_k1

            rho = cov[k, k + 1] / (sd_k * sd_k1)
            rho = max(-0.9999, min(0.9999, rho))

            # Bivariate CDF for the pair
            bvn = bivariate_normal_cdf(z_k, z_k1, rho)
            prob *= max(1e-300, bvn)

            if prob < 1e-300:
                return 0.0

            # Compute truncated bivariate moments for conditioning
            mu_pair = np.array([mu[k], mu[k + 1]])
            cov_pair = np.array([
                [cov[k, k], cov[k, k + 1]],
                [cov[k + 1, k], cov[k + 1, k + 1]],
            ])
            a_pair = np.array([a_ord[k], a_ord[k + 1]])

            trunc_mu = truncated_bivariate_mean(mu_pair, cov_pair, a_pair)
            trunc_cov = truncated_bivariate_cov(mu_pair, cov_pair, a_pair)

            # Update remaining variables using bivariate conditioning
            pair_idx = np.array([k, k + 1])
            remaining = np.arange(k + 2, K)

            if len(remaining) > 0:
                cov_pair_safe = cov_pair.copy()
                det = np.linalg.det(cov_pair_safe)
                if abs(det) < 1e-30:
                    cov_pair_safe += np.eye(2) * 1e-10
                cov_pair_inv = np.linalg.inv(cov_pair_safe)

                for j in remaining:
                    cov_j_pair = np.array([cov[k, j], cov[k + 1, j]])
                    # Conditional mean
                    mu[j] += cov_j_pair @ cov_pair_inv @ (trunc_mu - mu_pair)

                    # Conditional covariance update
                    for j2 in remaining:
                        cov_j2_pair = np.array([cov[k, j2], cov[k + 1, j2]])
                        # Standard Schur complement
                        cov[j, j2] -= cov_j_pair @ cov_pair_inv @ cov_j2_pair
                        # Truncation correction
                        cov[j, j2] += cov_j_pair @ cov_pair_inv @ trunc_cov @ cov_pair_inv @ cov_j2_pair

            k += 2
        else:
            # Odd variable: univariate conditioning (like ME)
            sd_k = np.sqrt(max(cov[k, k], 1e-30))
            z_k = (a_ord[k] - mu[k]) / sd_k
            p_k = float(norm.cdf(z_k))
            prob *= max(1e-300, p_k)
            k += 1

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# TVBS: BME + quadrivariate screening (Bhat 2018, Section 2.3.2.2)
# ---------------------------------------------------------------------------

def _mvncd_tvbs(a: np.ndarray, sigma: np.ndarray) -> float:
    """TVBS: Bivariate conditioning with quadrivariate screening.

    Like BME, but after each bivariate conditioning step, screen remaining
    pairs and compute a quadrivariate CDF for improved accuracy.
    """
    from pybhatlib.gradmvn._univariate import (
        bivariate_normal_cdf,
        quadrivariate_normal_cdf,
    )
    from pybhatlib.gradmvn._bivariate_trunc import (
        truncated_bivariate_cov,
        truncated_bivariate_mean,
    )

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    if K == 3:
        # For K=3, TVBS uses trivariate + screening = effectively BME + extra
        return _mvncd_bme(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    prob = 1.0
    mu = np.zeros(K)
    cov = sigma_ord.copy()

    k = 0
    while k < K:
        if k + 1 < K:
            # Process pair (k, k+1)
            sd_k = np.sqrt(max(cov[k, k], 1e-30))
            sd_k1 = np.sqrt(max(cov[k + 1, k + 1], 1e-30))

            z_k = (a_ord[k] - mu[k]) / sd_k
            z_k1 = (a_ord[k + 1] - mu[k + 1]) / sd_k1

            rho_kk1 = cov[k, k + 1] / (sd_k * sd_k1)
            rho_kk1 = max(-0.9999, min(0.9999, rho_kk1))

            bvn_kk1 = bivariate_normal_cdf(z_k, z_k1, rho_kk1)

            # Screen for best remaining pair
            remaining = np.arange(k + 2, K)
            if len(remaining) >= 2:
                # Find the pair in remaining with strongest correlation to current pair
                best_pair = None
                best_score = -1.0
                for i_idx in range(len(remaining)):
                    for j_idx in range(i_idx + 1, len(remaining)):
                        ri = remaining[i_idx]
                        rj = remaining[j_idx]
                        sd_ri = np.sqrt(max(cov[ri, ri], 1e-30))
                        sd_rj = np.sqrt(max(cov[rj, rj], 1e-30))
                        # Score by correlation magnitude to current pair
                        score = (abs(cov[k, ri]) / (sd_k * sd_ri)
                                 + abs(cov[k, rj]) / (sd_k * sd_rj)
                                 + abs(cov[k + 1, ri]) / (sd_k1 * sd_ri)
                                 + abs(cov[k + 1, rj]) / (sd_k1 * sd_rj))
                        if score > best_score:
                            best_score = score
                            best_pair = (ri, rj)

                if best_pair is not None and best_score > 0.04:
                    ri, rj = best_pair
                    sd_ri = np.sqrt(max(cov[ri, ri], 1e-30))
                    sd_rj = np.sqrt(max(cov[rj, rj], 1e-30))
                    z_ri = (a_ord[ri] - mu[ri]) / sd_ri
                    z_rj = (a_ord[rj] - mu[rj]) / sd_rj

                    # Build 4x4 correlation matrix for quadrivariate CDF
                    indices = [k, k + 1, ri, rj]
                    sds = np.array([sd_k, sd_k1, sd_ri, sd_rj])
                    zs = np.array([z_k, z_k1, z_ri, z_rj])

                    corr_4 = np.eye(4)
                    for ii in range(4):
                        for jj in range(ii + 1, 4):
                            r = cov[indices[ii], indices[jj]] / (sds[ii] * sds[jj])
                            r = max(-0.9999, min(0.9999, r))
                            corr_4[ii, jj] = r
                            corr_4[jj, ii] = r

                    qvn = quadrivariate_normal_cdf(zs[0], zs[1], zs[2], zs[3], corr_4)

                    # Bivariate CDF for the screening pair
                    rho_rirj = cov[ri, rj] / (sd_ri * sd_rj)
                    rho_rirj = max(-0.9999, min(0.9999, rho_rirj))
                    bvn_rirj = bivariate_normal_cdf(z_ri, z_rj, rho_rirj)

                    # Screening adjustment: P(k,k+1) ~ QVN / BVN(ri,rj)
                    if bvn_rirj > 1e-300:
                        p_pair = qvn / bvn_rirj
                    else:
                        p_pair = bvn_kk1
                else:
                    p_pair = bvn_kk1
            else:
                p_pair = bvn_kk1

            prob *= max(1e-300, p_pair)

            if prob < 1e-300:
                return 0.0

            # Update conditional moments using bivariate truncated moments
            mu_pair = np.array([mu[k], mu[k + 1]])
            cov_pair = np.array([
                [cov[k, k], cov[k, k + 1]],
                [cov[k + 1, k], cov[k + 1, k + 1]],
            ])
            a_pair = np.array([a_ord[k], a_ord[k + 1]])

            trunc_mu = truncated_bivariate_mean(mu_pair, cov_pair, a_pair)
            trunc_cov = truncated_bivariate_cov(mu_pair, cov_pair, a_pair)

            remaining = np.arange(k + 2, K)
            if len(remaining) > 0:
                cov_pair_safe = cov_pair.copy()
                det = np.linalg.det(cov_pair_safe)
                if abs(det) < 1e-30:
                    cov_pair_safe += np.eye(2) * 1e-10
                cov_pair_inv = np.linalg.inv(cov_pair_safe)

                for j in remaining:
                    cov_j_pair = np.array([cov[k, j], cov[k + 1, j]])
                    mu[j] += cov_j_pair @ cov_pair_inv @ (trunc_mu - mu_pair)

                    for j2 in remaining:
                        cov_j2_pair = np.array([cov[k, j2], cov[k + 1, j2]])
                        cov[j, j2] -= cov_j_pair @ cov_pair_inv @ cov_j2_pair
                        cov[j, j2] += cov_j_pair @ cov_pair_inv @ trunc_cov @ cov_pair_inv @ cov_j2_pair

            k += 2
        else:
            sd_k = np.sqrt(max(cov[k, k], 1e-30))
            z_k = (a_ord[k] - mu[k]) / sd_k
            p_k = float(norm.cdf(z_k))
            prob *= max(1e-300, p_k)
            k += 1

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# OVBS: OVUS + trivariate screening (Bhat 2018, Section 2.3.1.2)
# ---------------------------------------------------------------------------

def _mvncd_ovbs(a: np.ndarray, sigma: np.ndarray) -> float:
    """OVBS: Sequential conditioning with trivariate screening.

    Like OVUS, but at each step screen for the two most correlated
    remaining variables and compute trivariate CDF for the triple.
    """
    from pybhatlib.gradmvn._univariate import (
        bivariate_normal_cdf,
        trivariate_normal_cdf,
    )

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    prob = 1.0
    mu = np.zeros(K)
    cov = sigma_ord.copy()

    processed = np.zeros(K, dtype=bool)

    for step in range(K):
        remaining = np.where(~processed)[0]
        if len(remaining) == 0:
            break

        # Select variable with smallest standardized limit
        z_vals = np.array([
            (a_ord[i] - mu[i]) / np.sqrt(max(cov[i, i], 1e-30))
            for i in remaining
        ])
        k = remaining[np.argmin(z_vals)]
        processed[k] = True

        sd_k = np.sqrt(max(cov[k, k], 1e-30))
        z_k = (a_ord[k] - mu[k]) / sd_k

        remaining_after = np.where(~processed)[0]

        if len(remaining_after) >= 2:
            # Find the two most correlated remaining variables
            corr_vals = []
            for j in remaining_after:
                sd_j = np.sqrt(max(cov[j, j], 1e-30))
                corr_kj = abs(cov[k, j]) / (sd_k * sd_j)
                corr_vals.append((corr_kj, j))
            corr_vals.sort(reverse=True)

            j1 = corr_vals[0][1]
            j2 = corr_vals[1][1]

            sd_j1 = np.sqrt(max(cov[j1, j1], 1e-30))
            sd_j2 = np.sqrt(max(cov[j2, j2], 1e-30))
            z_j1 = (a_ord[j1] - mu[j1]) / sd_j1
            z_j2 = (a_ord[j2] - mu[j2]) / sd_j2

            # Build 3x3 correlation matrix
            corr_3 = np.eye(3)
            indices_3 = [k, j1, j2]
            sds_3 = [sd_k, sd_j1, sd_j2]
            for ii in range(3):
                for jj in range(ii + 1, 3):
                    r = cov[indices_3[ii], indices_3[jj]] / (sds_3[ii] * sds_3[jj])
                    r = max(-0.9999, min(0.9999, r))
                    corr_3[ii, jj] = r
                    corr_3[jj, ii] = r

            tvn = trivariate_normal_cdf(z_k, z_j1, z_j2, corr_3)

            # Screening: P(k) = TVN / BVN(j1, j2)
            rho_j1j2 = cov[j1, j2] / (sd_j1 * sd_j2)
            rho_j1j2 = max(-0.9999, min(0.9999, rho_j1j2))
            bvn_j1j2 = bivariate_normal_cdf(z_j1, z_j2, rho_j1j2)

            if bvn_j1j2 > 1e-300:
                p_k = tvn / bvn_j1j2
            else:
                p_k = float(norm.cdf(z_k))

        elif len(remaining_after) == 1:
            # One remaining: use bivariate screening (like OVUS)
            j = remaining_after[0]
            sd_j = np.sqrt(max(cov[j, j], 1e-30))
            z_j = (a_ord[j] - mu[j]) / sd_j
            rho_kj = cov[k, j] / (sd_k * sd_j)
            rho_kj = max(-0.9999, min(0.9999, rho_kj))

            if abs(rho_kj) > 0.01:
                bvn = bivariate_normal_cdf(z_k, z_j, rho_kj)
                p_marginal_j = float(norm.cdf(z_j))
                if p_marginal_j > 1e-300:
                    p_k = bvn / p_marginal_j
                else:
                    p_k = float(norm.cdf(z_k))
            else:
                p_k = float(norm.cdf(z_k))
        else:
            p_k = float(norm.cdf(z_k))

        prob *= max(1e-300, p_k)

        if prob < 1e-300:
            return 0.0

        # Update conditional moments
        for j in np.where(~processed)[0]:
            sd_k_sq = max(cov[k, k], 1e-30)
            phi_k = norm.pdf(z_k)
            Phi_k = norm.cdf(z_k)
            if Phi_k > 1e-300:
                E_trunc_k = mu[k] - np.sqrt(sd_k_sq) * phi_k / Phi_k
            else:
                E_trunc_k = a_ord[k]

            mu[j] += cov[k, j] / sd_k_sq * (E_trunc_k - mu[k])

            V_trunc_k = sd_k_sq * (1.0 - z_k * phi_k / max(Phi_k, 1e-300)
                                    - (phi_k / max(Phi_k, 1e-300))**2)
            V_trunc_k = max(V_trunc_k, 1e-15)

            for j2 in np.where(~processed)[0]:
                cov[j, j2] -= cov[k, j] * cov[k, j2] / sd_k_sq
                cov[j, j2] += cov[k, j] * cov[k, j2] / sd_k_sq**2 * (V_trunc_k - sd_k_sq)

        cov[k, :] = 0.0
        cov[:, k] = 0.0

    return max(0.0, min(1.0, prob))
