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

All analytic methods use LDLT decomposition with rank-1/rank-2 updates
(Properties 3-4) for efficient O(H) sequential conditioning.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal as _scipy_mvn
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend
from pybhatlib.vecup._ldlt import ldlt_decompose, ldlt_rank1_update, ldlt_rank2_update


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
# Internal helpers
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


def _reorder_by_limits(a: np.ndarray, sigma: np.ndarray):
    """Reorder variables by |a_k / sqrt(Sigma_kk)| ascending for stability."""
    sd = np.sqrt(np.maximum(np.diag(sigma), 1e-30))
    z = np.abs(a / sd)
    order = np.argsort(z)
    return a[order], sigma[np.ix_(order, order)], order


def _extract_subcov(L: np.ndarray, D: np.ndarray, k: int) -> np.ndarray:
    """Extract k x k covariance sub-matrix from LDLT factors.

    Cov[0:k, 0:k] = L[0:k,0:k] @ diag(D[0:k]) @ L[0:k,0:k].T

    Parameters
    ----------
    L : ndarray, shape (H, H)
        Lower triangular with unit diagonal.
    D : ndarray, shape (H,)
        Diagonal elements.
    k : int
        Size of sub-matrix to extract.

    Returns
    -------
    cov : ndarray, shape (k, k)
    """
    Lk = L[:k, :k]
    return Lk @ np.diag(D[:k]) @ Lk.T


# ---------------------------------------------------------------------------
# ME: Sequential univariate conditioning (Bhat 2018, Property 3-4)
# ---------------------------------------------------------------------------

def _mvncd_me(a: np.ndarray, sigma: np.ndarray) -> float:
    """Bhat (2018) ME analytic approximation using LDLT decomposition.

    Algorithm (Property 4, p. 243-244):
    1. Reorder variables by |a_k / sqrt(Sigma_kk)| ascending
    2. LDLT decompose reordered Sigma -> (L, D)
    3. P = 1; pi = zeros(H)
    4. For h = 0..H-1:
       - sigma_h = D[0]
       - w_h = (a[0] - pi[0]) / sqrt(sigma_h)
       - P *= Phi(w_h)
       - mu_tilde = pi[0] + sqrt(sigma_h) * E[Z | Z <= w_h]
       - Omega_h = sigma_h * Var[Z | Z <= w_h]
       - pi[1:] += L[1:,0] * D[0] * (mu_tilde - pi[0]) / D[0]
                  = L[1:,0] * (mu_tilde - pi[0])
       - LDLT rank-1 update for Omega_h - D[0]
       - Trim: remove first row/col
    """
    K = len(a)

    if K == 1:
        sd = np.sqrt(sigma[0, 0])
        return float(norm.cdf(a[0] / sd))

    if K == 2:
        return _bvn_cdf(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    # LDLT decompose
    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    prob = 1.0
    pi = np.zeros(K, dtype=np.float64)
    H = K

    for h in range(H):
        n = H - h  # remaining dimension

        sigma_h = max(D[0], 1e-15)
        sd_h = np.sqrt(sigma_h)

        w_h = (a_ord[h] - pi[0]) / sd_h
        p_h = float(norm.cdf(w_h))
        prob *= max(1e-300, p_h)

        if prob < 1e-300:
            return 0.0

        if n <= 1:
            break

        # Truncated moments E[Z|Z<=w] and Var[Z|Z<=w] for standard normal
        phi_w = norm.pdf(w_h)
        Phi_w = max(norm.cdf(w_h), 1e-300)
        lambda_h = -phi_w / Phi_w  # E[Z | Z <= w_h]
        # Var[Z | Z <= w_h] = 1 - w_h * (phi/Phi) - (phi/Phi)^2
        #                    = 1 + w_h * lambda - lambda^2
        #                    = 1 + lambda * (w_h - lambda)
        var_z = 1.0 + lambda_h * (w_h - lambda_h)
        var_z = max(var_z, 1e-15)

        mu_tilde = pi[0] + sd_h * lambda_h
        Omega_h = sigma_h * var_z

        # Update pi for remaining variables
        # pi_j += L[j,0] * D[0] * (mu_tilde - pi[0]) / D[0]
        #       = L[j,0] * (mu_tilde - pi[0])
        shift = mu_tilde - pi[0]
        for j in range(1, n):
            pi[j] += L[j, 0] * shift

        # LDLT rank-1 update: A_new = A + alpha * v v^T
        # The sub-LDLT (L_sub @ D_sub @ L_sub^T) does NOT include the
        # column-0 contribution D[0]*v*v^T from the full matrix.
        # So the conditional covariance is: L_sub@D_sub@L_sub^T + Omega_h*v*v^T
        # Hence alpha = Omega_h (not Omega_h - D[0]).
        alpha = Omega_h
        v = L[1:n, 0].copy()

        L_sub = L[1:n, 1:n].copy()
        D_sub = D[1:n].copy()

        L_sub, D_sub = ldlt_rank1_update(L_sub, D_sub, v, alpha)

        # Trim: shift arrays
        a_ord_rest = a_ord[h + 1:]
        pi_new = pi[1:n].copy()

        # Prepare for next iteration
        n_new = n - 1
        L = np.eye(n_new, dtype=np.float64)
        L[:n_new, :n_new] = L_sub[:n_new, :n_new]
        D = D_sub[:n_new].copy()
        pi = np.zeros(n_new, dtype=np.float64)
        pi[:n_new] = pi_new[:n_new]

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# OVUS: ME + bivariate screening (Bhat 2018, Section 2.3.1.1)
# ---------------------------------------------------------------------------

def _univariate_truncate_and_update(
    L: np.ndarray, D: np.ndarray, pi: np.ndarray,
    w_h: float, sigma_h: float, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Univariate truncation of first variable + LDLT rank-1 update + trim.

    Returns updated (L, D, pi) with dimension reduced by 1.
    """
    sd_h = np.sqrt(sigma_h)
    phi_w = norm.pdf(w_h)
    Phi_w = max(norm.cdf(w_h), 1e-300)
    lambda_h = -phi_w / Phi_w
    var_z = max(1.0 + lambda_h * (w_h - lambda_h), 1e-15)

    mu_tilde = pi[0] + sd_h * lambda_h
    Omega_h = sigma_h * var_z

    shift = mu_tilde - pi[0]
    for j in range(1, n):
        pi[j] += L[j, 0] * shift

    alpha = Omega_h  # Sub-LDLT excludes D[0]*v*v^T, so alpha = Omega_h
    v = L[1:n, 0].copy()
    L_sub, D_sub = ldlt_rank1_update(L[1:n, 1:n].copy(), D[1:n].copy(), v, alpha)

    n_new = n - 1
    L_new = np.eye(n_new, dtype=np.float64)
    L_new[:n_new, :n_new] = L_sub[:n_new, :n_new]
    D_new = D_sub[:n_new].copy()
    pi_new = pi[1:n].copy()
    pi_out = np.zeros(n_new, dtype=np.float64)
    pi_out[:n_new] = pi_new[:n_new]

    return L_new, D_new, pi_out


def _mvncd_ovus(a: np.ndarray, sigma: np.ndarray) -> float:
    """OVUS: ME with bivariate screening (Bhat 2018, Section 2.3.1.1).

    Paper pseudocode (p. 244-245):
    (1) P_1 = Phi_2(w_1, w_2; rho_12). If H=2, STOP.
    (2) LDLT decompose Sigma.
    (3) For h = 1 to H-2:
    (4)   Truncate X_h univariately, rank-1 update.
    (5)   LDLT rank-1 update of Sigma_{h+1}.
    (6)   Extract 2x2 sub-cov of (Y_{h+1}, Y_{h+2}).
          P_{h+1} = BVN(w_{h+1}', w_{h+2}'; rho') / Phi(w_{h+2}')
    (7) End for
    (8) Return P_1 * prod(P_{h+1}).
    """
    from pybhatlib.gradmvn._univariate import bivariate_normal_cdf

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    pi = np.zeros(K, dtype=np.float64)

    # Step 1: P_1 = Phi_2(w_1, w_2; rho_12)
    w_0, w_1, rho_01 = _get_bvn_params(L, D, pi, a_ord, 0)
    prob = bivariate_normal_cdf(w_0, w_1, rho_01)
    if prob < 1e-300:
        return 0.0

    # Steps 3-7: For h = 1 to H-2 (0-indexed: h_idx = 0 to K-3)
    # At each step: univariate truncation of first variable, then BVN screening
    for h_idx in range(K - 2):
        n = K - h_idx  # remaining dimensions before truncation

        sigma_h = max(D[0], 1e-15)
        sd_h = np.sqrt(sigma_h)
        w_h = (a_ord[h_idx] - pi[0]) / sd_h

        # Univariate truncation + rank-1 update (reduces dimension by 1)
        L, D, pi = _univariate_truncate_and_update(L, D, pi, w_h, sigma_h, n)

        # After truncation, extract 2x2 sub-cov for screening
        n_new = n - 1  # remaining after truncation
        if n_new >= 2:
            cov2 = _extract_subcov(L[:2, :2], D[:2], 2)
            sd_0 = np.sqrt(max(cov2[0, 0], 1e-15))
            sd_1 = np.sqrt(max(cov2[1, 1], 1e-15))
            rho = cov2[0, 1] / (sd_0 * sd_1) if sd_0 > 1e-15 and sd_1 > 1e-15 else 0.0
            rho = max(-0.9999, min(0.9999, rho))
            w_next_0 = (a_ord[h_idx + 1] - pi[0]) / sd_0
            w_next_1 = (a_ord[h_idx + 2] - pi[1]) / sd_1
            bvn = bivariate_normal_cdf(w_next_0, w_next_1, rho)
            # Denominator is Phi of FIRST variable (the one being estimated),
            # not the second (the screened variable). See paper step (1):
            # P_1 = Phi(w_1) * BVN(w_1,w_2)/Phi(w_1) — denominator is Phi(w_1).
            p_denom = float(norm.cdf(w_next_0))
            p_h = bvn / p_denom if p_denom > 1e-300 else float(norm.cdf(w_next_0))
        else:
            # Only 1 variable left — just Phi
            w_last = (a_ord[h_idx + 1] - pi[0]) / np.sqrt(max(D[0], 1e-15))
            p_h = float(norm.cdf(w_last))

        prob *= max(1e-300, p_h)
        if prob < 1e-300:
            return 0.0

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# BME: Bivariate sequential conditioning (Bhat 2018, Section 2.3.2.1)
# ---------------------------------------------------------------------------

def _mvncd_bme(a: np.ndarray, sigma: np.ndarray) -> float:
    """BME: Process variables in pairs with bivariate conditioning.

    Algorithm (p. 246):
    1. Reorder by |a_k/sqrt(Sigma_kk)| ascending
    2. LDLT decompose -> (L, D)
    3. Process pairs: for h = 0, 2, 4, ...:
       - Extract 2x2 sub-cov from LDLT
       - P *= Phi_2(w_1, w_2; rho)
       - Bivariate truncated moments (Property 1)
       - Update pi using 2-variable conditioning
       - LDLT rank-2 update
    4. If H odd, last variable: P *= Phi(w_last), rank-1 update
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

    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    prob = 1.0
    pi = np.zeros(K, dtype=np.float64)
    h_idx = 0  # index into a_ord

    while h_idx < K:
        n = K - h_idx  # remaining dimensions

        if n >= 2:
            # Extract 2x2 sub-covariance from LDLT
            cov2 = _extract_subcov(L[:2, :2], D[:2], 2)

            sd_0 = np.sqrt(max(cov2[0, 0], 1e-15))
            sd_1 = np.sqrt(max(cov2[1, 1], 1e-15))
            rho = cov2[0, 1] / (sd_0 * sd_1) if sd_0 > 1e-15 and sd_1 > 1e-15 else 0.0
            rho = max(-0.9999, min(0.9999, rho))

            w_0 = (a_ord[h_idx] - pi[0]) / sd_0
            w_1 = (a_ord[h_idx + 1] - pi[1]) / sd_1

            bvn = bivariate_normal_cdf(w_0, w_1, rho)
            prob *= max(1e-300, bvn)

            if prob < 1e-300:
                return 0.0

            if n <= 2:
                break

            # Bivariate truncated moments for conditioning
            mu_pair = np.array([pi[0], pi[1]])
            a_pair = np.array([a_ord[h_idx], a_ord[h_idx + 1]])

            trunc_mu = truncated_bivariate_mean(mu_pair, cov2, a_pair)
            trunc_cov = truncated_bivariate_cov(mu_pair, cov2, a_pair)

            # Update pi for remaining variables (j >= 2)
            # pi_j += sum_k L[j,k]*D[k] * sum_i (L^{-1})_{ki} * (trunc_mu_i - pi_i)
            # Simplified: pi_j += C[j, 0:2] @ inv(C[0:2, 0:2]) @ (trunc_mu - mu_pair)
            # But with LDLT, we compute this via the L factors directly.
            # C[j, 0:2] = sum_k L[j,k] D[k] L[0:2, k].T for k < 2
            # With LDLT: the first two columns of L and D give us what we need.
            # Actually: cov[j, 0:2] for j>=2 from LDLT factors
            cov2_inv = np.linalg.inv(cov2)
            shift_vec = trunc_mu - mu_pair

            for j in range(2, n):
                # Cross-covariance between variable j and the pair (0, 1)
                cov_j_pair = np.array([
                    sum(L[j, k] * D[k] * L[0, k] for k in range(min(j + 1, 2))),
                    sum(L[j, k] * D[k] * L[1, k] for k in range(min(j + 1, 2))),
                ])
                pi[j] += cov_j_pair @ cov2_inv @ shift_vec

            # LDLT rank-2 update for remaining (n-2) x (n-2) block
            # The update is: Sigma_new = Sigma_old + L[:,0:2] @ (Omega - diag(D[0:2])) @ L[:,0:2].T
            # But we need to be more careful: the conditioning update for LDLT.
            # For BME, after conditioning on the pair, the remaining covariance is:
            # Sigma_{rem|pair} = Sigma_rem - Sigma_{rem,pair} @ inv(Sigma_pair) @ Sigma_{pair,rem}
            #                  + Sigma_{rem,pair} @ inv(Sigma_pair) @ Omega_pair @ inv(Sigma_pair) @ Sigma_{pair,rem}
            # = Sigma_rem + Sigma_{rem,pair} @ inv(Sigma_pair) @ (Omega_pair - Sigma_pair) @ inv(Sigma_pair) @ Sigma_{pair,rem}
            #
            # For LDLT: we compute this via two rank-1 updates.
            # Delta = inv(Sigma_pair) @ (Omega_pair - Sigma_pair) @ inv(Sigma_pair)
            # = cov2_inv @ (trunc_cov - cov2) @ cov2_inv
            Delta = cov2_inv @ (trunc_cov - cov2) @ cov2_inv

            # Build the cross-covariance matrix C_{rem, pair}: shape (n-2, 2)
            C_rem_pair = np.zeros((n - 2, 2), dtype=np.float64)
            for j in range(2, n):
                for col in range(2):
                    C_rem_pair[j - 2, col] = sum(
                        L[j, k] * D[k] * L[col, k] for k in range(min(j + 1, 2))
                    )

            # New remaining covariance = old_rem + C_rem_pair @ Delta @ C_rem_pair.T
            # We need to decompose this as LDLT.
            # Extract old remaining covariance
            old_rem_cov = _extract_subcov(L[2:n, :n], D[:n], n)[:n - 2, :n - 2]
            # Actually, more precisely: we need Sigma[2:n, 2:n] from the LDLT
            Sigma_rem = np.zeros((n - 2, n - 2), dtype=np.float64)
            for i in range(2, n):
                for j in range(i, n):
                    val = sum(L[i, k] * D[k] * L[j, k] for k in range(min(i + 1, n)))
                    Sigma_rem[i - 2, j - 2] = val
                    Sigma_rem[j - 2, i - 2] = val

            # Updated remaining covariance
            Sigma_new = Sigma_rem + C_rem_pair @ Delta @ C_rem_pair.T

            # Ensure symmetry and positive-definiteness
            Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
            np.fill_diagonal(Sigma_new, np.maximum(np.diag(Sigma_new), 1e-15))

            # Re-decompose LDLT for remaining block
            n_new = n - 2
            L_new, D_new = ldlt_decompose(Sigma_new)
            L = np.asarray(L_new, dtype=np.float64)
            D = np.asarray(D_new, dtype=np.float64)

            pi_new = pi[2:n].copy()
            pi = np.zeros(n_new, dtype=np.float64)
            pi[:n_new] = pi_new

            h_idx += 2
        else:
            # Odd variable: univariate conditioning (like ME)
            sigma_h = max(D[0], 1e-15)
            sd_h = np.sqrt(sigma_h)
            w_h = (a_ord[h_idx] - pi[0]) / sd_h
            p_h = float(norm.cdf(w_h))
            prob *= max(1e-300, p_h)
            h_idx += 1

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# Bivariate truncation + LDLT update helper (used by BME, TVBS)
# ---------------------------------------------------------------------------

def _bivariate_truncate_and_update(
    L: np.ndarray, D: np.ndarray, pi: np.ndarray,
    a_pair: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bivariate truncation of first two variables + LDLT update + trim.

    Returns updated (L, D, pi) with dimension reduced by 2.
    """
    from pybhatlib.gradmvn._bivariate_trunc import (
        truncated_bivariate_cov,
        truncated_bivariate_mean,
    )

    cov2 = _extract_subcov(L[:2, :2], D[:2], 2)
    mu_pair = np.array([pi[0], pi[1]])

    trunc_mu = truncated_bivariate_mean(mu_pair, cov2, a_pair)
    trunc_cov = truncated_bivariate_cov(mu_pair, cov2, a_pair)

    cov2_inv = np.linalg.inv(cov2)
    shift_vec = trunc_mu - mu_pair

    C_rem_pair = np.zeros((n - 2, 2), dtype=np.float64)
    for j in range(2, n):
        for col in range(2):
            C_rem_pair[j - 2, col] = sum(
                L[j, k] * D[k] * L[col, k] for k in range(min(j + 1, 2))
            )
        pi[j] += C_rem_pair[j - 2] @ cov2_inv @ shift_vec

    Delta = cov2_inv @ (trunc_cov - cov2) @ cov2_inv
    Sigma_rem = np.zeros((n - 2, n - 2), dtype=np.float64)
    for i in range(2, n):
        for j2 in range(i, n):
            val = sum(L[i, k] * D[k] * L[j2, k] for k in range(min(i + 1, n)))
            Sigma_rem[i - 2, j2 - 2] = val
            Sigma_rem[j2 - 2, i - 2] = val

    Sigma_new = Sigma_rem + C_rem_pair @ Delta @ C_rem_pair.T
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    np.fill_diagonal(Sigma_new, np.maximum(np.diag(Sigma_new), 1e-15))

    n_new = n - 2
    L_new, D_new = ldlt_decompose(Sigma_new)
    L_out = np.asarray(L_new, dtype=np.float64)
    D_out = np.asarray(D_new, dtype=np.float64)

    pi_new = pi[2:n].copy()
    pi_out = np.zeros(n_new, dtype=np.float64)
    pi_out[:n_new] = pi_new

    return L_out, D_out, pi_out


def _get_bvn_params(L, D, pi, a_ord, h_idx):
    """Extract standardized limits and correlation for first 2 variables."""
    cov2 = _extract_subcov(L[:2, :2], D[:2], 2)
    sd_0 = np.sqrt(max(cov2[0, 0], 1e-15))
    sd_1 = np.sqrt(max(cov2[1, 1], 1e-15))
    rho = cov2[0, 1] / (sd_0 * sd_1) if sd_0 > 1e-15 and sd_1 > 1e-15 else 0.0
    rho = max(-0.9999, min(0.9999, rho))
    w_0 = (a_ord[h_idx] - pi[0]) / sd_0
    w_1 = (a_ord[h_idx + 1] - pi[1]) / sd_1
    return w_0, w_1, rho


# ---------------------------------------------------------------------------
# TVBS: BME + quadrivariate screening (Bhat 2018, Section 2.3.2.2)
# ---------------------------------------------------------------------------

def _mvncd_tvbs(a: np.ndarray, sigma: np.ndarray) -> float:
    """TVBS: Quadrivariate initialization + bivariate sequential conditioning.

    Paper algorithm (Section 2.3.2.2, p. 247):
    1. P = Phi_4(w_0, w_1, w_2, w_3) for first 4 variables
    2. Bivariate truncation of (0,1), rank-2 update
    3. Bivariate truncation of (2,3), rank-2 update (no prob factor)
    4. Remaining: BME bivariate pairs
    For K=3: falls back to BME.

    The initial Phi_4 factor provides higher accuracy than BME's Phi_2 start.
    """
    from pybhatlib.gradmvn._univariate import (
        bivariate_normal_cdf,
        quadrivariate_normal_cdf,
    )

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    if K == 3:
        return _mvncd_bme(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    prob = 1.0
    pi = np.zeros(K, dtype=np.float64)
    h_idx = 0

    # Step 1: P = Phi_4(w_0, w_1, w_2, w_3) for first 4 variables
    cov4 = _extract_subcov(L[:4, :4], D[:4], 4)
    sds_4 = np.sqrt(np.maximum(np.diag(cov4), 1e-15))
    ws_4 = np.array([(a_ord[i] - pi[i]) / sds_4[i] for i in range(4)])

    corr4 = np.eye(4)
    for i in range(4):
        for j in range(i + 1, 4):
            r = cov4[i, j] / (sds_4[i] * sds_4[j])
            r = max(-0.9999, min(0.9999, r))
            corr4[i, j] = r
            corr4[j, i] = r

    qvn = quadrivariate_normal_cdf(ws_4[0], ws_4[1], ws_4[2], ws_4[3], corr4)
    prob *= max(1e-300, qvn)

    if prob < 1e-300:
        return 0.0

    # Step 2: Bivariate truncation of (0,1)
    n = K - h_idx
    a_pair = np.array([a_ord[h_idx], a_ord[h_idx + 1]])
    L, D, pi = _bivariate_truncate_and_update(L, D, pi, a_pair, n)
    h_idx += 2

    # Step 3: Bivariate truncation of (2,3) — NO probability factor (Phi_4 covers it)
    n = K - h_idx
    if n >= 2:
        a_pair = np.array([a_ord[h_idx], a_ord[h_idx + 1]])
        if n > 2:
            L, D, pi = _bivariate_truncate_and_update(L, D, pi, a_pair, n)
        h_idx += 2

    # Step 4: Remaining pairs with BME (bivariate conditioning)
    # Note: QVN/BVN screening was removed because the conditional screening
    # factor always >= BVN (conditioning on next pair being satisfied only
    # increases probability), causing systematic overestimation at H >= 10.
    # TVBS accuracy comes from the initial Phi_4 factor, not from screening.
    while h_idx < K:
        n = K - h_idx

        if n >= 2:
            w_0, w_1, rho = _get_bvn_params(L, D, pi, a_ord, h_idx)
            bvn = bivariate_normal_cdf(w_0, w_1, rho)
            prob *= max(1e-300, bvn)

            if prob < 1e-300:
                return 0.0

            if n > 2:
                a_pair = np.array([a_ord[h_idx], a_ord[h_idx + 1]])
                L, D, pi = _bivariate_truncate_and_update(L, D, pi, a_pair, n)

            h_idx += 2
        else:
            # Last odd variable
            sigma_h = max(D[0], 1e-15)
            sd_h = np.sqrt(sigma_h)
            w_h = (a_ord[h_idx] - pi[0]) / sd_h
            prob *= max(1e-300, float(norm.cdf(w_h)))
            h_idx += 1

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# OVBS: OVUS + trivariate screening (Bhat 2018, Section 2.3.1.2)
# ---------------------------------------------------------------------------

def _mvncd_ovbs(a: np.ndarray, sigma: np.ndarray) -> float:
    """OVBS: ME with trivariate screening (Bhat 2018, Section 2.3.1.2).

    Paper pseudocode (p. 245):
    (1) P_1 = Phi(w_1) * Phi_2(w_1,w_2)/Phi(w_1) * Phi_3(w_1,w_2,w_3)/Phi_2(w_1,w_2)
            = Phi_3(w_1, w_2, w_3; Lambda_1). If H=3, STOP.
    (2) LDLT decompose Sigma.
    (3) For h = 1 to H-3:
    (4)   Truncate X_h univariately, rank-1 update.
    (5)   LDLT rank-1 update of Sigma_{h+1}.
    (6)   Extract 3x3 (or 2x2) sub-cov.
          If n >= 3: P_{h+1} = Phi_3(w',w'',w''') / Phi_2(w'',w''')
          If n == 2: P_{h+1} = BVN(w',w'') / Phi(w'')
    (7) End for
    (8) Return P_1 * prod(P_{h+1}).
    """
    from pybhatlib.gradmvn._univariate import (
        bivariate_normal_cdf,
        trivariate_normal_cdf,
    )

    K = len(a)
    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    pi = np.zeros(K, dtype=np.float64)

    # Step 1: P_1 = Phi_3(w_1, w_2, w_3; Lambda_1) — trivariate CDF
    cov3 = _extract_subcov(L[:3, :3], D[:3], 3)
    sds_3 = np.sqrt(np.maximum(np.diag(cov3), 1e-15))
    ws_3 = np.array([(a_ord[i] - pi[i]) / sds_3[i] for i in range(3)])
    corr3 = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            r = cov3[i, j] / (sds_3[i] * sds_3[j])
            r = max(-0.9999, min(0.9999, r))
            corr3[i, j] = r
            corr3[j, i] = r
    prob = trivariate_normal_cdf(ws_3[0], ws_3[1], ws_3[2], corr3)
    if prob < 1e-300:
        return 0.0

    if K == 3:
        return max(0.0, min(1.0, prob))

    # Steps 3-7: For h = 1 to H-3 (0-indexed: h_idx = 0 to K-4)
    for h_idx in range(K - 3):
        n = K - h_idx  # remaining dimensions before truncation

        sigma_h = max(D[0], 1e-15)
        sd_h = np.sqrt(sigma_h)
        w_h = (a_ord[h_idx] - pi[0]) / sd_h

        # Univariate truncation + rank-1 update
        L, D, pi = _univariate_truncate_and_update(L, D, pi, w_h, sigma_h, n)

        # After truncation, extract sub-cov for screening
        n_new = n - 1
        if n_new >= 3:
            # Trivariate screening: P *= Phi_3 / Phi_2
            cov3 = _extract_subcov(L[:3, :3], D[:3], 3)
            sds = np.sqrt(np.maximum(np.diag(cov3), 1e-15))
            ws = np.array([(a_ord[h_idx + 1 + i] - pi[i]) / sds[i] for i in range(3)])
            corr = np.eye(3)
            for i in range(3):
                for j in range(i + 1, 3):
                    r = cov3[i, j] / (sds[i] * sds[j])
                    r = max(-0.9999, min(0.9999, r))
                    corr[i, j] = r
                    corr[j, i] = r
            tvn = trivariate_normal_cdf(ws[0], ws[1], ws[2], corr)
            # Denominator is BVN of FIRST TWO variables (the ones already
            # accounted for), not the last two. See paper step (1):
            # P_1 = TVN(1,2,3)/BVN(1,2) — denominator is BVN of first two.
            bvn_denom = bivariate_normal_cdf(ws[0], ws[1], corr[0, 1])
            p_h = tvn / bvn_denom if bvn_denom > 1e-300 else float(norm.cdf(ws[0]))
        elif n_new == 2:
            # Bivariate screening: P *= BVN / Phi
            cov2 = _extract_subcov(L[:2, :2], D[:2], 2)
            sd_0 = np.sqrt(max(cov2[0, 0], 1e-15))
            sd_1 = np.sqrt(max(cov2[1, 1], 1e-15))
            rho = cov2[0, 1] / (sd_0 * sd_1) if sd_0 > 1e-15 and sd_1 > 1e-15 else 0.0
            rho = max(-0.9999, min(0.9999, rho))
            w0 = (a_ord[h_idx + 1] - pi[0]) / sd_0
            w1 = (a_ord[h_idx + 2] - pi[1]) / sd_1
            bvn = bivariate_normal_cdf(w0, w1, rho)
            # Denominator is Phi of FIRST variable (consistent with OVUS)
            p_denom = float(norm.cdf(w0))
            p_h = bvn / p_denom if p_denom > 1e-300 else float(norm.cdf(w0))
        else:
            # 1 variable left
            w_last = (a_ord[h_idx + 1] - pi[0]) / np.sqrt(max(D[0], 1e-15))
            p_h = float(norm.cdf(w_last))

        prob *= max(1e-300, p_h)
        if prob < 1e-300:
            return 0.0

    return max(0.0, min(1.0, prob))
