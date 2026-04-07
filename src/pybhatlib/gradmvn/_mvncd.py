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
from scipy.special import ndtr as _ndtr
from scipy.stats import multivariate_normal as _scipy_mvn

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)

def _std_npdf(x):
    """Standard normal PDF, faster than scipy.stats.norm.pdf."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)

from pybhatlib.backend._array_api import array_namespace, get_backend
from pybhatlib.vecup._ldlt import ldlt_decompose, ldlt_rank1_update, ldlt_rank2_update

# ---------------------------------------------------------------------------
# Numba JIT support
# ---------------------------------------------------------------------------
try:
    import numba
    from pybhatlib.vecup._ldlt import HAS_NUMBA, _ldlt_decompose_jit, _ldlt_rank1_update_jit
except ImportError:
    HAS_NUMBA = False

import math as _math

if HAS_NUMBA:
    @numba.njit(cache=True)
    def _ndtr_jit(x):
        """Numba-compatible standard normal CDF via erfc."""
        return 0.5 * _math.erfc(-x / _math.sqrt(2.0))

    @numba.njit(cache=True)
    def _std_npdf_jit(x):
        """Numba-compatible standard normal PDF."""
        return (1.0 / _math.sqrt(2.0 * _math.pi)) * _math.exp(-0.5 * x * x)

    @numba.njit(cache=True)
    def _mvncd_me_core_jit(a_ord, L, D, K):
        """JIT-compiled ME core conditioning loop.

        Parameters
        ----------
        a_ord : ndarray, shape (K,) — reordered upper limits
        L : ndarray, shape (K, K) — LDLT lower triangular factor
        D : ndarray, shape (K,) — LDLT diagonal factor
        K : int — number of dimensions

        Returns
        -------
        prob : float — MVNCD probability
        """
        prob = 1.0
        pi = np.zeros(K)
        H = K

        for h in range(H):
            n = H - h  # remaining dimension

            sigma_h = D[0]
            if sigma_h < 1e-15:
                sigma_h = 1e-15
            sd_h = _math.sqrt(sigma_h)

            w_h = (a_ord[h] - pi[0]) / sd_h
            p_h = _ndtr_jit(w_h)
            if p_h < 1e-300:
                p_h = 1e-300
            prob *= p_h

            if prob < 1e-300:
                return 0.0

            if n <= 1:
                break

            # Truncated moments
            phi_w = _std_npdf_jit(w_h)
            Phi_w = _ndtr_jit(w_h)
            if Phi_w < 1e-300:
                Phi_w = 1e-300
            lambda_h = -phi_w / Phi_w
            var_z = 1.0 + lambda_h * (w_h - lambda_h)
            if var_z < 1e-15:
                var_z = 1e-15

            mu_tilde = pi[0] + sd_h * lambda_h
            Omega_h = sigma_h * var_z

            # Update pi for remaining variables
            shift = mu_tilde - pi[0]
            for j in range(1, n):
                pi[j] += L[j, 0] * shift

            # LDLT rank-1 update
            alpha = Omega_h
            v = L[1:n, 0].copy()

            L_sub, D_sub = _ldlt_rank1_update_jit(
                L[1:n, 1:n].copy(), D[1:n].copy(), v, alpha
            )

            # Trim: prepare arrays for next iteration
            L = L_sub
            D = D_sub
            pi_new = pi[1:n].copy()
            pi = pi_new

        if prob < 0.0:
            return 0.0
        if prob > 1.0:
            return 1.0
        return prob

    @numba.njit(cache=True)
    def _univariate_truncate_and_update_jit(L, D, pi, w_h, sigma_h, n):
        """JIT-compiled univariate truncation + LDLT rank-1 update + trim.

        Returns updated (L_new, D_new, pi_out) with dimension reduced by 1.
        """
        sd_h = _math.sqrt(sigma_h)
        phi_w = _std_npdf_jit(w_h)
        Phi_w = _ndtr_jit(w_h)
        if Phi_w < 1e-300:
            Phi_w = 1e-300
        lambda_h = -phi_w / Phi_w
        var_z = 1.0 + lambda_h * (w_h - lambda_h)
        if var_z < 1e-15:
            var_z = 1e-15

        mu_tilde = pi[0] + sd_h * lambda_h
        Omega_h = sigma_h * var_z

        shift = mu_tilde - pi[0]
        for j in range(1, n):
            pi[j] += L[j, 0] * shift

        alpha = Omega_h
        v = L[1:n, 0].copy()
        L_sub, D_sub = _ldlt_rank1_update_jit(
            L[1:n, 1:n].copy(), D[1:n].copy(), v, alpha
        )

        n_new = n - 1
        L_new = np.eye(n_new)
        L_new[:n_new, :n_new] = L_sub[:n_new, :n_new]
        D_new = D_sub[:n_new].copy()
        pi_new = pi[1:n].copy()
        pi_out = np.zeros(n_new)
        pi_out[:n_new] = pi_new[:n_new]

        return L_new, D_new, pi_out

    # ------------------------------------------------------------------
    # Phase 2: Batch MVNCD with shared covariance (parallel via prange)
    # ------------------------------------------------------------------

    @numba.njit(cache=True)
    def _mvncd_single_shared_cov_jit(a, L_base, D_base, K):
        """Evaluate MVNCD for a single observation with pre-computed LDLT.

        Handles reordering internally and dispatches to bivariate or ME core.

        Parameters
        ----------
        a : ndarray, shape (K,)
            Upper integration limits.
        L_base : ndarray, shape (K, K)
            LDLT lower triangular factor of the shared covariance.
        D_base : ndarray, shape (K,)
            LDLT diagonal factor of the shared covariance.
        K : int
            Dimension.

        Returns
        -------
        prob : float
        """
        if K == 1:
            sd = _math.sqrt(max(L_base[0, 0] * D_base[0] * L_base[0, 0], 1e-30))
            return _ndtr_jit(a[0] / sd)

        if K == 2:
            # Reconstruct covariance for bivariate case
            # sigma = L @ diag(D) @ L.T
            s00 = L_base[0, 0] * D_base[0] * L_base[0, 0]
            s01 = L_base[0, 0] * D_base[0] * L_base[1, 0] + 0.0
            # More precisely: s[i,j] = sum_k L[i,k]*D[k]*L[j,k]
            s11 = L_base[1, 0] * D_base[0] * L_base[1, 0] + L_base[1, 1] * D_base[1] * L_base[1, 1]
            s01 = L_base[1, 0] * D_base[0] * L_base[0, 0]

            sd1 = _math.sqrt(max(s00, 1e-30))
            sd2 = _math.sqrt(max(s11, 1e-30))
            if sd1 > 1e-15 and sd2 > 1e-15:
                rho = s01 / (sd1 * sd2)
            else:
                rho = 0.0
            if rho > 0.9999:
                rho = 0.9999
            if rho < -0.9999:
                rho = -0.9999
            # Cannot call _bivariate_normal_cdf_jit here because it needs GL arrays
            # Use _bvn_cdf_from_params_jit instead (inlined below)
            return _bvn_cdf_standardized_jit(a[0] / sd1, a[1] / sd2, rho)

        # K >= 3: reorder and run ME core
        # Reorder by |a_k / sqrt(sigma_kk)| ascending
        # Compute sigma_kk from LDLT: sigma_kk = sum_j L[k,j]^2 * D[j]
        sds = np.empty(K)
        z = np.empty(K)
        for k in range(K):
            var_k = 0.0
            for j in range(k + 1):
                var_k += L_base[k, j] * L_base[k, j] * D_base[j]
            sds[k] = _math.sqrt(max(var_k, 1e-30))
            z[k] = abs(a[k]) / sds[k]

        # Simple insertion sort for small K (typically K <= 9)
        order = np.arange(K)
        for i in range(1, K):
            key_z = z[i]
            key_o = order[i]
            j = i - 1
            while j >= 0 and z[j] > key_z:
                z[j + 1] = z[j]
                order[j + 1] = order[j]
                j -= 1
            z[j + 1] = key_z
            order[j + 1] = key_o

        # Reorder a
        a_ord = np.empty(K)
        for i in range(K):
            a_ord[i] = a[order[i]]

        # Reorder sigma: sigma_ord = sigma[order, :][:, order]
        # Reconstruct full sigma from LDLT, then reorder and re-decompose
        sigma_full = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                val = 0.0
                for k in range(min(i, j) + 1):
                    val += L_base[i, k] * D_base[k] * L_base[j, k]
                sigma_full[i, j] = val

        sigma_ord = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                sigma_ord[i, j] = sigma_full[order[i], order[j]]

        # LDLT decompose reordered sigma
        L_ord, D_ord = _ldlt_decompose_jit(sigma_ord)

        return _mvncd_me_core_jit(a_ord, L_ord, D_ord, K)

    @numba.njit(cache=True)
    def _bvn_cdf_standardized_jit(x1, x2, rho):
        """Bivariate standard normal CDF for standardized inputs.

        Simplified Genz BVND algorithm inlined for Numba compatibility
        without GL array parameters.
        """
        TWOPI = 2.0 * _math.pi

        # Edge cases
        if abs(rho) < 1e-15:
            return _ndtr_jit(x1) * _ndtr_jit(x2)
        if rho > 1.0 - 1e-15:
            return _ndtr_jit(min(x1, x2))
        if rho < -1.0 + 1e-15:
            val = _ndtr_jit(x1) + _ndtr_jit(x2) - 1.0
            return max(0.0, val) if x1 + x2 >= 0 else 0.0

        dh = -x1
        dk = -x2
        r = rho
        abs_r = abs(r)

        # GL6 weights and abscissae (hardcoded for Numba)
        gl6_w0 = 0.1713244923791704
        gl6_w1 = 0.3607615730481386
        gl6_w2 = 0.4679139345726910
        gl6_x0 = 0.9324695142031521
        gl6_x1 = 0.6612093864662645
        gl6_x2 = 0.2386191860831969

        # GL12
        gl12_w = np.array([
            0.0471753363865118, 0.1069393259953184, 0.1600783285433462,
            0.2031674267230659, 0.2334925365383548, 0.2491470458134028,
        ])
        gl12_x = np.array([
            0.9815606342467192, 0.9041172563704749, 0.7699026741943047,
            0.5873179542866175, 0.3678314989981802, 0.1252334085114689,
        ])

        # GL20
        gl20_w = np.array([
            0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
            0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
            0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
            0.1527533871307258,
        ])
        gl20_x = np.array([
            0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
            0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
            0.5108670019508271, 0.3737060887154195, 0.2277858511416451,
            0.0765265211334973,
        ])

        if abs_r < 0.925:
            # Low-to-moderate correlation
            hk = dh * dk
            bvn = 0.0
            if abs_r > 0:
                hs = (dh * dh + dk * dk) / 2.0
                asr = _math.asin(r)
                if abs_r < 0.3:
                    # GL6 (3 symmetric pairs)
                    for i in range(3):
                        w_i = gl6_w0 if i == 0 else (gl6_w1 if i == 1 else gl6_w2)
                        x_i = gl6_x0 if i == 0 else (gl6_x1 if i == 1 else gl6_x2)
                        for isign in range(2):
                            sign = 1.0 if isign == 0 else -1.0
                            sn = _math.sin(asr * (sign * x_i + 1.0) / 2.0)
                            denom = 1.0 - sn * sn
                            if denom < 1e-30:
                                denom = 1e-30
                            bvn += w_i * _math.exp((sn * hk - hs) / denom)
                elif abs_r < 0.75:
                    # GL12 (6 symmetric pairs)
                    for i in range(6):
                        for isign in range(2):
                            sign = 1.0 if isign == 0 else -1.0
                            sn = _math.sin(asr * (sign * gl12_x[i] + 1.0) / 2.0)
                            denom = 1.0 - sn * sn
                            if denom < 1e-30:
                                denom = 1e-30
                            bvn += gl12_w[i] * _math.exp((sn * hk - hs) / denom)
                else:
                    # GL20 (10 symmetric pairs)
                    for i in range(10):
                        for isign in range(2):
                            sign = 1.0 if isign == 0 else -1.0
                            sn = _math.sin(asr * (sign * gl20_x[i] + 1.0) / 2.0)
                            denom = 1.0 - sn * sn
                            if denom < 1e-30:
                                denom = 1e-30
                            bvn += gl20_w[i] * _math.exp((sn * hk - hs) / denom)
                bvn *= asr / (2.0 * TWOPI)
            result = bvn + _ndtr_jit(-dh) * _ndtr_jit(-dk)
        else:
            # High correlation
            if r < 0:
                k = -dk
                hk = -dh * dk
            else:
                k = dk
                hk = dh * dk

            ass1 = (1.0 - r) * (1.0 + r)
            a_val = _math.sqrt(max(ass1, 0.0))
            bs = (dh - k) ** 2
            c = (4.0 - hk) / 8.0
            d = (12.0 - hk) / 16.0

            if ass1 > 0:
                asr = -(bs / ass1 + hk) / 2.0
            else:
                asr = -200.0

            if asr > -100:
                bvn = (
                    a_val
                    * _math.exp(asr)
                    * (
                        1.0
                        - c * (bs - ass1) * (1.0 - d * bs / 5.0) / 3.0
                        + c * d * ass1 * ass1 / 5.0
                    )
                )
            else:
                bvn = 0.0

            if -hk < 100:
                b = _math.sqrt(bs)
                if a_val > 1e-30:
                    bvn -= (
                        _math.exp(-hk / 2.0)
                        * _math.sqrt(TWOPI)
                        * _ndtr_jit(-b / a_val)
                        * b
                        * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
                    )

            a_half = a_val / 2.0
            for i in range(10):
                for isign in range(2):
                    sign = 1.0 if isign == 0 else -1.0
                    xs = (a_half * (sign * gl20_x[i] + 1.0)) ** 2
                    rs = _math.sqrt(max(1.0 - xs, 0.0))
                    if xs < 1e-30:
                        continue
                    asr2 = -(bs / xs + hk) / 2.0
                    if asr2 > -100:
                        if rs > 1e-30:
                            bvn += a_half * gl20_w[i] * _math.exp(asr2) * (
                                _math.exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                                - (1.0 + c * xs * (1.0 + d * xs))
                            )

            bvn = -bvn / TWOPI

            if r > 0:
                bvn += _ndtr_jit(-max(dh, k))
            else:
                bvn = -bvn
                if k > dh:
                    bvn += _ndtr_jit(k) - _ndtr_jit(dh)

            result = bvn

        if result < 0.0:
            return 0.0
        if result > 1.0:
            return 1.0
        return result

    @numba.njit(parallel=True, cache=True)
    def _mvncd_batch_shared_cov_jit(a_all, L_base, D_base, K):
        """Evaluate MVNCD for N observations with shared covariance in parallel.

        Parameters
        ----------
        a_all : ndarray, shape (N, K)
            Upper integration limits for each observation.
        L_base : ndarray, shape (K, K)
            LDLT lower triangular factor of the shared covariance.
        D_base : ndarray, shape (K,)
            LDLT diagonal factor of the shared covariance.
        K : int
            Dimension.

        Returns
        -------
        log_probs : ndarray, shape (N,)
            Log-probabilities for each observation.
        """
        N = a_all.shape[0]
        log_probs = np.empty(N)
        for q in numba.prange(N):
            prob = _mvncd_single_shared_cov_jit(a_all[q], L_base, D_base, K)
            if prob < 1e-300:
                prob = 1e-300
            log_probs[q] = _math.log(prob)
        return log_probs


    @numba.njit(cache=True)
    def _mvncd_batch_shared_cov_seq_jit(a_all, L_base, D_base, K):
        """Sequential JIT batch: evaluate MVNCD for all observations.

        Faster than prange for small N (<1000) because it avoids thread
        scheduling overhead while still benefiting from JIT compilation
        and eliminating per-observation Python dispatch.
        """
        N = a_all.shape[0]
        log_probs = np.empty(N)
        for q in range(N):
            prob = _mvncd_single_shared_cov_jit(a_all[q], L_base, D_base, K)
            if prob < 1e-300:
                prob = 1e-300
            log_probs[q] = _math.log(prob)
        return log_probs

    @numba.njit(cache=True)
    def _mvncd_batch_percov_seq_jit(a_all, sigma_all, K):
        """Sequential JIT batch with per-observation covariance."""
        N = a_all.shape[0]
        log_probs = np.empty(N)
        for q in range(N):
            L_q, D_q = _ldlt_decompose_jit(sigma_all[q])
            prob = _mvncd_single_shared_cov_jit(a_all[q], L_q, D_q, K)
            if prob < 1e-300:
                prob = 1e-300
            log_probs[q] = _math.log(prob)
        return log_probs

    @numba.njit(parallel=True, cache=True)
    def _mvncd_batch_percov_jit(a_all, sigma_all, K):
        """Evaluate MVNCD for N observations with per-observation covariance.

        Parameters
        ----------
        a_all : ndarray, shape (N, K)
            Upper integration limits for each observation.
        sigma_all : ndarray, shape (N, K, K)
            Per-observation covariance matrices.
        K : int
            Dimension.

        Returns
        -------
        log_probs : ndarray, shape (N,)
            Log-probabilities for each observation.
        """
        N = a_all.shape[0]
        log_probs = np.empty(N)
        for q in numba.prange(N):
            L_q, D_q = _ldlt_decompose_jit(sigma_all[q])
            prob = _mvncd_single_shared_cov_jit(a_all[q], L_q, D_q, K)
            if prob < 1e-300:
                prob = 1e-300
            log_probs[q] = _math.log(prob)
        return log_probs





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
        - "tg": Trivariate-Gaussian univariate conditioning (no rank-1 updates)
        - "tgbme": TG with bivariate conditioning (no rank-2 updates)
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
    elif method == "tg":
        return _mvncd_tg(a_np, sigma_np)
    elif method == "tgbme":
        return _mvncd_tgbme(a_np, sigma_np)
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


def mvncd_log_batch(
    a_all: NDArray,
    sigma: NDArray,
    *,
    method: str = "me",
    per_obs_sigma: NDArray | None = None,
) -> NDArray:
    """Compute log-MVNCD for N observations, parallelized via Numba prange.

    Optimized for the MNP observation loop: returns log-probabilities directly
    and avoids Python-level per-observation overhead.

    Parameters
    ----------
    a_all : ndarray, shape (N, K)
        Upper integration limits for each observation.
    sigma : ndarray, shape (K, K)
        Shared covariance matrix (used when per_obs_sigma is None).
    method : str
        MVNCD method. Parallel batch only supports "me" (uses JIT core).
        Other methods fall back to sequential evaluation.
    per_obs_sigma : ndarray, shape (N, K, K) or None
        Per-observation covariance matrices. If provided, each observation
        uses its own covariance (for mixed MNP with random coefficients).

    Returns
    -------
    log_probs : ndarray, shape (N,)
        Log-probability for each observation.
    """
    a_all = np.asarray(a_all, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    N = a_all.shape[0]
    K = a_all.shape[1]

    # Try JIT parallel batch (ME method only, NumPy arrays, Numba available)
    if HAS_NUMBA and method == "me":
        if per_obs_sigma is not None:
            sigma_all = np.asarray(per_obs_sigma, dtype=np.float64)
            if N < 50000:
                return _mvncd_batch_percov_seq_jit(a_all, sigma_all, K)
            else:
                return _mvncd_batch_percov_jit(a_all, sigma_all, K)
        else:
            # Shared covariance: pre-compute LDLT once
            from pybhatlib.vecup._ldlt import _ldlt_decompose_jit
            L_base, D_base = _ldlt_decompose_jit(sigma)
            # Sequential JIT is faster than prange for all tested N < 50000
            # (prange has ~15ms thread pool startup overhead).
            # Only use parallel for very large N where amortization pays off.
            if N < 50000:
                return _mvncd_batch_shared_cov_seq_jit(a_all, L_base, D_base, K)
            else:
                return _mvncd_batch_shared_cov_jit(a_all, L_base, D_base, K)

    # Fallback: sequential evaluation with any method
    log_probs = np.empty(N, dtype=np.float64)
    xp = get_backend("numpy")
    for q in range(N):
        if per_obs_sigma is not None:
            sig_q = per_obs_sigma[q]
        else:
            sig_q = sigma
        prob = mvncd(xp.array(a_all[q]), xp.array(sig_q), method=method, xp=xp)
        prob = max(prob, 1e-300)
        log_probs[q] = np.log(prob)
    return log_probs



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
        return float(_ndtr(a[0] / sd))

    if K == 2:
        return _bvn_cdf(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    # LDLT decompose
    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    # Dispatch to JIT core if available
    if HAS_NUMBA:
        return _mvncd_me_core_jit(a_ord, L, D, K)

    # Pure Python fallback
    prob = 1.0
    pi = np.zeros(K, dtype=np.float64)
    H = K

    for h in range(H):
        n = H - h  # remaining dimension

        sigma_h = max(D[0], 1e-15)
        sd_h = np.sqrt(sigma_h)

        w_h = (a_ord[h] - pi[0]) / sd_h
        p_h = float(_ndtr(w_h))
        prob *= max(1e-300, p_h)

        if prob < 1e-300:
            return 0.0

        if n <= 1:
            break

        # Truncated moments E[Z|Z<=w] and Var[Z|Z<=w] for standard normal
        phi_w = _std_npdf(w_h)
        Phi_w = max(_ndtr(w_h), 1e-300)
        lambda_h = -phi_w / Phi_w  # E[Z | Z <= w_h]
        var_z = 1.0 + lambda_h * (w_h - lambda_h)
        var_z = max(var_z, 1e-15)

        mu_tilde = pi[0] + sd_h * lambda_h
        Omega_h = sigma_h * var_z

        shift = mu_tilde - pi[0]
        for j in range(1, n):
            pi[j] += L[j, 0] * shift

        alpha = Omega_h
        v = L[1:n, 0].copy()

        L_sub, D_sub = ldlt_rank1_update(L[1:n, 1:n], D[1:n], v, alpha)

        L = L_sub
        D = D_sub
        pi_new = pi[1:n].copy()
        pi = pi_new

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
    if HAS_NUMBA:
        return _univariate_truncate_and_update_jit(
            L.copy(), D.copy(), pi.copy(), float(w_h), float(sigma_h), n
        )

    # Pure Python fallback
    sd_h = np.sqrt(sigma_h)
    phi_w = _std_npdf(w_h)
    Phi_w = max(_ndtr(w_h), 1e-300)
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
            p_denom = float(_ndtr(w_next_0))
            p_h = bvn / p_denom if p_denom > 1e-300 else float(_ndtr(w_next_0))
        else:
            # Only 1 variable left — just Phi
            w_last = (a_ord[h_idx + 1] - pi[0]) / np.sqrt(max(D[0], 1e-15))
            p_h = float(_ndtr(w_last))

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
            p_h = float(_ndtr(w_h))
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
            prob *= max(1e-300, float(_ndtr(w_h)))
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
            p_h = tvn / bvn_denom if bvn_denom > 1e-300 else float(_ndtr(ws[0]))
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
            p_denom = float(_ndtr(w0))
            p_h = bvn / p_denom if p_denom > 1e-300 else float(_ndtr(w0))
        else:
            # 1 variable left
            w_last = (a_ord[h_idx + 1] - pi[0]) / np.sqrt(max(D[0], 1e-15))
            p_h = float(_ndtr(w_last))

        prob *= max(1e-300, p_h)
        if prob < 1e-300:
            return 0.0

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# TG: Trivariate-Gaussian univariate conditioning (no rank-1 updates)
# ---------------------------------------------------------------------------

def _mvncd_tg(a: np.ndarray, sigma: np.ndarray) -> float:
    """TG: Sequential univariate conditioning using LDLT strip-down only.

    Unlike ME which uses LDLT rank-1 updates for conditioning, TG simply
    strips rows/columns from the LDLT factors and adjusts the conditional
    mean via the L factor. This is simpler and avoids numerical issues
    from rank-1 updates, at the cost of slightly less accurate conditioning.

    GAUSS reference: cdfmvnaTG, line 837.

    Algorithm:
    1. Reorder variables by ascending |a_k / sqrt(Sigma_kk)|
    2. LDLT decompose reordered Sigma -> (L, D)
    3. For h = 0..K-1:
       - sigma_h = D[0,0]
       - w_h = (a[h] - mu[0]) / sqrt(sigma_h)
       - P *= Phi(w_h)
       - Compute truncated mean
       - Update conditional mean: mu[1:] += L[1:,0] * (mu_trunc - mu[0])
       - Strip first row/col of L and D (NO rank-1 update)
    """
    K = len(a)

    if K == 1:
        sd = np.sqrt(sigma[0, 0])
        return float(_ndtr(a[0] / sd))

    if K == 2:
        return _bvn_cdf(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    # LDLT decompose
    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    prob = 1.0
    mu = np.zeros(K, dtype=np.float64)

    for h in range(K):
        n = K - h  # remaining dimension

        sigma_h = max(D[0], 1e-15)
        sd_h = np.sqrt(sigma_h)

        w_h = (a_ord[h] - mu[0]) / sd_h
        p_h = float(_ndtr(w_h))
        prob *= max(1e-300, p_h)

        if prob < 1e-300:
            return 0.0

        if n <= 1:
            break

        # Truncated moments: E[Z|Z<=w] for standard normal
        phi_w = _std_npdf(w_h)
        Phi_w = max(_ndtr(w_h), 1e-300)
        lambda_h = -phi_w / Phi_w  # E[Z | Z <= w_h]

        mu_tilde = mu[0] + sd_h * lambda_h

        # Update conditional mean for remaining variables
        shift = mu_tilde - mu[0]
        for j in range(1, n):
            mu[j] += L[j, 0] * shift

        # Strip first row/col: NO rank-1 update (key difference from ME)
        L = L[1:n, 1:n].copy()
        D = D[1:n].copy()
        mu = mu[1:n].copy()

    return max(0.0, min(1.0, prob))


# ---------------------------------------------------------------------------
# TGBME: TG with bivariate conditioning (no rank-2 updates)
# ---------------------------------------------------------------------------

def _mvncd_tgbme(a: np.ndarray, sigma: np.ndarray) -> float:
    """TGBME: Sequential bivariate conditioning using LDLT strip-down only.

    Like BME but without rank-2 updates: processes variables in pairs,
    uses BVN for each pair factor, and strips 2 rows/cols from LDLT.
    For odd K, the last variable uses univariate conditioning.

    GAUSS reference: cdfmvnaTGBME, line 1297.

    Algorithm:
    1. Reorder variables by ascending |a_k / sqrt(Sigma_kk)|
    2. LDLT decompose reordered Sigma -> (L, D)
    3. For each pair (2k, 2k+1):
       - Extract 2x2 sub-cov from LDLT
       - Standardize: w_i = (a[i] - mu[i]) / sd_i, compute rho
       - P *= BVN(w_0, w_1; rho)
       - Compute bivariate truncated moments
       - Update conditional mean via cross-covariance and Sigma_2^{-1}
       - Strip first 2 rows/cols of L and D (NO rank-2 update)
    4. If K is odd, last variable: P *= Phi(w_last)
    """
    from pybhatlib.gradmvn._univariate import bivariate_normal_cdf
    from pybhatlib.gradmvn._bivariate_trunc import (
        truncated_bivariate_mean,
    )

    K = len(a)

    if K <= 2:
        return _mvncd_me(a, sigma)

    a_ord, sigma_ord, _ = _reorder_by_limits(a, sigma)

    # LDLT decompose
    L, D = ldlt_decompose(sigma_ord)
    L = np.asarray(L, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    prob = 1.0
    mu = np.zeros(K, dtype=np.float64)
    h_idx = 0

    while h_idx < K:
        n = K - h_idx  # remaining dimensions

        if n >= 2:
            # Extract 2x2 sub-covariance from LDLT
            cov2 = _extract_subcov(L[:2, :2], D[:2], 2)

            sd_0 = np.sqrt(max(cov2[0, 0], 1e-15))
            sd_1 = np.sqrt(max(cov2[1, 1], 1e-15))
            rho = cov2[0, 1] / (sd_0 * sd_1) if sd_0 > 1e-15 and sd_1 > 1e-15 else 0.0
            rho = max(-0.9999, min(0.9999, rho))

            w_0 = (a_ord[h_idx] - mu[0]) / sd_0
            w_1 = (a_ord[h_idx + 1] - mu[1]) / sd_1

            bvn = bivariate_normal_cdf(w_0, w_1, rho)
            prob *= max(1e-300, bvn)

            if prob < 1e-300:
                return 0.0

            if n <= 2:
                break

            # Bivariate truncated moments for conditioning
            mu_pair = np.array([mu[0], mu[1]])
            a_pair = np.array([a_ord[h_idx], a_ord[h_idx + 1]])

            trunc_mu = truncated_bivariate_mean(mu_pair, cov2, a_pair)

            # Compute cross-covariance C_{rem,pair} from LDLT factors
            # C[j, col] = sum_k L[j,k] * D[k] * L[col,k] for k < min(j+1, 2)
            cov2_inv = np.linalg.inv(cov2)
            shift_vec = trunc_mu - mu_pair

            for j in range(2, n):
                cov_j_pair = np.array([
                    sum(L[j, k] * D[k] * L[0, k] for k in range(min(j + 1, 2))),
                    sum(L[j, k] * D[k] * L[1, k] for k in range(min(j + 1, 2))),
                ])
                mu[j] += cov_j_pair @ cov2_inv @ shift_vec

            # Strip first 2 rows/cols: NO rank-2 update (key difference from BME)
            L = L[2:n, 2:n].copy()
            D = D[2:n].copy()
            mu_new = mu[2:n].copy()
            n_new = n - 2
            mu = np.zeros(n_new, dtype=np.float64)
            mu[:n_new] = mu_new

            h_idx += 2
        else:
            # Odd last variable: univariate conditioning
            sigma_h = max(D[0], 1e-15)
            sd_h = np.sqrt(sigma_h)
            w_h = (a_ord[h_idx] - mu[0]) / sd_h
            prob *= max(1e-300, float(_ndtr(w_h)))
            h_idx += 1

    return max(0.0, min(1.0, prob))
