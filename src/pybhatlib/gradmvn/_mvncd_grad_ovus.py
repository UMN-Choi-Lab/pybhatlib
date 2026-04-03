"""Analytic gradients for MVNCD OVUS method via backward/adjoint differentiation.

Computes gradients of the OVUS-approximated multivariate normal CDF with respect
to upper integration limits (a) and covariance matrix elements (sigma, vech form).

OVUS = ME + bivariate screening (Bhat 2018, Section 2.3.1.1). The probability
factors are:
    P_init = BVN(w_0, w_1; rho_01)
    P_h = BVN(w_h_0, w_h_1; rho_h) / Phi(w_h_0)  for each screening step
so that P_OVUS = P_init * prod(P_h).

The conditioning steps (univariate truncation + Schur complement) are identical
to ME. The only difference is the probability factor at each step: bivariate
screening ratios instead of simple univariate Phi factors.

For K=1 and K=2, delegates to the exact analytic formulas in _mvncd_grad_analytic.
For K>=3, the OVUS forward pass is differentiated via reverse-mode AD.

References
----------
Bhat, C. R. (2018). New matrix-based methods for the analytic evaluation of
the multivariate cumulative normal distribution function. Transportation
Research Part B, 109, 238-256.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr as _ndtr

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _std_npdf(x: float) -> float:
    """Standard normal PDF, faster than scipy.stats.norm.pdf."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


from pybhatlib.gradmvn._mvncd import _bvn_cdf, _reorder_by_limits
from pybhatlib.gradmvn._mvncd_grad_analytic import (
    _grad_me_k1,
    _grad_me_k2,
)
from pybhatlib.gradmvn._trunc_grads import grad_cdf_bvn
from pybhatlib.gradmvn._univariate import bivariate_normal_cdf


def mvncd_grad_ovus_analytic(
    a: NDArray,
    sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """Analytic gradient of OVUS-approximated MVNCD.

    Parameters
    ----------
    a : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix (symmetric positive definite).

    Returns
    -------
    prob : float
        MVNCD probability (OVUS approximation for K >= 3).
    grad_a : ndarray, shape (K,)
        d(prob)/d(a_k).
    grad_sigma : ndarray, shape (K*(K+1)//2,)
        d(prob)/d(sigma_vech[idx]) for upper-triangular elements (row-based).
        Off-diagonal (i < j): accounts for both sigma_{ij} and sigma_{ji}.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    sigma = np.asarray(sigma, dtype=np.float64)
    K = len(a)

    if K == 0:
        return 1.0, np.array([]), np.array([])
    if K == 1:
        return _grad_me_k1(a, sigma)
    if K == 2:
        return _grad_me_k2(a, sigma)
    return _grad_ovus_adjoint(a, sigma)


def _extract_bvn_params(X, mu, a0, a1):
    """Extract standardized BVN parameters from state (X, mu) and limits a0, a1.

    Returns
    -------
    w0, w1 : float
        Standardized limits.
    rho : float
        Correlation.
    sd0, sd1 : float
        Standard deviations.
    sig0, sig1 : float
        Variances.
    """
    sig0 = max(X[0, 0], 1e-15)
    sig1 = max(X[1, 1], 1e-15)
    sd0 = np.sqrt(sig0)
    sd1 = np.sqrt(sig1)
    rho = X[0, 1] / (sd0 * sd1) if sd0 > 1e-7 and sd1 > 1e-7 else 0.0
    rho = max(-0.9999, min(0.9999, rho))
    w0 = (a0 - mu[0]) / sd0
    w1 = (a1 - mu[1]) / sd1
    return w0, w1, rho, sd0, sd1, sig0, sig1


def _bvn_screening_log_derivs(w0, w1, rho):
    """Compute BVN/Phi screening factor and derivatives of log(BVN/Phi).

    Returns
    -------
    log_factor : float
        log(BVN(w0, w1, rho) / Phi(w0))
    dlf_dw0 : float
        d(log_factor)/d(w0)
    dlf_dw1 : float
        d(log_factor)/d(w1)
    dlf_drho : float
        d(log_factor)/d(rho)
    """
    bvn = bivariate_normal_cdf(w0, w1, rho)
    phi_w0 = _std_npdf(w0)
    Phi_w0 = max(float(_ndtr(w0)), 1e-300)

    bvn = max(bvn, 1e-300)
    factor = bvn / Phi_w0
    factor = max(factor, 1e-300)
    log_factor = np.log(factor)

    # d(BVN)/d(w0, w1, rho) via standard grad
    gw0_bvn, gw1_bvn, grho_bvn = grad_cdf_bvn(w0, w1, rho)

    # d(log BVN)/d(w0, w1, rho)
    inv_bvn = 1.0 / bvn
    dlog_bvn_dw0 = gw0_bvn * inv_bvn
    dlog_bvn_dw1 = gw1_bvn * inv_bvn
    dlog_bvn_drho = grho_bvn * inv_bvn

    # d(log Phi(w0))/d(w0) = phi(w0)/Phi(w0)
    imr0 = phi_w0 / Phi_w0

    # d(log(BVN/Phi))/d(w0) = d(log BVN)/d(w0) - d(log Phi)/d(w0)
    dlf_dw0 = dlog_bvn_dw0 - imr0
    dlf_dw1 = dlog_bvn_dw1
    dlf_drho = dlog_bvn_drho

    return log_factor, dlf_dw0, dlf_dw1, dlf_drho


def _bvn_init_log_derivs(w0, w1, rho):
    """Compute initial BVN factor and derivatives of log(BVN).

    Returns
    -------
    log_factor : float
        log(BVN(w0, w1, rho))
    dlf_dw0 : float
        d(log_factor)/d(w0)
    dlf_dw1 : float
        d(log_factor)/d(w1)
    dlf_drho : float
        d(log_factor)/d(rho)
    """
    bvn = bivariate_normal_cdf(w0, w1, rho)
    bvn = max(bvn, 1e-300)
    log_factor = np.log(bvn)

    gw0_bvn, gw1_bvn, grho_bvn = grad_cdf_bvn(w0, w1, rho)
    inv_bvn = 1.0 / bvn

    dlf_dw0 = gw0_bvn * inv_bvn
    dlf_dw1 = gw1_bvn * inv_bvn
    dlf_drho = grho_bvn * inv_bvn

    return log_factor, dlf_dw0, dlf_dw1, dlf_drho


def _inject_bvn_adjoint(
    adj_X, adj_mu, grad_a_ord,
    dlf_dw0, dlf_dw1, dlf_drho,
    w0, w1, rho, sd0, sd1, sig0, sig1,
    a_idx0, a_idx1,
):
    """Inject BVN factor gradient into state adjoints (adj_X, adj_mu) and grad_a.

    The BVN factor depends on:
        w0 = (a[a_idx0] - mu[0]) / sd0
        w1 = (a[a_idx1] - mu[1]) / sd1
        rho = X[0,1] / (sd0 * sd1)
    where sd0 = sqrt(X[0,0]), sd1 = sqrt(X[1,1]).

    Propagate d(log_factor)/d(w0, w1, rho) back to d/d(X, mu, a).
    """
    # d/d(a[a_idx0]) = dlf_dw0 / sd0
    grad_a_ord[a_idx0] += dlf_dw0 / sd0

    # d/d(a[a_idx1]) = dlf_dw1 / sd1
    grad_a_ord[a_idx1] += dlf_dw1 / sd1

    # d/d(mu[0]) = dlf_dw0 * (-1/sd0)
    adj_mu[0] += dlf_dw0 * (-1.0 / sd0)

    # d/d(mu[1]) = dlf_dw1 * (-1/sd1)
    adj_mu[1] += dlf_dw1 * (-1.0 / sd1)

    # d/d(X[0,0]) via w0: dlf_dw0 * d(w0)/d(sig0) = dlf_dw0 * (-w0/(2*sig0))
    # d/d(X[0,0]) via rho: dlf_drho * d(rho)/d(sig0) = dlf_drho * (-rho/(2*sig0))
    adj_X[0, 0] += dlf_dw0 * (-w0 / (2.0 * sig0)) + dlf_drho * (-rho / (2.0 * sig0))

    # d/d(X[1,1]) via w1: dlf_dw1 * (-w1/(2*sig1))
    # d/d(X[1,1]) via rho: dlf_drho * (-rho/(2*sig1))
    adj_X[1, 1] += dlf_dw1 * (-w1 / (2.0 * sig1)) + dlf_drho * (-rho / (2.0 * sig1))

    # d/d(X[0,1]) via rho: dlf_drho * 1/(sd0*sd1)
    adj_X[0, 1] += dlf_drho / (sd0 * sd1)


def _grad_ovus_adjoint(
    a: NDArray, sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """OVUS gradient via backward/adjoint differentiation for K >= 3.

    The OVUS method approximates P(X <= a) by:
        P = BVN_init * prod_{h=0}^{K-3} [BVN_h / Phi_h]
    where each step does univariate conditioning (same as ME) followed by
    bivariate screening.

    Forward pass stores intermediate values; backward pass computes
    d(log P)/d(params) by reverse chain rule, then multiplies by P.
    """
    K = len(a)
    a_ord, sigma_ord, order = _reorder_by_limits(a, sigma)

    # Storage for conditioning step intermediates (same as ME)
    sigs = np.zeros(K)
    sds = np.zeros(K)
    ws = np.zeros(K)
    imrs = np.zeros(K)
    lams = np.zeros(K)
    deltas = np.zeros(K)
    vs = [None] * K
    shifts = np.zeros(K)
    gammas = np.zeros(K)

    # Storage for BVN/screening factor parameters at each step
    # bvn_params[h] = (w0, w1, rho, sd0, sd1, sig0, sig1, type)
    # type: 'init' for initial BVN, 'screen' for BVN/Phi, 'univ' for Phi
    bvn_params_list = []

    # ---- Forward pass ----
    mu = np.zeros(K, dtype=np.float64)
    X = sigma_ord.copy()
    log_prob = 0.0

    # Initial BVN factor (uses pre-conditioning state X_0, mu_0)
    w0, w1, rho, sd0, sd1, sig0, sig1 = _extract_bvn_params(
        X, mu, a_ord[0], a_ord[1]
    )
    log_init, dlf_dw0_init, dlf_dw1_init, dlf_drho_init = _bvn_init_log_derivs(
        w0, w1, rho
    )
    log_prob += log_init

    # Store initial BVN params and derivs
    init_bvn = {
        'w0': w0, 'w1': w1, 'rho': rho,
        'sd0': sd0, 'sd1': sd1, 'sig0': sig0, 'sig1': sig1,
        'dlf_dw0': dlf_dw0_init, 'dlf_dw1': dlf_dw1_init,
        'dlf_drho': dlf_drho_init,
    }

    if log_prob < -690:
        return 0.0, np.zeros(K), np.zeros(K * (K + 1) // 2)

    # Conditioning loop: h = 0 to K-3
    # After conditioning at h, compute BVN/Phi screening factor
    n_cond_steps = K - 2  # number of conditioning steps

    # Store post-conditioning screening info for each step
    screen_info = []  # list of dicts for each conditioning step

    for h in range(n_cond_steps):
        n = K - h  # dimension before conditioning

        # ---- Univariate conditioning (same as ME) ----
        sig = max(X[0, 0], 1e-15)
        sd = np.sqrt(sig)
        w = (a_ord[h] - mu[0]) / sd

        phi_w = _std_npdf(w)
        Phi_w = max(float(_ndtr(w)), 1e-300)
        imr = phi_w / Phi_w
        lam = -imr
        delta = lam * (lam - w)

        sigs[h] = sig
        sds[h] = sd
        ws[h] = w
        imrs[h] = imr
        lams[h] = lam
        deltas[h] = delta

        if n <= 1:
            break

        # Conditioning: compute v, shift, gamma and update state
        v = X[1:, 0] / sig
        shift = sd * lam
        gamma = max(sig * delta, 0.0)

        vs[h] = v.copy()
        shifts[h] = shift
        gammas[h] = gamma

        # Update state
        mu = mu[1:] + v * shift
        X = X[1:, 1:] - gamma * np.outer(v, v)

        # ---- Screening factor after conditioning ----
        n_new = n - 1  # = K - h - 1
        si = {}
        if n_new >= 2:
            w0_s, w1_s, rho_s, sd0_s, sd1_s, sig0_s, sig1_s = _extract_bvn_params(
                X, mu, a_ord[h + 1], a_ord[h + 2]
            )
            log_f, dlf_dw0, dlf_dw1, dlf_drho = _bvn_screening_log_derivs(
                w0_s, w1_s, rho_s
            )
            log_prob += log_f
            si = {
                'type': 'screen',
                'w0': w0_s, 'w1': w1_s, 'rho': rho_s,
                'sd0': sd0_s, 'sd1': sd1_s,
                'sig0': sig0_s, 'sig1': sig1_s,
                'dlf_dw0': dlf_dw0, 'dlf_dw1': dlf_dw1, 'dlf_drho': dlf_drho,
                'a_idx0': h + 1, 'a_idx1': h + 2,
            }
        else:
            # n_new == 1: just Phi for the last variable
            sig_last = max(X[0, 0], 1e-15)
            sd_last = np.sqrt(sig_last)
            w_last = (a_ord[h + 1] - mu[0]) / sd_last
            Phi_last = max(float(_ndtr(w_last)), 1e-300)
            log_prob += np.log(Phi_last)
            imr_last = _std_npdf(w_last) / Phi_last
            si = {
                'type': 'univ',
                'w': w_last, 'sd': sd_last, 'sig': sig_last,
                'imr': imr_last,
                'a_idx': h + 1,
            }

        screen_info.append(si)

        if log_prob < -690:
            return 0.0, np.zeros(K), np.zeros(K * (K + 1) // 2)

    prob = np.exp(log_prob)
    if prob < 1e-300:
        return 0.0, np.zeros(K), np.zeros(K * (K + 1) // 2)

    # ---- Backward pass ----
    grad_a_ord = np.zeros(K, dtype=np.float64)

    # Start from the last screening factor's adjoint on the post-conditioning state
    last_h = n_cond_steps - 1
    si_last = screen_info[last_h]

    if si_last['type'] == 'univ':
        # Last factor is Phi(w_last): same as ME terminal step
        w_last = si_last['w']
        sd_last = si_last['sd']
        sig_last = si_last['sig']
        imr_last = si_last['imr']

        # adj of log Phi w.r.t. w: imr
        adj_w = imr_last
        grad_a_ord[si_last['a_idx']] += adj_w / sd_last

        # Initialize adj_X_next, adj_mu_next for the 1x1 post-conditioning state
        adj_X_next = np.array([[-adj_w * w_last / (2.0 * sig_last)]])
        adj_mu_next = np.array([-adj_w / sd_last])

    elif si_last['type'] == 'screen':
        # Last factor is BVN/Phi screening
        si = si_last
        n_post = K - last_h - 1  # dimension of post-conditioning state (should be >= 2)

        adj_X_next = np.zeros((n_post, n_post), dtype=np.float64)
        adj_mu_next = np.zeros(n_post, dtype=np.float64)

        _inject_bvn_adjoint(
            adj_X_next, adj_mu_next, grad_a_ord,
            si['dlf_dw0'], si['dlf_dw1'], si['dlf_drho'],
            si['w0'], si['w1'], si['rho'],
            si['sd0'], si['sd1'], si['sig0'], si['sig1'],
            si['a_idx0'], si['a_idx1'],
        )

    # Reverse through conditioning steps
    for h in range(last_h, -1, -1):
        n = K - h  # dimension before conditioning at step h
        nm1 = n - 1

        sig, sd, w = sigs[h], sds[h], ws[h]
        lam, delta = lams[h], deltas[h]
        v, shift, gamma = vs[h], shifts[h], gammas[h]

        # If this is not the last step, we need to also inject the screening
        # factor adjoint for step h (it was computed using the state AFTER
        # conditioning at step h, which is also the state BEFORE conditioning
        # at step h+1). But we already injected all screening factors' adjoints
        # into their respective post-conditioning states.
        #
        # Wait — we only injected the LAST factor's adjoint. For earlier steps,
        # we need to inject them during the backward traversal.
        #
        # At this point, adj_X_next and adj_mu_next represent the total adjoint
        # of the (h+1)-th state (post-conditioning at h). This includes:
        # - Contributions from the screening factor at step h (already injected
        #   if h == last_h, or will be combined below for h < last_h)
        # - Contributions from the conditioning at step h+1 (propagated backward)
        #
        # For h < last_h: we need to ADD the screening factor adjoint for step h
        # to adj_X_next, adj_mu_next BEFORE reversing through the conditioning.
        #
        # Actually, the screening factor at step h uses the state AFTER
        # conditioning at step h. The backward pass naturally propagates
        # the adjoint from step h+1's conditioning backward to the state
        # AFTER conditioning at h. At that point, we add the screening factor
        # h's adjoint contribution. Then we reverse through conditioning at h.

        if h < last_h:
            # Add screening factor adjoint for step h
            si = screen_info[h]
            if si['type'] == 'screen':
                _inject_bvn_adjoint(
                    adj_X_next, adj_mu_next, grad_a_ord,
                    si['dlf_dw0'], si['dlf_dw1'], si['dlf_drho'],
                    si['w0'], si['w1'], si['rho'],
                    si['sd0'], si['sd1'], si['sig0'], si['sig1'],
                    si['a_idx0'], si['a_idx1'],
                )
            elif si['type'] == 'univ':
                # Shouldn't happen for h < last_h with K >= 3, but handle anyway
                si_u = si
                adj_w_u = si_u['imr']
                grad_a_ord[si_u['a_idx']] += adj_w_u / si_u['sd']
                adj_mu_next[0] += -adj_w_u / si_u['sd']
                adj_X_next[0, 0] += -adj_w_u * si_u['w'] / (2.0 * si_u['sig'])

        # ---- Reverse through conditioning step h (same as ME adjoint) ----
        adj_mu = np.zeros(n, dtype=np.float64)
        adj_X = np.zeros((n, n), dtype=np.float64)

        # Reverse: X_next = X[1:,1:] - gamma * outer(v, v)
        adj_X[1:, 1:] += adj_X_next
        adj_gamma = -np.dot(v, adj_X_next @ v)
        adj_v = -gamma * (adj_X_next + adj_X_next.T) @ v

        # Reverse: mu_next = mu[1:] + v * shift
        adj_mu[1:] += adj_mu_next
        adj_v += adj_mu_next * shift
        adj_shift = np.dot(adj_mu_next, v)

        # Reverse: v = X[1:,0] / sig
        adj_X[1:, 0] += adj_v / sig
        adj_sig = -np.dot(adj_v, v) / sig

        # Reverse: gamma = sig * delta
        adj_sig += adj_gamma * delta
        adj_delta_val = adj_gamma * sig

        # Reverse: shift = sqrt(sig) * lam
        adj_lam = adj_shift * sd
        adj_sig += adj_shift * lam / (2.0 * sd)

        # Reverse: delta = lam * (lam - w)
        adj_lam += adj_delta_val * (2.0 * lam - w)
        adj_w_h = -adj_delta_val * lam

        # Reverse: lam = -phi(w)/Phi(w), d(lam)/d(w) = delta
        adj_w_h += adj_lam * delta

        # NOTE: In ME, we add the factor contribution d(log Phi(w_h))/dw_h = imr_h
        # In OVUS, there is NO Phi(w_h) factor at the conditioning step.
        # The probability factors are the BVN and BVN/Phi screening factors,
        # which are handled separately via _inject_bvn_adjoint.
        # So we do NOT add imr_h here.

        # Reverse: w = (a_ord[h] - mu[0]) / sd
        grad_a_ord[h] += adj_w_h / sd
        adj_mu[0] -= adj_w_h / sd
        adj_sig -= adj_w_h * w / (2.0 * sig)

        # sig = X[0,0]
        adj_X[0, 0] += adj_sig

        adj_X_next = adj_X
        adj_mu_next = adj_mu

    # After reversing all conditioning steps, adj_X_next/adj_mu_next are
    # adjoints w.r.t. the initial state (X_0 = sigma_ord, mu_0 = 0).
    # We still need to add the initial BVN factor's adjoint contribution.
    _inject_bvn_adjoint(
        adj_X_next, adj_mu_next, grad_a_ord,
        init_bvn['dlf_dw0'], init_bvn['dlf_dw1'], init_bvn['dlf_drho'],
        init_bvn['w0'], init_bvn['w1'], init_bvn['rho'],
        init_bvn['sd0'], init_bvn['sd1'], init_bvn['sig0'], init_bvn['sig1'],
        0, 1,  # a indices for initial BVN
    )

    # adj_mu_next gives d(log P)/d(mu_0), but mu_0 = 0 (no parameters), so
    # it only contributes to grad_a via the initial BVN factor (already handled).

    # ---- Un-reorder to original variable ordering ----
    grad_a_log = np.zeros(K)
    grad_a_log[order] = grad_a_ord

    adj_sigma_full = np.zeros((K, K))
    adj_sigma_full[np.ix_(order, order)] = adj_X_next

    # ---- Convert full matrix adjoint to vech (row-based upper triangular) ----
    n_vech = K * (K + 1) // 2
    grad_sigma_log = np.zeros(n_vech)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            if i == j:
                grad_sigma_log[idx] = adj_sigma_full[i, i]
            else:
                grad_sigma_log[idx] = adj_sigma_full[i, j] + adj_sigma_full[j, i]
            idx += 1

    # Convert d(log P) to d(P)
    return prob, prob * grad_a_log, prob * grad_sigma_log
