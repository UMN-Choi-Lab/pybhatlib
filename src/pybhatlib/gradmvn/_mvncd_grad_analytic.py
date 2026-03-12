"""Analytic gradients for MVNCD ME method via backward/adjoint differentiation.

Computes gradients of the ME-approximated multivariate normal CDF with respect
to upper integration limits (a) and covariance matrix elements (sigma, vech form).

For K=1 and K=2, exact analytic formulas are used.
For K>=3, the ME sequential conditioning is differentiated via reverse-mode AD
(adjoint/backpropagation), giving O(K^2) cost per step.

Uses full covariance matrices in the forward pass (mathematically equivalent to
the LDLT-based _mvncd_me, but simpler to differentiate). Equivalence:
    X_{h+1} = X_h[1:,1:] - gamma_h * v_h v_h^T
matches the LDLT rank-1 update result via the Schur complement identity.

References
----------
Bhat, C. R. (2018). New matrix-based methods for the analytic evaluation of
the multivariate cumulative normal distribution function. Transportation
Research Part B, 109, 238-256.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.gradmvn._mvncd import _bvn_cdf, _reorder_by_limits
from pybhatlib.gradmvn._trunc_grads import grad_noncdfbvn


def mvncd_grad_me_analytic(
    a: NDArray,
    sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """Analytic gradient of ME-approximated MVNCD.

    Parameters
    ----------
    a : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix (symmetric positive definite).

    Returns
    -------
    prob : float
        MVNCD probability (ME approximation for K >= 3).
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
    return _grad_me_adjoint(a, sigma)


def _grad_me_k1(
    a: NDArray, sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """Exact gradient for K=1: P = Phi(a / sqrt(sigma))."""
    sig = sigma[0, 0]
    sd = np.sqrt(max(sig, 1e-30))
    w = a[0] / sd
    prob = float(norm.cdf(w))
    phi_w = norm.pdf(w)
    grad_a = np.array([phi_w / sd])
    grad_sigma = np.array([phi_w * (-w / (2.0 * sig))])
    return prob, grad_a, grad_sigma


def _grad_me_k2(
    a: NDArray, sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """Exact gradient for K=2 via bivariate normal CDF gradient."""
    mu = np.zeros(2)
    _gmu, gcov, gx = grad_noncdfbvn(mu, sigma, a)
    prob = _bvn_cdf(a, sigma)
    return prob, gx, gcov


def _grad_me_adjoint(
    a: NDArray, sigma: NDArray,
) -> tuple[float, NDArray, NDArray]:
    """ME gradient via backward/adjoint differentiation for K >= 3.

    The ME method approximates P(X <= a) by sequential univariate conditioning:
        P ~ prod_{h=0}^{K-1} Phi(w_h)
    where w_h is the standardized limit at step h after conditioning on
    previous variables being below their limits.

    Forward pass stores intermediate values; backward pass computes
    d(log P)/d(params) by reverse chain rule, then multiplies by P.
    """
    K = len(a)
    a_ord, sigma_ord, order = _reorder_by_limits(a, sigma)

    # Storage for backward pass
    sigs = np.zeros(K)
    sds = np.zeros(K)
    ws = np.zeros(K)
    imrs = np.zeros(K)         # phi(w)/Phi(w) = inverse Mills ratio
    lams = np.zeros(K)         # lambda = -phi/Phi
    deltas = np.zeros(K)       # delta = lambda*(lambda - w) = 1 - Var[Z|Z<=w]
    vs = [None] * K            # v_h = X[1:,0] / sig_h
    shifts = np.zeros(K)       # sqrt(sig) * lambda
    gammas = np.zeros(K)       # sig * delta

    # ---- Forward pass ----
    mu = np.zeros(K, dtype=np.float64)
    X = sigma_ord.copy()
    log_prob = 0.0

    for h in range(K):
        n = K - h
        sig = max(X[0, 0], 1e-15)
        sd = np.sqrt(sig)
        w = (a_ord[h] - mu[0]) / sd

        phi_w = norm.pdf(w)
        Phi_w = max(norm.cdf(w), 1e-300)
        imr = phi_w / Phi_w
        lam = -imr
        delta = lam * (lam - w)

        sigs[h] = sig
        sds[h] = sd
        ws[h] = w
        imrs[h] = imr
        lams[h] = lam
        deltas[h] = delta
        log_prob += np.log(Phi_w)

        if log_prob < -690:  # prob < ~1e-300
            return 0.0, np.zeros(K), np.zeros(K * (K + 1) // 2)

        if n <= 1:
            break

        # Conditioning step
        v = X[1:, 0] / sig
        shift = sd * lam
        gamma = max(sig * delta, 0.0)

        vs[h] = v.copy()
        shifts[h] = shift
        gammas[h] = gamma

        # Update state for next step
        mu = mu[1:] + v * shift
        X = X[1:, 1:] - gamma * np.outer(v, v)

    prob = np.exp(log_prob)
    if prob < 1e-300:
        return 0.0, np.zeros(K), np.zeros(K * (K + 1) // 2)

    # ---- Backward pass ----
    # Propagate d(log P)/d(state) from innermost step outward.
    grad_a_ord = np.zeros(K, dtype=np.float64)

    # Initialize from last step h=K-1 (1x1 state, factor only)
    h = K - 1
    adj_w = imrs[h]
    grad_a_ord[h] = adj_w / sds[h]
    adj_X_next = np.array([[-adj_w * ws[h] / (2.0 * sigs[h])]])
    adj_mu_next = np.array([-adj_w / sds[h]])

    # Reverse through conditioning steps K-2 down to 0
    for h in range(K - 2, -1, -1):
        n = K - h
        sig, sd, w = sigs[h], sds[h], ws[h]
        lam, delta = lams[h], deltas[h]
        v, shift, gamma = vs[h], shifts[h], gammas[h]

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

        # Factor contribution: d(log Phi(w))/dw = phi/Phi = imr
        adj_w_h += imrs[h]

        # Reverse: w = (a_ord[h] - mu[0]) / sd
        grad_a_ord[h] = adj_w_h / sd
        adj_mu[0] -= adj_w_h / sd
        adj_sig -= adj_w_h * w / (2.0 * sig)

        # sig = X[0,0]
        adj_X[0, 0] += adj_sig

        adj_X_next = adj_X
        adj_mu_next = adj_mu

    # adj_X_next is d(log P)/d(sigma_ord_{ij}) treating entries independently

    # ---- Un-reorder to original variable ordering ----
    grad_a_log = np.zeros(K)
    grad_a_log[order] = grad_a_ord

    adj_sigma_full = np.zeros((K, K))
    adj_sigma_full[np.ix_(order, order)] = adj_X_next

    # ---- Convert full matrix adjoint to vech (row-based upper triangular) ----
    # For symmetric sigma: vech entry (i,j) with i<=j maps to both
    # sigma_{ij} and sigma_{ji}, so grad_vech = adj[i,j] + adj[j,i] for i<j.
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
