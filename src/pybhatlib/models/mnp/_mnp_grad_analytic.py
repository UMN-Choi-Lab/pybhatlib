"""Analytic gradient of MNP log-likelihood via adjoint/backpropagation.

Implements Phase D1 from GAUSS_INTEGRATION_PLAN.md. Uses mvncd_grad_me_analytic
for per-observation probability gradients and chains them through the MNP
parameter transformations (beta, Lambda scales/correlations, Omega Cholesky).

Supports: IID, heteroscedastic, full covariance, mixed MNP (nseg=1), and
mixture-of-normals (nseg > 1) with analytic gradients.
Requires method="me" or "ovus" for the MVNCD approximation.

The gradient chain:
    d(-LL)/d(theta) = -(1/N) sum_q (1/P_q) dP_q/d(theta)

where dP_q/d(theta) decomposes as:
    dP/d(beta) via differenced design matrix
    dP/d(lambda_params) via Lambda = D @ corr @ D parameterization
    dP/d(omega_params) via Omega = L @ L.T Cholesky parameterization

For mixture-of-normals (nseg > 1):
    P_q = sum_h pi_h * P_q_h(beta_h, Lambda, Omega_h)
    d(log P_q)/d(theta) = (1/P_q) * sum_h [
        pi_h * d(P_q_h)/d(theta_h) + P_q_h * d(pi_h)/d(segment_params)
    ]

Phase 2 optimization: Pre-computes Lambda_diff, differencing matrices, and
Lambda_full outside the observation loop. Groups observations by chosen
alternative to reuse shared differencing structures.

Phase 1.4 optimization: Groups observations by chosen alternative for
vectorized diff_V/X_diff construction and batched gradient accumulation.

References
----------
Bhat, C. R. (2018). New matrix-based methods for the analytic evaluation of
the multivariate cumulative normal distribution function. Transportation
Research Part B, 109, 238-256.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pybhatlib.utils._safe_reparam import safe_exp

from pybhatlib.gradmvn._mvncd_grad_analytic import (
    mvncd_grad_me_analytic,
    mvncd_grad_batch_k2,
    mvncd_grad_batch_k2_perobs,
)
from pybhatlib.gradmvn._mvncd_grad_ovus import mvncd_grad_ovus_analytic
from pybhatlib.matgradient._spherical import grad_corr_theta, theta_to_corr
from pybhatlib.models.mnp._mnp_control import MNPControl


def mnp_analytic_gradient(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    avail: NDArray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
) -> tuple[float, NDArray]:
    """Compute MNP negative mean log-likelihood and its analytic gradient.

    Uses the ME or OVUS method for MVNCD and backward/adjoint differentiation
    for the gradient chain through all parameter transformations.

    Supports single-segment and mixture-of-normals (nseg > 1) models.

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector.
    X : ndarray, shape (N, n_alts, n_vars)
        Design matrix for all observations and alternatives.
    y : ndarray, shape (N,)
        Chosen alternative index (0-based) for each observation.
    avail : ndarray, shape (N, n_alts) or None
        Availability matrix (1=available, 0=unavailable). None means all.
    n_alts : int
        Number of alternatives.
    n_beta : int
        Number of beta coefficients.
    control : MNPControl
        Model control structure.
    ranvar_indices : list of int or None
        Indices in beta vector that are random coefficients.

    Returns
    -------
    nll : float
        Negative mean log-likelihood (ME/OVUS approximation).
    grad : ndarray, shape (n_params,)
        Gradient of negative mean log-likelihood.
    """
    if control.nseg > 1:
        return _mixture_analytic_gradient(
            theta, X, y, avail, n_alts, n_beta, control, ranvar_indices,
        )

    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    N = X.shape[0]
    I = n_alts
    dim_lambda = I - 1  # dimension of Lambda (alts 1..I-1)
    n_params = len(theta)

    # ---- Unpack parameters ----
    idx = 0
    beta = theta[idx:idx + n_beta]
    idx += n_beta

    lambda_params = None
    n_scale = 0
    n_corr = 0
    n_lambda = 0
    if not control.iid:
        # GAUSS homogeneous form: I-2 FREE scales (first diff variance pinned).
        n_scale = max(dim_lambda - 1, 0)
        if control.heteronly:
            n_lambda = n_scale
        else:
            n_corr = dim_lambda * (dim_lambda - 1) // 2
            n_lambda = n_scale + n_corr
        lambda_params = theta[idx:idx + n_lambda]
        idx += n_lambda

    omega_params = None
    n_rand = 0
    n_omega = 0
    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            n_omega = n_rand
        else:
            n_omega = n_rand * (n_rand + 1) // 2
        omega_params = theta[idx:idx + n_omega]
        idx += n_omega

    # ---- Build covariance components ----
    # ``Lambda`` here is the DIFFERENCED kernel K (K[0,0] == 1 pinned).
    Lambda, scales, corr = _build_lambda_components(
        lambda_params, dim_lambda, control
    )

    Omega_L = None
    Omega = None
    if control.mix and omega_params is not None:
        Omega_L, Omega = _build_omega_components(omega_params, n_rand, control)

    # Pre-compute correlation Jacobian (once per gradient call, not per obs)
    corr_jac = None
    if not control.iid and not control.heteronly and n_corr > 0:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr_jac = grad_corr_theta(corr_theta, dim_lambda)

    # ---- Pre-compute shared quantities outside the loop ----
    need_sigma_chain = (not control.iid) or (control.mix and Omega is not None)
    all_avail = avail is None or np.all(avail > 0.5)
    dim = I - 1  # dimension when all alternatives available
    has_random_coeff = control.mix and Omega is not None

    # Pre-compute utilities for all observations: V_all = X @ beta
    V_all = np.einsum('nij,j->ni', X, beta)

    # GAUSS homogeneous form: kernel covariance lives in the undifferenced
    # I-space as ``covker = blockdiag(0, K)`` (covker[0,0] = 0, K[0,0] pinned
    # to 1). This replaces the old floor ``Lambda_full = eye; [1:,1:]=Lambda+eye``.
    # Built ALWAYS (incl. IID, which routes the frozen startker K through the
    # same covker path), so the gradient's Lambda_diff matches the forward model.
    Lambda_full = np.zeros((I, I), dtype=np.float64)
    Lambda_full[1:, 1:] = Lambda

    # ---- Dispatch to batched or sequential path ----
    if all_avail and not has_random_coeff:
        return _batched_gradient_shared_cov(
            X, y, V_all, beta, Lambda, Lambda_full, Omega, Omega_L,
            omega_params, scales, corr, corr_jac,
            ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
            n_params, n_scale, n_corr, dim_lambda, n_rand,
            need_sigma_chain,
        )
    else:
        return _sequential_gradient(
            X, y, V_all, avail, beta, Lambda, Lambda_full, Omega, Omega_L,
            omega_params, scales, corr, corr_jac,
            ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
            n_params, n_scale, n_corr, dim_lambda, n_rand,
            need_sigma_chain, all_avail,
        )


def mnp_per_obs_scores(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    avail: NDArray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
) -> NDArray:
    """Per-observation score matrix ``S[q, k] = d log P_q / d theta_k``.

    This is the single-analytic-pass replacement for the central-finite-
    difference BHHH/sandwich score loop (which costs ``2 * n_params`` full-data
    per-observation log-likelihood evaluations). It reuses the same adjoint
    machinery as :func:`mnp_analytic_gradient`, but stores each observation's
    score row instead of summing them. By construction
    ``S.sum(axis=0) == -N * grad`` where ``grad`` is the gradient returned by
    :func:`mnp_analytic_gradient` (negative mean log-likelihood).

    The returned scores are in the **parameterized** ``theta`` space (the
    optimiser's space). Callers that need unparameterized-space scores apply the
    par->unpar Jacobian (BHHH covariance transforms exactly under
    reparameterisation, so ``cov_unpar = J_pu @ inv(S^T S) @ J_pu.T``).

    Parameters mirror :func:`mnp_analytic_gradient`. Requires ``method in
    {"me", "ovus"}`` and single-segment models (``nseg == 1``); mixture models
    raise :class:`NotImplementedError` so the caller can fall back to finite
    differences.

    Returns
    -------
    scores : ndarray, shape (N, n_params)
        Per-observation scores ``d log P_q / d theta``.
    """
    if control.nseg > 1:
        raise NotImplementedError(
            "mnp_per_obs_scores does not support mixture models (nseg > 1); "
            "fall back to finite-difference scoring."
        )
    if control.method not in ("me", "ovus"):
        raise NotImplementedError(
            f"mnp_per_obs_scores requires method in {{me, ovus}}, "
            f"got {control.method!r}."
        )

    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    N = X.shape[0]
    I = n_alts
    dim_lambda = I - 1
    n_params = len(theta)

    # ---- Unpack parameters (same layout as mnp_analytic_gradient) ----
    idx = 0
    beta = theta[idx:idx + n_beta]
    idx += n_beta

    lambda_params = None
    n_scale = 0
    n_corr = 0
    n_lambda = 0
    if not control.iid:
        # GAUSS homogeneous form: I-2 FREE scales (first diff variance pinned).
        n_scale = max(dim_lambda - 1, 0)
        if control.heteronly:
            n_lambda = n_scale
        else:
            n_corr = dim_lambda * (dim_lambda - 1) // 2
            n_lambda = n_scale + n_corr
        lambda_params = theta[idx:idx + n_lambda]
        idx += n_lambda

    omega_params = None
    n_rand = 0
    n_omega = 0
    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            n_omega = n_rand
        else:
            n_omega = n_rand * (n_rand + 1) // 2
        omega_params = theta[idx:idx + n_omega]
        idx += n_omega

    # ``Lambda`` here is the DIFFERENCED kernel K (K[0,0] == 1 pinned).
    Lambda, scales, corr = _build_lambda_components(
        lambda_params, dim_lambda, control
    )

    Omega_L = None
    Omega = None
    if control.mix and omega_params is not None:
        Omega_L, Omega = _build_omega_components(omega_params, n_rand, control)

    corr_jac = None
    if not control.iid and not control.heteronly and n_corr > 0:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr_jac = grad_corr_theta(corr_theta, dim_lambda)

    has_random_coeff = control.mix and Omega is not None
    need_sigma_chain = (not control.iid) or has_random_coeff

    # GAUSS homogeneous form: covker = blockdiag(0, K) (covker[0,0] = 0,
    # K[0,0] pinned). Built ALWAYS (incl. IID) so Lambda_diff matches forward.
    Lambda_full = np.zeros((I, I), dtype=np.float64)
    Lambda_full[1:, 1:] = Lambda

    V_all = np.einsum('nij,j->ni', X, beta)
    lam_lo = n_beta
    lam_hi = n_beta + n_lambda
    om_lo = lam_hi
    om_hi = lam_hi + n_omega

    scores = np.zeros((N, n_params), dtype=np.float64)

    for q in range(N):
        chosen = int(y[q])
        if avail is None:
            avail_alts = [j for j in range(I) if j != chosen]
        else:
            avail_q = avail[q]
            avail_alts = [
                j for j in range(I) if j != chosen and avail_q[j] > 0.5
            ]
        dim_q = len(avail_alts)
        if dim_q == 0:
            continue

        M = np.zeros((dim_q, I), dtype=np.float64)
        for k, j in enumerate(avail_alts):
            M[k, j] = 1.0
            M[k, chosen] = -1.0

        diff_V = np.array([V_all[q, chosen] - V_all[q, j] for j in avail_alts])
        X_diff = np.array([X[q, chosen] - X[q, j] for j in avail_alts])

        # GAUSS homogeneous form: covker = blockdiag(0, K). IID routes through
        # the same covker path. Random coefficients add Omega_tilde in the
        # undifferenced I-space: Xi_full = Omega_tilde + covker.
        Xi_base = Lambda_full if Lambda_full is not None else np.eye(I)
        if has_random_coeff:
            X_rand = X[q][:, ranvar_indices]
            Xi_full = X_rand @ Omega @ X_rand.T + Xi_base
        else:
            Xi_full = Xi_base
        Lambda_diff = M @ Xi_full @ M.T
        Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

        if control.method == "ovus":
            prob, grad_a, grad_sigma_vech = mvncd_grad_ovus_analytic(
                diff_V, Lambda_diff
            )
        else:
            prob, grad_a, grad_sigma_vech = mvncd_grad_me_analytic(
                diff_V, Lambda_diff
            )
        prob = max(prob, 1e-300)
        inv_p = 1.0 / prob

        # Beta score
        scores[q, :n_beta] = inv_p * (X_diff.T @ grad_a)

        # Covariance parameter scores
        if need_sigma_chain:
            adj_Lambda_diff = _vech_to_symmetric(grad_sigma_vech, dim_q)
            adj_Xi_full = M.T @ adj_Lambda_diff @ M

            if not control.iid and n_lambda > 0:
                adj_Lambda = adj_Xi_full[1:, 1:]
                adj_lp = _adj_lambda_to_params(
                    adj_Lambda, scales, corr, corr_jac,
                    dim_lambda, n_scale, n_corr, control,
                )
                scores[q, lam_lo:lam_hi] = inv_p * adj_lp

            if has_random_coeff and n_omega > 0:
                X_rand = X[q][:, ranvar_indices]
                adj_Omega = X_rand.T @ adj_Xi_full @ X_rand
                adj_op = _adj_omega_to_params(
                    adj_Omega, Omega_L, omega_params, n_rand, control,
                )
                scores[q, om_lo:om_hi] = inv_p * adj_op

    return scores


# ---------------------------------------------------------------------------
# Mixture-of-normals analytic gradient (Phase 3.2)
# ---------------------------------------------------------------------------


def _mixture_analytic_gradient(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    avail: NDArray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
) -> tuple[float, NDArray]:
    """Analytic gradient for mixture-of-normals MNP (nseg > 1).

    Decomposes the mixture likelihood gradient as:
        d(log P_q)/d(theta) = (1/P_q) * sum_h [
            pi_h * d(P_q_h)/d(theta_h) + P_q_h * d(pi_h)/d(segment_params)
        ]

    For each segment h, reuses the single-segment MVNCD gradient machinery
    to compute d(P_q_h)/d(theta_h).

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Full parameter vector for mixture model.
    X : ndarray, shape (N, n_alts, n_vars)
        Design matrix.
    y : ndarray, shape (N,)
        Chosen alternative (0-based).
    avail : ndarray, shape (N, n_alts) or None
        Availability matrix.
    n_alts : int
        Number of alternatives.
    n_beta : int
        Number of beta coefficients.
    control : MNPControl
        Model control structure.
    ranvar_indices : list of int or None
        Indices of random coefficients in beta.

    Returns
    -------
    nll : float
        Negative mean log-likelihood.
    grad : ndarray, shape (n_params,)
        Gradient of negative mean log-likelihood.
    """
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    N = X.shape[0]
    I = n_alts
    nseg = control.nseg
    dim_lambda = I - 1
    n_params = len(theta)

    # ---- Compute dimension sizes ----
    n_lambda = 0
    n_scale = 0
    n_corr = 0
    if not control.iid:
        # GAUSS homogeneous form: I-2 FREE scales (first diff variance pinned).
        n_scale = max(dim_lambda - 1, 0)
        if control.heteronly:
            n_lambda = n_scale
        else:
            n_corr = dim_lambda * (dim_lambda - 1) // 2
            n_lambda = n_scale + n_corr

    n_rand = 0
    n_omega = 0
    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            n_omega = n_rand
        else:
            n_omega = n_rand * (n_rand + 1) // 2

    n_seg_params = nseg - 1

    # ---- Unpack parameter vector ----
    # Layout: [beta_1 | lambda_params | omega_params_1 | segment_params |
    #          beta_2 | omega_params_2 | ... | beta_H | omega_params_H]
    idx = 0
    betas = []
    omega_params_list = []

    # Segment 1
    beta_1 = theta[idx:idx + n_beta]
    idx += n_beta
    betas.append(beta_1)

    lambda_params = None
    if not control.iid:
        lambda_params = theta[idx:idx + n_lambda]
        idx += n_lambda

    omega_params_1 = None
    if control.mix and ranvar_indices is not None:
        omega_params_1 = theta[idx:idx + n_omega]
        idx += n_omega
    omega_params_list.append(omega_params_1)

    # Segment probabilities
    seg_params_start = idx
    segment_params = theta[idx:idx + n_seg_params]
    idx += n_seg_params

    # Additional segments
    for h in range(1, nseg):
        beta_h = theta[idx:idx + n_beta]
        idx += n_beta
        betas.append(beta_h)

        omega_h = None
        if control.mix and ranvar_indices is not None:
            omega_h = theta[idx:idx + n_omega]
            idx += n_omega
        omega_params_list.append(omega_h)

    # ---- Compute segment probabilities via softmax ----
    raw = np.concatenate([[0.0], segment_params])
    raw_max = raw.max()
    exp_raw = np.exp(raw - raw_max)
    pi_h = exp_raw / exp_raw.sum()

    # ---- Build shared Lambda components (shared across segments) ----
    Lambda, scales, corr = _build_lambda_components(
        lambda_params, dim_lambda, control,
    )

    corr_jac = None
    if not control.iid and not control.heteronly and n_corr > 0:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr_jac = grad_corr_theta(corr_theta, dim_lambda)

    # GAUSS homogeneous form: covker = blockdiag(0, K) (covker[0,0] = 0,
    # K[0,0] pinned). Built ALWAYS (incl. IID) so Lambda_diff matches forward.
    Lambda_full = np.zeros((I, I), dtype=np.float64)
    Lambda_full[1:, 1:] = Lambda

    # ---- Build per-segment Omega components ----
    Omega_Ls = []
    Omegas = []
    for h in range(nseg):
        if control.mix and omega_params_list[h] is not None:
            L_h, Om_h = _build_omega_components(
                omega_params_list[h], n_rand, control,
            )
            Omega_Ls.append(L_h)
            Omegas.append(Om_h)
        else:
            Omega_Ls.append(None)
            Omegas.append(None)

    # ---- Per-observation gradient accumulation ----
    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)
    dim = I - 1

    # Determine parameter index ranges for each segment
    # Segment 1: beta at [0, n_beta), lambda at [n_beta, n_beta+n_lambda),
    #            omega at [n_beta+n_lambda, n_beta+n_lambda+n_omega)
    # segment_params at [n_beta+n_lambda+n_omega, n_beta+n_lambda+n_omega+nseg-1)
    # Segment h (h>=2): beta at seg_h_start, omega at seg_h_start + n_beta
    beta_1_start = 0
    lambda_start = n_beta
    omega_1_start = n_beta + n_lambda

    seg_extra_starts = []  # (beta_start, omega_start) for segments 2..H
    extra_idx = seg_params_start + n_seg_params
    for h in range(1, nseg):
        beta_h_start = extra_idx
        extra_idx += n_beta
        omega_h_start = extra_idx if (control.mix and ranvar_indices is not None) else extra_idx
        if control.mix and ranvar_indices is not None:
            extra_idx += n_omega
        seg_extra_starts.append((beta_h_start, omega_h_start))

    all_avail = avail is None or np.all(avail > 0.5)
    has_random = control.mix and ranvar_indices is not None

    # ---- Fast vectorized path for K=2, all available, no random coeff ----
    if dim == 2 and all_avail and not has_random:
        return _mixture_vectorized_k2(
            X, y, betas, Lambda, Lambda_full, pi_h,
            scales, corr, corr_jac,
            control, I, N, dim, n_beta, n_lambda, n_params,
            n_scale, n_corr, dim_lambda, nseg, n_seg_params,
            beta_1_start, lambda_start, seg_params_start,
            seg_extra_starts,
        )

    for q in range(N):
        chosen = y[q]

        if all_avail:
            avail_alts = [j for j in range(I) if j != chosen]
        else:
            avail_q = avail[q] if avail is not None else np.ones(I)
            avail_alts = [j for j in range(I) if j != chosen and avail_q[j] > 0.5]

        dim_q = len(avail_alts)
        if dim_q == 0:
            continue

        # Build differencing matrix M
        M = np.zeros((dim_q, I), dtype=np.float64)
        for k, j in enumerate(avail_alts):
            M[k, j] = 1.0
            M[k, chosen] = -1.0

        # X_diff for this observation (needed for beta gradient)
        X_diff = np.array([X[q, chosen] - X[q, j] for j in avail_alts])

        # X_rand for this observation (needed for Omega gradient)
        X_rand = None
        if control.mix and ranvar_indices is not None:
            X_rand = X[q][:, ranvar_indices]  # (I, n_rand)

        # ---- Compute per-segment P_q_h and gradient ----
        P_q_h_vals = np.zeros(nseg)
        # Per-segment gradient contributions (to their own parameter slots)
        seg_grad_beta = []
        seg_grad_lambda = []
        seg_grad_omega = []

        for h in range(nseg):
            beta_h = betas[h]
            Omega_h = Omegas[h]
            Omega_L_h = Omega_Ls[h]
            omega_params_h = omega_params_list[h]

            # Differenced utilities
            V_h = X[q] @ beta_h  # (I,)
            diff_V = np.array([V_h[chosen] - V_h[j] for j in avail_alts])

            # Build differenced covariance for this segment.
            # GAUSS homogeneous form: covker = blockdiag(0, K) (covker[0,0]=0,
            # K[0,0] pinned). IID routes through the same covker path; random
            # coefficients add Omega_tilde in the undifferenced I-space.
            has_random = Omega_h is not None
            Xi_base = Lambda_full if Lambda_full is not None else np.eye(I)
            if has_random:
                Omega_tilde = X_rand @ Omega_h @ X_rand.T  # (I, I)
                Xi_full = Omega_tilde + Xi_base
            else:
                Xi_full = Xi_base
            Lambda_diff = M @ Xi_full @ M.T
            Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

            # MVNCD probability + gradient
            if control.method == "ovus":
                prob_h, grad_a_h, grad_sigma_vech_h = mvncd_grad_ovus_analytic(
                    diff_V, Lambda_diff,
                )
            else:
                prob_h, grad_a_h, grad_sigma_vech_h = mvncd_grad_me_analytic(
                    diff_V, Lambda_diff,
                )
            prob_h = max(prob_h, 1e-300)
            P_q_h_vals[h] = prob_h

            # Beta gradient: dP_h/d(beta_h) = X_diff.T @ grad_a_h
            grad_beta_h = X_diff.T @ grad_a_h
            seg_grad_beta.append(grad_beta_h)

            # Lambda gradient (shared across segments)
            need_sigma = (not control.iid) or has_random
            grad_lambda_h = np.zeros(n_lambda) if n_lambda > 0 else None
            grad_omega_h = np.zeros(n_omega) if n_omega > 0 else None

            if need_sigma:
                adj_Lambda_diff = _vech_to_symmetric(grad_sigma_vech_h, dim_q)
                adj_Xi_full = M.T @ adj_Lambda_diff @ M

                if not control.iid and n_lambda > 0:
                    adj_Lambda = adj_Xi_full[1:, 1:]
                    grad_lambda_h = _adj_lambda_to_params(
                        adj_Lambda, scales, corr, corr_jac,
                        dim_lambda, n_scale, n_corr, control,
                    )

                if has_random and n_omega > 0:
                    adj_Omega = X_rand.T @ adj_Xi_full @ X_rand
                    grad_omega_h = _adj_omega_to_params(
                        adj_Omega, Omega_L_h, omega_params_h, n_rand, control,
                    )

            seg_grad_lambda.append(grad_lambda_h)
            seg_grad_omega.append(grad_omega_h)

        # ---- Mixture probability ----
        P_q = np.dot(pi_h, P_q_h_vals)
        P_q = max(P_q, 1e-300)
        total_ll += np.log(P_q)
        inv_P_q = 1.0 / P_q

        # ---- Accumulate weighted gradients ----
        # Segment 1 beta
        grad[beta_1_start:beta_1_start + n_beta] += (
            inv_P_q * pi_h[0] * seg_grad_beta[0]
        )

        # Shared lambda params: accumulate from ALL segments
        if n_lambda > 0:
            total_grad_lambda = np.zeros(n_lambda)
            for h in range(nseg):
                if seg_grad_lambda[h] is not None:
                    total_grad_lambda += pi_h[h] * seg_grad_lambda[h]
            grad[lambda_start:lambda_start + n_lambda] += (
                inv_P_q * total_grad_lambda
            )

        # Segment 1 omega
        if n_omega > 0 and seg_grad_omega[0] is not None:
            grad[omega_1_start:omega_1_start + n_omega] += (
                inv_P_q * pi_h[0] * seg_grad_omega[0]
            )

        # Segment params gradient (softmax Jacobian)
        # d(pi_h)/d(s_k) = pi_h * (delta_{hk} - pi_k) for k=1..nseg-1
        # d(P_q)/d(s_k) = sum_h P_q_h * d(pi_h)/d(s_k)
        #               = sum_h P_q_h * pi_h * (delta_{hk} - pi_k)
        for k_idx in range(n_seg_params):
            k = k_idx + 1  # segment index (1-based, since s_0=0 is reference)
            dPq_dsk = 0.0
            for h in range(nseg):
                delta_hk = 1.0 if h == k else 0.0
                dPq_dsk += P_q_h_vals[h] * pi_h[h] * (delta_hk - pi_h[k])
            grad[seg_params_start + k_idx] += inv_P_q * dPq_dsk

        # Additional segments (h >= 2): beta and omega
        for h in range(1, nseg):
            h_beta_start, h_omega_start = seg_extra_starts[h - 1]

            # Beta for segment h
            grad[h_beta_start:h_beta_start + n_beta] += (
                inv_P_q * pi_h[h] * seg_grad_beta[h]
            )

            # Omega for segment h
            if n_omega > 0 and seg_grad_omega[h] is not None:
                grad[h_omega_start:h_omega_start + n_omega] += (
                    inv_P_q * pi_h[h] * seg_grad_omega[h]
                )

    nll = -total_ll / N
    grad = -grad / N
    return nll, grad


def _mixture_vectorized_k2(
    X, y, betas, Lambda, Lambda_full,
    pi_h, scales, corr, corr_jac,
    control, I, N, dim, n_beta, n_lambda, n_params,
    n_scale, n_corr, dim_lambda, nseg, n_seg_params,
    beta_1_start, lambda_start, seg_params_start,
    seg_extra_starts,
):
    """Vectorized mixture gradient for K=2, all available, no random coeff.

    Batches the MVNCD gradient across observations within each
    (segment, chosen_alt) group, eliminating per-observation Python loops.
    """
    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)

    unique_chosen = np.unique(y)

    # Pre-compute per-segment utilities
    V_all_segs = []
    for h in range(nseg):
        V_all_segs.append(np.einsum('nij,j->ni', X, betas[h]))

    # Per-segment, per-chosen-alt: compute batch MVNCD gradient
    # P_q_h[q, h] = MVNCD prob for obs q under segment h
    P_q_h_all = np.empty((N, nseg), dtype=np.float64)

    # Per-segment beta gradient contribution (before weighting by pi_h/P_q)
    seg_grad_beta_all = []

    need_sigma_chain = not control.iid

    # Cache for sigma gradient data (per segment/chosen group)
    grad_sv_cache = {}

    for h in range(nseg):
        V_all_h = V_all_segs[h]

        # Per-obs beta gradient for this segment
        grad_beta_h_all = np.zeros((N, n_beta), dtype=np.float64)

        for c in unique_chosen:
            mask = y == c
            obs_indices = np.where(mask)[0]
            N_c = len(obs_indices)
            if N_c == 0:
                continue

            avail_alts_c = [j for j in range(I) if j != c]
            M_c = np.zeros((dim, I), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                M_c[k, j] = 1.0
                M_c[k, c] = -1.0

            # GAUSS homogeneous form: covker = blockdiag(0, K). IID routes
            # through the same covker path (K = 0.5*(eye+ones)).
            Xi_full = Lambda_full if Lambda_full is not None else np.eye(I)
            Lambda_diff_c = M_c @ Xi_full @ M_c.T
            Lambda_diff_c = 0.5 * (Lambda_diff_c + Lambda_diff_c.T)

            # Vectorized diff_V
            V_group = V_all_h[obs_indices]
            diff_V_group = np.empty((N_c, dim), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                diff_V_group[:, k] = V_group[:, c] - V_group[:, j]

            # Vectorized X_diff
            X_group = X[obs_indices]
            X_diff_group = np.empty((N_c, dim, X.shape[2]), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                X_diff_group[:, k, :] = X_group[:, c, :] - X_group[:, j, :]

            # Batch MVNCD gradient
            prob_all, grad_a_all, grad_sv_all = mvncd_grad_batch_k2(
                diff_V_group, Lambda_diff_c
            )

            P_q_h_all[obs_indices, h] = prob_all

            # Per-obs beta gradient: vectorized via einsum
            grad_beta_h_all[obs_indices] = np.einsum(
                'nkv,nk->nv', X_diff_group, grad_a_all
            )

            # Cache sigma gradient for later weighting
            if need_sigma_chain:
                grad_sv_cache[(h, c)] = (obs_indices, grad_sv_all, M_c)

        seg_grad_beta_all.append(grad_beta_h_all)

    # ---- Compute mixture probabilities and accumulate gradients ----
    # P_q = sum_h pi_h * P_q_h
    P_q_all = P_q_h_all @ pi_h  # (N,)
    P_q_all = np.maximum(P_q_all, 1e-300)
    total_ll = float(np.sum(np.log(P_q_all)))
    inv_P_q = 1.0 / P_q_all  # (N,)

    # Beta gradients per segment
    for h in range(nseg):
        # Weight: (1/P_q) * pi_h * dP_q_h/d(beta_h)
        weights = inv_P_q * pi_h[h]  # (N,)
        weighted_grad = weights[:, None] * seg_grad_beta_all[h]  # (N, n_beta)
        grad_beta_h = weighted_grad.sum(axis=0)

        if h == 0:
            grad[beta_1_start:beta_1_start + n_beta] += grad_beta_h
        else:
            h_beta_start = seg_extra_starts[h - 1][0]
            grad[h_beta_start:h_beta_start + n_beta] += grad_beta_h

    # Lambda gradient (shared across segments)
    if need_sigma_chain and n_lambda > 0:
        for h in range(nseg):
            for c in unique_chosen:
                key = (h, c)
                if key not in grad_sv_cache:
                    continue
                obs_indices, grad_sv_all, M_c = grad_sv_cache[key]

                # Weight by (1/P_q) * pi_h
                weights = inv_P_q[obs_indices] * pi_h[h]  # (N_c,)
                weighted_gsv = weights[:, None] * grad_sv_all  # (N_c, 3)
                sum_gsv = weighted_gsv.sum(axis=0)  # (3,)

                adj_Lambda_diff = np.array([
                    [sum_gsv[0], sum_gsv[1] / 2.0],
                    [sum_gsv[1] / 2.0, sum_gsv[2]],
                ])
                adj_Xi_full = M_c.T @ adj_Lambda_diff @ M_c

                if not control.iid:
                    adj_Lambda = adj_Xi_full[1:, 1:]
                    adj_lp = _adj_lambda_to_params(
                        adj_Lambda, scales, corr, corr_jac,
                        dim_lambda, n_scale, n_corr, control,
                    )
                    grad[lambda_start:lambda_start + n_lambda] += adj_lp

    # Segment probability gradients (softmax Jacobian, vectorized)
    for k_idx in range(n_seg_params):
        k = k_idx + 1
        # dP_q/ds_k = sum_h P_q_h * pi_h * (delta_{hk} - pi_k)
        softmax_jac = pi_h * (np.eye(nseg)[k] - pi_h[k])  # (nseg,)
        dPq_dsk = P_q_h_all @ softmax_jac  # (N,)
        grad[seg_params_start + k_idx] += float(np.sum(inv_P_q * dPq_dsk))

    nll = -total_ll / N
    grad = -grad / N
    return nll, grad


def _batched_gradient_shared_cov(
    X, y, V_all, beta, Lambda, Lambda_full, Omega, Omega_L,
    omega_params, scales, corr, corr_jac,
    ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
    n_params, n_scale, n_corr, dim_lambda, n_rand,
    need_sigma_chain,
):
    """Batched gradient for shared covariance (no random coefficients, full avail).

    Groups observations by chosen alternative for vectorized diff_V/X_diff
    construction and shared Lambda_diff/M matrices.

    For K=2 (dim=2), uses fully vectorized BVN gradient computation
    that eliminates the per-observation Python loop entirely (~200x speedup).
    """
    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)

    unique_chosen = np.unique(y)

    for c in unique_chosen:
        mask = y == c
        obs_indices = np.where(mask)[0]
        N_c = len(obs_indices)
        if N_c == 0:
            continue

        # Build shared structures for chosen=c
        avail_alts_c = [j for j in range(I) if j != c]
        M_c = np.zeros((dim, I), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            M_c[k, j] = 1.0
            M_c[k, c] = -1.0

        # GAUSS homogeneous form: Lambda_diff = M @ covker @ M.T with
        # covker = blockdiag(0, K). IID routes through the SAME covker path
        # (K = 0.5*(eye+ones)), matching _mnp_loglik._batch_loglik_shared_cov.
        Xi_full = Lambda_full if Lambda_full is not None else np.eye(I)
        Lambda_diff_c = M_c @ Xi_full @ M_c.T
        Lambda_diff_c = 0.5 * (Lambda_diff_c + Lambda_diff_c.T)

        # Vectorized diff_V: shape (N_c, dim)
        V_group = V_all[obs_indices]  # (N_c, I)
        diff_V_group = np.empty((N_c, dim), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            diff_V_group[:, k] = V_group[:, c] - V_group[:, j]

        # Vectorized X_diff: shape (N_c, dim, n_beta)
        X_group = X[obs_indices]  # (N_c, I, n_beta)
        X_diff_group = np.empty((N_c, dim, X.shape[2]), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            X_diff_group[:, k, :] = X_group[:, c, :] - X_group[:, j, :]

        # --- Vectorized path for K=2 (bivariate) ---
        if dim == 2:
            # Batch all N_c observations in one vectorized call
            prob_all, grad_a_all, grad_sv_all = mvncd_grad_batch_k2(
                diff_V_group, Lambda_diff_c
            )

            total_ll += float(np.sum(np.log(prob_all)))
            inv_p_all = 1.0 / prob_all  # (N_c,)

            # Vectorized beta gradient:
            #   grad[:n_beta] += sum_q inv_p_q * X_diff_q.T @ grad_a_q
            # = einsum('nkv,nk->v', X_diff, inv_p * grad_a)
            weighted_grad_a = inv_p_all[:, None] * grad_a_all  # (N_c, 2)
            grad[:n_beta] += np.einsum(
                'nkv,nk->v', X_diff_group, weighted_grad_a
            )

            # Covariance parameter gradients (vectorized)
            if need_sigma_chain:
                # grad_sv_all: (N_c, 3) with entries (ds11, ds12, ds22)
                # Weight by 1/prob
                weighted_gsv = inv_p_all[:, None] * grad_sv_all  # (N_c, 3)
                # Sum over observations
                sum_gsv = weighted_gsv.sum(axis=0)  # (3,)

                # Build adjoint of Lambda_diff from summed vech gradient.
                # Off-diagonal vech entry accounts for both (i,j) and (j,i),
                # so each matrix entry gets half (matching _vech_to_symmetric).
                adj_Lambda_diff = np.array([
                    [sum_gsv[0], sum_gsv[1] / 2.0],
                    [sum_gsv[1] / 2.0, sum_gsv[2]],
                ])
                adj_Xi_full = M_c.T @ adj_Lambda_diff @ M_c

                if not control.iid:
                    adj_Lambda = adj_Xi_full[1:, 1:]
                    adj_lp = _adj_lambda_to_params(
                        adj_Lambda, scales, corr, corr_jac,
                        dim_lambda, n_scale, n_corr, control,
                    )
                    grad[n_beta:n_beta + n_lambda] += adj_lp

            continue

        # --- Per-observation loop for K >= 3 ---
        for qi in range(N_c):
            q = obs_indices[qi]
            diff_V = diff_V_group[qi]
            X_diff = X_diff_group[qi]

            if control.method == "ovus":
                prob, grad_a, grad_sigma_vech = mvncd_grad_ovus_analytic(
                    diff_V, Lambda_diff_c
                )
            else:
                prob, grad_a, grad_sigma_vech = mvncd_grad_me_analytic(
                    diff_V, Lambda_diff_c
                )
            prob = max(prob, 1e-300)
            total_ll += np.log(prob)
            inv_p = 1.0 / prob

            # Beta gradient: dP/d(beta) = X_diff.T @ grad_a
            grad[:n_beta] += inv_p * (X_diff.T @ grad_a)

            # Covariance parameter gradients
            if need_sigma_chain:
                adj_Lambda_diff = _vech_to_symmetric(grad_sigma_vech, dim)
                adj_Xi_full = M_c.T @ adj_Lambda_diff @ M_c

                if not control.iid:
                    adj_Lambda = adj_Xi_full[1:, 1:]
                    adj_lp = _adj_lambda_to_params(
                        adj_Lambda, scales, corr, corr_jac,
                        dim_lambda, n_scale, n_corr, control,
                    )
                    grad[n_beta:n_beta + n_lambda] += inv_p * adj_lp

    nll = -total_ll / N
    grad = -grad / N
    return nll, grad


def _vectorized_gradient_mixed_k2(
    X, y, V_all, beta, Lambda, Lambda_full, Omega, Omega_L,
    omega_params, scales, corr, corr_jac,
    ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
    n_params, n_scale, n_corr, dim_lambda, n_rand,
    need_sigma_chain,
):
    """Vectorized gradient for K=2 with random coefficients and full availability.

    Computes per-observation covariance as (N, 2, 2) tensors and uses
    the vectorized BVN gradient with per-obs sigma. Eliminates the
    per-observation Python loop.
    """
    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)

    unique_chosen = np.unique(y)

    # Pre-compute X_rand for all observations: (N, I, n_rand)
    X_rand_all = X[:, :, ranvar_indices]  # (N, I, n_rand)
    # Per-obs Omega_tilde: X_rand @ Omega @ X_rand.T → (N, I, I)
    # Using einsum: Omega_tilde[n, i, j] = sum_{r,s} X_rand[n,i,r] * Omega[r,s] * X_rand[n,j,s]
    Omega_tilde_all = np.einsum('nir,rs,njs->nij', X_rand_all, Omega, X_rand_all)

    Xi_base = Lambda_full if Lambda_full is not None else np.eye(I, dtype=np.float64)
    # Per-obs Xi_full = Omega_tilde + Xi_base
    Xi_full_all = Omega_tilde_all + Xi_base[None, :, :]  # (N, I, I)

    for c in unique_chosen:
        mask = y == c
        obs_indices = np.where(mask)[0]
        N_c = len(obs_indices)
        if N_c == 0:
            continue

        avail_alts_c = [j for j in range(I) if j != c]
        M_c = np.zeros((dim, I), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            M_c[k, j] = 1.0
            M_c[k, c] = -1.0

        # Per-obs Lambda_diff: M @ Xi_full_q @ M.T for each obs
        Xi_group = Xi_full_all[obs_indices]  # (N_c, I, I)
        # Lambda_diff_group[n] = M_c @ Xi_group[n] @ M_c.T
        Lambda_diff_group = np.einsum('di,nij,ej->nde', M_c, Xi_group, M_c)
        # Symmetrize
        Lambda_diff_group = 0.5 * (Lambda_diff_group + Lambda_diff_group.transpose(0, 2, 1))

        # Vectorized diff_V: shape (N_c, dim)
        V_group = V_all[obs_indices]
        diff_V_group = np.empty((N_c, dim), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            diff_V_group[:, k] = V_group[:, c] - V_group[:, j]

        # Vectorized X_diff: shape (N_c, dim, n_beta)
        X_group = X[obs_indices]
        X_diff_group = np.empty((N_c, dim, X.shape[2]), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            X_diff_group[:, k, :] = X_group[:, c, :] - X_group[:, j, :]

        # Batch MVNCD gradient with per-obs covariance
        prob_all, grad_a_all, grad_sv_all = mvncd_grad_batch_k2_perobs(
            diff_V_group, Lambda_diff_group
        )

        total_ll += float(np.sum(np.log(prob_all)))
        inv_p_all = 1.0 / prob_all  # (N_c,)

        # Beta gradient: vectorized
        weighted_grad_a = inv_p_all[:, None] * grad_a_all  # (N_c, 2)
        grad[:n_beta] += np.einsum('nkv,nk->v', X_diff_group, weighted_grad_a)

        # Covariance parameter gradients
        if need_sigma_chain:
            # Per-obs sigma gradient, weighted by 1/prob
            weighted_gsv = inv_p_all[:, None] * grad_sv_all  # (N_c, 3)

            # Reconstruct per-obs adj_Lambda_diff as (N_c, 2, 2)
            adj_Lambda_diff_all = np.zeros((N_c, dim, dim), dtype=np.float64)
            adj_Lambda_diff_all[:, 0, 0] = weighted_gsv[:, 0]
            adj_Lambda_diff_all[:, 0, 1] = weighted_gsv[:, 1] / 2.0
            adj_Lambda_diff_all[:, 1, 0] = weighted_gsv[:, 1] / 2.0
            adj_Lambda_diff_all[:, 1, 1] = weighted_gsv[:, 2]

            # Per-obs adj_Xi_full = M.T @ adj_Lambda_diff @ M
            # M_c is (dim, I), so M_c.T is (I, dim)
            adj_Xi_all = np.einsum('di,nde,ej->nij', M_c, adj_Lambda_diff_all, M_c)

            # Lambda gradient (shared across observations)
            if not control.iid and n_lambda > 0:
                # Sum adj_Xi over observations, then extract Lambda block
                sum_adj_Xi = adj_Xi_all.sum(axis=0)  # (I, I)
                adj_Lambda = sum_adj_Xi[1:, 1:]
                adj_lp = _adj_lambda_to_params(
                    adj_Lambda, scales, corr, corr_jac,
                    dim_lambda, n_scale, n_corr, control,
                )
                grad[n_beta:n_beta + n_lambda] += adj_lp

            # Omega gradient (per-obs X_rand modulates the gradient)
            if n_omega > 0:
                # adj_Omega_q = X_rand_q.T @ adj_Xi_full_q @ X_rand_q
                X_rand_group = X_rand_all[obs_indices]  # (N_c, I, n_rand)
                # Batched: adj_Omega[n] = X_rand[n].T @ adj_Xi[n] @ X_rand[n]
                adj_Omega_all = np.einsum(
                    'nir,nij,njs->rs', X_rand_group, adj_Xi_all, X_rand_group
                )  # (n_rand, n_rand) — summed over observations

                adj_op = _adj_omega_to_params(
                    adj_Omega_all, Omega_L, omega_params, n_rand, control,
                )
                grad[n_beta + n_lambda:n_beta + n_lambda + n_omega] += adj_op

    nll = -total_ll / N
    grad = -grad / N
    return nll, grad


def _sequential_gradient(
    X, y, V_all, avail, beta, Lambda, Lambda_full, Omega, Omega_L,
    omega_params, scales, corr, corr_jac,
    ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
    n_params, n_scale, n_corr, dim_lambda, n_rand,
    need_sigma_chain, all_avail,
):
    """Sequential gradient for varying availability or random coefficients.

    Falls back to per-observation loop when batching is not possible.
    For K=2 with random coefficients and all_avail, uses a vectorized
    path with per-observation covariance.
    """
    has_random_coeff = control.mix and Omega is not None

    # --- Fast vectorized path for K=2, all_avail, random coefficients ---
    if dim == 2 and all_avail and has_random_coeff:
        return _vectorized_gradient_mixed_k2(
            X, y, V_all, beta, Lambda, Lambda_full, Omega, Omega_L,
            omega_params, scales, corr, corr_jac,
            ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
            n_params, n_scale, n_corr, dim_lambda, n_rand,
            need_sigma_chain,
        )

    # Pre-compute per-chosen-alt structures for common case
    precomputed = {}
    if all_avail:
        for c in range(I):
            avail_alts_c = [j for j in range(I) if j != c]
            M_c = np.zeros((dim, I), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                M_c[k, j] = 1.0
                M_c[k, c] = -1.0

            # GAUSS homogeneous form: covker = blockdiag(0, K). IID routes
            # through the same covker path (K = 0.5*(eye+ones)).
            Xi_base = Lambda_full if Lambda_full is not None else np.eye(I)
            Lambda_diff_c = M_c @ Xi_base @ M_c.T
            Lambda_diff_c = 0.5 * (Lambda_diff_c + Lambda_diff_c.T)

            precomputed[c] = {
                'avail_alts': avail_alts_c,
                'M': M_c,
                'Lambda_diff_base': Lambda_diff_c,
            }

    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)

    for q in range(N):
        chosen = y[q]

        if all_avail:
            pc = precomputed[chosen]
            avail_alts = pc['avail_alts']
            M = pc['M']

            diff_V = np.array([V_all[q, chosen] - V_all[q, j] for j in avail_alts])
            X_diff = np.array([X[q, chosen] - X[q, j] for j in avail_alts])

            if has_random_coeff:
                X_rand = X[q][:, ranvar_indices]
                Omega_tilde = X_rand @ Omega @ X_rand.T
                Xi_full = Omega_tilde + (
                    Lambda_full if Lambda_full is not None else np.eye(I)
                )
                Lambda_diff = M @ Xi_full @ M.T
                Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)
            else:
                Lambda_diff = pc['Lambda_diff_base']
        else:
            avail_q = avail[q] if avail is not None else np.ones(I)
            avail_alts = [j for j in range(I) if j != chosen and avail_q[j] > 0.5]
            dim_q = len(avail_alts)

            if dim_q == 0:
                continue

            diff_V = np.array([V_all[q, chosen] - V_all[q, j] for j in avail_alts])
            X_diff = np.array([X[q, chosen] - X[q, j] for j in avail_alts])

            M = np.zeros((dim_q, I), dtype=np.float64)
            for k, j in enumerate(avail_alts):
                M[k, j] = 1.0
                M[k, chosen] = -1.0

            # GAUSS homogeneous form: covker = blockdiag(0, K). IID routes
            # through the same covker path. Random coefficients add
            # Omega_tilde in the undifferenced I-space: Xi_full = Omega_tilde
            # + covker.
            if Lambda_full is None:
                Lambda_full_local = np.eye(I, dtype=np.float64)
            else:
                Lambda_full_local = Lambda_full
            if has_random_coeff:
                X_rand = X[q][:, ranvar_indices]
                Omega_tilde = X_rand @ Omega @ X_rand.T
                Xi_full = Omega_tilde + Lambda_full_local
            else:
                Xi_full = Lambda_full_local
            Lambda_diff = M @ Xi_full @ M.T

            Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

        # ---- MVNCD probability and analytic gradient ----
        if control.method == "ovus":
            prob, grad_a, grad_sigma_vech = mvncd_grad_ovus_analytic(
                diff_V, Lambda_diff
            )
        else:
            prob, grad_a, grad_sigma_vech = mvncd_grad_me_analytic(
                diff_V, Lambda_diff
            )
        prob = max(prob, 1e-300)
        total_ll += np.log(prob)
        inv_p = 1.0 / prob

        # ---- Beta gradient ----
        grad[:n_beta] += inv_p * (X_diff.T @ grad_a)

        # ---- Covariance parameter gradients ----
        if need_sigma_chain:
            dim_q = len(avail_alts)
            adj_Lambda_diff = _vech_to_symmetric(grad_sigma_vech, dim_q)
            adj_Xi_full = M.T @ adj_Lambda_diff @ M

            if not control.iid:
                adj_Lambda = adj_Xi_full[1:, 1:]
                adj_lp = _adj_lambda_to_params(
                    adj_Lambda, scales, corr, corr_jac,
                    dim_lambda, n_scale, n_corr, control,
                )
                grad[n_beta:n_beta + n_lambda] += inv_p * adj_lp

            if has_random_coeff:
                X_rand = X[q][:, ranvar_indices]
                adj_Omega = X_rand.T @ adj_Xi_full @ X_rand
                adj_op = _adj_omega_to_params(
                    adj_Omega, Omega_L, omega_params, n_rand, control,
                )
                grad[n_beta + n_lambda:n_beta + n_lambda + n_omega] += (
                    inv_p * adj_op
                )

    nll = -total_ll / N
    grad = -grad / N

    return nll, grad


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_lambda_components(
    lambda_params: NDArray | None,
    dim: int,
    control: MNPControl,
) -> tuple[NDArray, NDArray | None, NDArray | None]:
    """Build the differenced kernel ``K`` and return (K, scales, corr).

    GAUSS homogeneous "first-differenced-variance = 1" form (mirrors
    ``_mnp_loglik._build_lambda``): the first differenced variance ``K[0,0]``
    is PINNED to 1, so ``scales = [1.0] + exp(free_log_scales)`` (length
    ``dim``, first entry literally 1.0), and there are only ``dim - 1`` FREE
    scale parameters. The number of correlations is ``dim*(dim-1)//2``
    (unchanged). ``K = W1 @ corr @ W1`` with ``W1 = diag(scales)``.

    The returned ``K`` is the (I-1)x(I-1) DIFFERENCED kernel (NOT the old
    floor ``Lambda + eye``). Callers embed it via ``covker = blockdiag(0, K)``.

    For IID the differenced kernel is the GAUSS frozen ``startker``
    ``K = 0.5*(eye + ones)`` (so ``K[0,0] == 1``), matching ``_build_lambda``.
    """
    if control.iid or lambda_params is None:
        # GAUSS IID kernel (frozen startker): K = 0.5*(eye + ones), K[0,0] == 1.
        K = 0.5 * np.eye(dim, dtype=np.float64) + 0.5 * np.ones(
            (dim, dim), dtype=np.float64
        )
        return K, None, None

    n_scale = max(dim - 1, 0)  # I-2 FREE scales (first diff variance pinned)
    free = safe_exp(lambda_params[:n_scale]) if n_scale > 0 else np.empty(0)
    scales = np.concatenate([[1.0], free])  # length dim, scales[0] == 1.0

    if control.heteronly:
        K = np.diag(scales ** 2)
        return K, scales, None

    n_corr = dim * (dim - 1) // 2
    if n_corr > 0 and len(lambda_params) > n_scale:
        corr_theta = lambda_params[n_scale:n_scale + n_corr]
        corr = theta_to_corr(corr_theta, dim)
    else:
        corr = np.eye(dim, dtype=np.float64)

    W1 = np.diag(scales)
    K = W1 @ corr @ W1
    return K, scales, corr


def _build_omega_components(
    omega_params: NDArray,
    n_rand: int,
    control: MNPControl,
) -> tuple[NDArray, NDArray]:
    """Build Cholesky factor L and Omega = L @ L.T."""
    if control.randdiag:
        L = np.diag(safe_exp(omega_params[:n_rand]))
    else:
        L = np.zeros((n_rand, n_rand), dtype=np.float64)
        idx = 0
        for i in range(n_rand):
            for j in range(i + 1):
                L[i, j] = omega_params[idx]
                idx += 1
    return L, L @ L.T


def _vech_to_symmetric(vech: NDArray, K: int) -> NDArray:
    """Convert row-based upper-triangular vech to symmetric matrix.

    For off-diagonal entries: vech[idx] accounts for both (i,j) and (j,i),
    so each matrix entry gets half the vech value.
    """
    S = np.zeros((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            if i == j:
                S[i, i] = vech[idx]
            else:
                S[i, j] = vech[idx] / 2.0
                S[j, i] = vech[idx] / 2.0
            idx += 1
    return S


def _adj_lambda_to_params(
    adj_Lambda: NDArray,
    scales: NDArray,
    corr: NDArray | None,
    corr_jac: NDArray | None,
    dim: int,
    n_scale: int,
    n_corr: int,
    control: MNPControl,
) -> NDArray:
    """Chain ``adj_K`` through the differenced kernel ``K = W1 @ corr @ W1`` to
    ``adj_lambda_params``, in the GAUSS homogeneous "first-differenced-variance
    = 1" parameterization.

    ``W1 = diag(scales)`` with ``scales = [1.0] + exp(free_log_scales)`` (length
    ``dim``); the FIRST scale is PINNED to 1 and has NO free theta column, so
    only the FREE indices ``k = 1..dim-1`` produce a scale-gradient entry. The
    output length is ``n_scale + n_corr`` with ``n_scale = dim - 1`` (free
    scales). This mirrors GAUSS ``lgd1``: the first row/col of the scale
    Jacobian is dropped because the first differenced variance is normalized to
    1.

    For heteroscedastic: ``K = diag(scales^2)``; only the free diagonal entries
    ``k = 1..dim-1`` get a gradient. For full: ``K = W1 @ corr @ W1``; the
    correlation block (``corr_jac @ adj_corr_vech``) is unchanged.
    """
    if control.heteronly:
        # K = diag(scales^2), free scales k=1..dim-1: d(exp(2x))/dx = 2*scale^2.
        # The pinned scale[0] contributes to K but gets NO gradient entry; free
        # index k maps to output position k-1.
        adj = np.zeros(n_scale, dtype=np.float64)
        for k in range(1, dim):
            adj[k - 1] = adj_Lambda[k, k] * 2.0 * scales[k] ** 2
        return adj

    adj = np.zeros(n_scale + n_corr, dtype=np.float64)

    # Scale gradient over FREE kernel indices k=1..dim-1 (k=0 is pinned, no
    # theta column). For free index k: adj_scale[k] = 2 * sum_j adj_K[k,j] *
    # corr[k,j] * scales[j], chained to log-scale via * scales[k], mapped to
    # output position k-1.
    for k in range(1, dim):
        s = 0.0
        for j in range(dim):
            s += adj_Lambda[k, j] * corr[k, j] * scales[j]
        adj[k - 1] = 2.0 * s * scales[k]

    # Correlation gradient (UNCHANGED).
    if n_corr > 0 and corr_jac is not None:
        # adj_corr[i,j] = adj_Lambda[i,j] * scales[i] * scales[j]
        # Build adj_corr_vech (K*(K+1)//2 upper-tri including diag)
        n_corr_upper = dim * (dim + 1) // 2
        adj_corr_vech = np.zeros(n_corr_upper, dtype=np.float64)
        vidx = 0
        for i in range(dim):
            for j in range(i, dim):
                if i == j:
                    adj_corr_vech[vidx] = 0.0  # diagonal of corr is fixed
                else:
                    # vech accounts for both (i,j) and (j,i)
                    adj_corr_vech[vidx] = (
                        2.0 * adj_Lambda[i, j] * scales[i] * scales[j]
                    )
                vidx += 1

        # Chain through spherical parameterization Jacobian
        # corr_jac: (n_corr, n_corr_upper), maps d(theta) -> d(corr_vech)
        # adj_theta[p] = sum_q corr_jac[p, q] * adj_corr_vech[q]
        adj[n_scale:n_scale + n_corr] = corr_jac @ adj_corr_vech

    return adj


def _adj_omega_to_params(
    adj_Omega: NDArray,
    Omega_L: NDArray,
    omega_params: NDArray,
    n_rand: int,
    control: MNPControl,
) -> NDArray:
    """Chain adj_Omega through Omega = L @ L.T to adj_omega_params.

    For diagonal: Omega = diag(exp(2*params)), d/d(params) = 2*exp(2*params).
    For full: Omega = L @ L.T, adj_L = 2*adj_Omega @ L (lower-tri part).
    """
    if control.randdiag:
        adj = np.zeros(n_rand, dtype=np.float64)
        for k in range(n_rand):
            exp_val = safe_exp(omega_params[k])
            adj[k] = adj_Omega[k, k] * 2.0 * exp_val ** 2
        return adj

    # Full Cholesky: adj_L = 2 * adj_Omega @ L (adj_Omega is symmetric)
    adj_L = 2.0 * adj_Omega @ Omega_L

    # Extract lower-triangular entries (direct params, no transformation)
    n_omega = n_rand * (n_rand + 1) // 2
    adj = np.zeros(n_omega, dtype=np.float64)
    oidx = 0
    for i in range(n_rand):
        for j in range(i + 1):
            adj[oidx] = adj_L[i, j]
            oidx += 1
    return adj
