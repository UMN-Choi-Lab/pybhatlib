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

from pybhatlib.gradmvn._mvncd_grad_analytic import mvncd_grad_me_analytic
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
        if control.heteronly:
            n_scale = dim_lambda
            n_lambda = n_scale
        else:
            n_scale = dim_lambda
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

    # Pre-compute Lambda_full (shared, built once)
    Lambda_full = None
    if need_sigma_chain or (not control.iid):
        Lambda_full = np.eye(I, dtype=np.float64)
        if not control.iid:
            Lambda_full[1:, 1:] = Lambda + np.eye(dim_lambda)

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
        if control.heteronly:
            n_scale = dim_lambda
            n_lambda = n_scale
        else:
            n_scale = dim_lambda
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

    Lambda_full = None
    if (not control.iid) or (control.mix and ranvar_indices is not None):
        Lambda_full = np.eye(I, dtype=np.float64)
        if not control.iid:
            Lambda_full[1:, 1:] = Lambda + np.eye(dim_lambda)

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

            # Build differenced covariance for this segment
            has_random = Omega_h is not None
            if control.iid and not has_random:
                Lambda_diff = (
                    np.ones((dim_q, dim_q), dtype=np.float64)
                    + np.eye(dim_q, dtype=np.float64)
                )
            else:
                if has_random:
                    Omega_tilde = X_rand @ Omega_h @ X_rand.T  # (I, I)
                    Xi_base = Lambda_full if Lambda_full is not None else np.eye(I)
                    Xi_full = Omega_tilde + Xi_base
                else:
                    Xi_full = Lambda_full if Lambda_full is not None else np.eye(I)
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

        if control.iid:
            Lambda_diff_c = (
                np.ones((dim, dim), dtype=np.float64)
                + np.eye(dim, dtype=np.float64)
            )
        else:
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

        # Per-observation MVNCD gradient calls (sequential, but with shared cov)
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


def _sequential_gradient(
    X, y, V_all, avail, beta, Lambda, Lambda_full, Omega, Omega_L,
    omega_params, scales, corr, corr_jac,
    ranvar_indices, control, I, N, dim, n_beta, n_lambda, n_omega,
    n_params, n_scale, n_corr, dim_lambda, n_rand,
    need_sigma_chain, all_avail,
):
    """Sequential gradient for varying availability or random coefficients.

    Falls back to per-observation loop when batching is not possible.
    """
    has_random_coeff = control.mix and Omega is not None

    # Pre-compute per-chosen-alt structures for common case
    precomputed = {}
    if all_avail:
        for c in range(I):
            avail_alts_c = [j for j in range(I) if j != c]
            M_c = np.zeros((dim, I), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                M_c[k, j] = 1.0
                M_c[k, c] = -1.0

            if control.iid and not has_random_coeff:
                Lambda_diff_c = (
                    np.ones((dim, dim), dtype=np.float64)
                    + np.eye(dim, dtype=np.float64)
                )
            else:
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

            if control.iid and not has_random_coeff:
                Lambda_diff = (
                    np.ones((dim_q, dim_q), dtype=np.float64)
                    + np.eye(dim_q, dtype=np.float64)
                )
            else:
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
    """Build Lambda and return (Lambda, scales, corr) for gradient chain."""
    if control.iid or lambda_params is None:
        return np.eye(dim, dtype=np.float64), None, None

    if control.heteronly:
        scales = np.exp(lambda_params[:dim])
        Lambda = np.diag(scales**2)
        return Lambda, scales, None

    scales = np.exp(lambda_params[:dim])
    n_corr = dim * (dim - 1) // 2
    if n_corr > 0 and len(lambda_params) > dim:
        corr_theta = lambda_params[dim:dim + n_corr]
        corr = theta_to_corr(corr_theta, dim)
    else:
        corr = np.eye(dim, dtype=np.float64)

    D = np.diag(scales)
    Lambda = D @ corr @ D
    return Lambda, scales, corr


def _build_omega_components(
    omega_params: NDArray,
    n_rand: int,
    control: MNPControl,
) -> tuple[NDArray, NDArray]:
    """Build Cholesky factor L and Omega = L @ L.T."""
    if control.randdiag:
        L = np.diag(np.exp(omega_params[:n_rand]))
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
    """Chain adj_Lambda through Lambda = D @ corr @ D to adj_lambda_params.

    For heteroscedastic: Lambda = diag(scales^2), scales = exp(lambda_params).
    For full: Lambda = D @ corr @ D, D = diag(exp(lambda_params[:n_scale])),
    corr = theta_to_corr(lambda_params[n_scale:]).
    """
    if control.heteronly:
        # Lambda = diag(scales^2), d(exp(2x))/dx = 2*exp(2x) = 2*scale^2
        adj = np.zeros(n_scale, dtype=np.float64)
        for k in range(dim):
            adj[k] = adj_Lambda[k, k] * 2.0 * scales[k] ** 2
        return adj

    adj = np.zeros(n_scale + n_corr, dtype=np.float64)

    # Scale gradient: adj_scales[k] = 2 * sum_j adj_Lambda[k,j]*corr[k,j]*scales[j]
    # Then chain: adj_log_scale[k] = adj_scales[k] * scales[k]
    for k in range(dim):
        s = 0.0
        for j in range(dim):
            s += adj_Lambda[k, j] * corr[k, j] * scales[j]
        adj[k] = 2.0 * s * scales[k]

    # Correlation gradient
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
            exp_val = np.exp(omega_params[k])
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
