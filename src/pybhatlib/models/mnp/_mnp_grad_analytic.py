"""Analytic gradient of MNP log-likelihood via adjoint/backpropagation.

Implements Phase D1 from GAUSS_INTEGRATION_PLAN.md. Uses mvncd_grad_me_analytic
for per-observation probability gradients and chains them through the MNP
parameter transformations (beta, Lambda scales/correlations, Omega Cholesky).

Supports: IID, heteroscedastic, full covariance, and mixed MNP (nseg=1).
Requires method="me" for the MVNCD approximation.

The gradient chain:
    d(-LL)/d(theta) = -(1/N) sum_q (1/P_q) dP_q/d(theta)

where dP_q/d(theta) decomposes as:
    dP/d(beta) via differenced design matrix
    dP/d(lambda_params) via Lambda = D @ corr @ D parameterization
    dP/d(omega_params) via Omega = L @ L.T Cholesky parameterization

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

    Uses the ME method for MVNCD and backward/adjoint differentiation for
    the gradient chain through all parameter transformations.

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
        Negative mean log-likelihood (ME approximation).
    grad : ndarray, shape (n_params,)
        Gradient of negative mean log-likelihood.
    """
    if control.nseg > 1:
        raise NotImplementedError(
            "Analytic gradient not supported for mixture-of-normals (nseg > 1)."
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

    # ---- Accumulate log-likelihood and gradient ----
    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)
    need_sigma_chain = (not control.iid) or (control.mix and Omega is not None)

    for q in range(N):
        avail_q = avail[q] if avail is not None else np.ones(I)
        chosen = y[q]
        avail_alts = [j for j in range(I) if j != chosen and avail_q[j] > 0.5]
        dim = len(avail_alts)

        if dim == 0:
            continue

        # Differenced utilities and design matrix
        V_q = X[q] @ beta
        diff_V = np.array([V_q[chosen] - V_q[j] for j in avail_alts])
        X_diff = np.array([X[q, chosen] - X[q, j] for j in avail_alts])

        # Differencing matrix M: dim x I
        M = np.zeros((dim, I), dtype=np.float64)
        for k, j in enumerate(avail_alts):
            M[k, j] = 1.0
            M[k, chosen] = -1.0

        # Build differenced covariance
        if control.iid and not (control.mix and Omega is not None):
            Lambda_diff = np.ones((dim, dim), dtype=np.float64) + np.eye(dim, dtype=np.float64)
        else:
            Lambda_full = np.eye(I, dtype=np.float64)
            if not control.iid:
                Lambda_full[1:, 1:] = Lambda + np.eye(dim_lambda)

            if control.mix and Omega is not None:
                X_rand = X[q][:, ranvar_indices]
                Omega_tilde = X_rand @ Omega @ X_rand.T
                Xi_full = Omega_tilde + Lambda_full
            else:
                Xi_full = Lambda_full

            Lambda_diff = M @ Xi_full @ M.T

        Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(Lambda_diff)
        if eigvals.min() < 1e-10:
            Lambda_diff += np.eye(dim) * (1e-10 - eigvals.min())

        # ---- MVNCD probability and analytic gradient ----
        prob, grad_a, grad_sigma_vech = mvncd_grad_me_analytic(
            diff_V, Lambda_diff
        )
        prob = max(prob, 1e-300)
        total_ll += np.log(prob)
        inv_p = 1.0 / prob

        # ---- Beta gradient: dP/d(beta) = X_diff.T @ grad_a ----
        grad[:n_beta] += inv_p * (X_diff.T @ grad_a)

        # ---- Covariance parameter gradients ----
        if need_sigma_chain:
            adj_Lambda_diff = _vech_to_symmetric(grad_sigma_vech, dim)
            adj_Xi_full = M.T @ adj_Lambda_diff @ M

            # Lambda gradient
            if not control.iid:
                adj_Lambda = adj_Xi_full[1:, 1:]
                adj_lp = _adj_lambda_to_params(
                    adj_Lambda, scales, corr, corr_jac,
                    dim_lambda, n_scale, n_corr, control,
                )
                grad[n_beta:n_beta + n_lambda] += inv_p * adj_lp

            # Omega gradient
            if control.mix and Omega is not None:
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
