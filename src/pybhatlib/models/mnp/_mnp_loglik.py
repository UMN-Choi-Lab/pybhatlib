"""MNP log-likelihood function and gradient.

Implements the log-likelihood for Multinomial Probit models with:
- IID errors
- Flexible covariance (heteroscedastic, correlated errors)
- Random coefficients (mixed MNP)
- Mixture-of-normals random coefficients

The probability for each observation is computed via the MVNCD function,
which evaluates (I-1)-dimensional multivariate normal CDFs.

Phase 2 optimization: For single-segment models (IID, flexible, mixed),
the observation loop is vectorized via Numba prange when available.
Pre-computed shared quantities (Lambda_diff, differencing matrices) are
reused across observations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd, mvncd_log_batch
from pybhatlib.matgradient._spherical import theta_to_corr
from pybhatlib.models.mnp._mnp_control import MNPControl
from pybhatlib.models.mnp._mnp_grad_analytic import mnp_analytic_gradient


def mnp_loglik(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    avail: NDArray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
    *,
    return_gradient: bool = False,
    xp=None,
) -> float | tuple[float, NDArray]:
    """Compute negative mean log-likelihood for MNP model.

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector (parametrized form).
    X : ndarray, shape (N, n_alts, n_vars)
        Design matrix for all observations and alternatives.
    y : ndarray, shape (N,)
        Chosen alternative index (0-based) for each observation.
    avail : ndarray, shape (N, n_alts) or None
        Availability matrix (1=available, 0=unavailable). None means all available.
    n_alts : int
        Number of alternatives.
    n_beta : int
        Number of beta coefficients.
    control : MNPControl
        Model control structure.
    ranvar_indices : list of int or None
        Indices in the beta vector that are random coefficients.
    return_gradient : bool
        If True, also return gradient.
    xp : backend, optional

    Returns
    -------
    nll : float
        Negative mean log-likelihood.
    grad : ndarray (only if return_gradient=True)
        Gradient of negative mean log-likelihood.
    """
    if xp is None:
        xp = get_backend("numpy")

    theta_np = np.asarray(theta, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.int64)
    N = X_np.shape[0]
    I = n_alts  # number of alternatives

    if return_gradient:
        # Analytic gradient computes both nll and grad in one pass,
        # avoiding a redundant forward-only evaluation.
        if (
            control.analytic_grad
            and control.method in ("me", "ovus")
        ):
            return mnp_analytic_gradient(
                theta_np, X_np, y_np, avail, n_alts, n_beta,
                control, ranvar_indices,
            )

    # Unpack parameters
    params = _unpack_params(theta_np, n_beta, I, control, ranvar_indices)
    beta = params["beta"]
    lambda_params = params.get("lambda_params", None)
    omega_params = params.get("omega_params", None)
    segment_params = params.get("segment_params", None)

    # Build kernel error covariance matrix Lambda
    Lambda = _build_lambda(lambda_params, I, control)

    # Build random coefficient covariance if needed
    Omega_L = None
    if control.mix and omega_params is not None:
        Omega_L = _build_omega_cholesky(omega_params, ranvar_indices, control)

    # --- Phase 2: Try vectorized batch path ---
    if control.nseg <= 1:
        total_ll = _batch_loglik_single_segment(
            X_np, y_np, avail, beta, Lambda, Omega_L,
            ranvar_indices, control, I, N, xp,
        )
    else:
        # Mixture of normals: per-observation loop with segment mixing
        total_ll = _sequential_loglik_mixture(
            X_np, y_np, avail, params, Lambda, ranvar_indices, control, I, N, xp,
        )

    mean_ll = total_ll / N
    nll = -mean_ll

    if return_gradient:
        # Numerical gradient fallback (analytic path handled above)
        grad = _numerical_gradient(
            theta_np, X_np, y_np, avail, n_alts, n_beta,
            control, ranvar_indices, xp
        )
        return nll, grad

    return nll


# ---------------------------------------------------------------------------
# Phase 2: Vectorized batch log-likelihood for single-segment models
# ---------------------------------------------------------------------------

def _batch_loglik_single_segment(
    X_np: np.ndarray,
    y_np: np.ndarray,
    avail: np.ndarray | None,
    beta: np.ndarray,
    Lambda: np.ndarray,
    Omega_L: np.ndarray | None,
    ranvar_indices: list[int] | None,
    control: MNPControl,
    I: int,
    N: int,
    xp,
) -> float:
    """Compute total log-likelihood for single-segment models using batch MVNCD.

    Pre-computes shared quantities (Lambda_diff, diff_V_all) and dispatches
    to mvncd_log_batch for parallel evaluation when possible.
    """
    has_random_coeff = Omega_L is not None and ranvar_indices is not None

    # Check if all observations have full availability (common case)
    all_avail = avail is None or np.all(avail > 0.5)

    if all_avail and not has_random_coeff:
        return _batch_loglik_shared_cov(
            X_np, y_np, beta, Lambda, control, I, N, xp,
        )
    elif all_avail and has_random_coeff:
        return _batch_loglik_mixed(
            X_np, y_np, beta, Lambda, Omega_L, ranvar_indices, control, I, N, xp,
        )
    else:
        # Varying availability: fall back to per-observation loop
        # (different observations may have different avail_alts dimensions)
        return _sequential_loglik_single_segment(
            X_np, y_np, avail, beta, Lambda, Omega_L,
            ranvar_indices, control, I, N, xp,
        )


def _batch_loglik_shared_cov(
    X_np: np.ndarray,
    y_np: np.ndarray,
    beta: np.ndarray,
    Lambda: np.ndarray,
    control: MNPControl,
    I: int,
    N: int,
    xp,
) -> float:
    """Batch log-likelihood for IID and flexible covariance (no random coeff).

    Lambda_diff is shared across all observations. Pre-compute it once,
    then vectorize diff_V computation and batch MVNCD evaluation.
    """
    dim = I - 1  # all alternatives available, so dim = I-1

    # Pre-compute Lambda_diff once (shared across all observations)
    if control.iid:
        Lambda_diff = np.ones((dim, dim), dtype=np.float64) + np.eye(dim, dtype=np.float64)
    else:
        # Build differencing matrix for the "worst case" ordering
        # Since all alts are available, avail_alts for chosen=c is all j != c
        # We compute Lambda_diff per unique chosen alternative
        Lambda_full = np.eye(I, dtype=np.float64)
        Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)
        Lambda_diff = None  # Will be computed per unique chosen alt

    # Compute utilities for all observations: V = X @ beta, shape (N, I)
    V_all = np.einsum('nij,j->ni', X_np, beta)

    # Group observations by chosen alternative for vectorized diff_V computation
    unique_chosen = np.unique(y_np)

    if control.iid:
        # IID: Lambda_diff is the same regardless of chosen alt
        # Build diff_V_all: shape (N, dim)
        # For each observation q with chosen=c:
        #   avail_alts = [0, 1, ..., c-1, c+1, ..., I-1]  (sorted)
        #   diff_V[k] = V[q, c] - V[q, avail_alts[k]]
        diff_V_all = _compute_diff_V_all(V_all, y_np, I, N, dim)
        log_probs = mvncd_log_batch(diff_V_all, Lambda_diff, method="me")
        return float(np.sum(log_probs))
    else:
        # Flexible cov: Lambda_diff depends on which alt is chosen
        # (because the differencing matrix M depends on chosen)
        # Group by chosen alt and batch each group
        total_ll = 0.0
        for c in unique_chosen:
            mask = y_np == c
            N_c = int(np.sum(mask))
            if N_c == 0:
                continue

            # Build differencing matrix for chosen=c
            avail_alts_c = [j for j in range(I) if j != c]
            M_c = np.zeros((dim, I), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                M_c[k, j] = 1.0
                M_c[k, c] = -1.0

            Lambda_diff_c = M_c @ Lambda_full @ M_c.T
            Lambda_diff_c = 0.5 * (Lambda_diff_c + Lambda_diff_c.T)

            # Compute diff_V for this group
            V_group = V_all[mask]  # (N_c, I)
            diff_V_group = np.empty((N_c, dim), dtype=np.float64)
            for k, j in enumerate(avail_alts_c):
                diff_V_group[:, k] = V_group[:, c] - V_group[:, j]

            log_probs = mvncd_log_batch(diff_V_group, Lambda_diff_c, method="me")
            total_ll += float(np.sum(log_probs))

        return total_ll


def _compute_diff_V_all(
    V_all: np.ndarray,
    y_np: np.ndarray,
    I: int,
    N: int,
    dim: int,
) -> np.ndarray:
    """Compute differenced utilities for all observations (IID case).

    For IID models, the ordering of avail_alts doesn't affect Lambda_diff
    (it's always I + 11^T), so we use a consistent ordering:
    avail_alts = [0, 1, ..., chosen-1, chosen+1, ..., I-1].

    Parameters
    ----------
    V_all : ndarray, shape (N, I)
        Utilities for all observations and alternatives.
    y_np : ndarray, shape (N,)
        Chosen alternative for each observation.
    I : int
        Number of alternatives.
    N : int
        Number of observations.
    dim : int
        I - 1 (number of non-chosen alternatives).

    Returns
    -------
    diff_V_all : ndarray, shape (N, dim)
        Differenced utilities: V[q, chosen] - V[q, j] for each j != chosen.
    """
    diff_V_all = np.empty((N, dim), dtype=np.float64)

    # Vectorized: for each observation, extract V_chosen and subtract non-chosen
    V_chosen = V_all[np.arange(N), y_np]  # shape (N,)

    for q in range(N):
        c = y_np[q]
        k = 0
        for j in range(I):
            if j != c:
                diff_V_all[q, k] = V_chosen[q] - V_all[q, j]
                k += 1

    return diff_V_all


def _batch_loglik_mixed(
    X_np: np.ndarray,
    y_np: np.ndarray,
    beta: np.ndarray,
    Lambda: np.ndarray,
    Omega_L: np.ndarray,
    ranvar_indices: list[int],
    control: MNPControl,
    I: int,
    N: int,
    xp,
) -> float:
    """Batch log-likelihood for mixed MNP (random coefficients).

    Xi_diff varies per observation (because X_rand varies), so we build
    per-observation covariance matrices and use mvncd_log_batch with per_obs_sigma.
    """
    dim = I - 1
    Omega = Omega_L @ Omega_L.T

    # Compute utilities for all observations
    V_all = np.einsum('nij,j->ni', X_np, beta)

    # Build Lambda_full
    if control.iid:
        Lambda_full = np.eye(I, dtype=np.float64)
    else:
        Lambda_full = np.eye(I, dtype=np.float64)
        Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)

    # Group by chosen alternative
    unique_chosen = np.unique(y_np)
    total_ll = 0.0

    for c in unique_chosen:
        mask = y_np == c
        N_c = int(np.sum(mask))
        if N_c == 0:
            continue

        # Build differencing matrix for chosen=c
        avail_alts_c = [j for j in range(I) if j != c]
        M_c = np.zeros((dim, I), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            M_c[k, j] = 1.0
            M_c[k, c] = -1.0

        # Compute diff_V for this group
        V_group = V_all[mask]
        diff_V_group = np.empty((N_c, dim), dtype=np.float64)
        for k, j in enumerate(avail_alts_c):
            diff_V_group[:, k] = V_group[:, c] - V_group[:, j]

        # Build per-observation Xi_diff
        X_group = X_np[mask]  # (N_c, I, n_vars)
        sigma_all = np.empty((N_c, dim, dim), dtype=np.float64)
        for qi in range(N_c):
            X_rand_q = X_group[qi][:, ranvar_indices]  # (I, n_rand)
            Omega_tilde_q = X_rand_q @ Omega @ X_rand_q.T  # (I, I)
            Xi_full_q = Omega_tilde_q + Lambda_full
            Xi_diff_q = M_c @ Xi_full_q @ M_c.T
            Xi_diff_q = 0.5 * (Xi_diff_q + Xi_diff_q.T)
            sigma_all[qi] = Xi_diff_q

        log_probs = mvncd_log_batch(
            diff_V_group, sigma_all[0], method="me", per_obs_sigma=sigma_all,
        )
        total_ll += float(np.sum(log_probs))

    return total_ll


def per_obs_loglik(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    avail: np.ndarray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
    xp=None,
) -> np.ndarray:
    """Return per-observation log-likelihood contributions (length N).

    Used by BHHH and sandwich SE computation: per-obs scores are obtained by
    numerical differencing of this function.
    """
    if xp is None:
        xp = get_backend("numpy")
    theta_np = np.asarray(theta, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.int64)
    N = X_np.shape[0]
    I = n_alts

    params = _unpack_params(theta_np, n_beta, I, control, ranvar_indices)
    beta = params["beta"]
    Lambda = _build_lambda(params.get("lambda_params"), I, control)
    Omega_L = None
    if control.mix and params.get("omega_params") is not None:
        Omega_L = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    ll_per_obs = np.zeros(N, dtype=np.float64)
    for q in range(N):
        avail_q = avail[q] if avail is not None else np.ones(I)
        chosen = int(y_np[q])
        if control.nseg > 1:
            prob_q = _compute_mixture_prob(
                X_np[q], params, chosen, avail_q, Lambda,
                ranvar_indices, control, xp,
            )
        else:
            prob_q = _compute_choice_prob(
                X_np[q], beta, chosen, avail_q, Lambda, Omega_L,
                ranvar_indices, control, xp,
            )
        ll_per_obs[q] = np.log(max(prob_q, 1e-300))
    return ll_per_obs


def _sequential_loglik_single_segment(
    X_np: np.ndarray,
    y_np: np.ndarray,
    avail: np.ndarray | None,
    beta: np.ndarray,
    Lambda: np.ndarray,
    Omega_L: np.ndarray | None,
    ranvar_indices: list[int] | None,
    control: MNPControl,
    I: int,
    N: int,
    xp,
) -> float:
    """Sequential per-observation log-likelihood (fallback for varying availability)."""
    total_ll = 0.0
    for q in range(N):
        if avail is not None:
            avail_q = avail[q]
        else:
            avail_q = np.ones(I)

        chosen = y_np[q]
        prob_q = _compute_choice_prob(
            X_np[q], beta, chosen, avail_q, Lambda, Omega_L,
            ranvar_indices, control, xp
        )
        prob_q = max(prob_q, 1e-300)
        total_ll += np.log(prob_q)

    return total_ll


def _sequential_loglik_mixture(
    X_np: np.ndarray,
    y_np: np.ndarray,
    avail: np.ndarray | None,
    params: dict,
    Lambda: np.ndarray,
    ranvar_indices: list[int] | None,
    control: MNPControl,
    I: int,
    N: int,
    xp,
) -> float:
    """Sequential per-observation log-likelihood for mixture-of-normals models."""
    total_ll = 0.0
    for q in range(N):
        if avail is not None:
            avail_q = avail[q]
        else:
            avail_q = np.ones(I)

        chosen = y_np[q]
        prob_q = _compute_mixture_prob(
            X_np[q], params, chosen, avail_q, Lambda,
            ranvar_indices, control, xp
        )
        prob_q = max(prob_q, 1e-300)
        total_ll += np.log(prob_q)

    return total_ll


# ---------------------------------------------------------------------------
# Parameter unpacking and covariance construction
# ---------------------------------------------------------------------------

def _unpack_params(
    theta: np.ndarray,
    n_beta: int,
    n_alts: int,
    control: MNPControl,
    ranvar_indices: list[int] | None,
) -> dict:
    """Unpack flat parameter vector into structured components."""
    params = {}
    idx = 0
    I = n_alts

    # Beta coefficients
    params["beta"] = theta[idx: idx + n_beta]
    idx += n_beta

    if not control.iid:
        # Lambda (kernel error covariance) parameters
        if control.heteronly:
            # Only variances: I-1 scale parameters (first alt is reference)
            n_lambda = I - 1
        else:
            # Full: (I-1) scale params + (I-1)*(I-2)/2 correlation params
            n_scale = I - 1
            n_corr = (I - 1) * (I - 2) // 2
            n_lambda = n_scale + n_corr
        params["lambda_params"] = theta[idx: idx + n_lambda]
        idx += n_lambda

    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            # Diagonal: just n_rand variance params
            n_omega = n_rand
        else:
            # Full Cholesky: n_rand*(n_rand+1)/2
            n_omega = n_rand * (n_rand + 1) // 2
        params["omega_params"] = theta[idx: idx + n_omega]
        idx += n_omega

    if control.nseg > 1:
        # Segment probability parameters (nseg-1 free params, softmax)
        n_seg_params = control.nseg - 1
        params["segment_params"] = theta[idx: idx + n_seg_params]
        idx += n_seg_params

        # Additional beta and omega for extra segments
        params["segment_betas"] = []
        params["segment_omegas"] = []
        for h in range(1, control.nseg):
            seg_beta = theta[idx: idx + n_beta]
            idx += n_beta
            params["segment_betas"].append(seg_beta)

            if control.mix and ranvar_indices is not None:
                n_rand = len(ranvar_indices)
                if control.randdiag:
                    n_omega = n_rand
                else:
                    n_omega = n_rand * (n_rand + 1) // 2
                seg_omega = theta[idx: idx + n_omega]
                idx += n_omega
                params["segment_omegas"].append(seg_omega)

    return params


def _build_lambda(
    lambda_params: np.ndarray | None,
    n_alts: int,
    control: MNPControl,
) -> np.ndarray:
    """Build the (I-1) x (I-1) differenced kernel error covariance matrix."""
    I = n_alts
    dim = I - 1  # differenced dimension

    if control.iid or lambda_params is None:
        # IID: Lambda = I (identity)
        return np.eye(dim, dtype=np.float64)

    if control.heteronly:
        # Only heteroscedastic: Lambda = diag(exp(params))
        scales = np.exp(lambda_params[:dim])
        return np.diag(scales**2)

    # Full covariance: scales + correlations via spherical parameterization
    n_scale = dim
    scales = np.exp(lambda_params[:n_scale])

    n_corr = dim * (dim - 1) // 2
    if n_corr > 0 and len(lambda_params) > n_scale:
        corr_theta = lambda_params[n_scale: n_scale + n_corr]
        corr = theta_to_corr(corr_theta, dim)
    else:
        corr = np.eye(dim)

    # Omega = diag(scales) @ corr @ diag(scales)
    D = np.diag(scales)
    return D @ corr @ D


def _build_omega_cholesky(
    omega_params: np.ndarray,
    ranvar_indices: list[int],
    control: MNPControl,
) -> np.ndarray:
    """Build the Cholesky factor L of the random coefficient covariance Omega = LL'."""
    n_rand = len(ranvar_indices)

    if control.randdiag:
        # Diagonal: L = diag(exp(params))
        return np.diag(np.exp(omega_params[:n_rand]))

    # Full lower triangular Cholesky
    L = np.zeros((n_rand, n_rand), dtype=np.float64)
    idx = 0
    for i in range(n_rand):
        for j in range(i + 1):
            L[i, j] = omega_params[idx]
            idx += 1

    return L


def _compute_choice_prob(
    X_q: np.ndarray,
    beta: np.ndarray,
    chosen: int,
    avail_q: np.ndarray,
    Lambda: np.ndarray,
    Omega_L: np.ndarray | None,
    ranvar_indices: list[int] | None,
    control: MNPControl,
    xp,
) -> float:
    """Compute choice probability for one observation.

    P(chosen | X_q, beta, Lambda) = P(V_chosen + eps_chosen > V_j + eps_j for all j != chosen)

    This is equivalent to P(diff_eps < diff_V) where differencing is done
    w.r.t. the chosen alternative.
    """
    I = X_q.shape[0]  # number of alternatives
    n_vars = X_q.shape[1]

    # Systematic utility for each alternative
    V_q = X_q @ beta  # shape (I,)

    if Omega_L is not None and ranvar_indices is not None:
        # With random coefficients: need to integrate over beta distribution
        # Use MVNCD on the expanded covariance
        return _compute_mixed_choice_prob(
            X_q, V_q, chosen, avail_q, Lambda, Omega_L,
            ranvar_indices, control, xp
        )

    # Standard MNP: difference utilities w.r.t. chosen alternative
    # P(chosen) = P(V_chosen - V_j > eps_j - eps_chosen for all j != chosen)
    # = P(diff_eps < diff_V) where diff is w.r.t. chosen

    avail_alts = [j for j in range(I) if j != chosen and avail_q[j] > 0.5]
    dim = len(avail_alts)

    if dim == 0:
        return 1.0

    # Differenced utilities: V_chosen - V_j for each available j != chosen
    diff_V = np.array([V_q[chosen] - V_q[j] for j in avail_alts])

    # Differenced covariance: Lambda_diff for the differences (eps_j - eps_chosen)
    # For IID: Lambda_diff = 2 * I (since Var(eps_j - eps_chosen) = 2*sigma^2)
    # For general Lambda (already differenced w.r.t. first alternative):
    # Need to apply appropriate differencing transformation

    if control.iid:
        # IID errors: Var(eps_j - eps_chosen) = 2, Cov(eps_j-eps_c, eps_k-eps_c) = 1
        Lambda_diff = np.ones((dim, dim), dtype=np.float64) + np.eye(dim, dtype=np.float64)
    else:
        # Build differencing matrix M such that diff_eps = M @ eps
        # where eps ~ MVN(0, Lambda_full)
        # M is dim x I, M[k, chosen] = -1, M[k, avail_alts[k]] = 1
        M = np.zeros((dim, I), dtype=np.float64)
        for k, j in enumerate(avail_alts):
            M[k, j] = 1.0
            M[k, chosen] = -1.0

        # Full Lambda: need I x I covariance
        # The Lambda we have is (I-1) x (I-1) differenced w.r.t. first alt
        # Reconstruct full I x I covariance (first alt has variance 1, no correlation)
        Lambda_full = np.eye(I, dtype=np.float64)
        # Place the estimated covariance for alts 1..I-1
        Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)  # add back reference variance

        Lambda_diff = M @ Lambda_full @ M.T

    # Symmetrize
    Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

    # PD check skipped: Lambda_diff = M @ Lambda_full @ M.T is always PD
    # because Lambda_full is PD (constructed from exp(params)) and M has full row rank.
    # For IID, Lambda_diff = I + 11^T has eigenvalues dim+1 and 1.

    # P(diff_eps < diff_V) = P(Z < diff_V) for Z ~ MVN(0, Lambda_diff)
    prob = mvncd(xp.array(diff_V), xp.array(Lambda_diff), method=control.method, xp=xp)

    return max(prob, 1e-300)


def _compute_mixed_choice_prob(
    X_q: np.ndarray,
    V_q_base: np.ndarray,
    chosen: int,
    avail_q: np.ndarray,
    Lambda: np.ndarray,
    Omega_L: np.ndarray,
    ranvar_indices: list[int],
    control: MNPControl,
    xp,
) -> float:
    """Compute choice probability with random coefficients.

    For mixed MNP, the utility covariance has two components:
    Xi_q = Omega_tilde_q + Lambda
    where Omega_tilde_q = X_q @ Omega @ X_q.T captures random coefficient variation
    and Lambda captures kernel error variation.

    The key advantage of the MNP formulation is that both are normal, so the
    total covariance is still normal and can be evaluated analytically via MVNCD.
    """
    I = X_q.shape[0]

    # Random coefficient covariance in utility space
    # Omega = L @ L.T (covariance of random coefficients)
    Omega = Omega_L @ Omega_L.T

    # Build random coefficient design matrix (only random coefficient columns)
    X_rand = X_q[:, ranvar_indices]  # (I, n_rand)

    # Utility covariance from random coefficients: X_rand @ Omega @ X_rand.T
    Omega_tilde = X_rand @ Omega @ X_rand.T  # (I, I)

    # Total utility covariance
    avail_alts = [j for j in range(I) if j != chosen and avail_q[j] > 0.5]
    dim = len(avail_alts)

    if dim == 0:
        return 1.0

    # Differenced utilities
    diff_V = np.array([V_q_base[chosen] - V_q_base[j] for j in avail_alts])

    # Differencing matrix
    M = np.zeros((dim, I), dtype=np.float64)
    for k, j in enumerate(avail_alts):
        M[k, j] = 1.0
        M[k, chosen] = -1.0

    # Full error covariance
    if control.iid:
        Lambda_full = np.eye(I, dtype=np.float64)
    else:
        Lambda_full = np.eye(I, dtype=np.float64)
        Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)

    # Total covariance in utility space
    Xi_full = Omega_tilde + Lambda_full

    # Differenced covariance
    Xi_diff = M @ Xi_full @ M.T
    Xi_diff = 0.5 * (Xi_diff + Xi_diff.T)

    # PD check skipped: Xi_full = Omega_tilde + Lambda_full is PD
    # and M @ Xi_full @ M.T is PD since M has full row rank.

    prob = mvncd(xp.array(diff_V), xp.array(Xi_diff), method=control.method, xp=xp)
    return max(prob, 1e-300)


def _compute_mixture_prob(
    X_q: np.ndarray,
    params: dict,
    chosen: int,
    avail_q: np.ndarray,
    Lambda: np.ndarray,
    ranvar_indices: list[int] | None,
    control: MNPControl,
    xp,
) -> float:
    """Compute choice probability under mixture-of-normals specification.

    P(chosen) = sum_h pi_h * P(chosen | beta_h, Omega_h)
    """
    I = X_q.shape[0]
    nseg = control.nseg

    # Segment probabilities via softmax (reference segment 0 at 0)
    seg_params = params.get("segment_params", np.array([]))

    if len(seg_params) == 0:
        pi_h = np.array([1.0])
    else:
        raw = np.concatenate([[0.0], seg_params])
        raw_max = raw.max()
        exp_raw = np.exp(raw - raw_max)
        pi_h = exp_raw / exp_raw.sum()

    prob = 0.0

    # Segment 1: use base parameters
    beta_1 = params["beta"]
    Omega_L_1 = None
    if control.mix and ranvar_indices is not None and "omega_params" in params:
        Omega_L_1 = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    V_1 = X_q @ beta_1
    prob += pi_h[0] * _compute_mixed_choice_prob(
        X_q, V_1, chosen, avail_q, Lambda, Omega_L_1 if Omega_L_1 is not None else np.eye(1),
        ranvar_indices or [], control, xp
    ) if control.mix else pi_h[0] * _compute_choice_prob(
        X_q, beta_1, chosen, avail_q, Lambda, None, None, control, xp
    )

    # Additional segments
    for h in range(1, nseg):
        if h - 1 < len(params.get("segment_betas", [])):
            beta_h = params["segment_betas"][h - 1]
        else:
            beta_h = beta_1

        Omega_L_h = None
        if control.mix and ranvar_indices is not None:
            if h - 1 < len(params.get("segment_omegas", [])):
                Omega_L_h = _build_omega_cholesky(
                    params["segment_omegas"][h - 1], ranvar_indices, control
                )

        V_h = X_q @ beta_h

        if h < len(pi_h):
            if control.mix and Omega_L_h is not None:
                p_h = _compute_mixed_choice_prob(
                    X_q, V_h, chosen, avail_q, Lambda, Omega_L_h,
                    ranvar_indices, control, xp
                )
            else:
                p_h = _compute_choice_prob(
                    X_q, beta_h, chosen, avail_q, Lambda, None, None, control, xp
                )
            prob += pi_h[h] * p_h

    return max(prob, 1e-300)


def _numerical_gradient(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    avail: np.ndarray | None,
    n_alts: int,
    n_beta: int,
    control: MNPControl,
    ranvar_indices: list[int] | None,
    xp,
) -> np.ndarray:
    """Compute gradient via central finite differences."""
    eps = 1e-6
    n = len(theta)
    grad = np.zeros(n, dtype=np.float64)

    f0 = mnp_loglik(theta, X, y, avail, n_alts, n_beta, control, ranvar_indices, xp=xp)

    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        f_plus = mnp_loglik(
            theta_plus, X, y, avail, n_alts, n_beta, control, ranvar_indices, xp=xp
        )

        theta_minus = theta.copy()
        theta_minus[i] -= eps
        f_minus = mnp_loglik(
            theta_minus, X, y, avail, n_alts, n_beta, control, ranvar_indices, xp=xp
        )

        grad[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad


def count_params(
    n_beta: int,
    n_alts: int,
    control: MNPControl,
    ranvar_indices: list[int] | None = None,
) -> int:
    """Count total number of parameters for the MNP model."""
    I = n_alts
    n = n_beta

    if not control.iid:
        dim = I - 1
        if control.heteronly:
            n += dim  # scales only
        else:
            n += dim + dim * (dim - 1) // 2  # scales + correlations

    if control.mix and ranvar_indices is not None:
        n_rand = len(ranvar_indices)
        if control.randdiag:
            n += n_rand
        else:
            n += n_rand * (n_rand + 1) // 2

    if control.nseg > 1:
        n += control.nseg - 1  # segment probability params
        # Additional beta and omega for each extra segment
        for _ in range(1, control.nseg):
            n += n_beta
            if control.mix and ranvar_indices is not None:
                n_rand = len(ranvar_indices)
                if control.randdiag:
                    n += n_rand
                else:
                    n += n_rand * (n_rand + 1) // 2

    return n
