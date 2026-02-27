"""MNP log-likelihood function and gradient.

Implements the log-likelihood for Multinomial Probit models with:
- IID errors
- Flexible covariance (heteroscedastic, correlated errors)
- Random coefficients (mixed MNP)
- Mixture-of-normals random coefficients

The probability for each observation is computed via the MVNCD function,
which evaluates (I-1)-dimensional multivariate normal CDFs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd
from pybhatlib.matgradient._spherical import theta_to_corr
from pybhatlib.models.mnp._mnp_control import MNPControl


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

    # Compute log-likelihood
    total_ll = 0.0

    for q in range(N):
        # Available alternatives for this observation
        if avail is not None:
            avail_q = avail[q]
        else:
            avail_q = np.ones(I)

        chosen = y_np[q]

        if control.nseg <= 1:
            # Single segment
            prob_q = _compute_choice_prob(
                X_np[q], beta, chosen, avail_q, Lambda, Omega_L,
                ranvar_indices, control, xp
            )
        else:
            # Mixture of normals
            prob_q = _compute_mixture_prob(
                X_np[q], params, chosen, avail_q, Lambda,
                ranvar_indices, control, xp
            )

        # Safeguard against log(0)
        prob_q = max(prob_q, 1e-300)
        total_ll += np.log(prob_q)

    mean_ll = total_ll / N
    nll = -mean_ll

    if return_gradient:
        # Numerical gradient
        grad = _numerical_gradient(
            theta_np, X_np, y_np, avail, n_alts, n_beta,
            control, ranvar_indices, xp
        )
        return nll, grad

    return nll


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
        # IID errors: differenced covariance = 2*I_dim (if sigma^2=1)
        Lambda_diff = 2.0 * np.eye(dim, dtype=np.float64)
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

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(Lambda_diff)
    if eigvals.min() < 1e-10:
        Lambda_diff += np.eye(dim) * (1e-10 - eigvals.min())

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

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(Xi_diff)
    if eigvals.min() < 1e-10:
        Xi_diff += np.eye(dim) * (1e-10 - eigvals.min())

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

    # Segment probabilities via ordered softmax (pi_1 < pi_2 < ... < pi_H)
    seg_params = params.get("segment_params", np.array([]))

    if len(seg_params) == 0:
        pi_h = np.array([1.0])
    else:
        # Use cumulative logistic for ordered probabilities
        raw = np.concatenate([[0.0], np.cumsum(np.exp(seg_params))])
        pi_h = np.exp(raw) / np.sum(np.exp(raw))
        # Normalize
        pi_h = pi_h / pi_h.sum()

    prob = 0.0

    # Segment 1: use base parameters
    beta_1 = params["beta"]
    Omega_L_1 = None
    if control.mix and ranvar_indices is not None and "omega_params" in params:
        Omega_L_1 = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    V_1 = X_q @ beta_1
    prob += pi_h[0] * _compute_mixed_choice_prob(
        X_q, V_1, chosen, avail_q, Lambda, Omega_L_1 or np.eye(1),
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
