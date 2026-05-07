"""MORP log-likelihood function.

Implements the log-likelihood for the Multivariate Ordered Response Probit
model. Each observation has D ordinal outcomes (dimensions), each with
J_d categories. The probability of observing category j_d in dimension d
is a rectangular MVNCD probability:

    P(y) = P(tau_{j_d - 1} < Y_d* <= tau_{j_d} for all d = 1..D)

where Y* ~ MVN(X @ beta, Sigma) is the latent utility vector.

Threshold parameterization ensures ordering:
    tau_1 is free, tau_{j} = tau_{j-1} + exp(delta_j) for j >= 2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.matgradient._spherical import theta_to_corr
from pybhatlib.models.morp._morp_control import MORPControl
from pybhatlib.models.morp._morp_grad_analytic import morp_analytic_gradient


def morp_loglik(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    control: MORPControl,
    *,
    return_gradient: bool = False,
    xp=None,
) -> float | tuple[float, NDArray]:
    """Compute negative mean log-likelihood for MORP model.

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector.
    X : ndarray, shape (N, D, n_vars)
        Design matrix for all observations and dimensions.
    y : ndarray, shape (N, D)
        Observed ordinal outcomes (0-based category indices).
    n_dims : int
        Number of ordinal dimensions (D).
    n_categories : list of int
        Number of categories per dimension.
    n_beta : int
        Number of regression coefficients.
    control : MORPControl
        Model control structure.
    return_gradient : bool
        If True, also return numerical gradient.
    xp : backend, optional

    Returns
    -------
    nll : float
        Negative mean log-likelihood.
    grad : ndarray (only if return_gradient=True)
    """
    if xp is None:
        xp = get_backend("numpy")

    theta_np = np.asarray(theta, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.int64)
    N = X_np.shape[0]

    # Analytic gradient computes both nll and grad in one pass when the
    # MVNCD method has an analytic-gradient implementation. Fall through to
    # the forward-only path + numerical FD otherwise.
    if (
        return_gradient
        and getattr(control, "analytic_grad", False)
        and control.method in ("me", "ovus")
    ):
        return morp_analytic_gradient(
            theta_np, X_np, y_np, n_dims, n_categories, n_beta, control,
        )

    # Per-observation log-likelihoods (also used by BHHH score computation).
    ll_per_obs = _per_obs_loglik(
        theta_np, X_np, y_np, n_dims, n_categories, n_beta, control, xp,
    )
    total_ll = float(ll_per_obs.sum())

    mean_ll = total_ll / N
    nll = -mean_ll

    if return_gradient:
        grad = _numerical_gradient_morp(
            theta_np, X_np, y_np, n_dims, n_categories, n_beta, control, xp
        )
        return nll, grad

    return nll


def _per_obs_loglik(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    control: MORPControl,
    xp=None,
) -> NDArray:
    """Per-observation log-likelihood vector for MORP.

    Same model as ``morp_loglik`` but returns the length-N array
    ``log P(y_q | theta, X_q)`` instead of the scalar mean. Used by the
    BHHH / sandwich SE computation in MORPModel — per-obs scores are
    obtained by finite-differencing this function.

    Returns
    -------
    ll_per_obs : ndarray, shape (N,)
        Log-probability for each observation under the current ``theta``.
    """
    if xp is None:
        xp = get_backend("numpy")

    theta_np = np.asarray(theta, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.int64)
    N = X_np.shape[0]

    beta, thresholds, sigma = _unpack_morp_params(
        theta_np, n_beta, n_dims, n_categories, control
    )

    ll_per_obs = np.zeros(N, dtype=np.float64)

    for q in range(N):
        # Latent utility mean: mu_d = X_d @ beta for each dimension
        mu_q = np.array([X_np[q, d] @ beta for d in range(n_dims)])

        # Build lower and upper limits from thresholds
        lower = np.zeros(n_dims, dtype=np.float64)
        upper = np.zeros(n_dims, dtype=np.float64)

        for d in range(n_dims):
            j = y_np[q, d]  # observed category (0-based)
            tau_d = thresholds[d]  # threshold values for dimension d

            if j == 0:
                lower[d] = -np.inf
            else:
                lower[d] = tau_d[j - 1] - mu_q[d]

            if j == n_categories[d] - 1:
                upper[d] = np.inf
            else:
                upper[d] = tau_d[j] - mu_q[d]

        # P(lower <= eps <= upper) where eps ~ MVN(0, sigma).
        # We collapse +inf/-inf bounds out of the integration before
        # passing to mvncd_rect to avoid the NaN that mvncd produces for
        # +inf entries when sigma is non-diagonal (the rectangle CDF on
        # the surviving "alive" sub-block of sigma is mathematically
        # equivalent to the original integral).
        prob_q = _rect_prob_finite_only(lower, upper, sigma, control, xp)

        prob_q = max(prob_q, 1e-300)
        ll_per_obs[q] = np.log(prob_q)

    return ll_per_obs


def _unpack_morp_params(
    theta: np.ndarray,
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Unpack flat parameter vector into beta, thresholds, sigma."""
    idx = 0

    # Beta coefficients
    beta = theta[idx: idx + n_beta]
    idx += n_beta

    # Thresholds per dimension
    # tau_1 is free; tau_j = tau_{j-1} + exp(delta_j) for j >= 2
    thresholds = []
    for d in range(n_dims):
        n_thresh = n_categories[d] - 1  # J_d - 1 thresholds
        if n_thresh <= 0:
            thresholds.append(np.array([]))
            continue

        tau_d = np.zeros(n_thresh, dtype=np.float64)
        tau_d[0] = theta[idx]
        idx += 1
        for j in range(1, n_thresh):
            tau_d[j] = tau_d[j - 1] + np.exp(theta[idx])
            idx += 1
        thresholds.append(tau_d)

    # Covariance parameters
    if control.iid:
        sigma = np.eye(n_dims, dtype=np.float64)
    elif control.heteronly:
        # Diagonal: D scale parameters (first dimension is reference)
        scales = np.ones(n_dims, dtype=np.float64)
        for d in range(1, n_dims):
            scales[d] = np.exp(theta[idx])
            idx += 1
        sigma = np.diag(scales**2)
    else:
        # Full: (D-1) scale params + D*(D-1)/2 correlation params
        scales = np.ones(n_dims, dtype=np.float64)
        for d in range(1, n_dims):
            scales[d] = np.exp(theta[idx])
            idx += 1

        n_corr = n_dims * (n_dims - 1) // 2
        if n_corr > 0:
            corr_theta = theta[idx: idx + n_corr]
            idx += n_corr
            if control.spherical:
                corr = theta_to_corr(corr_theta, n_dims)
            else:
                # Direct parameterization via tanh
                corr = np.eye(n_dims, dtype=np.float64)
                c_idx = 0
                for i in range(n_dims):
                    for j in range(i + 1, n_dims):
                        corr[i, j] = np.tanh(corr_theta[c_idx])
                        corr[j, i] = corr[i, j]
                        c_idx += 1
        else:
            corr = np.eye(n_dims)

        D = np.diag(scales)
        sigma = D @ corr @ D

    return beta, thresholds, sigma


def count_morp_params(
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
) -> int:
    """Count total number of parameters for MORP model."""
    n = n_beta

    # Thresholds: sum of (J_d - 1) for each dimension
    for d in range(n_dims):
        n += max(0, n_categories[d] - 1)

    # Covariance
    if not control.iid:
        if control.heteronly:
            n += n_dims - 1  # scale params
        else:
            n += n_dims - 1  # scale params
            n += n_dims * (n_dims - 1) // 2  # correlation params

    return n


def _rect_prob_finite_only(
    lower: np.ndarray,
    upper: np.ndarray,
    sigma: np.ndarray,
    control: MORPControl,
    xp,
) -> float:
    """Compute ``P(lower <= eps <= upper)`` while tolerating ``+/-inf`` bounds.

    Since PR #9 ``mvncd_rect`` natively collapses ``+inf`` upper-bound
    dimensions via the ``_drop_inf_dims`` guard inside ``mvncd``.  We
    therefore drop fully-open dims (``[-inf, +inf]``) here for the cheap
    case, short-circuit on empty intervals, and delegate everything else
    directly to ``mvncd_rect`` — which now uses the *same* marginalize-
    via-submatrix approach as the analytic path
    (``_morp_grad_analytic._rect_prob_and_grad``).  This eliminates the
    silent forward/analytic asymmetry under approximate MVNCD methods
    (ME / OVUS) that the previous sign-flip implementation introduced.

    See PR #8 review (Opus, P0): the prior sign-flip + sub-Sigma was
    individually correct under exact MVN-CDF but produced numerically
    different ``P`` values from the analytic path under ME/OVUS, since
    the two paths used different K-dimensional kernels.
    """
    K = len(lower)
    keep = []
    new_lower = []
    new_upper = []
    for d in range(K):
        u_d = upper[d]
        l_d = lower[d]
        if np.isposinf(u_d) and np.isneginf(l_d):
            # Whole real line for this dim: integrates to 1, drop it.
            continue
        if np.isposinf(l_d) or np.isneginf(u_d):
            # Empty interval: probability is exactly 0.
            return 0.0
        keep.append(d)
        new_lower.append(l_d)
        new_upper.append(u_d)

    if len(keep) == 0:
        return 1.0

    sub_idx = np.array(keep, dtype=np.int64)
    sigma_sub = sigma[np.ix_(sub_idx, sub_idx)].copy()
    lower_a = np.asarray(new_lower, dtype=np.float64)
    upper_a = np.asarray(new_upper, dtype=np.float64)

    # ``mvncd_rect`` handles any remaining ``+inf`` upper bound natively
    # (via ``mvncd``'s ``_drop_inf_dims`` guard) — same path the analytic
    # gradient takes — so the forward and analytic K-D evaluations now
    # share kernels regardless of method.
    return mvncd_rect(
        xp.array(lower_a), xp.array(upper_a), xp.array(sigma_sub),
        method=control.method, xp=xp,
    )


def _numerical_gradient_morp(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    control: MORPControl,
    xp,
) -> np.ndarray:
    """Compute gradient via central finite differences."""
    eps = 1e-6
    n = len(theta)
    grad = np.zeros(n, dtype=np.float64)

    for i in range(n):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        f_plus = morp_loglik(
            theta_plus, X, y, n_dims, n_categories, n_beta, control, xp=xp
        )

        theta_minus = theta.copy()
        theta_minus[i] -= eps
        f_minus = morp_loglik(
            theta_minus, X, y, n_dims, n_categories, n_beta, control, xp=xp
        )

        grad[i] = (f_plus - f_minus) / (2.0 * eps)

    return grad
