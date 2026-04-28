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

    # Unpack parameters
    beta, thresholds, sigma = _unpack_morp_params(
        theta_np, n_beta, n_dims, n_categories, control
    )

    total_ll = 0.0

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
        total_ll += np.log(prob_q)

    mean_ll = total_ll / N
    nll = -mean_ll

    if return_gradient:
        grad = _numerical_gradient_morp(
            theta_np, X_np, y_np, n_dims, n_categories, n_beta, control, xp
        )
        return nll, grad

    return nll


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

    The shipped ``mvncd_rect`` skips inclusion-exclusion vertices that
    select a ``-inf`` lower bound but does not collapse ``+inf`` upper
    bounds; the underlying ``mvncd`` returns ``NaN`` when the limit
    vector contains ``+inf`` and ``sigma`` has off-diagonal mass, which
    silently corrupts the MORP forward log-likelihood.

    This helper marginalizes any dimension whose upper bound is
    ``+inf`` (and sets the integration to 0 if its lower is also
    ``+inf``), then delegates to ``mvncd_rect`` on the surviving
    sub-block. The marginal of an MVN over ``Sigma`` is the MVN with
    the corresponding sub-block of ``Sigma``.
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
        if np.isposinf(u_d):
            # One-sided lower truncation: split into 1 - P(eps <= l).
            # Easier: keep dim, but mvncd_rect can't take +inf upper.
            # We rewrite via P(lower <= eps) = 1 - P(eps < lower); but
            # the multi-dim case is messier. Use sign flip: replace
            # eps_d -> -eps_d (flip sigma's row/col d signs), which
            # turns the half-line [l, inf) into (-inf, -l]. We avoid
            # this complexity for now by collapsing only fully-open
            # dims; for half-open with finite lower we need a fix.
            keep.append(d)
            new_lower.append(l_d)
            new_upper.append(u_d)
        else:
            keep.append(d)
            new_lower.append(l_d)
            new_upper.append(u_d)

    if len(keep) == 0:
        return 1.0

    # Convert (-inf, +inf) cases on a per-dim basis using sign flip:
    # If a dim has +inf upper but finite lower, flip eps_d -> -eps_d so
    # the integration becomes (-inf, -lower]. This requires flipping the
    # sign of row/col d in sigma but preserves the MVN structure.
    K_a = len(keep)
    sub_idx = np.array(keep, dtype=np.int64)
    sigma_sub = sigma[np.ix_(sub_idx, sub_idx)].copy()
    lower_a = np.asarray(new_lower, dtype=np.float64)
    upper_a = np.asarray(new_upper, dtype=np.float64)

    flip = np.isposinf(upper_a)
    if np.any(flip):
        # Apply -1 to flipped rows/cols of sigma_sub; the (i,i) double
        # flip cancels, so diagonal entries are unaffected.
        sign_vec = np.where(flip, -1.0, 1.0)
        sigma_sub = sigma_sub * np.outer(sign_vec, sign_vec)
        # Swap and negate bounds where flipped: new_lower = -inf,
        # new_upper = -old_lower.
        new_upper_arr = np.where(flip, -lower_a, upper_a)
        new_lower_arr = np.where(flip, -np.inf, lower_a)
        lower_a, upper_a = new_lower_arr, new_upper_arr

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

    f0 = morp_loglik(theta, X, y, n_dims, n_categories, n_beta, control, xp=xp)

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
