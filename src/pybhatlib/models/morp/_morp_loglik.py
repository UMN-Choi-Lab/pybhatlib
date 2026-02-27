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
from scipy.stats import norm

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.matgradient._spherical import theta_to_corr
from pybhatlib.models.morp._morp_control import MORPControl


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

        # P(lower <= eps <= upper) where eps ~ MVN(0, sigma)
        # Convert to standard form: P(lower/sd <= Z <= upper/sd)
        # with correlation matrix
        prob_q = mvncd_rect(
            xp.array(lower), xp.array(upper), xp.array(sigma),
            method=control.method, xp=xp,
        )

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
    if control.indep:
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
    if not control.indep:
        if control.heteronly:
            n += n_dims - 1  # scale params
        else:
            n += n_dims - 1  # scale params
            n += n_dims * (n_dims - 1) // 2  # correlation params

    return n


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
