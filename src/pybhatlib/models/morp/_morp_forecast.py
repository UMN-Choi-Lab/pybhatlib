"""MORP model prediction and forecasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import get_backend
from pybhatlib.models.morp._morp_loglik import _unpack_morp_params
from pybhatlib.models.morp._morp_results import MORPResults


def morp_predict(
    results: MORPResults,
    X_new: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
) -> list[NDArray]:
    """Predict ordinal probabilities for new observations.

    Parameters
    ----------
    results : MORPResults
        Fitted model results.
    X_new : ndarray, shape (N, D, n_vars)
        Design matrix for new observations.
    n_dims : int
        Number of dimensions.
    n_categories : list of int
        Number of categories per dimension.
    n_beta : int
        Number of beta coefficients.

    Returns
    -------
    probs : list of ndarray
        probs[d] has shape (N, n_categories[d]) with predicted probabilities
        for each category in dimension d.
    """
    theta_hat = results.params
    control = results.control

    beta, thresholds, sigma = _unpack_morp_params(
        theta_hat, n_beta, n_dims, n_categories, control
    )

    X_np = np.asarray(X_new, dtype=np.float64)
    N = X_np.shape[0]

    probs = [
        np.zeros((N, n_categories[d]), dtype=np.float64)
        for d in range(n_dims)
    ]

    for q in range(N):
        mu_q = np.array([X_np[q, d] @ beta for d in range(n_dims)])

        for d in range(n_dims):
            tau_d = thresholds[d]
            sd_d = np.sqrt(max(sigma[d, d], 1e-30))

            for j in range(n_categories[d]):
                if j == 0:
                    z_lower = -np.inf
                else:
                    z_lower = (tau_d[j - 1] - mu_q[d]) / sd_d

                if j == n_categories[d] - 1:
                    z_upper = np.inf
                else:
                    z_upper = (tau_d[j] - mu_q[d]) / sd_d

                p_upper = norm.cdf(z_upper) if np.isfinite(z_upper) else 1.0
                p_lower = norm.cdf(z_lower) if np.isfinite(z_lower) else 0.0
                probs[d][q, j] = max(0.0, p_upper - p_lower)

            # Normalize
            row_sum = probs[d][q].sum()
            if row_sum > 0:
                probs[d][q] /= row_sum

    return probs


def morp_predict_category(
    results: MORPResults,
    X_new: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
) -> NDArray:
    """Predict most likely category for new observations.

    Parameters
    ----------
    results : MORPResults
        Fitted model results.
    X_new : ndarray, shape (N, D, n_vars)
        Design matrix.
    n_dims : int
        Number of dimensions.
    n_categories : list of int
        Number of categories per dimension.
    n_beta : int
        Number of beta coefficients.

    Returns
    -------
    categories : ndarray, shape (N, D)
        Predicted category (0-based) for each observation and dimension.
    """
    probs = morp_predict(results, X_new, n_dims, n_categories, n_beta)
    N = probs[0].shape[0]

    categories = np.zeros((N, n_dims), dtype=np.int64)
    for d in range(n_dims):
        categories[:, d] = np.argmax(probs[d], axis=1)

    return categories
