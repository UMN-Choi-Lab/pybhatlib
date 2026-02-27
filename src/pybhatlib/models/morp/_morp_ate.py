"""Average Treatment Effect (ATE) post-estimation for MORP.

Computes predicted ordinal probabilities under base and treatment
scenarios to enable ATE analysis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.models.morp._morp_loglik import _unpack_morp_params
from pybhatlib.models.morp._morp_results import MORPResults


@dataclass
class MORPATEResult:
    """MORP ATE analysis results.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_probs : list[NDArray]
        Mean predicted probability for each category in each dimension.
        predicted_probs[d] has shape (n_categories[d],).
    base_probs : list[NDArray] or None
        Predicted probabilities at base level.
    treatment_probs : list[NDArray] or None
        Predicted probabilities at treatment level.
    """

    n_obs: int
    predicted_probs: list[NDArray]
    base_probs: list[NDArray] | None = None
    treatment_probs: list[NDArray] | None = None


def morp_ate(
    results: MORPResults,
    X: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
) -> MORPATEResult:
    """Compute predicted ordinal probabilities for MORP model.

    Parameters
    ----------
    results : MORPResults
        Fitted MORP model results.
    X : ndarray, shape (N, D, n_vars)
        Design matrix.
    n_dims : int
        Number of dimensions.
    n_categories : list of int
        Number of categories per dimension.
    n_beta : int
        Number of beta coefficients.

    Returns
    -------
    result : MORPATEResult
    """
    xp = get_backend("numpy")

    theta_hat = results.params
    control = results.control
    method = control.method if control else "ovus"

    beta, thresholds, sigma = _unpack_morp_params(
        theta_hat, n_beta, n_dims, n_categories, control
    )

    X_np = np.asarray(X, dtype=np.float64)
    N = X_np.shape[0]

    # Predicted probabilities per dimension and category
    pred_probs = [
        np.zeros((N, n_categories[d]), dtype=np.float64)
        for d in range(n_dims)
    ]

    for q in range(N):
        mu_q = np.array([X_np[q, d] @ beta for d in range(n_dims)])

        for d in range(n_dims):
            for j in range(n_categories[d]):
                # P(Y_d = j) = P(tau_{j-1} < Y_d* <= tau_j)
                # For the univariate marginal of dimension d
                tau_d = thresholds[d]
                sd_d = np.sqrt(max(sigma[d, d], 1e-30))

                if j == 0:
                    z_lower = -np.inf
                else:
                    z_lower = (tau_d[j - 1] - mu_q[d]) / sd_d

                if j == n_categories[d] - 1:
                    z_upper = np.inf
                else:
                    z_upper = (tau_d[j] - mu_q[d]) / sd_d

                from scipy.stats import norm
                p_upper = norm.cdf(z_upper) if np.isfinite(z_upper) else 1.0
                p_lower = norm.cdf(z_lower) if np.isfinite(z_lower) else 0.0
                pred_probs[d][q, j] = max(0.0, p_upper - p_lower)

    # Mean across observations
    mean_probs = [pred_probs[d].mean(axis=0) for d in range(n_dims)]

    return MORPATEResult(
        n_obs=N,
        predicted_probs=mean_probs,
    )
