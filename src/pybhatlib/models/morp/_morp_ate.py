"""Average Treatment Effect (ATE) post-estimation for MORP.

Computes predicted ordinal probabilities under base and treatment
scenarios to enable ATE analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import NDArray

from scipy.special import ndtr as _ndtr

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.models.morp._morp_loglik import (
    _rect_prob_finite_only,
    _unpack_morp_params,
)
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

                p_upper = _ndtr(z_upper) if np.isfinite(z_upper) else 1.0
                p_lower = _ndtr(z_lower) if np.isfinite(z_lower) else 0.0
                pred_probs[d][q, j] = max(0.0, p_upper - p_lower)

    # Mean across observations
    mean_probs = [pred_probs[d].mean(axis=0) for d in range(n_dims)]

    return MORPATEResult(
        n_obs=N,
        predicted_probs=mean_probs,
    )


@dataclass
class MORPJointATEResult:
    """Mean joint category-combination probabilities (GAUSS ATE equivalent).

    Mirrors the ``combo ~ meancombprob`` matrix that GAUSS BHATLIB's MORP ATE
    script writes (e.g. ``ate1.csv``): one row per joint outcome combination,
    averaged over observations.

    Attributes
    ----------
    n_obs : int
        Number of observations averaged over.
    combos : NDArray, shape (n_combos, n_dims)
        Category combinations, **1-based** to match GAUSS output.
    probs : NDArray, shape (n_combos,)
        Mean joint probability of each combination across observations.
    """

    n_obs: int
    combos: NDArray
    probs: NDArray

    def marginal(self, dim: int) -> NDArray:
        """Marginal P(Y_dim = k) for each category k (collapsing the joint)."""
        n_cat = int(self.combos[:, dim].max())
        out = np.zeros(n_cat, dtype=np.float64)
        for k in range(1, n_cat + 1):
            out[k - 1] = self.probs[self.combos[:, dim] == k].sum()
        return out


def morp_joint_probs(
    results: MORPResults,
    X: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
) -> MORPJointATEResult:
    """Mean joint probability of every ordinal outcome combination.

    Reproduces GAUSS BHATLIB's MORP ATE computation: for each joint category
    combination it evaluates the MVN rectangle probability per observation
    (the same kernel as the log-likelihood) and averages over observations.
    Use with :meth:`MORPResults.from_estimates` to compute ATEs from the final
    reported coefficients without re-fitting, exactly like the GAUSS workflow.

    For an ATE, call this under a base-scenario design matrix and a
    treatment-scenario design matrix and difference the marginals, e.g.::

        base = morp_joint_probs(res, X_base, ...).marginal(0)
        treat = morp_joint_probs(res, X_treat, ...).marginal(0)
        ate = treat - base

    Parameters
    ----------
    results : MORPResults
        Fitted results, or one built via :meth:`MORPResults.from_estimates`.
    X : ndarray, shape (N, D, n_vars)
        Design matrix.
    n_dims, n_categories, n_beta
        Model structure.

    Returns
    -------
    MORPJointATEResult
    """
    xp = get_backend("numpy")
    control = results.control
    beta, thresholds, sigma = _unpack_morp_params(
        results.params, n_beta, n_dims, n_categories, control
    )

    X_np = np.asarray(X, dtype=np.float64)
    N = X_np.shape[0]
    # mu[q, d] = X[q, d] @ beta
    mu = np.stack([X_np[:, d, :] @ beta for d in range(n_dims)], axis=1)

    combos = list(product(*[range(n_categories[d]) for d in range(n_dims)]))
    probs = np.zeros(len(combos), dtype=np.float64)

    for ci, combo in enumerate(combos):
        acc = 0.0
        for q in range(N):
            lower = np.empty(n_dims, dtype=np.float64)
            upper = np.empty(n_dims, dtype=np.float64)
            for d in range(n_dims):
                j = combo[d]
                tau_d = thresholds[d]
                lower[d] = -np.inf if j == 0 else tau_d[j - 1] - mu[q, d]
                upper[d] = (
                    np.inf if j == n_categories[d] - 1 else tau_d[j] - mu[q, d]
                )
            acc += _rect_prob_finite_only(lower, upper, sigma, control, xp)
        probs[ci] = acc / N

    combos_arr = np.asarray(combos, dtype=np.int64) + 1  # 1-based like GAUSS
    return MORPJointATEResult(n_obs=N, combos=combos_arr, probs=probs)


def morp_ate_from_params(
    beta: NDArray,
    thresholds: list[NDArray],
    correlation: NDArray | None,
    X: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    *,
    dep_vars: list[str] | None = None,
    joint: bool = False,
):
    """Compute MORP ATE predictions directly from natural-space coefficients.

    Convenience wrapper that builds a results object via
    :meth:`MORPResults.from_estimates` and dispatches to either the marginal
    predicted probabilities (``joint=False``, default) or the full joint
    distribution (``joint=True``, GAUSS ``ate1.csv`` equivalent).

    Parameters
    ----------
    beta, thresholds, correlation
        Final reported coefficients (see ``MORPResults.from_estimates``).
    X : ndarray, shape (N, D, n_vars)
        Design matrix.
    n_dims, n_categories, n_beta
        Model structure.
    dep_vars : list of str, optional
        Outcome names (defaults to ``y1, y2, ...``).
    joint : bool, default False
        Return the joint combination distribution instead of marginals.

    Returns
    -------
    MORPATEResult or MORPJointATEResult
    """
    if dep_vars is None:
        dep_vars = [f"y{d + 1}" for d in range(n_dims)]
    results = MORPResults.from_estimates(
        beta, thresholds, correlation,
        dep_vars=dep_vars, n_categories=n_categories,
    )
    if joint:
        return morp_joint_probs(results, X, n_dims, n_categories, n_beta)
    return morp_ate(results, X, n_dims, n_categories, n_beta)
