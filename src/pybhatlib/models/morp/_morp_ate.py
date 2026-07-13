"""Average Treatment Effect (ATE) post-estimation for MORP.

Computes predicted ordinal probabilities under base and treatment
scenarios to enable ATE analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from scipy.special import ndtr as _ndtr

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._ate_common import (
    ScenarioSpec,
    apply_scenario_overrides as _apply_scenario_overrides,
    scenarios_to_dict as _scenarios_to_dict,
)
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
    shares_per_scenario: "dict[str, list[NDArray]] | None" = None

    def comparison(self, base: str, treatment: str) -> "list[NDArray]":
        """Percentage change between two scenarios, per outcome dimension.

        Mirrors :meth:`MNPATEResult.comparison`, but because MORP predicts one
        ordinal-category distribution per dimension, the result is a list of
        arrays — one per dimension ``d``, shape ``(n_categories[d],)``:
        ``100 * (treatment_probs[d] - base_probs[d]) / base_probs[d]``.

        Raises
        ------
        ValueError
            If ``shares_per_scenario`` is None or a scenario name is missing.
        """
        if self.shares_per_scenario is None:
            raise ValueError(
                "comparison() requires shares_per_scenario; run morp_ate with "
                "scenarios=..."
            )
        if base not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{base}' not found in shares_per_scenario")
        if treatment not in self.shares_per_scenario:
            raise ValueError(
                f"Scenario '{treatment}' not found in shares_per_scenario"
            )

        b_list = self.shares_per_scenario[base]
        t_list = self.shares_per_scenario[treatment]
        out: list[NDArray] = []
        for b, t in zip(b_list, t_list):
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(np.where(b > 0, 100.0 * (t - b) / b, np.nan))
        return out


def _compute_morp_predicted_probs(
    X_np: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    results: MORPResults,
) -> list[NDArray]:
    """Mean per-dimension ordinal-category probabilities for one design matrix.

    Extracted from :func:`morp_ate` so the same computation drives both the
    single-design path and the per-scenario loop.

    Returns
    -------
    list[NDArray]
        ``out[d]`` has shape ``(n_categories[d],)`` (mean over observations).
    """
    control = results.control
    beta, thresholds, sigma = _unpack_morp_params(
        results.params, n_beta, n_dims, n_categories, control
    )

    N = X_np.shape[0]
    pred_probs = [
        np.zeros((N, n_categories[d]), dtype=np.float64)
        for d in range(n_dims)
    ]

    for q in range(N):
        mu_q = np.array([X_np[q, d] @ beta for d in range(n_dims)])
        for d in range(n_dims):
            tau_d = thresholds[d]
            sd_d = np.sqrt(max(sigma[d, d], 1e-30))
            for j in range(n_categories[d]):
                z_lower = -np.inf if j == 0 else (tau_d[j - 1] - mu_q[d]) / sd_d
                z_upper = (
                    np.inf if j == n_categories[d] - 1
                    else (tau_d[j] - mu_q[d]) / sd_d
                )
                p_upper = _ndtr(z_upper) if np.isfinite(z_upper) else 1.0
                p_lower = _ndtr(z_lower) if np.isfinite(z_lower) else 0.0
                pred_probs[d][q, j] = max(0.0, p_upper - p_lower)

    return [pred_probs[d].mean(axis=0) for d in range(n_dims)]


def morp_ate(
    results: MORPResults,
    X: NDArray | None = None,
    n_dims: int | None = None,
    n_categories: list[int] | None = None,
    n_beta: int | None = None,
    *,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    dep_vars: list[str] | None = None,
    scenarios: "ScenarioSpec | None" = None,
) -> MORPATEResult:
    """Compute predicted ordinal probabilities for MORP model.

    Supports two modes, mirroring :func:`mnp_ate`:

    **Single-design mode** (default): pass a pre-built ``X`` (or
    ``data`` + ``spec`` + ``dep_vars`` to rebuild it) and get mean per-dimension
    ordinal probabilities in ``predicted_probs``.

    **Scenario mode** (``scenarios=``): pass ``data`` + ``spec`` + ``dep_vars``
    and a dict/DataFrame of per-variable overrides.  The design matrix is
    rebuilt for each scenario and mean probabilities are returned in
    ``shares_per_scenario`` (with a ``.comparison(base, treatment)`` helper).

    Parameters
    ----------
    results : MORPResults
        Fitted MORP model results (or one from ``from_estimates``).
    X : ndarray, shape (N, D, n_vars), optional
        Pre-built design matrix.  Optional when ``data``+``spec``+``dep_vars``
        are given (or in scenario mode).
    n_dims : int, optional
        Number of dimensions.  Inferred from ``dep_vars``, else derived from
        ``results``, when omitted.
    n_categories : list of int, optional
        Number of categories per dimension.  Derived from ``results`` when
        omitted.
    n_beta : int, optional
        Number of beta coefficients.  Derived from ``results`` when omitted
        (falling back to ``len(spec)`` for legacy results objects).
    data : pd.DataFrame, optional
        Dataset — required to rebuild ``X`` (and for scenario mode).
    spec : dict, optional
        Variable specification mapping (as passed to ``MORPModel``).
    dep_vars : list of str, optional
        Outcome-dimension column names (the ``parse_spec`` "alternatives" slot).
    scenarios : dict or pd.DataFrame, optional
        Per-variable overrides keyed by scenario name (see :func:`mnp_ate`).

    Returns
    -------
    result : MORPATEResult
    """
    if n_dims is None and dep_vars is not None:
        n_dims = len(dep_vars)
    if n_beta is None and results.n_beta is None and spec is not None:
        # Only reached for results objects predating stored structural
        # metadata; one spec entry contributes one beta coefficient.
        n_beta = len(spec)
    n_dims, n_categories, n_beta = results._structure(n_dims, n_categories, n_beta)

    # --- Scenario mode ---
    if scenarios is not None:
        if data is None or spec is None or dep_vars is None:
            raise ValueError(
                "data, spec, and dep_vars are required when using scenarios"
            )
        X_base, _ = parse_spec(spec, data, dep_vars, nseg=1)
        X_base_np = np.asarray(X_base, dtype=np.float64)
        baseline = _compute_morp_predicted_probs(
            X_base_np, n_dims, n_categories, n_beta, results
        )
        shares_per_scenario: dict[str, list[NDArray]] = {}
        for name, overrides in _scenarios_to_dict(scenarios).items():
            data_mod = _apply_scenario_overrides(data, overrides)
            X_mod, _ = parse_spec(spec, data_mod, dep_vars, nseg=1)
            shares_per_scenario[name] = _compute_morp_predicted_probs(
                np.asarray(X_mod, dtype=np.float64),
                n_dims, n_categories, n_beta, results,
            )
        return MORPATEResult(
            n_obs=X_base_np.shape[0],
            predicted_probs=baseline,
            shares_per_scenario=shares_per_scenario,
        )

    # --- Single-design mode ---
    if X is None:
        if data is None or spec is None or dep_vars is None:
            raise ValueError(
                "Either X, or data+spec+dep_vars, must be provided"
            )
        X, _ = parse_spec(spec, data, dep_vars, nseg=1)

    X_np = np.asarray(X, dtype=np.float64)
    mean_probs = _compute_morp_predicted_probs(
        X_np, n_dims, n_categories, n_beta, results
    )
    return MORPATEResult(n_obs=X_np.shape[0], predicted_probs=mean_probs)


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
    n_dims: int | None = None,
    n_categories: list[int] | None = None,
    n_beta: int | None = None,
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
    n_dims, n_categories, n_beta = results._structure(n_dims, n_categories, n_beta)
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
    X: NDArray | None = None,
    n_dims: int | None = None,
    n_categories: list[int] | None = None,
    n_beta: int | None = None,
    *,
    dep_vars: list[str] | None = None,
    joint: bool = False,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    scenarios: "ScenarioSpec | None" = None,
):
    """Compute MORP ATE predictions directly from natural-space coefficients.

    Convenience wrapper that builds a results object via
    :meth:`MORPResults.from_estimates` and dispatches to :func:`morp_ate`
    (marginals, default; or the ``scenarios=`` multi-counterfactual API) or to
    :func:`morp_joint_probs` (``joint=True``, GAUSS ``ate1.csv`` equivalent).

    Parameters
    ----------
    beta, thresholds, correlation
        Final reported coefficients (see ``MORPResults.from_estimates``).
    X : ndarray, shape (N, D, n_vars), optional
        Design matrix.  Optional when ``data``+``spec``+``dep_vars`` are given.
    n_dims, n_categories, n_beta
        Model structure (``n_dims`` inferred from ``dep_vars`` when omitted).
    dep_vars : list of str, optional
        Outcome names (defaults to ``y1, y2, ...``).
    joint : bool, default False
        Return the joint combination distribution instead of marginals.
    data, spec, scenarios
        Forwarded to :func:`morp_ate` (same counterfactual/scenario API).

    Returns
    -------
    MORPATEResult or MORPJointATEResult
    """
    if n_dims is None and dep_vars is not None:
        n_dims = len(dep_vars)
    if dep_vars is None:
        if n_dims is None:
            raise ValueError("Provide n_dims or dep_vars")
        dep_vars = [f"y{d + 1}" for d in range(n_dims)]
    results = MORPResults.from_estimates(
        beta, thresholds, correlation,
        dep_vars=dep_vars, n_categories=n_categories,
    )
    if scenarios is not None:
        if joint:
            raise ValueError(
                "joint=True is not supported with scenarios=; call "
                "morp_joint_probs per scenario instead."
            )
        return morp_ate(
            results, X, n_dims, n_categories, n_beta,
            data=data, spec=spec, dep_vars=dep_vars, scenarios=scenarios,
        )
    if joint:
        return morp_joint_probs(results, X, n_dims, n_categories, n_beta)
    return morp_ate(
        results, X, n_dims, n_categories, n_beta,
        data=data, spec=spec, dep_vars=dep_vars,
    )
