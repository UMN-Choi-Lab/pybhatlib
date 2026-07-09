"""Average Treatment Effect (ATE) post-estimation for the MDCEV model.

Computes predicted consumption shares and percentage ATEs by comparing
base-level vs treatment-level scenarios, following the same pattern used
for MNP and MNL in pybhatlib.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models._ate_common import (
    ATEResultMixin,
    scenarios_to_dict as _scenarios_to_dict,
)
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults
from pybhatlib.models.mdcev._mdcev_forecast import (
    mdcev_predict,
    prepare_mdcev_forecast_data,
)

# Narrower than the shared ``_ate_common.ScenarioSpec``: MDCEV overrides are
# scalar-valued (broadcast to a column via prepare_mdcev_forecast_data), with
# no source-column string mode as MNP and MORP have.
ScenarioSpec = Union[
    "dict[str, dict[str, float]]",
    "pd.DataFrame",
]


@dataclass
class MDCEVATEResult(ATEResultMixin):
    """Average Treatment Effect analysis results for MDCEV.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_shares : NDArray, shape (nc,)
        Mean predicted consumption share for each alternative at the
        observed data values.
    base_shares : NDArray or None, shape (nc,)
        Predicted shares at the base level of the treatment variable.
    treatment_shares : NDArray or None, shape (nc,)
        Predicted shares at the treatment level.
    pct_ate : NDArray or None, shape (nc,)
        Percentage ATE = (treatment_shares - base_shares) / base_shares * 100.
    alternative_names : list[str] or None
        Names of the alternatives, in order.
    """

    _model_label = "MDCEV"
    _ate_func_name = "mdcev_ate"

    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None = None
    treatment_shares: NDArray | None = None
    pct_ate: NDArray | None = None
    alternative_names: list[str] | None = None
    shares_per_scenario: "dict[str, NDArray] | None" = None


def mdcev_ate(
    results: MDCEVResults,
    X: NDArray | None = None,
    X_gam: NDArray | None = None,
    price: NDArray | None = None,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
    n_draws: int = 1000,
    seed: int = 1234,
    *,
    model=None,
    data: "pd.DataFrame | None" = None,
    scenarios: "ScenarioSpec | None" = None,
    budget_col: str = "tot",
) -> MDCEVATEResult:
    """Compute predicted consumption shares and ATE for the MDCEV model.

    Supports two modes:

    **Legacy / single-override mode** (default): pass pre-built ``X`` /
    ``X_gam`` / ``price``.  With ``changevar_idx`` + ``base_val`` +
    ``treatment_val`` the indexed element of ``X`` is set to each value and the
    shares compared.

    **Scenario mode** (``scenarios=``): pass the fitted ``model`` and its
    ``data`` (DataFrame) plus a dict/DataFrame of ``{column: scalar}`` overrides
    per scenario.  Each scenario's design matrices are rebuilt via
    :func:`prepare_mdcev_forecast_data` (both the utility and satiation
    matrices) and mean shares are returned in ``shares_per_scenario`` (with a
    ``.comparison(base, treatment)`` helper), mirroring :func:`mnp_ate`.

    Parameters
    ----------
    results : MDCEVResults
        Fitted MDCEV model results.
    X : NDArray, shape (N, nc, nvarm), optional
        Baseline utility design matrix (legacy mode).
    X_gam : NDArray, shape (N, nc, nvargam), optional
        Satiation utility design matrix (legacy mode).
    price : NDArray, shape (N, nc), optional
        Price matrix (legacy mode).
    changevar_idx : tuple (alt_idx, var_idx) or None
        Which element of X to modify: ``X[:, alt_idx, var_idx]``.
        Pass None to skip ATE computation.
    base_val : float or None
        Value to set ``changevar_idx`` to for the base scenario.
    treatment_val : float or None
        Value to set ``changevar_idx`` to for the treatment scenario.
    alternative_names : list[str] or None
        Names of the alternatives, in order.
    n_draws : int
        Monte Carlo draws for share computation.
    seed : int
        Random seed.
    model : MDCEVModel, optional
        Fitted model — required for ``scenarios=`` (carries utility_spec /
        gamma_spec / availability used to rebuild the matrices).
    data : pd.DataFrame, optional
        Dataset to rebuild scenario matrices from (required for ``scenarios=``).
    scenarios : dict or pd.DataFrame, optional
        Per-column scalar overrides keyed by scenario name.
    budget_col : str, default "tot"
        Budget column name passed to :func:`prepare_mdcev_forecast_data`.

    Returns
    -------
    MDCEVATEResult
    """
    # --- Scenario mode ---
    if scenarios is not None:
        if model is None or data is None:
            raise ValueError(
                "model and data are required when using scenarios="
            )
        Xb, Xgb, pb, _, _ = prepare_mdcev_forecast_data(
            model, data, None, None, budget_col
        )
        baseline = mdcev_predict(
            results, Xb, Xgb, pb, n_draws=n_draws, seed=seed,
        ).mean(axis=0)
        shares_per_scenario: dict[str, NDArray] = {}
        for name, overrides in _scenarios_to_dict(scenarios).items():
            cv = list(overrides.keys())
            cval = [float(v) for v in overrides.values()]
            Xs, Xgs, ps, _, _ = prepare_mdcev_forecast_data(
                model, data, cv, cval, budget_col
            )
            shares_per_scenario[name] = mdcev_predict(
                results, Xs, Xgs, ps, n_draws=n_draws, seed=seed,
            ).mean(axis=0)
        return MDCEVATEResult(
            n_obs=Xb.shape[0],
            predicted_shares=baseline,
            alternative_names=alternative_names,
            shares_per_scenario=shares_per_scenario,
        )

    # --- Legacy / single-override mode ---
    if X is None or X_gam is None or price is None:
        raise ValueError(
            "X, X_gam and price are required unless scenarios=/model=/data= "
            "are provided"
        )
    N = X.shape[0]

    predicted_shares = mdcev_predict(
        results, X, X_gam, price, n_draws=n_draws, seed=seed,
    ).mean(axis=0)

    base_shares      = None
    treatment_shares = None
    pct_ate          = None

    if changevar_idx is not None and base_val is not None and treatment_val is not None:
        alt_idx, var_idx = changevar_idx

        X_base = X.copy()
        X_base[:, alt_idx, var_idx] = base_val
        base_shares = mdcev_predict(
            results, X_base, X_gam, price, n_draws=n_draws, seed=seed,
        ).mean(axis=0)

        X_treat = X.copy()
        X_treat[:, alt_idx, var_idx] = treatment_val
        treatment_shares = mdcev_predict(
            results, X_treat, X_gam, price, n_draws=n_draws, seed=seed,
        ).mean(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct_ate = np.where(
                base_shares > 0,
                (treatment_shares - base_shares) / base_shares * 100.0,
                0.0,
            )

    return MDCEVATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
        base_shares=base_shares,
        treatment_shares=treatment_shares,
        pct_ate=pct_ate,
        alternative_names=alternative_names,
    )


def mdcev_ate_from_params(
    b_reported: NDArray,
    sigma: float,
    X: NDArray | None = None,
    X_gam: NDArray | None = None,
    price: NDArray | None = None,
    *,
    control=None,
    param_names: list[str] | None = None,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
    n_draws: int = 1000,
    seed: int = 1234,
    model=None,
    data: "pd.DataFrame | None" = None,
    scenarios: "ScenarioSpec | None" = None,
    budget_col: str = "tot",
) -> MDCEVATEResult:
    """Compute MDCEV ATE predictions directly from natural-space coefficients.

    Convenience wrapper mirroring :func:`morp_ate_from_params`: it builds a
    results object via :meth:`MDCEVResults.from_estimates` and dispatches to
    :func:`mdcev_ate`, so ATEs can be computed from manually entered (e.g.
    GAUSS) estimates without re-fitting.

    Parameters
    ----------
    b_reported : ndarray
        Reported coefficient vector ``[beta, gamma, sigma]`` in natural units
        (see :meth:`MDCEVResults.from_estimates`).
    sigma : float
        Scale parameter.
    X : ndarray, shape (N, nc, nvarm)
        Baseline utility design matrix.
    X_gam : ndarray, shape (N, nc, nvargam)
        Satiation utility design matrix.
    price : ndarray, shape (N, nc)
        Price matrix.
    control : MDCEVControl, optional
        Control structure (drives ``outside_good_gamma`` / utility type).
    param_names : list of str, optional
        Names aligned with ``b_reported``.
    changevar_idx, base_val, treatment_val, alternative_names, n_draws, seed
        Forwarded to :func:`mdcev_ate`.

    Returns
    -------
    MDCEVATEResult
    """
    results = MDCEVResults.from_estimates(
        b_reported, sigma, control=control, param_names=param_names,
    )
    return mdcev_ate(
        results, X, X_gam, price,
        changevar_idx=changevar_idx,
        base_val=base_val,
        treatment_val=treatment_val,
        alternative_names=alternative_names,
        n_draws=n_draws,
        seed=seed,
        model=model,
        data=data,
        scenarios=scenarios,
        budget_col=budget_col,
    )
