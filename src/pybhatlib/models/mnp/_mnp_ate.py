"""Average Treatment Effect (ATE) post-estimation for MNP.

Implements the mnpATEFit procedure from BHATLIB for computing predicted
mode shares and Average Treatment Effects.

MNP-004 extends the API to accept a ``scenarios`` specification — a mapping
of scenario name to per-variable overrides — so that multiple counterfactuals
can be evaluated in a single call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models.mnp._mnp_loglik import (
    _build_lambda,
    _build_omega_cholesky,
    _compute_choice_prob,
    _compute_mixture_prob,
    _unpack_params,
)
from pybhatlib.models.mnp._mnp_results import MNPResults

# Type alias: dict or DataFrame keyed/indexed by scenario name,
# values are variable overrides (column → scalar | column-name-string).
ScenarioSpec = Union[
    "dict[str, dict[str, float | str]]",
    "pd.DataFrame",
]


@dataclass
class MNPATEResult:
    """Average Treatment Effect analysis results.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_shares : NDArray
        Mean predicted probability for each alternative (unconditional on
        scenario; computed from unmodified data, or from the treatment data
        when the legacy ``changevar``/``changeval`` path is used).
    base_shares : NDArray or None
        Predicted shares at base level (legacy path only).
    treatment_shares : NDArray or None
        Predicted shares at treatment level (legacy path only).
    pct_ate : NDArray or None
        Percentage ATE = (treatment − base) / base × 100 (legacy path only).
    shares_per_scenario : dict[str, NDArray] or None
        Per-scenario mean predicted shares (scenarios path only).
        Keys are scenario names; values have shape ``(n_alts,)``.
    """

    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None = None
    treatment_shares: NDArray | None = None
    pct_ate: NDArray | None = None
    shares_per_scenario: dict[str, NDArray] | None = None

    def comparison(self, base: str, treatment: str) -> NDArray:
        """Compute percentage change between two scenarios.

        Parameters
        ----------
        base : str
            Scenario name to use as the denominator (base case).
        treatment : str
            Scenario name to use as the numerator (treatment case).

        Returns
        -------
        pct_change : NDArray, shape (n_alts,)
            ``100 * (treatment_shares − base_shares) / base_shares``

        Raises
        ------
        ValueError
            If ``shares_per_scenario`` is None or a scenario name is missing.
        """
        if self.shares_per_scenario is None:
            raise ValueError(
                "comparison() requires shares_per_scenario; run mnp_ate with scenarios=..."
            )
        if base not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{base}' not found in shares_per_scenario")
        if treatment not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{treatment}' not found in shares_per_scenario")

        b = self.shares_per_scenario[base]
        t = self.shares_per_scenario[treatment]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, 100.0 * (t - b) / b, np.nan)


# Backwards-compatible alias.  The class was historically named ``ATEResult``
# (unprefixed, unlike MNLATEResult / MDCEVATEResult / MORPATEResult); it is now
# ``MNPATEResult`` for naming parity.  ``ATEResult`` remains importable.
ATEResult = MNPATEResult


def _apply_scenario_overrides(
    data: pd.DataFrame,
    overrides: dict[str, float | str],
) -> pd.DataFrame:
    """Apply a single scenario's overrides to a copy of *data*.

    Parameters
    ----------
    data : pd.DataFrame
        Original (unmodified) dataset.
    overrides : dict
        Mapping of ``{column_name: scalar_value | source_column_name}``.
        - scalar: broadcast the value to all rows of that column.
        - string: resolve as ``data[that_string]`` (identity or remap).
          Raises ``ValueError`` if the string is not a real column.

    Returns
    -------
    data_mod : pd.DataFrame
        Deep copy with overrides applied.

    Raises
    ------
    ValueError
        If a string override value is not a column in *data*.
    """
    data_mod = data.copy()
    for col, val in overrides.items():
        # Validate the override target column exists (PR-review P1).  Without
        # this guard a typo (e.g. ``{"AGEE45": 0}``) silently adds a new
        # column to ``data_mod`` and ``parse_spec`` keeps the original column
        # — producing baseline shares with no warning.
        if col not in data.columns:
            raise ValueError(
                f"Scenario override target '{col}' is not a column in data. "
                f"Available columns: {sorted(data.columns)[:20]}..."
                if len(data.columns) > 20
                else f"Available columns: {list(data.columns)}"
            )
        if isinstance(val, str):
            # String value is interpreted as a source column name — no label mode.
            if val not in data.columns:
                raise ValueError(
                    f"Scenario override for '{col}' references column '{val}', "
                    f"which is not present in data. Only real column names are "
                    f"accepted as string values (no label-mode)."
                )
            data_mod[col] = data[val].values
        else:
            data_mod[col] = float(val)
    return data_mod


def _scenarios_to_dict(
    scenarios: ScenarioSpec,
) -> dict[str, dict[str, float | str]]:
    """Normalise *scenarios* to canonical dict form.

    Parameters
    ----------
    scenarios : dict or pd.DataFrame
        - dict: ``{scenario_name: {col: val, ...}, ...}``
        - DataFrame: rows = scenarios (index = name), columns = variables,
          values = scalar or column-name string.

    Returns
    -------
    out : dict[str, dict[str, float | str]]
    """
    if isinstance(scenarios, pd.DataFrame):
        out: dict[str, dict[str, float | str]] = {}
        for name in scenarios.index:
            row = scenarios.loc[name]
            out[str(name)] = {col: row[col] for col in scenarios.columns}
        return out
    return dict(scenarios)


def _compute_predicted_shares(
    X_np: NDArray,
    N: int,
    I: int,
    n_vars: int,
    theta_hat: NDArray,
    control,
    ranvar_indices: list[int] | None,
    avail: NDArray | None,
    xp,
) -> NDArray:
    """Compute per-alternative mean predicted shares for one design matrix.

    Parameters
    ----------
    X_np : ndarray, shape (N, I, n_vars)
        Design matrix.
    N : int
        Number of observations.
    I : int
        Number of alternatives.
    n_vars : int
        Number of variables.
    theta_hat : ndarray
        Fitted parameter vector.
    control : MNPControl
        Model control structure.
    ranvar_indices : list[int] or None
        Indices of random coefficients in beta.
    avail : ndarray or None
        Availability matrix, shape (N, I).
    xp : backend
        Array backend (numpy).

    Returns
    -------
    shares : ndarray, shape (I,)
        Mean predicted probabilities, normalised to sum to 1.
    """
    params = _unpack_params(theta_hat, n_vars, I, control, ranvar_indices)
    beta = params["beta"]
    Lambda = _build_lambda(params.get("lambda_params"), I, control)

    Omega_L = None
    if control is not None and control.mix and params.get("omega_params") is not None:
        Omega_L = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    use_mixture = (control is not None and control.nseg > 1)

    pred_probs = np.zeros((N, I), dtype=np.float64)

    for q in range(N):
        avail_q = avail[q] if avail is not None else np.ones(I)

        for i in range(I):
            if avail_q[i] < 0.5:
                continue

            if use_mixture:
                pred_probs[q, i] = _compute_mixture_prob(
                    X_np[q], params, i, avail_q, Lambda,
                    ranvar_indices, control, xp,
                )
            else:
                pred_probs[q, i] = _compute_choice_prob(
                    X_np[q], beta, i, avail_q, Lambda, Omega_L,
                    ranvar_indices, control, xp,
                )

        row_sum = pred_probs[q].sum()
        if row_sum > 0:
            pred_probs[q] /= row_sum

    return pred_probs.mean(axis=0)


def mnp_ate(
    results: MNPResults,
    *,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    alternatives: list[str] | None = None,
    avail: NDArray | None = None,
    scenarios: "ScenarioSpec | None" = None,
    # Back-compat shim
    changevar: str | None = None,
    changeval: float | None = None,
    X: NDArray | None = None,
) -> MNPATEResult:
    """Compute predicted shares and ATE for an MNP model.

    Supports two modes:

    **Scenario mode** (MNP-004):
        Pass ``scenarios`` as a dict or DataFrame.  Each scenario specifies
        variable overrides; the function rebuilds the design matrix for each
        scenario, computes per-observation probabilities via
        ``_compute_choice_prob`` / ``_compute_mixture_prob``, and returns
        mean shares in ``MNPATEResult.shares_per_scenario``.

    **Legacy mode** (pre-MNP-004):
        Pass ``changevar`` + ``changeval`` to replicate the old single
        base/treatment ATE.  ``base_shares``, ``treatment_shares``, and
        ``pct_ate`` are populated.

    Parameters
    ----------
    results : MNPResults
        Fitted MNP model results.
    data : pd.DataFrame, optional
        Dataset.  Required when ``X`` is not provided.
    spec : dict, optional
        Variable specification mapping.  Required when ``X`` is not provided.
    alternatives : list of str, optional
        Alternative column names.  Required when ``X`` is not provided.
    avail : ndarray, shape (N, n_alts), optional
        Availability matrix.
    scenarios : dict or pd.DataFrame, optional
        Scenario specification.

        - dict form: ``{scenario_name: {col: scalar | col_name, ...}, ...}``
        - DataFrame form: rows are scenarios (index = scenario name),
          columns are variable names, values are scalars or column-name
          strings.

        String values are resolved as source column names from *data* —
        no label-mode.  Raises ``ValueError`` if the string is not a real
        column.

        Mutually exclusive with ``changevar``/``changeval``.
    changevar : str, optional
        Variable name to override (legacy API).
    changeval : float, optional
        Value for override (legacy API).
    X : ndarray, shape (N, n_alts, n_vars), optional
        Pre-built design matrix.  If provided, ``data`` / ``spec`` /
        ``alternatives`` are ignored for the base computation (but still
        required for scenario/legacy override rebuilds).

    Returns
    -------
    result : MNPATEResult

    Raises
    ------
    ValueError
        If both ``scenarios`` and ``changevar``/``changeval`` are supplied.
    ValueError
        If a string scenario value is not a column in *data*.
    ValueError
        If ``X`` is None and ``data``/``spec``/``alternatives`` are missing.
    """
    # --- Guard: mutually exclusive ---
    if scenarios is not None and (changevar is not None or changeval is not None):
        raise ValueError(
            "scenarios and changevar/changeval are mutually exclusive. "
            "Use one or the other."
        )

    xp = get_backend("numpy")
    theta_hat = np.asarray(results.params, dtype=np.float64)
    control = results.control
    ranvar_indices = getattr(results, "ranvar_indices", None)

    # --- Build base design matrix ---
    if X is None:
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "Either X must be provided, or data+spec+alternatives for reconstruction"
            )
        X_base, _ = parse_spec(spec, data, alternatives, nseg=1)
    else:
        X_base = X

    X_base_np = np.asarray(X_base, dtype=np.float64)
    N = X_base_np.shape[0]
    I = X_base_np.shape[1]
    n_vars = X_base_np.shape[2]

    # --- Scenario mode (MNP-004) ---
    if scenarios is not None:
        scenario_dict = _scenarios_to_dict(scenarios)
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "data, spec, and alternatives are required when using scenarios"
            )

        # Compute baseline predicted shares (from unmodified data)
        baseline_shares = _compute_predicted_shares(
            X_base_np, N, I, n_vars, theta_hat, control, ranvar_indices, avail, xp,
        )

        shares_per_scenario: dict[str, NDArray] = {}
        for scenario_name, overrides in scenario_dict.items():
            data_mod = _apply_scenario_overrides(data, overrides)
            X_mod, _ = parse_spec(spec, data_mod, alternatives, nseg=1)
            X_mod_np = np.asarray(X_mod, dtype=np.float64)

            scenario_shares = _compute_predicted_shares(
                X_mod_np, N, I, n_vars, theta_hat, control, ranvar_indices, avail, xp,
            )
            shares_per_scenario[scenario_name] = scenario_shares

        return MNPATEResult(
            n_obs=N,
            predicted_shares=baseline_shares,
            shares_per_scenario=shares_per_scenario,
        )

    # --- Legacy mode: changevar / changeval ---
    if changevar is not None and changeval is not None:
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "data, spec, and alternatives are required for changevar/changeval ATE"
            )

        # Base shares: from unmodified data
        base_shares = _compute_predicted_shares(
            X_base_np, N, I, n_vars, theta_hat, control, ranvar_indices, avail, xp,
        )

        # Treatment shares: override variable
        data_modified = data.copy()
        if changevar in data_modified.columns:
            data_modified[changevar] = changeval
        X_treat, _ = parse_spec(spec, data_modified, alternatives, nseg=1)
        X_treat_np = np.asarray(X_treat, dtype=np.float64)

        treatment_shares = _compute_predicted_shares(
            X_treat_np, N, I, n_vars, theta_hat, control, ranvar_indices, avail, xp,
        )

        # Percentage ATE: avoid div-by-zero
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_ate = np.where(
                base_shares > 0,
                100.0 * (treatment_shares - base_shares) / base_shares,
                np.nan,
            )

        # ``predicted_shares`` historically reflected the unconditional /
        # unmodified-data shares, NOT the treatment-data shares.  Aliasing it
        # to ``treatment_shares`` would silently change semantics for callers
        # that read ``result.predicted_shares`` (PR-review P1).
        return MNPATEResult(
            n_obs=N,
            predicted_shares=base_shares,
            base_shares=base_shares,
            treatment_shares=treatment_shares,
            pct_ate=pct_ate,
        )

    # --- No modification: plain predicted shares ---
    predicted_shares = _compute_predicted_shares(
        X_base_np, N, I, n_vars, theta_hat, control, ranvar_indices, avail, xp,
    )

    return MNPATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
    )


def mnp_ate_from_params(
    beta: NDArray,
    *,
    kernel_cov: NDArray | None = None,
    control=None,
    ranvar_indices: list[int] | None = None,
    param_names: list[str] | None = None,
    n_alts: int | None = None,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    alternatives: list[str] | None = None,
    avail: NDArray | None = None,
    scenarios: "ScenarioSpec | None" = None,
    changevar: str | None = None,
    changeval: float | None = None,
    X: NDArray | None = None,
) -> MNPATEResult:
    """Compute MNP ATE predictions from the model's *reported* coefficients.

    Convenience wrapper mirroring :func:`morp_ate_from_params`: it builds a
    results object via :meth:`MNPResults.from_estimates` and dispatches to
    :func:`mnp_ate`, so ATEs can be computed from manually entered estimates —
    the coefficients exactly as the model reports them (unparameterized,
    ``sum(scale**2)=1``) — without re-fitting (issue #44).

    Parameters
    ----------
    beta : ndarray, shape (n_beta,)
        Reported slope coefficients (``results.b_original`` slope rows).
    kernel_cov : ndarray, shape (I-1, I-1), optional
        The **reported** differenced kernel covariance, built from the reported
        ``scale*``/``parker*`` rows as ``diag(scales) @ corr @ diag(scales)``
        (``None`` ⇒ IID).  See :meth:`MNPResults.from_estimates` for the exact
        reconstruction.  ``n_alts`` is required for IID.
    control : MNPControl, optional
        Control structure (defaults to ``MNPControl(iid=kernel_cov is None)``).
    ranvar_indices, param_names, n_alts
        Forwarded to :meth:`MNPResults.from_estimates`.
    data, spec, alternatives, avail, scenarios, changevar, changeval, X
        Forwarded to :func:`mnp_ate` (same counterfactual API).

    Returns
    -------
    MNPATEResult
    """
    results = MNPResults.from_estimates(
        beta,
        kernel_cov=kernel_cov,
        control=control,
        ranvar_indices=ranvar_indices,
        param_names=param_names,
        n_alts=n_alts,
    )
    return mnp_ate(
        results,
        data=data,
        spec=spec,
        alternatives=alternatives,
        avail=avail,
        scenarios=scenarios,
        changevar=changevar,
        changeval=changeval,
        X=X,
    )
