"""Average Treatment Effect (ATE) post-estimation for the MNL model.

Computes predicted market shares and percentage ATEs by comparing
base-level vs treatment-level covariate scenarios.

MNL-004 extends the API to accept a ``scenarios`` specification — a mapping
of scenario name to per-variable overrides — so that multiple counterfactuals
can be evaluated in a single call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models.mnl._mnl_results import MNLResults
from pybhatlib.models.mnl._mnl_forecast import mnl_predict

# Type alias: dict or DataFrame keyed/indexed by scenario name,
# values are variable overrides (column → scalar | column-name-string).
ScenarioSpec = Union[
    "dict[str, dict[str, float | str]]",
    "pd.DataFrame",
]


@dataclass
class MNLATEResult:
    """Average Treatment Effect analysis results for MNL.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_shares : NDArray, shape (nc,)
        Mean predicted probability for each alternative at observed values
        (unconditional on scenario; computed from unmodified data, or from
        the base data when the legacy ``changevar_idx``/``base_val`` path is
        used).
    base_shares : NDArray or None, shape (nc,)
        Predicted shares at the base level of the treatment variable
        (legacy path only).
    treatment_shares : NDArray or None, shape (nc,)
        Predicted shares at the treatment level (legacy path only).
    pct_ate : NDArray or None, shape (nc,)
        Percentage ATE = (treatment - base) / base * 100 (legacy path only).
    alternative_names : list[str] or None
        Names of the alternatives, in order.
    shares_per_scenario : dict[str, NDArray] or None
        Per-scenario mean predicted shares (scenarios path only).
        Keys are scenario names; values have shape ``(nc,)``.
    """

    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None = None
    treatment_shares: NDArray | None = None
    pct_ate: NDArray | None = None
    alternative_names: list[str] | None = None
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
        pct_change : NDArray, shape (nc,)
            ``100 * (treatment_shares − base_shares) / base_shares``

        Raises
        ------
        ValueError
            If ``shares_per_scenario`` is None or a scenario name is missing.
        """
        if self.shares_per_scenario is None:
            raise ValueError(
                "comparison() requires shares_per_scenario; run mnl_ate with scenarios=..."
            )
        if base not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{base}' not found in shares_per_scenario")
        if treatment not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{treatment}' not found in shares_per_scenario")

        b = self.shares_per_scenario[base]
        t = self.shares_per_scenario[treatment]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, 100.0 * (t - b) / b, np.nan)

    def summary(self) -> str:
        """Print a formatted ATE summary table.

        Returns
        -------
        text : str
        """
        lines = []
        sep = "=" * 65
        lines.append(sep)
        lines.append("  MNL Average Treatment Effect (ATE) Summary")
        lines.append(sep)
        lines.append(f"  N observations: {self.n_obs}")
        lines.append("")

        nc    = len(self.predicted_shares)
        names = self.alternative_names or [f"Alt {k}" for k in range(nc)]
        header = f"  {'Alternative':<16s} {'Pred. Share':>12s}"

        if self.base_shares is not None:
            header += f" {'Base':>10s} {'Treatment':>10s} {'%ATE':>10s}"
        lines.append(header)
        lines.append("  " + "-" * 60)

        for k in range(nc):
            row = f"  {names[k]:<16s} {self.predicted_shares[k]:>12.4f}"
            if self.base_shares is not None:
                ate = self.pct_ate[k] if self.pct_ate is not None else float("nan")
                row += (
                    f" {self.base_shares[k]:>10.4f}"
                    f" {self.treatment_shares[k]:>10.4f}"
                    f" {ate:>10.2f}"
                )
            lines.append(row)

        lines.append(sep)
        text = "\n".join(lines)
        print(text)
        return text


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


def mnl_ate(
    results: MNLResults,
    *,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    alternatives: list[str] | None = None,
    avail: NDArray | None = None,
    scenarios: "ScenarioSpec | None" = None,
    # Legacy path: direct design-matrix manipulation
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
    X: NDArray | None = None,
) -> MNLATEResult:
    """Compute predicted shares and ATE for the MNL model.

    For the MNL model the ATE is computed analytically (no simulation
    required) since probabilities are available in closed form.

    Supports three modes:

    **Scenario mode** (MNL-004):
        Pass ``scenarios`` as a dict or DataFrame along with ``data``,
        ``spec``, and ``alternatives``.  Each scenario specifies variable
        overrides applied to *data*; the function rebuilds the design matrix
        for each scenario via ``parse_spec`` and returns mean shares in
        ``MNLATEResult.shares_per_scenario``.  Use
        :meth:`MNLATEResult.comparison` to compute pairwise percentage
        changes between any two scenarios.

    **Legacy mode** (pre-MNL-004):
        Pass ``changevar_idx`` + ``base_val`` + ``treatment_val`` to
        replicate the old single base/treatment ATE.  ``base_shares``,
        ``treatment_shares``, and ``pct_ate`` are populated.

    **Plain mode**:
        Pass neither ``scenarios`` nor ``changevar_idx`` to obtain
        unconditional predicted shares only.

    Parameters
    ----------
    results : MNLResults
        Fitted MNL model results.
    data : pd.DataFrame, optional
        Dataset.  Required when ``X`` is not provided, or when
        ``scenarios`` is used.
    spec : dict, optional
        Variable specification mapping.  Required when ``X`` is not
        provided, or when ``scenarios`` is used.
    alternatives : list of str, optional
        Alternative column names.  Required when ``X`` is not provided,
        or when ``scenarios`` is used.
    avail : NDArray, shape (N, nc), optional
        Availability matrix.  Defaults to all ones.
    scenarios : dict or pd.DataFrame, optional
        Scenario specification.

        - dict form: ``{scenario_name: {col: scalar | col_name, ...}, ...}``
        - DataFrame form: rows are scenarios (index = scenario name),
          columns are variable names, values are scalars or column-name
          strings.

        String values are resolved as source column names from *data* —
        no label-mode.  Raises ``ValueError`` if the string is not a real
        column.

        Mutually exclusive with ``changevar_idx``/``base_val``/
        ``treatment_val``.
    changevar_idx : tuple (alt_idx, var_idx) or None
        Which element of X to modify: ``X[:, alt_idx, var_idx]``
        (legacy API).  Mutually exclusive with ``scenarios``.
    base_val : float or None
        Value to set ``changevar_idx`` to for the base scenario
        (legacy API).
    treatment_val : float or None
        Value to set ``changevar_idx`` to for the treatment scenario
        (legacy API).
    alternative_names : list[str] or None
        Names of the alternatives, in order.
    X : NDArray, shape (N, nc, numunord), optional
        Pre-built design matrix.  If provided, ``data`` / ``spec`` /
        ``alternatives`` are used only for scenario/legacy override
        rebuilds (not for the base computation).

    Returns
    -------
    MNLATEResult

    Raises
    ------
    ValueError
        If both ``scenarios`` and ``changevar_idx``/``base_val``/
        ``treatment_val`` are supplied.
    ValueError
        If a string scenario value is not a column in *data*.
    ValueError
        If ``X`` is None and ``data``/``spec``/``alternatives`` are
        missing.
    """
    # --- Guard: mutually exclusive ---
    if scenarios is not None and (
        changevar_idx is not None or base_val is not None or treatment_val is not None
    ):
        raise ValueError(
            "scenarios and changevar_idx/base_val/treatment_val are mutually exclusive. "
            "Use one or the other."
        )

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

    # --- Scenario mode (MNL-004) ---
    if scenarios is not None:
        scenario_dict = _scenarios_to_dict(scenarios)
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "data, spec, and alternatives are required when using scenarios"
            )

        # Compute baseline predicted shares (from unmodified data)
        baseline_shares = mnl_predict(results, X_base_np, avail).mean(axis=0)

        shares_per_scenario: dict[str, NDArray] = {}
        for scenario_name, overrides in scenario_dict.items():
            data_mod = _apply_scenario_overrides(data, overrides)
            X_mod, _ = parse_spec(spec, data_mod, alternatives, nseg=1)
            X_mod_np = np.asarray(X_mod, dtype=np.float64)

            shares_per_scenario[scenario_name] = (
                mnl_predict(results, X_mod_np, avail).mean(axis=0)
            )

        return MNLATEResult(
            n_obs=N,
            predicted_shares=baseline_shares,
            alternative_names=alternative_names,
            shares_per_scenario=shares_per_scenario,
        )

    # --- Legacy mode: changevar_idx / base_val / treatment_val ---
    if changevar_idx is not None and base_val is not None and treatment_val is not None:
        alt_idx, var_idx = changevar_idx

        X_base_mod = X_base_np.copy()
        X_base_mod[:, alt_idx, var_idx] = base_val
        base_shares = mnl_predict(results, X_base_mod, avail).mean(axis=0)

        X_treat = X_base_np.copy()
        X_treat[:, alt_idx, var_idx] = treatment_val
        treatment_shares = mnl_predict(results, X_treat, avail).mean(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct_ate = np.where(
                base_shares > 0,
                100.0 * (treatment_shares - base_shares) / base_shares,
                np.nan,
            )

        # ``predicted_shares`` reflects the unmodified-data shares, not
        # the treatment-data shares, to preserve historical semantics for
        # callers that read ``result.predicted_shares`` directly.
        return MNLATEResult(
            n_obs=N,
            predicted_shares=base_shares,
            base_shares=base_shares,
            treatment_shares=treatment_shares,
            pct_ate=pct_ate,
            alternative_names=alternative_names,
        )

    # --- No modification: plain predicted shares ---
    predicted_shares = mnl_predict(results, X_base_np, avail).mean(axis=0)

    return MNLATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
        alternative_names=alternative_names,
    )


def mnl_ate_from_params(
    beta: NDArray,
    *,
    param_names: list[str] | None = None,
    data: "pd.DataFrame | None" = None,
    spec: dict | None = None,
    alternatives: list[str] | None = None,
    avail: NDArray | None = None,
    scenarios: "ScenarioSpec | None" = None,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
    X: NDArray | None = None,
) -> MNLATEResult:
    """Compute MNL ATE predictions directly from natural-space coefficients.

    Convenience wrapper mirroring :func:`mnp_ate_from_params`: it builds a
    results object via :meth:`MNLResults.from_estimates` and dispatches to
    :func:`mnl_ate`, so ATEs can be computed from manually entered (e.g. GAUSS)
    estimates without re-fitting.

    Unlike the MNP equivalent there is no IID/kernel-covariance distinction —
    the MNL log-likelihood has no error covariance structure, so only the slope
    coefficients are required.

    Parameters
    ----------
    beta : ndarray, shape (n_beta,)
        Slope coefficients in natural (non-transformed) space.
    param_names : list[str] or None
        Forwarded to :meth:`MNLResults.from_estimates`.
    data, spec, alternatives, avail, scenarios, changevar_idx, base_val,
    treatment_val, alternative_names, X
        Forwarded to :func:`mnl_ate` (same counterfactual API).

    Returns
    -------
    MNLATEResult
    """
    results = MNLResults.from_estimates(
        beta,
        param_names=param_names,
    )
    return mnl_ate(
        results,
        data=data,
        spec=spec,
        alternatives=alternatives,
        avail=avail,
        scenarios=scenarios,
        changevar_idx=changevar_idx,
        base_val=base_val,
        treatment_val=treatment_val,
        alternative_names=alternative_names,
        X=X,
    )
