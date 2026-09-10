"""Analytical scenario effects on a continuous outcome's predicted mean.

``LRATEResult`` deliberately does **not** use :class:`ATEResultMixin`: the
outcome is a continuous mean in outcome units rather than a vector of
alternative shares, so ``comparison()`` returns a scalar in outcome units
(percentage change on request) and the per-scenario field is
``means_per_scenario``.
The scenario grammar itself (``scenarios_to_dict`` /
``apply_scenario_overrides``) is shared with the other models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pybhatlib.models._ate_common import (
    ScenarioSpec,
    apply_scenario_overrides,
    scenarios_to_dict,
)
from pybhatlib.models.lr._lr_control import LRControl
from pybhatlib.models.lr._lr_forecast import lr_predict
from pybhatlib.models.lr._lr_model import _build_design
from pybhatlib.models.lr._lr_results import LRResults


@dataclass
class LRATEResult:
    """Predicted outcome means, in outcome units (not probability shares).

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_mean : float
        Mean predicted outcome at observed covariate values.
    means_per_scenario : dict[str, float] or None
        Mean predicted outcome under each named scenario (``scenarios=`` path
        only).
    dep_var : str or None
        Outcome column name, for labelling.
    """

    n_obs: int
    predicted_mean: float
    means_per_scenario: dict[str, float] | None = None
    dep_var: str | None = None

    def comparison(self, base: str, treatment: str, *, percent: bool = False) -> float:
        """Change in the mean predicted outcome between two scenarios.

        Parameters
        ----------
        base : str
            Scenario name used as the reference.
        treatment : str
            Scenario name compared against *base*.
        percent : bool
            ``False`` (default) returns the effect in outcome units, which for
            a linear model equals the coefficient times the covariate change.
            ``True`` returns the percentage change
            ``100 * (treatment - base) / base`` of the predicted mean (the
            share-model convention) using the signed baseline, NaN for a zero
            baseline.  For a log outcome this is *not* the percentage change
            of the underlying quantity; use ``exp(effect) - 1``.

        Returns
        -------
        float

        Raises
        ------
        ValueError
            If ``means_per_scenario`` is None or a scenario name is unknown.
        """
        if self.means_per_scenario is None:
            raise ValueError("comparison() requires lr_ate with scenarios")
        if base not in self.means_per_scenario or treatment not in self.means_per_scenario:
            raise ValueError("Unknown base or treatment scenario")
        b, v = self.means_per_scenario[base], self.means_per_scenario[treatment]
        return (100 * (v - b) / b if b != 0 else np.nan) if percent else v - b

    def to_dataframe(self) -> pd.DataFrame:
        """Scenario means and absolute changes from the observed prediction.

        Returns
        -------
        df : pd.DataFrame
            Index = scenario name (``"observed"`` when no scenarios were
            given); columns ``Predicted Mean``, ``Change from Observed``.
        """
        means = self.means_per_scenario
        if means is None:
            means = {"observed": self.predicted_mean}
        frame = pd.DataFrame.from_dict(means, orient="index", columns=["Predicted Mean"])
        frame["Change from Observed"] = frame["Predicted Mean"] - self.predicted_mean
        return frame

    def summary(self) -> str:
        """Print a formatted ATE summary table.

        Returns
        -------
        text : str
        """
        lines = []
        sep = "=" * 65
        lines.append(sep)
        lines.append("  pybhatlib LR Average Treatment Effect (ATE) Summary")
        lines.append(sep)
        lines.append(f"  N observations: {self.n_obs}")
        if self.dep_var is not None:
            lines.append(f"  Outcome: {self.dep_var}")
        lines.append(f"  Observed predicted mean: {self.predicted_mean:>12.4f}")
        lines.append("")

        header = f"  {'Scenario':<16s} {'Pred. Mean':>12s} {'Change':>12s}"
        lines.append(header)
        lines.append("  " + "-" * 44)

        means = self.means_per_scenario or {"observed": self.predicted_mean}
        for name, mean in means.items():
            lines.append(
                f"  {name:<16s} {mean:>12.4f} {mean - self.predicted_mean:>12.4f}"
            )

        lines.append(sep)
        text = "\n".join(lines)
        print(text)
        return text


def lr_ate(
    results: LRResults,
    *,
    data: pd.DataFrame | None = None,
    spec: dict | None = None,
    dep_var: str | None = None,
    scenarios: ScenarioSpec | None = None,
    X: NDArray | None = None,
) -> LRATEResult:
    """Mean predicted outcome and optional named counterfactuals, analytically.

    Parameters
    ----------
    results : LRResults
        Fitted results, or :meth:`LRResults.from_estimates` output.
    data : pd.DataFrame, optional
        Dataset.  Required when ``X`` is not provided or ``scenarios`` is used.
    spec : dict, optional
        Coefficient specification (as passed to :class:`LRModel`).  Required
        with ``data``.
    dep_var : str, optional
        Outcome column name.  Required with ``data``.
    scenarios : dict or pd.DataFrame, optional
        Shared scenario grammar: ``{name: {column: scalar | source_column}}``
        or a DataFrame with one row per scenario.  Overrides are applied to a
        copy of *data* and the design is rebuilt through *spec*.
    X : ndarray, shape (N, K), optional
        Pre-built design matrix for the baseline prediction.  Regressor order
        must match the coefficients ``results.params[:-1]``.

    Returns
    -------
    LRATEResult

    Raises
    ------
    ValueError
        If neither ``X`` nor ``data`` / ``spec`` / ``dep_var`` are given, if
        ``scenarios`` is used without them, or if the scenario data row count
        differs from the baseline design.
    """
    if X is None or scenarios is not None:
        if data is None or spec is None or dep_var is None:
            raise ValueError("data, spec, and dep_var are required for design reconstruction")
    if X is None:
        X, _ = _build_design(data, spec, dep_var)
    predicted = lr_predict(results, X)
    if len(predicted) == 0:
        raise ValueError("ATE requires at least one observation")
    means = None
    if scenarios is not None:
        if len(data) != len(predicted):
            raise ValueError("X and scenario data must have the same number of rows")
        means = {}
        for name, overrides in scenarios_to_dict(scenarios).items():
            modified = apply_scenario_overrides(data, overrides)
            design, _ = _build_design(modified, spec, dep_var)
            means[name] = float(lr_predict(results, design).mean())
    return LRATEResult(len(predicted), float(predicted.mean()), means, dep_var)


def lr_ate_from_params(
    b_reported: ArrayLike,
    *,
    sigma: float | None = None,
    param_names: list[str] | None = None,
    control: LRControl | None = None,
    **kwargs,
) -> LRATEResult:
    """Scenario analysis from externally supplied estimates.

    Convenience wrapper mirroring :func:`mdcev_ate_from_params`: builds a
    results object via :meth:`LRResults.from_estimates` and dispatches to
    :func:`lr_ate`.

    Parameters
    ----------
    b_reported : array_like, shape (K + 1,)
        Reported vector ``[beta..., sigma]`` in the order of the spec, exactly
        as printed by ``summary()`` (``sigma`` does not affect the ATE).
    sigma : float or None
        Forwarded to :meth:`LRResults.from_estimates` (overrides the trailing
        slot).
    param_names : list[str] or None
        Forwarded to :meth:`LRResults.from_estimates`.
    control : LRControl or None
        Forwarded to :meth:`LRResults.from_estimates`.
    **kwargs
        ``data`` / ``spec`` / ``dep_var`` / ``scenarios`` / ``X``, forwarded
        to :func:`lr_ate`.

    Returns
    -------
    LRATEResult
    """
    return lr_ate(LRResults.from_estimates(
        b_reported, sigma, param_names=param_names, control=control,
    ), **kwargs)
