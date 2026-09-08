"""Analytical scenario effects on a continuous outcome's predicted mean."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pybhatlib.models._ate_common import apply_scenario_overrides, scenarios_to_dict
from ._lr_forecast import lr_predict
from ._lr_model import _build_design
from ._lr_results import LRResults


@dataclass
class LRATEResult:
    """Predicted means, in outcome units (not probability shares)."""

    n_obs: int
    predicted_mean: float
    means_per_scenario: dict[str, float] | None = None
    dep_var: str | None = None

    def comparison(self, base, treatment, *, percent=True):
        """Compare scenarios; percent defaults to True as in other models.

        Use ``percent=False`` for an effect in outcome units. Percentage change
        uses the signed baseline denominator and is NaN for a zero baseline.
        """
        if self.means_per_scenario is None:
            raise ValueError("comparison() requires lr_ate with scenarios")
        if base not in self.means_per_scenario or treatment not in self.means_per_scenario:
            raise ValueError("Unknown base or treatment scenario")
        b, v = self.means_per_scenario[base], self.means_per_scenario[treatment]
        return (100 * (v - b) / b if b != 0 else np.nan) if percent else v - b

    def to_dataframe(self):
        """Return scenario means and absolute changes from observed predictions."""
        means = self.means_per_scenario
        if means is None:
            means = {"observed": self.predicted_mean}
        frame = pd.DataFrame.from_dict(means, orient="index", columns=["Predicted Mean"])
        frame["Change from Observed"] = frame["Predicted Mean"] - self.predicted_mean
        return frame

    def summary(self):
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


def lr_ate(results, *, data=None, spec=None, dep_var=None, scenarios=None, X=None):
    """Compute mean forecasts and optional named counterfactuals analytically.

    Accepts an (N, K) ``X`` or ``data``/``spec``/``dep_var``. Scenarios require
    the latter and follow the shared dict or DataFrame convention: override
    data columns with scalar values or values from another named column.
    Data are copied before overrides. Regressor order must match coefficients.
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


def lr_ate_from_params(beta, *, param_names=None, control=None, **kwargs):
    """Run scenario analysis from externally supplied regression coefficients."""
    return lr_ate(LRResults.from_estimates(
        beta, param_names=param_names, control=control,
    ), **kwargs)
