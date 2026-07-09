"""Shared post-estimation ATE utilities.

MNP, MNL, and MDCEV independently grew near-identical scenario machinery and
ATE-result presentation helpers.  This module is the single home for that
shared surface so the per-model ``_*_ate.py`` files import rather than
duplicate it:

* :data:`ScenarioSpec` — the counterfactual specification type,
* :func:`apply_scenario_overrides` / :func:`scenarios_to_dict` — the
  scenario-override application and normalisation functions,
* :class:`ATEResultMixin` — the common ``.comparison()`` and ``.summary()``
  methods for the per-model ``*ATEResult`` dataclasses.

The per-outcome ordinal MORP result (``MORPATEResult``) has a different shape
(a list of per-dimension probability vectors) and deliberately does **not**
use this mixin.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Type alias: dict or DataFrame keyed/indexed by scenario name,
# values are variable overrides (column → scalar | column-name-string).
ScenarioSpec = Union[
    "dict[str, dict[str, float | str]]",
    "pd.DataFrame",
]


def apply_scenario_overrides(
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
        # Validate the override target column exists.  Without this guard a
        # typo (e.g. ``{"AGEE45": 0}``) silently adds a new column to
        # ``data_mod`` and ``parse_spec`` keeps the original column — producing
        # baseline shares with no warning.
        if col not in data.columns:
            # Parenthesise the conditional: implicit string concatenation binds
            # tighter than ``if/else``, so without it the prefix naming the bad
            # column is dropped whenever data has <= 20 columns.
            raise ValueError(
                f"Scenario override target '{col}' is not a column in data. "
                + (
                    f"Available columns: {sorted(data.columns)[:20]}..."
                    if len(data.columns) > 20
                    else f"Available columns: {list(data.columns)}"
                )
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


def scenarios_to_dict(
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


class ATEComparisonMixin:
    """Supplies ``.comparison()`` to scenario-capable ``*ATEResult`` classes.

    The host dataclass must carry a ``shares_per_scenario`` mapping (populated
    by the ``scenarios=`` path).  ``_ate_func_name`` is a plain (unannotated)
    class attribute — so ``@dataclass`` never mistakes it for a field — that
    each subclass overrides with its own function name for the error text.
    """

    # Deliberately unannotated: keeps @dataclass from treating it as a field.
    _ate_func_name = "the ate function"

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
        pct_change : NDArray
            ``100 * (treatment_shares − base_shares) / base_shares``

        Raises
        ------
        ValueError
            If ``shares_per_scenario`` is None or a scenario name is missing.
        """
        if self.shares_per_scenario is None:
            raise ValueError(
                f"comparison() requires shares_per_scenario; run "
                f"{self._ate_func_name} with scenarios=..."
            )
        if base not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{base}' not found in shares_per_scenario")
        if treatment not in self.shares_per_scenario:
            raise ValueError(f"Scenario '{treatment}' not found in shares_per_scenario")

        b = self.shares_per_scenario[base]
        t = self.shares_per_scenario[treatment]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(b > 0, 100.0 * (t - b) / b, np.nan)


class ATESummaryMixin:
    """Supplies ``.summary()`` to share-based ``*ATEResult`` classes.

    Host dataclasses provide ``n_obs`` / ``predicted_shares`` and, for the
    base/treatment path, ``base_shares`` / ``treatment_shares`` / ``pct_ate``.
    ``alternative_names`` is optional (defaults to ``Alt k``).  ``_model_label``
    is a plain (unannotated) class attribute overridden per model.
    """

    # Deliberately unannotated: keeps @dataclass from treating it as a field.
    _model_label = "Model"

    def summary(self) -> str:
        """Print a formatted ATE summary table.

        Returns
        -------
        text : str
        """
        lines = []
        sep = "=" * 65
        lines.append(sep)
        lines.append(f"  {self._model_label} Average Treatment Effect (ATE) Summary")
        lines.append(sep)
        lines.append(f"  N observations: {self.n_obs}")
        lines.append("")

        nc    = len(self.predicted_shares)
        names = getattr(self, "alternative_names", None) or [
            f"Alt {k}" for k in range(nc)
        ]
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


class ATEResultMixin(ATEComparisonMixin, ATESummaryMixin):
    """Both ``.comparison()`` and ``.summary()`` — for scenario-capable results.

    Used by ``MNPATEResult`` / ``MNLATEResult``, which expose the full
    ``scenarios=`` counterfactual surface.  Results without a
    ``shares_per_scenario`` field (e.g. ``MDCEVATEResult``) should inherit only
    :class:`ATESummaryMixin`.
    """
