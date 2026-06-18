"""Average Treatment Effect (ATE) post-estimation for the MNL model.

Computes predicted market shares and percentage ATEs by comparing
base-level vs treatment-level covariate scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models.mnl._mnl_results import MNLResults
from pybhatlib.models.mnl._mnl_forecast import mnl_predict


@dataclass
class MNLATEResult:
    """Average Treatment Effect analysis results for MNL.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_shares : NDArray, shape (nc,)
        Mean predicted probability for each alternative at observed values.
    base_shares : NDArray or None, shape (nc,)
        Predicted shares at the base level of the treatment variable.
    treatment_shares : NDArray or None, shape (nc,)
        Predicted shares at the treatment level.
    pct_ate : NDArray or None, shape (nc,)
        Percentage ATE = (treatment - base) / base * 100.
    alternative_names : list[str] or None
        Names of the alternatives, in order.
    """

    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None = None
    treatment_shares: NDArray | None = None
    pct_ate: NDArray | None = None
    alternative_names: list[str] | None = None

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


def mnl_ate(
    results: MNLResults,
    X: NDArray,
    avail: NDArray | None = None,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
) -> MNLATEResult:
    """Compute predicted shares and ATE for the MNL model.

    For the MNL model the ATE is computed analytically (no simulation
    required) since probabilities are available in closed form.

    When ``changevar_idx`` and both value arguments are provided, the
    specified element of X is set to ``base_val`` and ``treatment_val``
    for all observations, and predicted shares are compared.  When
    ``changevar_idx`` is None, only overall predicted shares at the
    observed data values are returned.

    Parameters
    ----------
    results : MNLResults
        Fitted MNL model results.
    X : NDArray, shape (N, nc, numunord)
        Design matrix.
    avail : NDArray, shape (N, nc), optional
        Availability matrix.  Defaults to all ones.
    changevar_idx : tuple (alt_idx, var_idx) or None
        Which element of X to modify: ``X[:, alt_idx, var_idx]``.
        Pass None to skip ATE computation.
    base_val : float or None
        Value to set ``changevar_idx`` to for the base scenario.
    treatment_val : float or None
        Value to set ``changevar_idx`` to for the treatment scenario.
    alternative_names : list[str] or None
        Names of the alternatives.

    Returns
    -------
    MNLATEResult
    """
    N = X.shape[0]

    # Overall predicted shares at observed values
    predicted_shares = mnl_predict(results, X, avail).mean(axis=0)

    base_shares      = None
    treatment_shares = None
    pct_ate          = None

    if changevar_idx is not None and base_val is not None and treatment_val is not None:
        alt_idx, var_idx = changevar_idx

        X_base = X.copy()
        X_base[:, alt_idx, var_idx] = base_val
        base_shares = mnl_predict(results, X_base, avail).mean(axis=0)

        X_treat = X.copy()
        X_treat[:, alt_idx, var_idx] = treatment_val
        treatment_shares = mnl_predict(results, X_treat, avail).mean(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            pct_ate = np.where(
                base_shares > 0,
                (treatment_shares - base_shares) / base_shares * 100.0,
                0.0,
            )

    return MNLATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
        base_shares=base_shares,
        treatment_shares=treatment_shares,
        pct_ate=pct_ate,
        alternative_names=alternative_names,
    )


def mnl_ate_from_params(
    beta: NDArray,
    X: NDArray,
    avail: NDArray | None = None,
    *,
    param_names: list[str] | None = None,
    control=None,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
) -> MNLATEResult:
    """Compute MNL ATE predictions directly from reported coefficients.

    Convenience wrapper mirroring :func:`morp_ate_from_params`: it builds a
    results object via :meth:`MNLResults.from_estimates` and dispatches to
    :func:`mnl_ate`, so ATEs can be computed from manually entered (e.g. GAUSS)
    estimates without re-fitting.

    Parameters
    ----------
    beta : ndarray, shape (numunord,)
        Estimated MNL coefficients.
    X : ndarray, shape (N, nc, numunord)
        Design matrix.
    avail : ndarray, shape (N, nc), optional
        Availability matrix.
    param_names : list of str, optional
        Names aligned with ``beta``.
    control : MNLControl, optional
        Control structure to carry along.
    changevar_idx, base_val, treatment_val, alternative_names
        Forwarded to :func:`mnl_ate`.

    Returns
    -------
    MNLATEResult
    """
    results = MNLResults.from_estimates(
        beta, param_names=param_names, control=control,
    )
    return mnl_ate(
        results, X, avail,
        changevar_idx=changevar_idx,
        base_val=base_val,
        treatment_val=treatment_val,
        alternative_names=alternative_names,
    )
