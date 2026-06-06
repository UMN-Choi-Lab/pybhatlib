"""Average Treatment Effect (ATE) post-estimation for the MDCEV model.

Computes predicted consumption shares and percentage ATEs by comparing
base-level vs treatment-level scenarios, following the same pattern used
for MNP and MNL in pybhatlib.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.models.mdcev._mdcev_results import MDCEVResults
from pybhatlib.models.mdcev._mdcev_forecast import mdcev_predict


@dataclass
class MDCEVATEResult:
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
        lines.append("  MDCEV Average Treatment Effect (ATE) Summary")
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


def mdcev_ate(
    results: MDCEVResults,
    X: NDArray,
    X_gam: NDArray,
    price: NDArray,
    changevar_idx: tuple[int, int] | None = None,
    base_val: float | None = None,
    treatment_val: float | None = None,
    alternative_names: list[str] | None = None,
    n_draws: int = 1000,
    seed: int = 1234,
) -> MDCEVATEResult:
    """Compute predicted consumption shares and ATE for the MDCEV model.

    When ``changevar_idx`` and both value arguments are provided, the
    specified element of X is set to ``base_val`` and ``treatment_val``
    for all observations and predicted shares are compared.  When
    ``changevar_idx`` is None, only overall predicted shares at the
    observed data values are returned.

    Parameters
    ----------
    results : MDCEVResults
        Fitted MDCEV model results.
    X : NDArray, shape (N, nc, nvarm)
        Baseline utility design matrix.
    X_gam : NDArray, shape (N, nc, nvargam)
        Satiation utility design matrix.
    price : NDArray, shape (N, nc)
        Price matrix.
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

    Returns
    -------
    MDCEVATEResult
    """
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
