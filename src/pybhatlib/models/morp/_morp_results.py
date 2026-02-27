"""MORP model results structure.

Contains estimated parameters, standard errors, test statistics,
threshold estimates, and model diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models.morp._morp_control import MORPControl


@dataclass
class MORPResults:
    """Results from MORP model estimation.

    Attributes
    ----------
    params : NDArray
        Estimated parameters (parametrized form).
    se : NDArray
        Standard errors.
    loglik : float
        Mean log-likelihood (per observation).
    n_obs : int
        Number of observations.
    n_params : int
        Total number of estimated parameters.
    converged : bool
        Whether optimization converged.
    n_iter : int
        Number of optimizer iterations.
    thresholds : list[NDArray]
        Estimated threshold parameters per dimension. Each array contains
        the category boundaries tau_1, tau_2, ..., tau_{J_d - 1}.
    correlation_matrix : NDArray or None
        Estimated error correlation matrix.
    param_names : list[str]
        Parameter names.
    t_stat : NDArray
        t-statistics.
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Gradient at convergence.
    cov_matrix : NDArray or None
        Variance-covariance matrix of parameter estimates.
    convergence_time : float
        Time in minutes to convergence.
    return_code : int
        Optimizer return code.
    control : MORPControl
        Control structure used for estimation.
    """

    params: NDArray
    se: NDArray
    loglik: float
    n_obs: int
    n_params: int
    converged: bool
    n_iter: int
    thresholds: list[NDArray]
    correlation_matrix: NDArray | None
    param_names: list[str]
    t_stat: NDArray
    p_value: NDArray
    gradient: NDArray
    cov_matrix: NDArray | None = None
    convergence_time: float = 0.0
    return_code: int = 0
    control: MORPControl | None = None

    def summary(self) -> str:
        """Print formatted estimation results.

        Returns
        -------
        text : str
            Formatted summary string.
        """
        lines = []
        sep = "=" * 70

        lines.append(sep)
        lines.append("  pybhatlib MORP Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = "normal convergence" if self.return_code == 0 else f"code {self.return_code}"
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.loglik:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append("")

        header = (
            f"  {'Parameters':<16s} {'Estimates':>10s} {'Std. err.':>10s} "
            f"{'Est./s.e.':>10s} {'Prob.':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 58)

        for i, name in enumerate(self.param_names):
            est = self.params[i]
            se_i = self.se[i] if i < len(self.se) else 0.0
            t_i = self.t_stat[i] if i < len(self.t_stat) else 0.0
            p_i = self.p_value[i] if i < len(self.p_value) else 0.0
            lines.append(f"  {name:<16s} {est:>10.4f} {se_i:>10.4f} {t_i:>10.3f} {p_i:>10.4f}")

        lines.append("")

        # Thresholds
        for d, tau_d in enumerate(self.thresholds):
            lines.append(f"  Thresholds (dimension {d + 1}): "
                         + ", ".join(f"{t:.4f}" for t in tau_d))

        lines.append("")

        # Correlation matrix
        if self.correlation_matrix is not None:
            lines.append("  Estimated error correlation matrix:")
            n = self.correlation_matrix.shape[0]
            for i in range(n):
                row_vals = [f"{self.correlation_matrix[i, j]:>7.3f}" for j in range(n)]
                lines.append("  " + " ".join(row_vals))

        lines.append("")
        lines.append(f"  Number of iterations   {self.n_iter:>10d}")
        lines.append(f"  Minutes to convergence {self.convergence_time:>10.5f}")
        lines.append(sep)

        text = "\n".join(lines)
        print(text)
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficient table to DataFrame."""
        n = len(self.param_names)
        return pd.DataFrame(
            {
                "Estimate": self.params[:n],
                "Std.Error": self.se[:n],
                "t-stat": self.t_stat[:n],
                "p-value": self.p_value[:n],
            },
            index=self.param_names,
        )
