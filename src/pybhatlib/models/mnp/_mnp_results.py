"""MNP model results structure.

Equivalent to BHATLIB's mnpResults struct. Contains estimated parameters,
standard errors, test statistics, and model diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models.mnp._mnp_control import MNPControl


@dataclass
class MNPResults:
    """Results from MNP model estimation.

    Attributes
    ----------
    b : NDArray
        Estimated parameters (parametrized form).
    b_original : NDArray
        Unparametrized coefficients.
    se : NDArray
        Standard errors.
    t_stat : NDArray
        t-statistics (b / se).
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Gradient at convergence.
    ll : float
        Mean log-likelihood (per observation).
    ll_total : float
        Total log-likelihood.
    n_obs : int
        Number of observations.
    param_names : list[str]
        Parameter names.
    corr_matrix : NDArray
        Correlation matrix of parameter estimates.
    cov_matrix : NDArray
        Variance-covariance matrix of parameter estimates.
    n_iterations : int
        Number of optimizer iterations.
    convergence_time : float
        Time in minutes to convergence.
    converged : bool
        Whether optimization converged.
    return_code : int
        Optimizer return code.
    lambda_hat : NDArray | None
        Estimated kernel error covariance (if IID=False).
    omega_hat : NDArray | None
        Random coefficient covariance (if mix=True).
    cholesky_L : NDArray | None
        Cholesky of Omega (if mix=True).
    segment_probs : NDArray | None
        Mixture segment probabilities (if nseg>1).
    segment_means : list[NDArray] | None
        Segment-specific means (if nseg>1).
    segment_covs : list[NDArray] | None
        Segment-specific covariances (if nseg>1).
    control : MNPControl
        Control structure used for estimation.
    data_path : str
        Path to data file used.
    """

    b: NDArray
    b_original: NDArray
    se: NDArray
    t_stat: NDArray
    p_value: NDArray
    gradient: NDArray
    ll: float
    ll_total: float
    n_obs: int
    param_names: list[str]
    corr_matrix: NDArray
    cov_matrix: NDArray
    n_iterations: int
    convergence_time: float
    converged: bool
    return_code: int
    lambda_hat: NDArray | None = None
    omega_hat: NDArray | None = None
    cholesky_L: NDArray | None = None
    segment_probs: NDArray | None = None
    segment_means: list[NDArray] | None = None
    segment_covs: list[NDArray] | None = None
    control: MNPControl | None = None
    data_path: str = ""

    def summary(self) -> str:
        """Print formatted estimation results (like BHATLIB Figure 10).

        Returns
        -------
        text : str
            Formatted summary string.
        """
        lines = []
        sep = "=" * 70

        lines.append(sep)
        lines.append(f"  pybhatlib MNP Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = "normal convergence" if self.return_code == 0 else f"code {self.return_code}"
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.ll:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append("")

        # Coefficient table
        lines.append("  Covariance matrix of the parameters computed by the following method:")
        lines.append("  Cross-product of first derivatives")
        lines.append("")

        header = f"  {'Parameters':<14s} {'Estimates':>10s} {'Std. err.':>10s} {'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        lines.append(header)
        lines.append("  " + "-" * 66)

        for i, name in enumerate(self.param_names):
            est = self.b_original[i] if i < len(self.b_original) else self.b[i]
            se = self.se[i] if i < len(self.se) else 0.0
            t = self.t_stat[i] if i < len(self.t_stat) else 0.0
            p = self.p_value[i] if i < len(self.p_value) else 0.0
            g = self.gradient[i] if i < len(self.gradient) else 0.0
            lines.append(
                f"  {name:<14s} {est:>10.4f} {se:>10.4f} {t:>10.3f} {p:>10.4f} {g:>10.4f}"
            )

        lines.append("")

        # Correlation matrix
        n_params = len(self.param_names)
        lines.append("  Correlation matrix of the parameters")
        for i in range(min(n_params, self.corr_matrix.shape[0])):
            row_vals = [
                f"{self.corr_matrix[i, j]:>7.3f}"
                for j in range(min(n_params, self.corr_matrix.shape[1]))
            ]
            lines.append("  " + " ".join(row_vals))

        lines.append("")
        lines.append(f"  Number of iterations   {self.n_iterations:>10d}")
        lines.append(f"  Minutes to convergence {self.convergence_time:>10.5f}")
        lines.append(sep)

        text = "\n".join(lines)
        print(text)
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficient table to pandas DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: Estimate, Std.Error, t-stat, p-value, Gradient.
        """
        n = len(self.param_names)
        return pd.DataFrame(
            {
                "Estimate": self.b_original[:n],
                "Std.Error": self.se[:n],
                "t-stat": self.t_stat[:n],
                "p-value": self.p_value[:n],
                "Gradient": self.gradient[:n],
            },
            index=self.param_names,
        )
