"""MDCEV model results structure.

Contains estimated parameters, standard errors, test statistics, and model
diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models.mdcev._mdcev_control import MDCEVControl


@dataclass
class MDCEVResults:
    """Results from MDCEV model estimation.

    Attributes
    ----------
    b : NDArray
        Raw optimised parameter vector (theta-space), shape (n_params,).
        The final element is log(sigma) as used during optimisation.
    b_reported : NDArray
        Reported parameter vector with sigma in natural units (i.e., the
        last element has been exponentiated from the log-scale used
        internally), aligned with ``param_names``.
    se : NDArray
        Standard errors aligned with ``b_reported``.
        Computed using the estimator named in ``control.se_method``.
        SEs are computed directly in the unparameterized parameter space
        (sigma, not log_sigma) using finite differences on the
        unparameterized log-likelihood.
    se_bhhh, se_hessian, se_sandwich : NDArray | None
        Standard errors under each of the three asymptotic-variance
        estimators, computed at the converged MLE for diagnostic
        comparison. Each matches ``se`` for the method named in
        ``control.se_method`` and provides the alternatives for the
        other two. ``None`` if a particular estimator failed (e.g.,
        observed-Hessian computation diverged). 
    t_stat : NDArray
        t-statistics (b_reported / se).
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Final gradient vector at convergence (reported-parameter space).
    ll : float
        Mean log-likelihood (per observation).
    ll_total : float
        Total log-likelihood.
    n_obs : int
        Number of observations.
    param_names : list[str]
        Parameter names: beta names, then gamma names, then ``"sigma"``.
        Mirrors ``_max_ParNames = varnam|varngam|"sigm"`` in GAUSS.
    corr_matrix : NDArray
        Correlation matrix of parameter estimates.
    cov_matrix : NDArray
        Variance-covariance matrix of parameter estimates.
    n_iterations : int
        Number of optimizer iterations.
    convergence_time : float
        Wall-clock time in minutes to convergence.
    converged : bool
        Whether optimisation converged.
    return_code : int
        Optimizer return code (0 = normal convergence).
    sigma : float
        Estimated scale parameter (exp of the last optimised value).
    control : MDCEVControl
        Control structure used for estimation.
    data_path : str
        Path to the data file used.
    """

    b: NDArray
    b_reported: NDArray
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
    sigma: float
    se_bhhh: NDArray | None = None
    se_hessian: NDArray | None = None
    se_sandwich: NDArray | None = None
    control: MDCEVControl | None = None
    data_path: str = ""

    def summary(self) -> str:
        """Print formatted estimation results mirroring GAUSS maxprt output.

        Returns
        -------
        text : str
            Formatted summary string.
        """
        lines = []
        sep = "=" * 70

        lines.append(sep)
        utility_label = (
            "Traditional" if (self.control is None or self.control.utility == "trad")
            else "Linear"
        )
        lines.append(f"  pybhatlib MDCEV ({utility_label}) Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = (
            "normal convergence" if self.return_code == 0 else f"code {self.return_code}"
        )
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.ll:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append("")

        lines.append(
            "  Covariance matrix of the parameters computed by the following method:"
        )
        method_desc = {
            "bhhh": "Cross-product of first derivatives",
            "hessian": "Inverse observed information",
            "sandwich": "Robust sandwich estimator",
        }
        primary_method = self.control.se_method if self.control else "bhhh"
        lines.append(f"  {method_desc.get(primary_method, 'Unknown method')}")
        lines.append("")

        header = (
            f"  {'Parameters':<16s} {'Estimates':>10s} {'Std. err.':>10s}"
            f" {'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 68)

        for i, name in enumerate(self.param_names):
            est = self.b_reported[i] if i < len(self.b_reported) else 0.0
            se  = self.se[i]         if i < len(self.se)         else 0.0
            t   = self.t_stat[i]     if i < len(self.t_stat)     else 0.0
            p   = self.p_value[i]    if i < len(self.p_value)    else 0.0
            g   = self.gradient[i]   if i < len(self.gradient)   else 0.0
            lines.append(
                f"  {name:<16s} {est:>10.4f} {se:>10.4f}"
                f" {t:>10.3f} {p:>10.4f} {g:>10.4f}"
            )

        lines.append("")

        n_params = len(self.param_names)
        lines.append("  Correlation matrix of the parameters")
        for i in range(min(n_params, self.corr_matrix.shape[0])):
            row_vals = [
                f"{self.corr_matrix[i, j]:>7.3f}"
                for j in range(min(n_params, self.corr_matrix.shape[1]))
            ]
            lines.append("  " + " ".join(row_vals))

        # Side-by-side SE diagnostic when more than one estimator is available.
        # Large BHHH/Hessian/Sandwich divergence on the same coordinate is a
        # classic misspecification signal — under correct specification the
        # three converge asymptotically (information-matrix equality).
        available = [
            (label, arr) for label, arr in (
                ("BHHH",     self.se_bhhh),
                ("Hessian",  self.se_hessian),
                ("Sandwich", self.se_sandwich),
            )
            if arr is not None
        ]
        if len(available) >= 2:
            lines.append("")
            lines.append("  Standard error diagnostic (alternative estimators)")
            lines.append("  " + "-" * 66)
            header_se = (
                "  " + f"{'Parameter':<14s}"
                + "".join(f"{lbl:>12s}" for lbl, _ in available)
                + f"{'Hess/BHHH':>12s}"
            )
            lines.append(header_se)
            n_p = len(self.param_names)
            se_bhhh = self.se_bhhh if self.se_bhhh is not None else None
            se_hess = self.se_hessian if self.se_hessian is not None else None
            for i, name in enumerate(self.param_names):
                if i >= n_p:
                    break
                cells = []
                for lbl, arr in available:
                    if i < len(arr):
                        cells.append(f"{arr[i]:>12.4f}")
                    else:
                        cells.append(f"{'-':>12s}")
                # Hessian/BHHH ratio when both are available
                if (
                    se_bhhh is not None and se_hess is not None
                    and i < len(se_bhhh) and i < len(se_hess)
                    and se_bhhh[i] > 0
                ):
                    ratio = se_hess[i] / se_bhhh[i]
                    ratio_str = f"{ratio:>12.3f}"
                else:
                    ratio_str = f"{'-':>12s}"
                lines.append(f"  {name:<14s}" + "".join(cells) + ratio_str)
            primary = self.control.se_method if self.control is not None else "?"
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
            DataFrame with columns: Estimate, Std.Error, t-stat, p-value,
            Gradient.
        """
        n = len(self.param_names)
        return pd.DataFrame(
            {
                "Estimate":  self.b_reported[:n],
                "Std.Error": self.se[:n],
                "t-stat":    self.t_stat[:n],
                "p-value":   self.p_value[:n],
                "Gradient":  self.gradient[:n],
            },
            index=self.param_names,
        )
