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
from pybhatlib.models.morp._morp_report import MORPReportTable


# Asymmetric with ``MNPResults`` (which is ``@dataclass(init=False)`` with a
# legacy-kwarg shim): MORPResults uses a plain ``@dataclass`` because MORP has
# no historical field renames to migrate — the canonical names are the
# original names, so the deprecation-shim infrastructure is unnecessary.


@dataclass
class MORPResults:
    """Results from MORP model estimation.

    Attributes
    ----------
    params : NDArray
        Estimated parameters (parametrized form).
    se : NDArray
        Standard errors. Computed using the estimator named in
        ``control.se_method`` (default ``"bhhh"`` to match GAUSS BHATLIB).
    se_bhhh, se_hessian, se_sandwich : NDArray | None
        Standard errors under each of the three asymptotic-variance
        estimators, computed at the converged MLE for diagnostic
        comparison. Each matches ``se`` for the method named in
        ``control.se_method`` and provides the alternatives for the
        other two. ``None`` if a particular estimator failed (e.g.,
        observed-Hessian computation diverged). Large divergence
        between BHHH, Hessian, and Sandwich on the same parameter is
        a classic misspecification signal — under correct specification
        the three converge asymptotically (information-matrix equality).
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
    report : MORPReportTable or None
        GAUSS-style reporting table in which the raw ``tau_*`` slots are
        replaced by the actual threshold cut-points (with delta-method
        standard errors) and a log-likelihood gradient column is added.
        Drives ``summary()`` / ``to_dataframe()`` when present; ``None`` for
        results constructed without it (raw ``tau``-space rendering is used
        as a fallback).
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
    se_bhhh: NDArray | None = None
    se_hessian: NDArray | None = None
    se_sandwich: NDArray | None = None
    report: MORPReportTable | None = None

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

        # Use the GAUSS-style reporting table (threshold cut-points instead of
        # the raw tau/delta slots, plus a gradient column) when available;
        # otherwise fall back to the raw tau-space arrays.
        if self.report is not None:
            names = self.report.names
            est_arr = self.report.estimate
            se_arr = self.report.se
            t_arr = self.report.t_stat
            p_arr = self.report.p_value
            grad_arr = self.report.gradient
        else:
            names = self.param_names
            est_arr = self.params
            se_arr = self.se
            t_arr = self.t_stat
            p_arr = self.p_value
            grad_arr = self.gradient

        name_w = max(16, max((len(n) for n in names), default=16) + 1)
        header = (
            f"  {'Parameters':<{name_w}s} {'Estimates':>10s} {'Std. err.':>10s} "
            f"{'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (name_w + 56))

        def _fmt(arr, i, width=10, prec=4):
            if arr is not None and i < len(arr) and np.isfinite(arr[i]):
                return f"{arr[i]:>{width}.{prec}f}"
            return f"{'.':>{width}s}"

        for i, name in enumerate(names):
            est_s = _fmt(est_arr, i)
            se_s = _fmt(se_arr, i)
            t_s = _fmt(t_arr, i, prec=3)
            p_s = _fmt(p_arr, i)
            g_s = _fmt(grad_arr, i)
            lines.append(f"  {name:<{name_w}s} {est_s} {se_s} {t_s} {p_s} {g_s}")

        lines.append("")

        # Correlation matrix
        if self.correlation_matrix is not None:
            lines.append("  Estimated error correlation matrix:")
            n = self.correlation_matrix.shape[0]
            for i in range(n):
                row_vals = [f"{self.correlation_matrix[i, j]:>7.3f}" for j in range(n)]
                lines.append("  " + " ".join(row_vals))

        # Side-by-side SE diagnostic when more than one estimator is available.
        # Mirrors the MNP-002c diagnostic block — see MORP_BHATLIB_PARITY plan.
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
                "  " + f"{'Parameter':<16s}"
                + "".join(f"{lbl:>12s}" for lbl, _ in available)
                + f"{'Hess/BHHH':>12s}"
            )
            lines.append(header_se)
            for i, name in enumerate(self.param_names):
                cells = []
                for _lbl, arr in available:
                    if i < len(arr):
                        cells.append(f"{arr[i]:>12.4f}")
                    else:
                        cells.append(f"{'-':>12s}")
                if (
                    self.se_bhhh is not None and self.se_hessian is not None
                    and i < len(self.se_bhhh) and i < len(self.se_hessian)
                    and self.se_bhhh[i] > 0
                ):
                    ratio_str = f"{self.se_hessian[i] / self.se_bhhh[i]:>12.3f}"
                else:
                    ratio_str = f"{'-':>12s}"
                lines.append(f"  {name:<16s}" + "".join(cells) + ratio_str)
            primary = self.control.se_method if self.control is not None else "?"
            lines.append("")
            lines.append(f"  Primary se_method = '{primary}' (controls .se / .t_stat / .p_value)")
            lines.append(
                "  Hess/BHHH ratios far from 1 indicate score variance and curvature disagree —"
            )
            lines.append(
                "  consider a richer covariance specification or se_method='sandwich'."
            )

        lines.append("")
        lines.append(f"  Number of iterations   {self.n_iter:>10d}")
        lines.append(f"  Minutes to convergence {self.convergence_time:>10.5f}")
        lines.append(sep)

        text = "\n".join(lines)
        print(text)
        return text

    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficient table to DataFrame.

        Uses the GAUSS-style reporting table (threshold cut-points, gradient
        column) when available; falls back to the raw ``tau``-space arrays.
        """
        if self.report is not None:
            r = self.report
            return pd.DataFrame(
                {
                    "Estimate": r.estimate,
                    "Std.Error": r.se,
                    "t-stat": r.t_stat,
                    "p-value": r.p_value,
                    "Gradient": r.gradient,
                },
                index=r.names,
            )
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
