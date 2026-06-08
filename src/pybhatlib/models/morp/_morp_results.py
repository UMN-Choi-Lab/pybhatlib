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

    @classmethod
    def from_estimates(
        cls,
        beta: NDArray,
        thresholds: list[NDArray],
        correlation: NDArray | None = None,
        *,
        dep_vars: list[str],
        n_categories: list[int],
        var_names: list[str] | None = None,
        control: MORPControl | None = None,
    ) -> "MORPResults":
        """Build a results object from final (natural-space) coefficients.

        This is the GAUSS-style "plug in the converged estimates" workflow: it
        lets you compute predictions / ATEs from the **reported** coefficients
        (betas, threshold cut-points, and the correlation matrix) without
        re-running the optimiser. See ``anna0605/email.md`` (UTA follow-up,
        2026-06): "match the GAUSS style, where users can input the final
        estimation coefficients and use those values to calculate the ATEs."

        The natural quantities are encoded into a self-consistent raw parameter
        vector (thresholds → log-increments; correlations → ``atanh`` under the
        direct parameterisation, which reproduces any valid PD input matrix
        exactly), so the resulting object drives ``morp_ate`` / ``morp_predict``
        / ``morp_joint_probs`` / ``summary`` unchanged. Standard errors are
        ``NaN`` (no covariance is supplied for fixed user inputs).

        Parameters
        ----------
        beta : array-like, shape (n_beta,)
            Slope coefficients, in the model's coefficient order.
        thresholds : list of array-like
            Per-dimension threshold **cut-points** (the values printed in the
            output table / GAUSS THRESH), each of length ``n_categories[d] - 1``
            and strictly increasing.
        correlation : array-like, shape (n_dims, n_dims), optional
            Error correlation matrix. ``None`` (default) builds an IID model.
        dep_vars : list of str
            Outcome names (defines ``n_dims = len(dep_vars)``).
        n_categories : list of int
            Number of ordinal categories per dimension.
        var_names : list of str, optional
            Names for the ``beta`` coefficients (defaults to ``b0, b1, ...``).
        control : MORPControl, optional
            Base control; its ``method`` (MVNCD approximation) is preserved.
            The correlation encoding is forced to a consistent unit-variance,
            direct-parameterisation form.

        Returns
        -------
        MORPResults
        """
        from pybhatlib.models.morp._morp_control import (
            MORPControl as _MC,
            morp_control_replace,
        )
        from pybhatlib.models.morp._morp_report import build_morp_report

        beta = np.asarray(beta, dtype=np.float64).ravel()
        n_beta = beta.shape[0]
        n_dims = len(dep_vars)
        thresholds = [np.asarray(t, dtype=np.float64).ravel() for t in thresholds]
        if len(thresholds) != n_dims or len(n_categories) != n_dims:
            raise ValueError(
                "thresholds and n_categories must each have length len(dep_vars)"
            )
        iid = correlation is None

        base = control if control is not None else _MC(iid=iid)
        control = morp_control_replace(
            base, iid=iid, spherical=False, fix_scales=True, heteronly=False
        )

        # Encode raw optimiser-space theta from the natural quantities.
        theta: list[float] = [float(b) for b in beta]
        for d in range(n_dims):
            m = n_categories[d] - 1
            t = thresholds[d]
            if m <= 0:
                continue
            if t.shape[0] != m:
                raise ValueError(
                    f"thresholds[{d}] must have {m} cut-points, got {t.shape[0]}"
                )
            incs = np.diff(t)
            if t.shape[0] >= 2 and np.any(incs <= 0):
                raise ValueError(
                    f"thresholds[{d}] must be strictly increasing: {t}"
                )
            theta.append(float(t[0]))
            theta.extend(np.log(incs).tolist())

        corr_mat: NDArray | None = None
        if not iid:
            corr_mat = np.asarray(correlation, dtype=np.float64)
            if corr_mat.shape != (n_dims, n_dims):
                raise ValueError(
                    f"correlation must be {n_dims}x{n_dims}, got {corr_mat.shape}"
                )
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    rho = float(corr_mat[i, j])
                    if not -1.0 < rho < 1.0:
                        raise ValueError(
                            f"correlation[{i},{j}]={rho} must be in (-1, 1)"
                        )
                    theta.append(float(np.arctanh(rho)))
        theta_arr = np.asarray(theta, dtype=np.float64)

        # Parameter names matching MORPModel._build_param_names convention.
        if var_names is None:
            var_names = [f"b{i}" for i in range(n_beta)]
        names = list(var_names)
        for d in range(n_dims):
            for j in range(n_categories[d] - 1):
                names.append(f"tau_{dep_vars[d]}_{j + 1}")
        if not iid:
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    names.append(f"corr_{dep_vars[i]}_{dep_vars[j]}")

        report = build_morp_report(
            theta_arr, None, None, n_beta, n_dims, n_categories, control,
            names, list(dep_vars),
        )

        nan = np.full(theta_arr.shape, np.nan, dtype=np.float64)
        return cls(
            params=theta_arr,
            se=nan.copy(),
            loglik=float("nan"),
            n_obs=0,
            n_params=theta_arr.shape[0],
            converged=True,
            n_iter=0,
            thresholds=thresholds,
            correlation_matrix=corr_mat,
            param_names=names,
            t_stat=nan.copy(),
            p_value=nan.copy(),
            gradient=nan.copy(),
            cov_matrix=None,
            control=control,
            report=report,
        )

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
