"""Linear regression coefficient inference and fit diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from pybhatlib.models.lr._lr_control import LRControl


@dataclass
class LRResults:
    """Results from analytical linear regression, using the canonical field names.

    Attributes
    ----------
    params : NDArray
        Regression coefficients, shape (K,).  The residual variance is not a
        parameter; see ``sigma2`` / ``residual_variance``.
    se : NDArray
        Standard errors aligned with ``params``.
    t_stat : NDArray
        ``params / se``.
    p_value : NDArray
        Two-sided p-values: Student's t with ``df_resid`` degrees of freedom
        for ``se_method="hessian"``, normal otherwise.
    gradient : NDArray
        Mean per-observation score at the solution (zero up to round-off).
    loglik : float
        Mean Gaussian log-likelihood (per observation) at ``sigma2``.  For an
        exact fit the variance MLE lies on the zero boundary and ``loglik``
        is ``inf``.
    n_obs : int
        Number of observations.
    param_names : list[str]
        Coefficient names aligned with ``params``.
    corr_matrix, cov_matrix : NDArray
        Correlation / covariance matrix of the coefficient estimates.  NaN
        when ``want_covariance=False``.
    n_iter, convergence_time, converged, return_code
        Facade-parity fields (``0`` / minutes / ``True`` / ``0``): there is no
        optimizer.
    control : LRControl
        Control structure used for estimation.
    data_path : str
        Path to the data file used (``"<DataFrame>"`` for in-memory data).
    message : str
        Estimation note.
    sigma2 : float
        Gaussian MLE residual variance ``SSE / N``.
    residual_variance : float
        Unbiased residual variance ``SSE / (N - K)``.
    df_resid : int
        ``N - K``.
    r_squared, adj_r_squared : float
        Coefficient of determination; centered TSS when the design spans a
        constant, uncentered otherwise.
    fitted_values, residuals : NDArray or None
        In-sample fitted values and residuals, shape (N,).
    sst, ssr, sse : float
        Total, explained (model), and residual sums of squares on the same
        centered / uncentered basis as ``r_squared`` (``ssr + sse == sst``).
    df_model : int
        Number of non-constant regressors.
    f_stat : float
        Joint test of all slopes against zero,
        ``(ssr / df_model) / (sse / df_resid)``; NaN with no non-constant
        regressor, ``inf`` for an exact fit.
    """

    params: NDArray
    se: NDArray
    t_stat: NDArray
    p_value: NDArray
    gradient: NDArray
    loglik: float
    n_obs: int
    param_names: list[str]
    corr_matrix: NDArray
    cov_matrix: NDArray
    n_iter: int = 0
    convergence_time: float = 0.0
    converged: bool = True
    return_code: int = 0
    control: LRControl = field(default_factory=LRControl)
    data_path: str = ""
    message: str = "Analytical least squares"
    sigma2: float = np.nan
    residual_variance: float = np.nan
    df_resid: int = 0
    r_squared: float = np.nan
    adj_r_squared: float = np.nan
    fitted_values: NDArray | None = None
    residuals: NDArray | None = None
    sst: float = np.nan
    ssr: float = np.nan
    sse: float = np.nan
    df_model: int = 0
    f_stat: float = np.nan

    @classmethod
    def from_estimates(
        cls,
        beta: ArrayLike,
        *,
        param_names: list[str] | None = None,
        control: LRControl | None = None,
    ) -> LRResults:
        """Construct a minimal ``LRResults`` from externally supplied coefficients.

        Intended for post-estimation use (prediction, :func:`lr_ate`) when a
        full fit object is not available.  Inference fields are NaN and must
        not be interpreted, mirroring :meth:`MNLResults.from_estimates`.

        Parameters
        ----------
        beta : array_like, shape (K,)
            Regression coefficients.
        param_names : list[str] or None
            Names for each element of *beta*; defaults to ``["b1", "b2", ...]``.
        control : LRControl or None
            Control structure to carry through (defaults to ``LRControl()``).

        Returns
        -------
        LRResults
        """
        beta = np.asarray(beta, dtype=float)
        if beta.ndim != 1 or not len(beta) or not np.isfinite(beta).all():
            raise ValueError("beta must be a nonempty finite vector")
        names = list(param_names) if param_names is not None else [
            f"b{i + 1}" for i in range(len(beta))
        ]
        if len(names) != len(beta):
            raise ValueError("param_names must match beta length")
        v = np.full(len(beta), np.nan)
        m = np.full((len(beta), len(beta)), np.nan)
        return cls(
            params=beta.copy(), se=v.copy(), t_stat=v.copy(), p_value=v.copy(),
            gradient=v.copy(), loglik=np.nan, n_obs=0, param_names=names,
            corr_matrix=m.copy(), cov_matrix=m.copy(),
            control=control or LRControl(), message="External estimates",
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Coefficient table as a DataFrame.

        Returns
        -------
        df : pd.DataFrame
            Columns ``Estimate``, ``Std.Error``, ``t-stat``, ``p-value``,
            ``Gradient``; index = ``param_names``.
        """
        return pd.DataFrame({
            "Estimate": self.params, "Std.Error": self.se,
            "t-stat": self.t_stat, "p-value": self.p_value,
            "Gradient": self.gradient,
        }, index=self.param_names)

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
        lines.append("  pybhatlib LR Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = "normal convergence" if self.return_code == 0 else f"code {self.return_code}"
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.loglik:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append(f"  Residual df            {self.df_resid:>14d}")
        lines.append(f"  R-squared              {self.r_squared:>14.6f}")
        lines.append(f"  Adjusted R-squared     {self.adj_r_squared:>14.6f}")
        lines.append(f"  Residual variance      {self.residual_variance:>14.6f}")
        lines.append(f"  SST (total)            {self.sst:>14.6f}")
        lines.append(f"  SSR (explained)        {self.ssr:>14.6f}")
        lines.append(f"  SSE (residual)         {self.sse:>14.6f}")
        f_label = f"F({self.df_model}, {self.df_resid})"
        lines.append(f"  {f_label:<23s}{self.f_stat:>14.6f}")
        lines.append("")

        lines.append(
            "  Covariance matrix of the parameters computed by the following method:"
        )
        method_labels = {
            "hessian": "Inverse of observed information (Hessian)",
            "sandwich": "Huber-White robust sandwich",
            "bhhh": "Cross-product of first derivatives (BHHH)",
        }
        label = method_labels.get(self.control.se_method, self.control.se_method)
        if self.control.df_correction:
            label += " (df-corrected)"
        lines.append(f"  {label}" if self.control.want_covariance else "  disabled")
        lines.append("")

        header = (
            f"  {'Parameters':<16s} {'Estimates':>10s} {'Std. err.':>10s}"
            f" {'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 68)

        for i, name in enumerate(self.param_names):
            lines.append(
                f"  {name:<16s} {self.params[i]:>10.4f} {self.se[i]:>10.4f}"
                f" {self.t_stat[i]:>10.3f} {self.p_value[i]:>10.4f} {self.gradient[i]:>10.4f}"
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

        lines.append("")
        lines.append(f"  Number of iterations   {self.n_iter:>10d}")
        lines.append(f"  Minutes to convergence {self.convergence_time:>10.5f}")
        if self.message:
            lines.append(f"  Message: {self.message}")
        lines.append(sep)

        text = "\n".join(lines)
        print(text)
        return text
