"""Linear regression coefficient inference and fit diagnostics."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._lr_control import LRControl


@dataclass
class LRResults:
    """Analytical LR results using the package's canonical field names.

    ``params`` contains regression coefficients only. ``sigma2`` is the
    Gaussian MLE residual variance (SSE/N); ``residual_variance`` is SSE/(N-K).
    ``loglik`` is the mean Gaussian log-likelihood. For an exact fit the
    Gaussian variance MLE lies on the zero boundary and loglik is infinite.
    R-squared uses centered TSS when the design spans a constant, otherwise
    uncentered TSS. ``sst``/``ssr``/``sse`` are the total, explained (model),
    and residual (error) sums of squares on the same centered/uncentered
    basis as ``r_squared``, so ``ssr + sse == sst``. ``f_stat`` tests all
    slopes jointly against zero: ``(ssr / df_model) / (sse / df_resid)``,
    where ``df_model`` excludes the constant (NaN when there are no
    non-constant regressors). For an exact fit ``f_stat`` is infinite,
    matching ``loglik``. Inference unavailable by request is represented
    by NaN.
    """

    params: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    gradient: np.ndarray
    loglik: float
    n_obs: int
    param_names: list[str]
    corr_matrix: np.ndarray
    cov_matrix: np.ndarray
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
    fitted_values: np.ndarray | None = None
    residuals: np.ndarray | None = None
    sst: float = np.nan
    ssr: float = np.nan
    sse: float = np.nan
    df_model: int = 0
    f_stat: float = np.nan

    @classmethod
    def from_estimates(cls, beta, *, param_names=None, control=None):
        """Wrap external coefficients for prediction and scenario analysis."""
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
        return cls(beta.copy(), v.copy(), v.copy(), v.copy(), v.copy(),
                   np.nan, 0, names, m.copy(), m.copy(),
                   control=control or LRControl(), message="External estimates")

    def to_dataframe(self):
        """Return the standard package coefficient table."""
        return pd.DataFrame({
            "Estimate": self.params, "Std.Error": self.se,
            "t-stat": self.t_stat, "p-value": self.p_value,
            "Gradient": self.gradient,
        }, index=self.param_names)

    def summary(self):
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
