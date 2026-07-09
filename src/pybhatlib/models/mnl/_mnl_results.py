"""MNL model results structure.

Equivalent to the ``maxprt`` output ``{x, f, g, cov, retcode}`` from
MNLcasenew.gss.  Contains estimated parameters, standard errors, test
statistics, and model diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models._results_common import (
    attach_deprecated_aliases,
    attach_ll_total_alias,
    legacy_init,
)
from pybhatlib.models.mnl._mnl_control import MNLControl


# Legacy construction kwargs → canonical field names.
_MNLRESULTS_LEGACY_KWARGS: dict[str, str] = {
    "b": "params",
    "ll": "loglik",
    "n_iterations": "n_iter",
}


@dataclass(init=False)
class MNLResults:
    """Results from MNL model estimation.

    Attributes
    ----------
    params : NDArray
        Optimised parameter vector, shape (n_params,).
    se : NDArray
        Standard errors aligned with ``params``.
    t_stat : NDArray
        t-statistics (params / se).
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Final gradient vector at convergence.
    loglik : float
        Mean log-likelihood (per observation).
    n_obs : int
        Number of observations.
    param_names : list[str]
        Parameter names, aligned with ``params`` (mirrors ``_max_ParNames``
        in GAUSS).
    corr_matrix : NDArray
        Correlation matrix of parameter estimates.
    cov_matrix : NDArray
        Variance-covariance matrix of parameter estimates.
    n_iter : int
        Number of optimizer iterations.
    convergence_time : float
        Wall-clock time in minutes to convergence.
    converged : bool
        Whether optimisation converged.
    return_code : int
        Optimizer return code (0 = normal convergence).
    control : MNLControl
        Control structure used for estimation.
    data_path : str
        Path to the data file used.
    message : str | None
        Solver message from the optimizer.

    Notes
    -----
    Field renames (canonical name ← deprecated alias): ``params`` ← ``b``,
    ``loglik`` ← ``ll`` (mean log-likelihood), ``n_iter`` ← ``n_iterations``.
    Reading a legacy alias still works but emits a ``DeprecationWarning``.
    ``ll_total`` is no longer stored; reading it returns ``loglik * n_obs``
    (with a warning).
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
    n_iter: int
    convergence_time: float
    converged: bool
    return_code: int
    control: MNLControl | None = None
    data_path: str = ""
    message: str | None = None

    def __init__(self, **kwargs: object) -> None:
        """Construct MNLResults, accepting both canonical and legacy kwargs.

        Legacy kwargs (``b``, ``ll``, ``n_iterations``, ``ll_total``) are
        translated to their canonical counterparts and emit a
        ``DeprecationWarning``.  Unknown kwargs raise ``TypeError``.
        """
        legacy_init(self, kwargs, _MNLRESULTS_LEGACY_KWARGS, "MNLResults")

    @classmethod
    def from_estimates(
        cls,
        beta: NDArray,
        *,
        param_names: list[str] | None = None,
        control: "MNLControl | None" = None,
    ) -> "MNLResults":
        """Construct a minimal ``MNLResults`` from externally supplied coefficients.

        Intended for post-estimation use (e.g. computing ATEs from GAUSS
        estimates) where a full fit object is not available.  Inference fields
        (standard errors, test statistics, log-likelihood) are filled with
        ``nan`` and should not be interpreted.  Following the cross-model
        convention (:meth:`MORPResults.from_estimates` /
        :meth:`MNPResults.from_estimates`), the supplied estimates are treated
        as a converged solution: ``converged=True`` and ``return_code=0``.

        Parameters
        ----------
        beta : ndarray, shape (n_params,)
            Slope coefficients in natural (non-transformed) space.
        param_names : list[str] or None
            Names for each element of *beta*.  Defaults to
            ``["b1", "b2", ...]`` when not provided.
        control : MNLControl or None
            Control structure to carry through (defaults to ``MNLControl()``).

        Returns
        -------
        MNLResults
        """
        beta = np.asarray(beta, dtype=np.float64).ravel()
        n = len(beta)

        names = param_names if param_names is not None else [f"b{i + 1}" for i in range(n)]

        nan_vec  = np.full(n, np.nan)
        nan_mat  = np.full((n, n), np.nan)

        return cls(
            params            = beta,
            se                = nan_vec.copy(),
            t_stat            = nan_vec.copy(),
            p_value           = nan_vec.copy(),
            gradient          = nan_vec.copy(),
            loglik            = float("nan"),
            n_obs             = 0,
            param_names       = list(names),
            corr_matrix       = nan_mat.copy(),
            cov_matrix        = nan_mat.copy(),
            n_iter            = 0,
            convergence_time  = float("nan"),
            converged         = True,
            return_code       = 0,
            control           = control if control is not None else MNLControl(),
        )

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
        lines.append("  pybhatlib MNL Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = (
            "normal convergence" if self.return_code == 0 else f"code {self.return_code}"
        )
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.loglik:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append("")

        lines.append(
            "  Covariance matrix of the parameters computed by the following method:"
        )
        method_labels = {
            "bhhh": "Cross-product of first derivatives (BHHH)",
            "hessian": "Inverse of observed information (Hessian)",
            "sandwich": "Huber-White robust sandwich",
        }
        if self.control is not None:
            label = method_labels.get(
                self.control.se_method.lower(), self.control.se_method
            )
            lines.append(f"  {label}")
        else:
            lines.append("  Cross-product of first derivatives")
        lines.append("")

        header = (
            f"  {'Parameters':<16s} {'Estimates':>10s} {'Std. err.':>10s}"
            f" {'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * 68)

        for i, name in enumerate(self.param_names):
            est = self.params[i]  if i < len(self.params)   else 0.0
            se  = self.se[i]      if i < len(self.se)       else 0.0
            t   = self.t_stat[i]  if i < len(self.t_stat)   else 0.0
            p   = self.p_value[i] if i < len(self.p_value)  else 0.0
            g   = self.gradient[i]if i < len(self.gradient) else 0.0
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

        lines.append("")
        lines.append(f"  Number of iterations   {self.n_iter:>10d}")
        lines.append(f"  Minutes to convergence {self.convergence_time:>10.5f}")
        if self.message is not None:
            lines.append(f"  Optimizer message: {self.message}")
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
                "Estimate":  self.params[:n],
                "Std.Error": self.se[:n],
                "t-stat":    self.t_stat[:n],
                "p-value":   self.p_value[:n],
                "Gradient":  self.gradient[:n],
            },
            index=self.param_names,
        )


# ----------------------------------------------------------------------
# Deprecated property aliases (b → params, ll → loglik,
# n_iterations → n_iter, ll_total → loglik * n_obs).  Attached after class
# construction so ``@dataclass`` does not treat them as fields.  Shared
# mechanism: ``pybhatlib.models._results_common``.
# ----------------------------------------------------------------------
attach_deprecated_aliases(MNLResults, _MNLRESULTS_LEGACY_KWARGS)
attach_ll_total_alias(MNLResults)
