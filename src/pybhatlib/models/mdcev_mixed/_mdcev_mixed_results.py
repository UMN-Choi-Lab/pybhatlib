"""Mixed MDCEV model results structure.

Harmonized results object for the mixed / panel MDCEV model. Mirrors
:class:`~pybhatlib.models.mixmnl._mixmnl_results.MixMNLResults`: canonical
estimation fields ``params`` (estimate vector), ``loglik`` (mean log-likelihood)
and ``n_iter`` (iteration count), with the deprecated per-model aliases (``b`` /
``ll`` / ``n_iterations`` and the computed ``ll_total``) installed through the
shared shims in :mod:`pybhatlib.models._results_common`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models._results_common import (
    MixingSummary,
    attach_deprecated_aliases,
    attach_ll_total_alias,
    legacy_init,
    mixing_summary_from_state,
)
from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl


# Legacy construction kwargs → canonical field names.
_MDCEVMIXEDRESULTS_LEGACY_KWARGS: dict[str, str] = {
    "b": "params",
    "ll": "loglik",
    "n_iterations": "n_iter",
}


@dataclass(init=False)
class MDCEVMixedResults:
    """Results from mixed MDCEV model estimation.

    Attributes
    ----------
    params : NDArray
        Optimised parameter vector in *reporting* space, shape (n_params,).
        Layout ``[beta | gamma | rcor | kern | scal | lam]``: baseline-utility
        coefficients, translation (``gamma``) parameters, random-coefficient
        correlation entries, the MDCEV kernel error scale, the random-coefficient
        scale (std-dev) vector, and the Yeo-Johnson ``lambda`` values in
        ``(0, 2)``.
    se : NDArray
        Standard errors aligned with ``params``.
    t_stat : NDArray
        t-statistics (params / se).
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Final (summed) score vector at convergence, estimation space.
    loglik : float
        Mean simulated log-likelihood (per individual).
    n_obs : int
        Number of observations.
    n_ind : int
        Number of individuals (panel persons).
    param_names : list[str]
        Parameter names, aligned with ``params``.
    corr_matrix : NDArray
        Correlation matrix of the parameter estimates.
    cov_matrix : NDArray
        Variance-covariance matrix of the parameter estimates.
    n_iter : int
        Number of optimizer iterations.
    convergence_time : float
        Wall-clock time in minutes to convergence.
    converged : bool
        Whether optimisation converged.
    return_code : int
        Optimizer return code (0 = normal convergence).
    control : MDCEVMixedControl or None
        Control structure used for estimation.
    data_path : str
        Path to the data file used.
    message : str | None
        Solver message from the optimizer.
    mixing_summary : MixingSummary | None
        Fitted random-coefficient distribution details, when available.

    Notes
    -----
    Field renames (canonical name ← deprecated alias): ``params`` ← ``b``,
    ``loglik`` ← ``ll`` (mean log-likelihood), ``n_iter`` ← ``n_iterations``.
    Reading a legacy alias still works but emits a ``DeprecationWarning``.
    ``ll_total`` is no longer stored; reading it returns ``loglik * n_obs``.
    """

    params: NDArray
    se: NDArray
    t_stat: NDArray
    p_value: NDArray
    gradient: NDArray
    loglik: float
    n_obs: int
    n_ind: int
    param_names: list[str]
    corr_matrix: NDArray
    cov_matrix: NDArray
    n_iter: int
    convergence_time: float
    converged: bool
    return_code: int
    control: MDCEVMixedControl | None = None
    data_path: str = ""
    message: str | None = None
    mixing_summary: MixingSummary | None = None

    def __init__(self, **kwargs: object) -> None:
        """Construct MDCEVMixedResults, accepting both canonical and legacy kwargs.

        Legacy kwargs (``b``, ``ll``, ``n_iterations``, ``ll_total``) are
        translated to their canonical counterparts and emit a
        ``DeprecationWarning``. Unknown kwargs raise ``TypeError``.
        """
        legacy_init(
            self, kwargs, _MDCEVMIXEDRESULTS_LEGACY_KWARGS, "MDCEVMixedResults"
        )

    @classmethod
    def from_estimates(
        cls,
        params: NDArray,
        *,
        param_names: list[str] | None = None,
        control: "MDCEVMixedControl | None" = None,
    ) -> "MDCEVMixedResults":
        """Construct a minimal ``MDCEVMixedResults`` from external estimates.

        Intended for post-estimation use (e.g. computing ATEs from GAUSS
        estimates) where a full fit object is not available.  Inference fields
        (standard errors, test statistics, log-likelihood, covariance) are filled
        with ``nan`` and should not be interpreted.  Following the cross-model
        convention (:meth:`MNLResults.from_estimates` /
        :meth:`MDCEVResults.from_estimates`), the supplied estimates are treated
        as a converged solution: ``converged=True`` and ``return_code=0``.

        The ``params`` stored here are directly consumable by
        :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_ate.mdcev_mixed_ate_from_params`,
        which pins the model's MSL engine under
        :class:`~pybhatlib.mixed._reparam.ReportingSpace` and computes
        draw-integrated ATEs at these natural (reporting-space) parameters --
        no reporting -> estimation inversion is required.

        Parameters
        ----------
        params : ndarray, shape (n_params,)
            Parameter estimates, in the model's reporting-space layout
            ``[beta | gamma | rcor | kern | scal | lam]``.
        param_names : list[str] or None
            Names for each element of *params*.  Defaults to
            ``["p1", "p2", ...]`` when not provided.
        control : MDCEVMixedControl or None
            Control structure to carry through (defaults to
            ``MDCEVMixedControl()``).

        Returns
        -------
        MDCEVMixedResults
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        n = len(params)
        names = (
            list(param_names)
            if param_names is not None
            else [f"p{i + 1}" for i in range(n)]
        )

        nan_vec = np.full(n, np.nan)
        nan_mat = np.full((n, n), np.nan)

        return cls(
            params=params,
            se=nan_vec.copy(),
            t_stat=nan_vec.copy(),
            p_value=nan_vec.copy(),
            gradient=nan_vec.copy(),
            loglik=float("nan"),
            n_obs=0,
            n_ind=0,
            param_names=names,
            corr_matrix=nan_mat.copy(),
            cov_matrix=nan_mat.copy(),
            n_iter=0,
            convergence_time=float("nan"),
            converged=True,
            return_code=0,
            control=control if control is not None else MDCEVMixedControl(),
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
        lines.append("  pybhatlib Mixed MDCEV Estimation Results")
        lines.append(sep)
        lines.append("")

        rc_msg = (
            "normal convergence"
            if self.return_code == 0
            else f"code {self.return_code}"
        )
        lines.append(f"  return code = {self.return_code:>5d}")
        lines.append(f"  {rc_msg}")
        lines.append("")
        lines.append(f"  Mean log-likelihood    {self.loglik:>14.6f}")
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append(f"  Number of individuals  {self.n_ind:>14d}")
        lines.append("")

        name_width = max(
            16,
            max((len(name) for name in self.param_names), default=16) + 1,
        )
        header = (
            f"  {'Parameters':<{name_width}s} {'Estimates':>10s} {'Std. err.':>10s}"
            f" {'Est./s.e.':>10s} {'Prob.':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for i, name in enumerate(self.param_names):
            est = self.params[i] if i < len(self.params) else 0.0
            se = self.se[i] if i < len(self.se) else 0.0
            t = self.t_stat[i] if i < len(self.t_stat) else 0.0
            p = self.p_value[i] if i < len(self.p_value) else 0.0
            lines.append(
                f"  {name:<{name_width}s} {est:>10.4f} {se:>10.4f}"
                f" {t:>10.3f} {p:>10.4f}"
            )

        mixing = self.mixing_summary
        if mixing is not None and mixing.names:
            width = max(20, max(map(len, mixing.names)))

            def append_moments(title: str, start: int, stop: int) -> None:
                lines.extend([
                    "", f"  {title}",
                    f"  {'Coefficient':<{width}s} {'Location':>12s}"
                    f" {'Scale':>12s} {'Shape':>12s}",
                    "  " + "-" * (width + 39),
                ])
                for i in range(start, stop):
                    lines.append(
                        f"  {mixing.names[i]:<{width}s} {mixing.means[i]:>12.5f}"
                        f" {mixing.scales[i]:>12.5f} {mixing.shapes[i]:>12.5f}"
                    )

            if mixing.n_random:
                append_moments(
                    "Random-coefficient locations, scales, and shapes",
                    0, mixing.n_random,
                )
            has_kernel = mixing.n_random < len(mixing.names)
            if has_kernel:
                lines.extend([
                    "", "  Kernel-error scales and shapes",
                    f"  {'Coefficient':<{width}s} {'Scale':>12s} {'Shape':>12s}",
                    "  " + "-" * (width + 26),
                ])
                for i in range(mixing.n_random, len(mixing.names)):
                    lines.append(
                        f"  {mixing.names[i]:<{width}s}"
                        f" {mixing.scales[i]:>12.5f} {mixing.shapes[i]:>12.5f}"
                    )

            def append_matrix(title: str, matrix: NDArray) -> None:
                col_width = max(11, max(map(len, mixing.names)))
                lines.extend([
                    "", f"  {title}",
                    f"  {'':<{width}s} "
                    + " ".join(f"{name:>{col_width}s}" for name in mixing.names),
                ])
                for name, row in zip(mixing.names, matrix):
                    values = [
                        f"{value:>{col_width}.4f}" if np.isfinite(value)
                        else f"{'--':>{col_width}s}"
                        for value in row
                    ]
                    lines.append(f"  {name:<{width}s} " + " ".join(values))

            subject = (
                "Joint random-coefficient / kernel-error" if has_kernel and mixing.n_random
                else "Kernel-error" if has_kernel else "Random-coefficient"
            )
            append_matrix(f"{subject} correlation matrix", mixing.corr_matrix)
            append_matrix(f"{subject} implied covariance matrix", mixing.cov_matrix)

        lines.extend(["", "  Correlation matrix of the parameters"])
        for row in self.corr_matrix[:len(self.param_names), :len(self.param_names)]:
            lines.append("  " + " ".join(f"{value:>7.3f}" for value in row))

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
        """Convert the coefficient table to a pandas DataFrame.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: Estimate, Std.Error, t-stat, p-value.
        """
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


# ----------------------------------------------------------------------
# Deprecated property aliases (b → params, ll → loglik,
# n_iterations → n_iter, ll_total → loglik * n_obs). Shared mechanism:
# ``pybhatlib.models._results_common``.
# ----------------------------------------------------------------------
attach_deprecated_aliases(MDCEVMixedResults, _MDCEVMIXEDRESULTS_LEGACY_KWARGS)
attach_ll_total_alias(MDCEVMixedResults)
