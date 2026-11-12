"""MNPKerCP model results structure.

Harmonized results object for the mixed-panel MNP (MNPKerCP) model.  Mirrors
:class:`~pybhatlib.models.mixmnl._mixmnl_results.MixMNLResults`: canonical
estimation fields ``params`` (estimate vector), ``loglik`` (mean log-likelihood)
and ``n_iter`` (iteration count), with the deprecated per-model aliases
(``b`` / ``ll`` / ``n_iterations`` and the computed ``ll_total``) installed
through the shared shims in :mod:`pybhatlib.models._results_common`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pybhatlib.models._fit_diagnostics import FitMetricsMixin
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr  # noqa: F401 - kept for API symmetry with siblings

from pybhatlib.models._results_common import (
    attach_deprecated_aliases,
    attach_ll_total_alias,
    legacy_init,
)
from pybhatlib.models.mnpkercp._mnpkercp_control import MNPKerCPControl


@dataclass
class MixingSummary:
    """Reported locations, scales and shapes, with latent joint dependence.

    Random coefficients precede kernel errors. Covariances are implied by
    the reported scales and latent Gaussian correlation, before nonlinear
    coefficient or kernel transformations.
    """

    names: list[str]
    n_random: int
    means: NDArray
    scales: NDArray
    shapes: NDArray
    corr_matrix: NDArray
    note: str = ""

    @property
    def cov_matrix(self) -> NDArray:
        return np.outer(self.scales, self.scales) * self.corr_matrix


def mixing_summary_from_state(
    var_names, spec, state, *, kernel_names=(), kernel_scales=None,
    kernel_shapes=None, joint_corr=None, copula=False, note="",
) -> MixingSummary:
    """Capture the fitted pipeline/kernel state without changing estimation."""
    rc_names = [var_names[int(p)] for p in spec.mixpos]
    k = len(rc_names)
    names = rc_names + list(kernel_names)
    nk = len(kernel_names)
    corr = np.array(
        state.omegastar if joint_corr is None else joint_corr,
        dtype=np.float64, copy=True,
    ).reshape(len(names), len(names))
    if nk and not copula:
        # The likelihood ignores RC/kernel dependence when copula is off.
        corr[:k, k:] = 0.0
        corr[k:, :k] = 0.0
    return MixingSummary(
        names=names,
        n_random=k,
        means=np.concatenate([
            np.asarray(state.xmu)[np.asarray(spec.mixpos, dtype=np.intp)],
            np.zeros(nk),
        ]),
        scales=np.concatenate([
            np.asarray(state.wscalrand).reshape(-1),
            np.ones(nk) if kernel_scales is None else kernel_scales,
        ]),
        shapes=np.concatenate([
            np.asarray(state.xlamrnd).reshape(-1),
            np.ones(nk) if kernel_shapes is None else kernel_shapes,
        ]),
        corr_matrix=corr,
        note=note,
    )


# Legacy construction kwargs -> canonical field names.
_MNPKERCPRESULTS_LEGACY_KWARGS: dict[str, str] = {
    "b": "params",
    "ll": "loglik",
    "n_iterations": "n_iter",
}


@dataclass(init=False)
class MNPKerCPResults(FitMetricsMixin):
    """Results from MNPKerCP (mixed-panel MNP) model estimation.

    Attributes
    ----------
    params : NDArray
        Optimised parameter vector in *reporting* space, shape (n_params,).
        Layout ``[beta | rcor | scal | kern | lam]`` (GAUSS ``b`` order for
        MNPKERCP): fixed coefficients, then the joint random-coefficient +
        differenced-kernel correlation entries, the random-coefficient scale
        (std-dev) vector, the kernel-scale parameters, and the Yeo-Johnson
        ``lambda`` values in ``(0, 2)``.
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
    loglik_constants, loglik_zero : float
        Mean independent empirical-share and uniform null log-likelihoods,
        normalized like ``loglik``. Shares are conditional on the available
        choice set; ordinal dimensions and panel occasions are independent
        in these benchmarks, rather than restricted fits of the full kernel.
    average_correct_probability : float
        Arithmetic mean fitted probability of the observed outcome (the joint
        category combination for ordinal models). In panels, average across
        occasions within each person, then across persons. It is not the
        exponential of the mean log-likelihood or a classification hit rate.
        These diagnostics are ``nan`` without fitting data.
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
    control : MNPKerCPControl or None
        Control structure used for estimation.
    data_path : str
        Path to the data file used.
    message : str | None
        Solver message from the optimizer.
    mixing_summary : MixingSummary | None
        Fitted random-coefficient and differenced-kernel distribution details.

    Notes
    -----
    Field renames (canonical name <- deprecated alias): ``params`` <- ``b``,
    ``loglik`` <- ``ll`` (mean log-likelihood), ``n_iter`` <- ``n_iterations``.
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
    control: MNPKerCPControl | None = None
    data_path: str = ""
    message: str | None = None
    mixing_summary: MixingSummary | None = None

    # Independent null benchmarks; not restricted fits of the full kernel.
    loglik_constants: float = np.nan
    loglik_zero: float = np.nan
    n_free_params: int = -1
    average_correct_probability: float = np.nan

    def __init__(self, **kwargs: object) -> None:
        """Construct MNPKerCPResults, accepting both canonical and legacy kwargs.

        Legacy kwargs (``b``, ``ll``, ``n_iterations``, ``ll_total``) are
        translated to their canonical counterparts and emit a
        ``DeprecationWarning``.  Unknown kwargs raise ``TypeError``.
        """
        legacy_init(
            self, kwargs, _MNPKERCPRESULTS_LEGACY_KWARGS, "MNPKerCPResults"
        )

    @classmethod
    def from_estimates(
        cls,
        params: NDArray,
        *,
        param_names: list[str] | None = None,
        control: "MNPKerCPControl | None" = None,
    ) -> "MNPKerCPResults":
        """Construct a minimal ``MNPKerCPResults`` from externally supplied params.

        Intended for post-estimation use (e.g. computing draw-integrated ATEs
        from GAUSS estimates) where a full fit object is not available.
        Inference fields (standard errors, test statistics, log-likelihood) are
        filled with ``nan`` and should not be interpreted.  Following the
        cross-model convention (:meth:`MNLResults.from_estimates` /
        :meth:`MNPResults.from_estimates`), the supplied estimates are treated as
        a converged solution: ``converged=True`` and ``return_code=0``.

        Parameters
        ----------
        params : ndarray, shape (n_params,)
            Reporting-space parameter vector in the GAUSS ``b`` layout
            ``[beta | rcor | scal | kern | lam]``.
        param_names : list[str] or None
            Names for each element of *params*.  Defaults to
            ``["b1", "b2", ...]`` when not provided.
        control : MNPKerCPControl or None
            Control structure to carry through (defaults to
            ``MNPKerCPControl()``).

        Returns
        -------
        MNPKerCPResults
        """
        params = np.asarray(params, dtype=np.float64).ravel()
        n = len(params)

        names = (
            param_names
            if param_names is not None
            else [f"b{i + 1}" for i in range(n)]
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
            param_names=list(names),
            corr_matrix=nan_mat.copy(),
            cov_matrix=nan_mat.copy(),
            n_iter=0,
            convergence_time=float("nan"),
            converged=True,
            return_code=0,
            control=control if control is not None else MNPKerCPControl(),
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
        lines.append("  pybhatlib MNPKerCP Estimation Results")
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
        for label, value in (
            ("Mean LL(C)", self.loglik_constants),
            ("Mean LL(0)", self.loglik_zero),
            ("APCP", self.average_correct_probability),
        ):
            formatted = f"{value:>14.6f}" if not np.isnan(value) else f"{'N/A':>14s}"
            lines.append(f"  {label:<23s}{formatted}")
        lines.extend(self.rho2_summary_lines())
        lines.append(f"  Number of cases        {self.n_obs:>14d}")
        lines.append(f"  Number of individuals  {self.n_ind:>14d}")
        lines.append("")

        name_width = max(
            16,
            max((len(name) for name in self.param_names), default=16) + 1,
        )
        header = (
            f"  {'Parameters':<{name_width}s} {'Estimates':>10s} {'Std. err.':>10s}"
            f" {'Est./s.e.':>10s} {'Prob.':>10s} {'Gradient':>10s}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for i, name in enumerate(self.param_names):
            est = self.params[i] if i < len(self.params) else 0.0
            se = self.se[i] if i < len(self.se) else 0.0
            t = self.t_stat[i] if i < len(self.t_stat) else 0.0
            p = self.p_value[i] if i < len(self.p_value) else 0.0
            g = self.gradient[i] if i < len(self.gradient) else 0.0
            lines.append(
                f"  {name:<{name_width}s} {est:>10.4f} {se:>10.4f}"
                f" {t:>10.3f} {p:>10.4f} {g:>10.4f}"
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
                    "", "  Kernel-error (differenced) scales and shapes",
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
                "Joint random-coefficient / kernel-error (differenced)" if has_kernel and mixing.n_random
                else "Kernel-error (differenced)" if has_kernel else "Random-coefficient"
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
# Deprecated property aliases (b -> params, ll -> loglik,
# n_iterations -> n_iter, ll_total -> loglik * n_obs).  Attached after class
# construction so ``@dataclass`` does not treat them as fields.  Shared
# mechanism: ``pybhatlib.models._results_common``.
# ----------------------------------------------------------------------
attach_deprecated_aliases(MNPKerCPResults, _MNPKERCPRESULTS_LEGACY_KWARGS)
attach_ll_total_alias(MNPKerCPResults)
