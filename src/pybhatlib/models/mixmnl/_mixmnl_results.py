"""MixMNL model results structure.

Harmonized results object for the Mixed Multinomial Logit model.  Mirrors
:class:`~pybhatlib.models.mnl._mnl_results.MNLResults`: canonical estimation
fields ``params`` (estimate vector), ``loglik`` (mean log-likelihood) and
``n_iter`` (iteration count), with the deprecated per-model aliases (``b`` /
``ll`` / ``n_iterations`` and the computed ``ll_total``) installed through the
shared shims in :mod:`pybhatlib.models._results_common`.
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
from pybhatlib.models.mixmnl._mixmnl_control import MixMNLControl


# Legacy construction kwargs → canonical field names.
_MIXMNLRESULTS_LEGACY_KWARGS: dict[str, str] = {
    "b": "params",
    "ll": "loglik",
    "n_iterations": "n_iter",
}


@dataclass(init=False)
class MixMNLResults:
    """Results from MixMNL model estimation.

    Attributes
    ----------
    params : NDArray
        Optimised parameter vector in *reporting* space, shape (n_params,).
        Layout ``[beta | rcor | scal | lam | kern]``: fixed coefficients, then
        the random-coefficient correlation entries, scale (std-dev) vector,
        Yeo-Johnson ``lambda`` values in ``(0, 2)``, and any kernel-only
        parameters (none for the softmax kernel).
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
    control : MixMNLControl or None
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
    n_ind: int
    param_names: list[str]
    corr_matrix: NDArray
    cov_matrix: NDArray
    n_iter: int
    convergence_time: float
    converged: bool
    return_code: int
    control: MixMNLControl | None = None
    data_path: str = ""
    message: str | None = None
    random_coef_names: list[str] | None = None
    random_coef_means: NDArray | None = None
    random_coef_scales: NDArray | None = None
    random_coef_shapes: NDArray | None = None
    random_coef_corr_matrix: NDArray | None = None
    random_coef_cov_matrix: NDArray | None = None

    def __init__(self, **kwargs: object) -> None:
        """Construct MixMNLResults, accepting both canonical and legacy kwargs.

        Legacy kwargs (``b``, ``ll``, ``n_iterations``, ``ll_total``) are
        translated to their canonical counterparts and emit a
        ``DeprecationWarning``.  Unknown kwargs raise ``TypeError``.
        """
        legacy_init(self, kwargs, _MIXMNLRESULTS_LEGACY_KWARGS, "MixMNLResults")

    @classmethod
    def from_estimates(
        cls,
        params: NDArray,
        *,
        param_names: list[str] | None = None,
        control: "MixMNLControl | None" = None,
    ) -> "MixMNLResults":
        """Construct a minimal ``MixMNLResults`` from externally supplied params.

        Intended for post-estimation use (e.g. computing ATEs from GAUSS
        estimates) where a full fit object is not available.  Inference fields
        (standard errors, test statistics, log-likelihood) are filled with
        ``nan`` and should not be interpreted.  Following the cross-model
        convention (:meth:`~pybhatlib.models.mnl.MNLResults.from_estimates`),
        the supplied estimates are treated as a converged solution:
        ``converged=True`` and ``return_code=0``.

        Parameters
        ----------
        params : ndarray, shape (n_params,)
            Parameter vector in *reporting* space (``[beta | rcor | scal |
            lam]`` layout, as stored on a fitted :class:`MixMNLResults`).
        param_names : list[str] or None
            Names for each element of *params*.  Defaults to
            ``["b1", "b2", ...]`` when not provided.
        control : MixMNLControl or None
            Control structure to carry through (defaults to ``MixMNLControl()``).

        Returns
        -------
        MixMNLResults
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
            control=control if control is not None else MixMNLControl(),
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
        lines.append("  pybhatlib MixMNL Estimation Results")
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


        rc_names = self.random_coef_names
        rc_means = self.random_coef_means
        rc_scales = self.random_coef_scales
        rc_shapes = self.random_coef_shapes
        rc_corr = self.random_coef_corr_matrix
        rc_cov = self.random_coef_cov_matrix
        if (
            rc_names
            and rc_means is not None
            and rc_scales is not None
            and rc_corr is not None
            and rc_cov is not None
        ):
            lines.append("")
            lines.append("  Random-coefficient locations, scales, and shapes")
            lines.append(
                f"  {'Coefficient':<20s} {'Location':>12s} {'Scale':>12s} {'Shape':>12s}"
            )
            lines.append("  " + "-" * 59)
            for i, (name, mean, scale) in enumerate(zip(rc_names, rc_means, rc_scales)):
                shape = (
                    f"{rc_shapes[i]:>12.5f}"
                    if rc_shapes is not None
                    else f"{'--':>12s}"
                )
                lines.append(f"  {name:<20s} {mean:>12.5f} {scale:>12.5f} {shape}")

            matrix_name_width = max(
                12, max((len(name) for name in rc_names), default=12)
            )

            def _append_rc_matrix(title: str, matrix: NDArray) -> None:
                lines.append("")
                lines.append(f"  {title}")
                lines.append(
                    f"  {'':<{matrix_name_width}s} "
                    + " ".join(f"{name:>11s}" for name in rc_names)
                )
                for i, name in enumerate(rc_names):
                    values = [
                        f"{matrix[i, j]:>11.4f}"
                        if np.isfinite(matrix[i, j])
                        else f"{'--':>11s}"
                        for j in range(len(rc_names))
                    ]
                    lines.append(
                        f"  {name:<{matrix_name_width}s} " + " ".join(values)
                    )

            _append_rc_matrix("Random-coefficient correlation matrix", rc_corr)
            _append_rc_matrix("Implied random-coefficient covariance matrix", rc_cov)

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
# n_iterations → n_iter, ll_total → loglik * n_obs).  Attached after class
# construction so ``@dataclass`` does not treat them as fields.  Shared
# mechanism: ``pybhatlib.models._results_common``.
# ----------------------------------------------------------------------
attach_deprecated_aliases(MixMNLResults, _MIXMNLRESULTS_LEGACY_KWARGS)
attach_ll_total_alias(MixMNLResults)
