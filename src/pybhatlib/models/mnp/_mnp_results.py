"""MNP model results structure.

Equivalent to BHATLIB's mnpResults struct. Contains estimated parameters,
standard errors, test statistics, and model diagnostics.
"""

from __future__ import annotations

import warnings
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
    params : NDArray
        Raw optimized parameters (theta-space).  Used internally for
        prediction and forecasting.  Length equals ``n_params``.
        (Previously named ``b``; the old attribute is still available
        as a deprecated alias.)
    b_original : NDArray
        BHATLIB-normalized reporting coefficients (Sigma_diff[0,0]=1).
        Aligned with ``param_names``.  For non-IID models this has
        one fewer element than ``params`` (one scale absorbed).
    se : NDArray
        Delta-method standard errors (aligned with ``b_original``).
    t_stat : NDArray
        t-statistics (b_original / se).
    p_value : NDArray
        Two-sided p-values.
    gradient : NDArray
        Gradient projected into reporting space.
    loglik : float
        Mean log-likelihood (per observation).  (Previously named
        ``ll``; the old attribute is still available as a deprecated
        alias.  ``ll_total`` is also a deprecated alias of
        ``loglik`` even though they referred to different quantities
        previously — see Notes.)
    n_obs : int
        Number of observations.
    param_names : list[str]
        Parameter names.
    corr_matrix : NDArray
        Correlation matrix of parameter estimates.
    cov_matrix : NDArray
        Variance-covariance matrix of parameter estimates.
    n_iter : int
        Number of optimizer iterations.  (Previously named
        ``n_iterations``; the old attribute is still available as a
        deprecated alias.)
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

    Notes
    -----
    Field renames (canonical name → deprecated alias):

    - ``params`` ← ``b``
    - ``loglik`` ← ``ll`` (mean log-likelihood)
    - ``n_iter`` ← ``n_iterations``

    Historically ``ll`` exposed the *mean* and ``ll_total`` the *total*
    log-likelihood.  In the harmonized API only the *mean* is stored
    (as ``loglik``).  Reading ``MNPResults.ll_total`` still returns
    ``loglik * n_obs`` (preserving its original numerical meaning) but
    emits a ``DeprecationWarning``; downstream code should switch to
    ``loglik`` or compute the total explicitly.
    """

    params: NDArray
    b_original: NDArray
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
    lambda_hat: NDArray | None = None
    omega_hat: NDArray | None = None
    cholesky_L: NDArray | None = None
    segment_probs: NDArray | None = None
    segment_means: list[NDArray] | None = None
    segment_covs: list[NDArray] | None = None
    control: MNPControl | None = None
    data_path: str = ""
    ranvar_indices: list[int] | None = None

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
        lines.append(f"  Mean log-likelihood    {self.loglik:>14.6f}")
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
            est = self.b_original[i] if i < len(self.b_original) else self.params[i]
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
        lines.append(f"  Number of iterations   {self.n_iter:>10d}")
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


# ----------------------------------------------------------------------
# Deprecated property aliases
# ----------------------------------------------------------------------
#
# Adding deprecated aliases to a ``@dataclass`` requires attaching
# ``property`` descriptors after class construction (descriptors set on
# the class body would otherwise be picked up by ``@dataclass`` as
# fields).  Each alias forwards reads/writes to the canonical field and
# emits a ``DeprecationWarning``.


def _make_alias(old_name: str, new_name: str) -> property:
    """Return a ``property`` that aliases ``old_name`` → ``new_name``."""

    def _getter(self):
        warnings.warn(
            f"MNPResults.{old_name} is deprecated; use "
            f"MNPResults.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(self, new_name)

    def _setter(self, value):
        warnings.warn(
            f"MNPResults.{old_name} is deprecated; use "
            f"MNPResults.{new_name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        setattr(self, new_name, value)

    return property(_getter, _setter)


MNPResults.b = _make_alias("b", "params")
MNPResults.ll = _make_alias("ll", "loglik")
MNPResults.n_iterations = _make_alias("n_iterations", "n_iter")


def _ll_total_getter(self):
    warnings.warn(
        "MNPResults.ll_total is deprecated; use "
        "MNPResults.loglik (mean log-likelihood) or "
        "``loglik * n_obs`` for the total.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.loglik * self.n_obs


def _ll_total_setter(self, value):
    warnings.warn(
        "MNPResults.ll_total is deprecated; use "
        "MNPResults.loglik (mean log-likelihood) or "
        "``loglik * n_obs`` for the total.",
        DeprecationWarning,
        stacklevel=2,
    )
    if self.n_obs:
        self.loglik = value / self.n_obs
    else:
        self.loglik = float(value)


MNPResults.ll_total = property(_ll_total_getter, _ll_total_setter)
