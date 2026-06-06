"""Reporting-space transform for MORP results.

The MORP log-likelihood is optimised in a *parameterised* threshold space
that guarantees ordering: for each dimension the first threshold is free and
each subsequent one is the previous plus ``exp(delta)``::

    t_0 = p_0
    t_j = t_{j-1} + exp(p_j)   for j >= 1

The raw optimiser parameters ``p`` (the ``tau_*`` slots) are therefore *not*
the threshold values for j >= 1 — ``p_j`` is a log-increment. GAUSS BHATLIB
reports the actual cut-points ``t_j`` with their standard errors, t-stats, and
the log-likelihood gradient. This module maps the raw estimate vector, its
covariance, and the objective gradient into that reporting space via the
delta method, leaving the beta and covariance blocks untouched.

See ``anna0605/email.md`` (UTA report, 2026-06): "the threshold values printed
at the end are correct ... would you be able to adjust the output so that the
threshold values are the ones printed in the table as opposed to the tau
values? Also ... add a gradient column similar to the GAUSS output."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr as _ndtr

from pybhatlib.models.morp._morp_control import MORPControl

__all__ = ["MORPReportTable", "reporting_jacobian", "build_morp_report"]


@dataclass
class MORPReportTable:
    """GAUSS-style reporting table (thresholds replace the raw tau slots).

    All arrays are length ``n_params`` and aligned with ``names``.

    Attributes
    ----------
    names : list[str]
        Display labels. Threshold rows are ``thresh_<dep>_<j>``; beta and
        covariance rows keep their raw names.
    estimate : NDArray
        Reporting-space point estimates (cumulative thresholds in the
        threshold blocks, raw values elsewhere).
    se : NDArray
        Delta-method standard errors. ``NaN`` where the covariance was
        unavailable.
    t_stat, p_value : NDArray
        ``estimate / se`` and the two-sided normal p-value.
    gradient : NDArray
        Gradient of the *mean* log-likelihood with respect to each reporting
        parameter, at the converged estimate (a convergence diagnostic — all
        entries near zero at a clean optimum).
    """

    names: list[str]
    estimate: NDArray
    se: NDArray
    t_stat: NDArray
    p_value: NDArray
    gradient: NDArray


def reporting_jacobian(
    theta: NDArray,
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
) -> NDArray:
    """Jacobian ``dr/dtheta`` of the reporting transform ``r = g(theta)``.

    Identity on the beta and covariance blocks; on each threshold block the
    cumulative-sum-of-exponentials map ``t_0 = p_0``,
    ``t_j = t_{j-1} + exp(p_j)``. The result is block lower-triangular with a
    strictly positive determinant, hence invertible.

    Parameters
    ----------
    theta : NDArray
        Raw (optimiser-space) parameter vector.
    n_beta, n_dims, n_categories, control
        Model dimensions, used to locate the threshold blocks.

    Returns
    -------
    G : NDArray, shape (n_params, n_params)
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.shape[0]
    G = np.eye(n, dtype=np.float64)

    idx = n_beta
    for d in range(n_dims):
        m = n_categories[d] - 1
        if m <= 0:
            continue
        p = theta[idx : idx + m]
        # Row j (threshold t_j): dt_j/dp_0 = 1; dt_j/dp_k = exp(p_k) for
        # 1 <= k <= j; 0 for k > j.
        for j in range(m):
            G[idx + j, idx] = 1.0
            for k in range(1, j + 1):
                G[idx + j, idx + k] = np.exp(p[k])
        idx += m

    return G


def _cumulative_thresholds(p: NDArray) -> NDArray:
    """``t_0 = p_0``, ``t_j = t_{j-1} + exp(p_j)`` for one dimension block."""
    m = p.shape[0]
    t = np.empty(m, dtype=np.float64)
    if m == 0:
        return t
    t[0] = p[0]
    for j in range(1, m):
        t[j] = t[j - 1] + np.exp(p[j])
    return t


def build_morp_report(
    theta: NDArray,
    cov: NDArray | None,
    grad_nll_mean: NDArray | None,
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
    param_names_raw: list[str],
    dep_vars: list[str],
) -> MORPReportTable:
    """Build the GAUSS-style reporting table from raw MLE quantities.

    Parameters
    ----------
    theta : NDArray
        Raw optimiser-space MLE.
    cov : NDArray or None
        Covariance of ``theta`` under the primary SE method (``None`` if it
        could not be computed — SEs become ``NaN``).
    grad_nll_mean : NDArray or None
        Gradient of the *mean negative* log-likelihood at ``theta`` (what the
        optimiser returns). Mapped to the gradient of the mean *positive*
        log-likelihood in reporting space.
    n_beta, n_dims, n_categories, control
        Model structure.
    param_names_raw : list[str]
        Raw parameter names (beta / tau / scale / corr). Threshold rows are
        relabelled; all others are passed through.
    dep_vars : list[str]
        Outcome names, used for the ``thresh_<dep>_<j>`` labels.

    Returns
    -------
    MORPReportTable
    """
    theta = np.asarray(theta, dtype=np.float64)
    n = theta.shape[0]

    G = reporting_jacobian(theta, n_beta, n_dims, n_categories, control)

    estimate = theta.copy()
    names = list(param_names_raw)
    idx = n_beta
    for d in range(n_dims):
        m = n_categories[d] - 1
        if m <= 0:
            continue
        estimate[idx : idx + m] = _cumulative_thresholds(theta[idx : idx + m])
        for j in range(m):
            names[idx + j] = f"thresh_{dep_vars[d]}_{j + 1}"
        idx += m

    # Delta-method SEs: Cov(r) = G Cov(theta) G^T.
    if cov is not None:
        report_cov = G @ np.asarray(cov, dtype=np.float64) @ G.T
        se = np.sqrt(np.maximum(np.diag(report_cov), 0.0))
    else:
        se = np.full(n, np.nan, dtype=np.float64)

    # Gradient in reporting space. With r = g(theta), the chain rule gives
    # grad_r L = (G^{-1})^T grad_theta L, i.e. solve(G^T, grad_theta L). We
    # report the gradient of the mean *log-likelihood* (= -grad of mean NLL),
    # matching GAUSS's sign/scale convention.
    if grad_nll_mean is not None:
        grad_ll_theta = -np.asarray(grad_nll_mean, dtype=np.float64)
        try:
            gradient = np.linalg.solve(G.T, grad_ll_theta)
        except np.linalg.LinAlgError:
            gradient = np.full(n, np.nan, dtype=np.float64)
    else:
        gradient = np.full(n, np.nan, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, estimate / se, 0.0)
        p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

    return MORPReportTable(
        names=names,
        estimate=estimate,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        gradient=gradient,
    )
