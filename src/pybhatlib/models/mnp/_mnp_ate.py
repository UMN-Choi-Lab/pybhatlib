"""Average Treatment Effect (ATE) post-estimation for MNP.

Implements the mnpATEFit procedure from BHATLIB for computing predicted
mode shares and Average Treatment Effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd
from pybhatlib.models.mnp._mnp_loglik import (
    _build_lambda,
    _build_omega_cholesky,
    _unpack_params,
)
from pybhatlib.models.mnp._mnp_results import MNPResults


@dataclass
class ATEResult:
    """Average Treatment Effect analysis results.

    Attributes
    ----------
    n_obs : int
        Number of observations.
    predicted_shares : NDArray
        Mean predicted probability for each alternative.
    base_shares : NDArray or None
        Predicted shares at base level.
    treatment_shares : NDArray or None
        Predicted shares at treatment level.
    pct_ate : NDArray or None
        Percentage ATE = (treatment - base) / base * 100.
    """

    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None = None
    treatment_shares: NDArray | None = None
    pct_ate: NDArray | None = None


def mnp_ate(
    results: MNPResults,
    X: NDArray | None = None,
    avail: NDArray | None = None,
    changevar: str | None = None,
    changeval: float | None = None,
    data: "pd.DataFrame | None" = None,
    alternatives: list[str] | None = None,
    spec: dict | None = None,
) -> ATEResult:
    """Compute predicted shares and ATE for MNP model.

    When changevar and changeval are provided, all observations have the
    specified variable set to changeval, and predicted shares are computed.
    This enables ATE analysis by comparing base vs treatment levels.

    When changevar is None, computes overall predicted shares using the
    actual data values.

    Parameters
    ----------
    results : MNPResults
        Fitted MNP model results.
    X : ndarray, shape (N, n_alts, n_vars), optional
        Design matrix. If None, reconstructed from data and spec.
    avail : ndarray, shape (N, n_alts), optional
        Availability matrix.
    changevar : str or None
        Variable name to modify for ATE analysis.
    changeval : float or None
        Value to set changevar to.
    data : pd.DataFrame, optional
        Data (needed if X not provided and changevar is set).
    alternatives : list of str, optional
        Alternative names.
    spec : dict, optional
        Variable specification.

    Returns
    -------
    result : ATEResult
    """
    xp = get_backend("numpy")

    theta_hat = results.b
    control = results.control
    n_alts = len(results.param_names)  # approximate

    # Determine dimensions from results
    n_beta = len([n for n in results.param_names
                  if not n.startswith(("scale", "parker", "CovCOv", "segunpar", "param"))])

    # If X not provided, we need data and spec to reconstruct
    if X is None:
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "Either X must be provided, or data+spec+alternatives for reconstruction"
            )
        from pybhatlib.io._spec_parser import parse_spec
        X, _ = parse_spec(spec, data, alternatives, nseg=control.nseg if control else 1)

        if changevar is not None and changeval is not None:
            # Modify the data column and reparse
            data_modified = data.copy()
            if changevar in data_modified.columns:
                data_modified[changevar] = changeval
            X, _ = parse_spec(spec, data_modified, alternatives,
                              nseg=control.nseg if control else 1)

    N = X.shape[0]
    I = X.shape[1]
    n_vars = X.shape[2]

    # Extract parameters
    beta = theta_hat[:n_vars]

    # Build Lambda
    ranvar_indices = None
    if control and control.mix:
        # Try to infer ranvar_indices
        pass

    if control is None:
        control_use = type('MNPControl', (), {'iid': True, 'mix': False, 'heteronly': False,
                                               'randdiag': False, 'nseg': 1})()
    else:
        control_use = control

    params = _unpack_params(theta_hat, n_vars, I, control_use, ranvar_indices)
    Lambda = _build_lambda(params.get("lambda_params"), I, control_use)

    # Compute predicted probabilities for each observation
    pred_probs = np.zeros((N, I), dtype=np.float64)

    for q in range(N):
        V_q = X[q] @ beta  # (I,)
        avail_q = avail[q] if avail is not None else np.ones(I)

        for i in range(I):
            if avail_q[i] < 0.5:
                pred_probs[q, i] = 0.0
                continue

            # P(choose i) = P(V_i + eps_i > V_j + eps_j for all j != i)
            avail_alts = [j for j in range(I) if j != i and avail_q[j] > 0.5]
            dim = len(avail_alts)

            if dim == 0:
                pred_probs[q, i] = 1.0
                continue

            diff_V = np.array([V_q[i] - V_q[j] for j in avail_alts])

            if control_use.iid:
                Lambda_diff = 2.0 * np.eye(dim, dtype=np.float64)
            else:
                M = np.zeros((dim, I), dtype=np.float64)
                for k, j in enumerate(avail_alts):
                    M[k, j] = 1.0
                    M[k, i] = -1.0
                Lambda_full = np.eye(I, dtype=np.float64)
                Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)
                Lambda_diff = M @ Lambda_full @ M.T
                Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

            eigvals = np.linalg.eigvalsh(Lambda_diff)
            if eigvals.min() < 1e-10:
                Lambda_diff += np.eye(dim) * (1e-10 - eigvals.min())

            method = control_use.method if hasattr(control_use, 'method') else "ovus"
            prob = mvncd(xp.array(diff_V), xp.array(Lambda_diff), method=method, xp=xp)
            pred_probs[q, i] = max(prob, 1e-300)

        # Normalize
        row_sum = pred_probs[q].sum()
        if row_sum > 0:
            pred_probs[q] /= row_sum

    # Mean predicted shares
    predicted_shares = pred_probs.mean(axis=0)

    return ATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
    )
