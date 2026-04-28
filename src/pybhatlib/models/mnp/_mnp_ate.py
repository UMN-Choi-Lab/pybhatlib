"""Average Treatment Effect (ATE) post-estimation for MNP.

Implements the mnpATEFit procedure from BHATLIB for computing predicted
mode shares and Average Treatment Effects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.models.mnp._mnp_loglik import (
    _build_lambda,
    _build_omega_cholesky,
    _compute_choice_prob,
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

    theta_hat = np.asarray(results.params, dtype=np.float64)
    control = results.control

    if X is None:
        if data is None or spec is None or alternatives is None:
            raise ValueError(
                "Either X must be provided, or data+spec+alternatives for reconstruction"
            )
        from pybhatlib.io._spec_parser import parse_spec
        X, _ = parse_spec(spec, data, alternatives, nseg=1)

        if changevar is not None and changeval is not None:
            data_modified = data.copy()
            if changevar in data_modified.columns:
                data_modified[changevar] = changeval
            X, _ = parse_spec(spec, data_modified, alternatives, nseg=1)

    X_np = np.asarray(X, dtype=np.float64)
    N = X_np.shape[0]
    I = X_np.shape[1]
    n_vars = X_np.shape[2]

    ranvar_indices = getattr(results, "ranvar_indices", None)
    params = _unpack_params(theta_hat, n_vars, I, control, ranvar_indices)
    beta = params["beta"]
    Lambda = _build_lambda(params.get("lambda_params"), I, control)
    Omega_L = None
    if control is not None and control.mix and params.get("omega_params") is not None:
        Omega_L = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    pred_probs = np.zeros((N, I), dtype=np.float64)

    for q in range(N):
        avail_q = avail[q] if avail is not None else np.ones(I)
        for i in range(I):
            if avail_q[i] < 0.5:
                continue
            pred_probs[q, i] = _compute_choice_prob(
                X_np[q], beta, i, avail_q, Lambda, Omega_L,
                ranvar_indices, control, xp,
            )

        row_sum = pred_probs[q].sum()
        if row_sum > 0:
            pred_probs[q] /= row_sum

    predicted_shares = pred_probs.mean(axis=0)

    return ATEResult(
        n_obs=N,
        predicted_shares=predicted_shares,
    )
