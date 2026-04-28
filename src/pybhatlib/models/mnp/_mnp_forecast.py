"""MNP model prediction and forecasting."""

from __future__ import annotations

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


def mnp_predict(
    results: MNPResults,
    X_new: NDArray,
    avail_new: NDArray | None = None,
) -> NDArray:
    """Predict choice probabilities for new observations.

    Parameters
    ----------
    results : MNPResults
        Fitted model results.
    X_new : ndarray, shape (N, n_alts, n_vars)
        Design matrix for new observations.
    avail_new : ndarray, shape (N, n_alts), optional
        Availability for new observations.

    Returns
    -------
    probs : ndarray, shape (N, n_alts)
        Predicted choice probabilities.
    """
    xp = get_backend("numpy")

    X_np = np.asarray(X_new, dtype=np.float64)
    N = X_np.shape[0]
    I = X_np.shape[1]
    n_vars = X_np.shape[2]

    theta_hat = np.asarray(results.b, dtype=np.float64)
    control = results.control

    ranvar_indices = getattr(results, "ranvar_indices", None)
    params = _unpack_params(theta_hat, n_vars, I, control, ranvar_indices)
    beta = params["beta"]
    Lambda = _build_lambda(params.get("lambda_params"), I, control)
    Omega_L = None
    if control is not None and control.mix and params.get("omega_params") is not None:
        Omega_L = _build_omega_cholesky(params["omega_params"], ranvar_indices, control)

    probs = np.zeros((N, I), dtype=np.float64)

    for q in range(N):
        avail_q = avail_new[q] if avail_new is not None else np.ones(I)
        for i in range(I):
            if avail_q[i] < 0.5:
                continue
            probs[q, i] = _compute_choice_prob(
                X_np[q], beta, i, avail_q, Lambda, Omega_L,
                ranvar_indices, control, xp,
            )

        row_sum = probs[q].sum()
        if row_sum > 0:
            probs[q] /= row_sum

    return probs


def mnp_predict_choice(
    results: MNPResults,
    X_new: NDArray,
    avail_new: NDArray | None = None,
) -> NDArray:
    """Predict most likely choice for new observations.

    Parameters
    ----------
    results : MNPResults
        Fitted model results.
    X_new : ndarray, shape (N, n_alts, n_vars)
        Design matrix.
    avail_new : ndarray, optional
        Availability matrix.

    Returns
    -------
    choices : ndarray, shape (N,)
        Predicted choice index (0-based) for each observation.
    """
    probs = mnp_predict(results, X_new, avail_new)
    return np.argmax(probs, axis=1)
