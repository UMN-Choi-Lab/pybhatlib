"""MNP model prediction and forecasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend
from pybhatlib.gradmvn._mvncd import mvncd
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

    theta_hat = results.b
    beta = theta_hat[:n_vars]

    control = results.control
    is_iid = control.iid if control else True

    # Build Lambda from results
    Lambda = results.lambda_hat if results.lambda_hat is not None else np.eye(I - 1)

    probs = np.zeros((N, I), dtype=np.float64)

    for q in range(N):
        V_q = X_np[q] @ beta
        avail_q = avail_new[q] if avail_new is not None else np.ones(I)

        for i in range(I):
            if avail_q[i] < 0.5:
                continue

            avail_alts = [j for j in range(I) if j != i and avail_q[j] > 0.5]
            dim = len(avail_alts)

            if dim == 0:
                probs[q, i] = 1.0
                continue

            diff_V = np.array([V_q[i] - V_q[j] for j in avail_alts])

            if is_iid:
                Lambda_diff = 2.0 * np.eye(dim)
            else:
                M = np.zeros((dim, I))
                for k, j in enumerate(avail_alts):
                    M[k, j] = 1.0
                    M[k, i] = -1.0
                Lambda_full = np.eye(I)
                Lambda_full[1:, 1:] = Lambda + np.eye(I - 1)
                Lambda_diff = M @ Lambda_full @ M.T
                Lambda_diff = 0.5 * (Lambda_diff + Lambda_diff.T)

            eigvals = np.linalg.eigvalsh(Lambda_diff)
            if eigvals.min() < 1e-10:
                Lambda_diff += np.eye(dim) * (1e-10 - eigvals.min())

            method = control.method if control and hasattr(control, 'method') else "ovus"
            prob = mvncd(xp.array(diff_V), xp.array(Lambda_diff), method=method, xp=xp)
            probs[q, i] = max(prob, 1e-300)

        # Normalize
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
