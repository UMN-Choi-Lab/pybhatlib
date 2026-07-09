"""MNL model prediction and forecasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.models.mnl._mnl_loglik import _compute_probabilities
from pybhatlib.models.mnl._mnl_results import MNLResults


def mnl_predict(
    results: MNLResults,
    X_new: NDArray,
    avail_new: NDArray | None = None,
) -> NDArray:
    """Predict choice probabilities for new observations.

    Parameters
    ----------
    results : MNLResults
        Fitted MNL model results.
    X_new : NDArray, shape (N, nc, numunord)
        Design matrix for new observations.  ``X_new[q, k, :]`` is the
        covariate vector for observation q and alternative k.
    avail_new : NDArray, shape (N, nc), optional
        Availability matrix.  Defaults to all ones (all alternatives
        available) if not provided.

    Returns
    -------
    probs : NDArray, shape (N, nc)
        Predicted choice probabilities.
    """
    x_opt    = results.params
    N        = X_new.shape[0]
    nc       = X_new.shape[1]
    numunord = X_new.shape[2]

    if avail_new is None:
        avail_new = np.ones((N, nc), dtype=np.float64)

    # Compute utilities and probabilities directly from design matrix
    v     = (X_new * x_opt[np.newaxis, np.newaxis, :]).sum(axis=2)  # (N, nc)
    p1    = np.exp(v) * avail_new
    denom = p1.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return p1 / denom


def mnl_predict_choice(
    results: MNLResults,
    X_new: NDArray,
    avail_new: NDArray | None = None,
) -> NDArray:
    """Predict most likely choice for new observations.

    Parameters
    ----------
    results : MNLResults
        Fitted MNL model results.
    X_new : NDArray, shape (N, nc, numunord)
        Design matrix for new observations.
    avail_new : NDArray, shape (N, nc), optional
        Availability matrix.

    Returns
    -------
    choices : NDArray, shape (N,)
        Predicted choice index (0-based) for each observation.
    """
    probs = mnl_predict(results, X_new, avail_new)
    return np.argmax(probs, axis=1)
