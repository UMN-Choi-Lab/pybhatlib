"""Analytical conditional-mean forecasts for linear regression."""

from __future__ import annotations

from numpy.typing import ArrayLike, NDArray

from pybhatlib.models.lr._lr_loglik import _validate_design
from pybhatlib.models.lr._lr_results import LRResults


def lr_predict(results: LRResults, X_new: ArrayLike) -> NDArray:
    """Predict continuous outcomes for new observations.

    Parameters
    ----------
    results : LRResults
        Fitted results, or :meth:`LRResults.from_estimates` output.
    X_new : array_like, shape (N, K)
        Design matrix in the same column order as ``results.params``,
        including the constant column if one was estimated.

    Returns
    -------
    y_hat : ndarray, shape (N,)
        Predicted conditional means ``X_new @ params``.
    """
    return _validate_design(X_new, len(results.params)) @ results.params
