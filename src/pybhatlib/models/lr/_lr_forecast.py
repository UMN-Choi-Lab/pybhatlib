"""Analytical conditional-mean forecasts for linear regression."""

from ._lr_loglik import _validate_design


def lr_predict(results, X_new):
    """Predict continuous outcomes from an (N, K) matrix, including constants."""
    return _validate_design(X_new, len(results.params)) @ results.params
