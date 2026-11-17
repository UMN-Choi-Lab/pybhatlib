"""Shared log-space satiation and Jacobian terms for both MDCEV engines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


def _log_satiation_terms(
    consumption: NDArray, price: NDArray, u: NDArray, *, utility: str,
) -> dict:
    """Compute draw-invariant terms from already mapped satiation indices.

    ``u`` includes the fixed outside-good index in column zero. Traditional
    utility includes that good in the Jacobian; linear utility uses inside
    goods only. The outside good is always marked consumed for the traditional
    Jacobian, while inside goods with zero consumption are excluded.

    No density floors, observation weights, or sigma transformations are
    applied here. Those conventions belong to the respective likelihood
    callers. Log quantities remain finite for large finite satiation indices
    without constructing ``exp(u)`` or products of Jacobian factors.
    """
    qty = np.asarray(consumption, dtype=np.float64)
    price = np.asarray(price, dtype=np.float64)
    if not np.all(np.isfinite(qty)) or not np.all(np.isfinite(price)):
        raise ValueError("MDCEV consumption and prices must be finite")
    if np.any(qty < 0) or np.any(price <= 0):
        raise ValueError("MDCEV requires nonnegative consumption and positive prices")
    if utility == "trad" and np.any(qty[:, 0] <= 0):
        raise ValueError("Traditional MDCEV requires positive outside consumption")

    log_qty = np.full_like(qty, -np.inf)
    np.log(qty, out=log_qty, where=qty > 0)
    log_total = np.logaddexp(log_qty, u)
    log_price = np.log(price)
    consumed = qty > 0
    consumed[:, 0] = True
    con = consumed[:, 1:]
    frac_qty = np.exp(log_qty[:, 1:] - log_total[:, 1:])
    frac_gamma = np.exp(u[:, 1:] - log_total[:, 1:])
    if utility == "trad":
        terms = np.where(consumed, log_price + log_total, -np.inf)
        log_sum = logsumexp(terms, axis=1)
        log_c2 = -log_price[:, 0] - np.where(consumed, log_total, 0.0).sum(axis=1)
        log_c = log_c2 + log_sum
        shares = np.exp(terms - log_sum[:, None])
        jac_gamma = (shares[:, 1:] - con) * frac_gamma
    else:
        log_c2 = None
        log_c = -np.where(
            con, log_total[:, 1:] + log_price[:, 1:], 0.0,
        ).sum(axis=1)
        jac_gamma = -(con * frac_gamma)
    return dict(
        con=con, log_c=log_c, log_c2=log_c2, log_total=log_total,
        adjustment=log_total[:, 1:] - u[:, 1:],
        log_price=log_price, log_qty_out=log_qty[:, 0],
        frac_qty=frac_qty, jac_gamma=jac_gamma,
    )
