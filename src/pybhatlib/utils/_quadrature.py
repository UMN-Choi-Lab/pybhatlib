"""Gauss-Hermite quadrature nodes and weights.

Port of the GAUSS ``intherabs`` procedure (Bhat, ``vecup.src`` /
``Intherabs.src``).  The original stores only the positive half of the
symmetric node/weight table and reconstructs the full set via
``e = (-e) | rev(e)`` and ``w = w | rev(w)``.  ``scipy.special.roots_hermite``
already returns the complete symmetric set, so this wrapper simply returns it
(after validating the requested order against the BHATLIB-supported list).

The weights satisfy ``sum(w) == sqrt(pi)``, matching the physicists'
Gauss-Hermite convention ``INT exp(-t^2) f(t) dt ~= sum_i w_i f(e_i)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import roots_hermite

# Integration orders supported by the GAUSS ``intherabs`` tables.
SUPPORTED_ORDERS: tuple[int, ...] = (
    2, 4, 6, 10, 20, 30, 40, 50, 60, 76, 92, 104, 120, 136,
)


def gauss_hermite(intord: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the full symmetric Gauss-Hermite nodes and weights.

    Mirrors GAUSS ``intherabs(intord)``: the returned set is symmetric about
    zero and the weights sum to ``sqrt(pi)``.

    Parameters
    ----------
    intord : int
        Order (number of nodes) of the quadrature.  Must be one of
        ``{2, 4, 6, 10, 20, 30, 40, 50, 60, 76, 92, 104, 120, 136}``.

    Returns
    -------
    nodes : ndarray, shape (intord,)
        Quadrature abscissae, symmetric about zero.
    weights : ndarray, shape (intord,)
        Quadrature weights; ``weights.sum() == sqrt(pi)``.

    Raises
    ------
    ValueError
        If ``intord`` is not one of the BHATLIB-supported orders.
    """
    if intord not in SUPPORTED_ORDERS:
        raise ValueError(
            f"intord={intord!r} not supported; use one of {SUPPORTED_ORDERS}."
        )
    nodes, weights = roots_hermite(intord)
    return np.asarray(nodes, dtype=np.float64), np.asarray(weights, dtype=np.float64)
