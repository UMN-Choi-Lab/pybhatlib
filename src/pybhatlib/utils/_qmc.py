"""Quasi-Monte Carlo sequences for simulated likelihood estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc


def halton_sequence(n: int, d: int, seed: int | None = None) -> NDArray:
    """Generate Halton quasi-random sequence.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimensionality.
    seed : int or None
        Random seed for scrambling.

    Returns
    -------
    points : (n, d) array
        Halton sequence in [0, 1]^d.
    """
    sampler = qmc.Halton(d=d, scramble=True, seed=seed)
    return sampler.random(n)


def halton_normal(n: int, d: int, seed: int | None = None) -> NDArray:
    """Generate standard normal draws via Halton sequence + inverse CDF.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimensionality.
    seed : int or None
        Random seed for scrambling.

    Returns
    -------
    draws : (n, d) array
        Standard normal quasi-random draws.
    """
    from scipy.stats import norm
    u = halton_sequence(n, d, seed=seed)
    # Clip to avoid infinite values at boundaries
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    return norm.ppf(u)
