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


def halton_normal(
    n: int, d: int, seed: int | None = None, *, xp=None
) -> NDArray:
    """Generate standard normal draws via Halton sequence + inverse CDF.

    A fresh scrambled Halton sampler is constructed on every call from the
    supplied ``seed`` (per-call seeding), so this function never mutates any
    global RNG state and is safe to call from parallel gradient evaluations.

    Parameters
    ----------
    n : int
        Number of points.
    d : int
        Dimensionality.
    seed : int or None
        Per-call random seed for scrambling. ``None`` draws a fresh
        (non-deterministic) scramble; an integer makes the draws reproducible.
    xp : backend, optional
        Array backend used to wrap the result (see
        :func:`pybhatlib.backend.get_backend`). Defaults to NumPy, in which
        case a plain :class:`numpy.ndarray` is returned (backward compatible).

    Returns
    -------
    draws : (n, d) array
        Standard normal quasi-random draws, in the requested backend.
    """
    from scipy.special import ndtri
    u = halton_sequence(n, d, seed=seed)
    # Clip to avoid infinite values at boundaries
    u = np.clip(u, 1e-10, 1.0 - 1e-10)
    z = ndtri(u)
    if xp is not None:
        z = xp.array(z, dtype=xp.float64)
    return z
