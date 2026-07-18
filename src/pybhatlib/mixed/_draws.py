"""DrawSource strategy for the shared mixed/panel MSL engine.

The MSL engine consumes standard-normal random draws (covariance ``I``) with a
**fixed output contract** so that runtime draws and GAUSS-verification fixtures
are swappable at construction time only -- the engine, pipeline, and kernels are
byte-identical under either source (Plan §2.4):

    draws(n_ind, n_rnd, n_rep, *, run=1) -> ndarray, shape (n_rep, n_ind, n_rnd)

Correlation between random coefficients is imposed downstream in the RC pipeline
(``errbeta3 = x11chol' @ errbeta2``), never here: every draw source emits
independent standard normals (``cov = I``), so the draw source is
model-independent.

Panel reshape (matches GAUSS exactly)
-------------------------------------
GAUSS stores the post-``createhalt`` draw block as::

    ass = (reshape(ass', nii*nrndcoef, nrep))'          # (nrep, nii*nrndcoef)

then, per replication ``r``, expands one person-by-coefficient slice::

    errbeta1temp = reshape(ass[r,.], nrndcoef, nii)'    # (nii, nrndcoef)

GAUSS ``reshape`` is row-major, so ``reshape(row, nrndcoef, nii)`` fills a
``(nrndcoef, nii)`` matrix row-by-row and the trailing ``'`` transposes it to
``(nii, nrndcoef)``. :func:`panel_reshape_gauss` reproduces this for the whole
``(nrep, nii*nrndcoef)`` block at once, returning ``(n_rep, n_ind, n_rnd)`` so
that ``draws()[r]`` is directly GAUSS ``errbeta1temp`` and
``PanelIndex.broadcast(draws()[r])`` is ``Dmask @ errbeta1temp``.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


# ---------------------------------------------------------------------------
# Panel reshape helper  (GAUSS ass -> (n_rep, n_ind, n_rnd))
# ---------------------------------------------------------------------------

def panel_reshape_gauss(
    ass2d: ArrayLike, n_ind: int, n_rnd: int, *, xp=None
) -> NDArray:
    """Reshape a stored GAUSS ``ass`` block to the DrawSource contract.

    Reproduces the GAUSS per-replication expansion
    ``errbeta1temp = reshape(ass[r,.], nrndcoef, nii)'`` for every replication
    at once.

    Parameters
    ----------
    ass2d : array-like, shape (n_rep, n_ind * n_rnd)
        The stored GAUSS draw block, exactly as written by
        ``ass = (reshape(ass', nii*nrndcoef, nrep))'`` (one row per
        replication, ``nii*nrndcoef`` columns laid out coefficient-major then
        individual-minor, i.e. row-major ``(nrndcoef, nii)``).
    n_ind : int
        Number of individuals ``nii``.
    n_rnd : int
        Number of random coefficients ``nrndcoef``.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    draws : ndarray, shape (n_rep, n_ind, n_rnd)
        ``draws[r]`` equals GAUSS ``errbeta1temp`` for replication ``r``:
        ``draws[r, i, k] = ass2d[r, k * n_ind + i]``.

    Raises
    ------
    ValueError
        If ``ass2d`` is not 2-D or its column count is not ``n_ind * n_rnd``.
    """
    ass2d = np.asarray(ass2d, dtype=np.float64)
    if ass2d.ndim != 2:
        raise ValueError(
            f"ass2d must be 2-D (n_rep, n_ind*n_rnd), got shape {ass2d.shape}"
        )
    n_rep, ncol = ass2d.shape
    if ncol != n_ind * n_rnd:
        raise ValueError(
            f"ass2d has {ncol} columns but n_ind*n_rnd = {n_ind * n_rnd} "
            f"(n_ind={n_ind}, n_rnd={n_rnd})"
        )
    # reshape(ass[r,.], nrndcoef, nii)' == row-major (n_rnd, n_ind) then swap.
    out = ass2d.reshape(n_rep, n_rnd, n_ind).transpose(0, 2, 1)
    out = np.ascontiguousarray(out)
    if xp is not None:
        out = xp.array(out, dtype=xp.float64)
    return out


# ---------------------------------------------------------------------------
# DrawSource protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DrawSource(Protocol):
    """Strategy producing standard-normal MSL draws (covariance ``I``).

    Implementations must honour the fixed output contract so that runtime and
    fixture draw sources are interchangeable at construction time only.
    """

    def draws(
        self, n_ind: int, n_rnd: int, n_rep: int, *, run: int = 1
    ) -> NDArray:
        """Return standard-normal draws, shape ``(n_rep, n_ind, n_rnd)``.

        Parameters
        ----------
        n_ind : int
            Number of individuals ``nii``.
        n_rnd : int
            Number of random coefficients ``nrndcoef``.
        n_rep : int
            Number of MSL replications ``nrep``.
        run : int, default 1
            1-indexed run number (GAUSS ``runno``). Draw sources tied to a
            single dump ignore it; scipy-generated sources offset the scramble.

        Returns
        -------
        draws : ndarray, shape (n_rep, n_ind, n_rnd)
            Independent standard normals (``cov = I``); ``draws[r]`` is the
            per-replication ``(n_ind, n_rnd)`` slice (GAUSS ``errbeta1temp``).
        """
        ...


# ---------------------------------------------------------------------------
# Fixture draw source  (GAUSS-authoritative, bit-exact for tests)
# ---------------------------------------------------------------------------

class FixtureDrawSource:
    """Return a stored post-``createhalt`` GAUSS ``ass`` block verbatim.

    The authoritative draw source for GAUSS-parity fixtures: it feeds the
    engine the identical ``ass`` GAUSS consumed, reshaped to the DrawSource
    contract via :func:`panel_reshape_gauss`. It **deliberately bypasses**
    ``createhalt`` so that engine parity and draw-generation parity remain two
    independent gates -- an engine test never fails because of a
    ``createhalt``-port bug.

    Parameters
    ----------
    ass : array-like or str or os.PathLike
        Either the stored GAUSS ``ass`` block as an array of shape
        ``(n_rep, n_ind * n_rnd)``, or a path to a CSV file holding it (one
        row per replication, comma-delimited, no header).

    Notes
    -----
    The block is stored 2-D and reshaped on each :meth:`draws` call using the
    ``n_ind``/``n_rnd`` supplied there, matching GAUSS's per-replication
    ``reshape(ass[r,.], nrndcoef, nii)'``.
    """

    def __init__(self, ass: ArrayLike | str | os.PathLike) -> None:
        if isinstance(ass, (str, os.PathLike)):
            arr = np.loadtxt(ass, delimiter=",", dtype=np.float64, ndmin=2)
        else:
            arr = np.asarray(ass, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(
                    "ass array must be 2-D (n_rep, n_ind*n_rnd), got shape "
                    f"{arr.shape}"
                )
        self._ass2d: NDArray = arr

    def draws(
        self, n_ind: int, n_rnd: int, n_rep: int, *, run: int = 1
    ) -> NDArray:
        """Return the stored ``ass`` reshaped to ``(n_rep, n_ind, n_rnd)``.

        Parameters
        ----------
        n_ind, n_rnd, n_rep : int
            Contract dimensions; must match the stored block's shape.
        run : int, default 1
            Ignored (a fixture dump is tied to a single GAUSS ``runno``).

        Raises
        ------
        ValueError
            If ``n_rep`` or ``n_ind * n_rnd`` disagree with the stored block.
        """
        del run  # a fixture dump is a single run
        if self._ass2d.shape[0] != n_rep:
            raise ValueError(
                f"stored ass has {self._ass2d.shape[0]} rows but n_rep={n_rep}"
            )
        return panel_reshape_gauss(self._ass2d, n_ind, n_rnd)


# ---------------------------------------------------------------------------
# Scipy scrambled-Halton draw source  (runtime default)
# ---------------------------------------------------------------------------

class ScipyHaltonDrawSource:
    """Runtime-default draw source: scipy scrambled Halton + inverse CDF.

    Uses ``scipy.stats.qmc.Halton(d=n_rnd, scramble=True, seed=...)`` per call,
    maps the uniforms to standard normals via ``scipy.special.ndtri``, and
    reshapes the ``n_rep * n_ind`` points (replication-major, individual-minor)
    to the DrawSource contract ``(n_rep, n_ind, n_rnd)``.

    The sampler is constructed fresh on every :meth:`draws` call from the
    per-call seed, so this **never mutates any global RNG** and is safe for
    parallel gradient evaluation.

    Parameters
    ----------
    seed : int or None, default None
        Base scramble seed. ``None`` is resolved to a concrete random integer
        **once at construction** (via ``np.random.SeedSequence``), so every
        :meth:`draws` call on this source returns identical draws -- the MSL
        requirement that draws stay fixed across optimizer iterations, and that
        an ATE baseline/scenario share draws so Monte-Carlo noise cancels. The
        resolved draws are still cov-``I`` standard normals but are *not*
        reproducible across separate processes/runs; pass an explicit integer
        seed for cross-run reproducibility. The effective seed for run ``r`` is
        ``seed + (run - 1)`` so distinct runs give distinct draws while
        ``run=1`` reproduces ``seed`` exactly.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.
    """

    def __init__(self, seed: int | None = None, *, xp=None) -> None:
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        self.seed = seed
        self._xp = xp

    def draws(
        self, n_ind: int, n_rnd: int, n_rep: int, *, run: int = 1
    ) -> NDArray:
        """Return scipy scrambled-Halton standard normals.

        Parameters
        ----------
        n_ind, n_rnd, n_rep : int
            Contract dimensions.
        run : int, default 1
            1-indexed run number; offsets the scramble seed by ``run - 1``.

        Returns
        -------
        draws : ndarray, shape (n_rep, n_ind, n_rnd)
            Standard-normal draws (``cov = I``).
        """
        from scipy.special import ndtri
        from scipy.stats import qmc

        if run < 1:
            raise ValueError(f"run must be >= 1, got {run}")

        eff_seed = None if self.seed is None else int(self.seed) + (run - 1)
        sampler = qmc.Halton(d=n_rnd, scramble=True, seed=eff_seed)
        u = sampler.random(n_rep * n_ind)                     # (n_rep*n_ind, n_rnd)
        u = np.clip(u, 1e-10, 1.0 - 1e-10)
        z = ndtri(u).reshape(n_rep, n_ind, n_rnd)
        z = np.ascontiguousarray(z, dtype=np.float64)
        if self._xp is not None:
            z = self._xp.array(z, dtype=self._xp.float64)
        return z


__all__ = [
    "DrawSource",
    "FixtureDrawSource",
    "ScipyHaltonDrawSource",
    "panel_reshape_gauss",
]
