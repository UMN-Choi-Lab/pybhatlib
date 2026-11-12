"""
Explicit Halton sequence generator
with digit/radical-inverse algorithm and
Bratten-Weller scrambling table.

Provides:
    gethalton(...)            -- direct port of the GAUSS function
    gethalton_default(...)    -- direct port with default settings
    ExplicitHaltonDrawSource     -- wraps gethalton() as a pybhatlib DrawSource
                                 (drop-in replacement for ScipyHaltonDrawSource)

Determinism note
-----------------
corbreak=0 (unscrambled) and corbreak=1 (Bratten-Weller) with rantuff=0 and
rancolumn=0 -- the library's stated default, via gethalton_default() -- are
FULLY DETERMINISTIC: no random-number generator is used anywhere in that
path, so results are bit-reproducible run to run, matching the deterministic
digit-permutation algorithm in the original HALTON.SRC.

corbreak=2 (random within-column scrambling), rantuff=1 (Tuffin shift), and
rancolumn=1 (random column/dimension assignment) all rely on native Generator, 
seeded off the same seed formula as the original (offset by runno). 
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

# ---------------------------------------------------------------------------
# 1. Prime list and digit-count constant 
# ---------------------------------------------------------------------------
_HALT_PRIM: NDArray = np.array(
    [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
     67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
     157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223],
    dtype=np.int64,
)
_HALT_MAXDIGIT = 22


def _halt_make_xiden() -> NDArray:
    """Identity (no-scrambling) digit-permutation table, shape (48, 223).

    Row m holds the digits 0, 1, ..., prim[m]-1 (zero-padded past that).
    """
    xiden = np.zeros((48, 223), dtype=np.int64)
    for i in range(48):
        p = int(_HALT_PRIM[i])
        xiden[i, :p] = np.arange(p)
    return xiden


# ---------------------------------------------------------------------------
# 2. Bratten-Weller scrambled digit-permutation table (16 x 53),
# ---------------------------------------------------------------------------
_XBRAT: NDArray = np.array([
    [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,3,1,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,4,2,6,1,5,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,5,8,2,10,3,6,1,9,7,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,6,10,2,8,4,12,1,9,5,11,3,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,8,13,3,11,5,16,1,10,7,14,4,12,2,15,6,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,9,14,3,17,6,11,1,15,7,12,4,18,8,2,16,10,5,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,11,17,4,20,7,13,2,22,9,15,5,18,1,14,10,21,6,16,3,19,8,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,15,7,24,11,20,2,27,9,18,4,22,13,26,5,16,10,23,1,19,28,6,14,17,3,25,12,8,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,15,23,5,27,9,18,2,29,12,20,7,25,11,17,3,30,14,22,1,21,8,26,10,16,28,4,19,6,24,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,18,28,6,23,11,34,3,25,14,31,8,20,36,1,16,27,10,22,13,32,4,29,17,7,35,19,2,26,12,30,9,24,15,33,5,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,20,31,7,26,12,38,3,23,34,14,17,29,5,40,10,24,1,35,18,28,9,33,15,21,4,37,13,30,8,39,22,2,27,16,32,11,25,6,36,19,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,21,32,7,38,13,25,3,35,17,28,10,41,5,23,30,15,37,1,19,33,11,26,42,8,18,29,4,39,14,22,34,6,24,12,40,2,31,20,27,9,36,16,0,0,0,0,0,0,0,0,0,0],
    [0,24,12,39,6,33,20,44,3,29,16,36,10,42,22,8,31,26,14,46,1,35,18,28,5,40,19,37,11,25,43,4,30,15,34,9,45,21,2,32,17,41,13,27,7,38,23,0,0,0,0,0,0],
    [0,26,40,9,33,16,49,4,36,21,45,12,29,6,51,23,38,14,43,1,30,19,47,10,34,24,42,3,27,52,15,18,39,7,46,31,11,35,20,48,2,28,41,8,22,50,13,32,17,44,5,37,25],
], dtype=np.int64)


# ---------------------------------------------------------------------------
# 3. Core Halton engine: vectorized scrambled radical inverse.
# ---------------------------------------------------------------------------
def _halt_rawgen(n: int, s: int, tbl: NDArray) -> NDArray:
    """Generate `n` raw (scrambled) Halton points in `s` dimensions, (n, s).

    Vectorized radical inverse of the 1-based point indices 1..n: digits of
    the index in base ``prim[m]`` are mapped through ``tbl[m]`` and summed
    least-significant first with weights ``prim[m]**-(l+1)``. Summation order
    and float operations match the original carry-counter loop, so the output
    is bit-identical. (Every row of ``tbl`` maps digit 0 to 0, so digits
    beyond the index's length contribute exactly zero.)
    """
    primer = _HALT_PRIM[:s]
    maxdigit = max(_HALT_MAXDIGIT, int(n).bit_length())
    h = np.zeros((n, s), dtype=np.float64)
    for m in range(s):
        c = int(primer[m])
        idx = np.arange(1, n + 1, dtype=np.int64)
        xx = float(c)
        row = tbl[m]
        acc = h[:, m]
        for _ in range(maxdigit):
            if not idx.any():
                break
            acc += row[idx % c] / xx
            idx //= c
            xx *= c
    return h


# ---------------------------------------------------------------------------
# 4. Master function: gethalton() 
# ---------------------------------------------------------------------------
def gethalton(
    nrep: int,
    nobs: int,
    ndim: int,
    mu: NDArray,
    cov: NDArray,
    runno: int,
    normal: int = 1,
    corbreak: int = 1,
    rantuff: int = 0,
    rancolumn: int = 0,
    rng: np.random.Generator | None = None,
) -> NDArray:
    """Direct port of GAUSS `gethalton`.

    Returns an (nobs*nrep) x ndim matrix: the first `nrep` rows correspond
    to the first observation, the next `nrep` rows to the second
    observation, and so on (same convention as the original).
    """
    if ndim > 48:
        raise ValueError("gethalton: ndim cannot exceed 48 (only 48 primes are tabulated).")
    if corbreak == 1 and ndim > 16:
        raise ValueError("gethalton: Bratten-Weller scrambling (corbreak=1) only supports ndim <= 16.")

    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64)

    if corbreak == 1:
        tbl = _XBRAT[:ndim, :]
    else:
        tbl = _halt_make_xiden()[:ndim, :]

    burnin = 10
    raw_full = _halt_rawgen(nrep * nobs + burnin, ndim, tbl)
    raw = raw_full[burnin: burnin + nrep * nobs, :]
    ass = raw

    needs_rng = corbreak == 2 or rantuff == 1 or rancolumn == 1
    if needs_rng and rng is None:
        rng = np.random.default_rng(9_836_506 + (runno - 1) * 40_000)

    if corbreak == 2:
        ass = ass.copy()
        for i in range(nobs):
            block = slice(i * nrep, (i + 1) * nrep)
            for j in range(ndim):
                perm = rng.permutation(nrep)
                ass[block, j] = raw[block, j][perm]

    if rantuff:
        shift = rng.uniform(size=ndim)
        ass = ass + shift
        ass = ass - np.floor(ass)

    if rancolumn == 1:
        seqq = rng.permutation(ndim)
    else:
        seqq = np.arange(ndim)
    ass = ass[:, seqq]

    if normal == 0:
        return ass
    else:
        z = ndtri(ass)                 # (nobs*nrep, ndim) standard normals
        L = np.linalg.cholesky(cov).T  # GAUSS chol() is upper-triangular; L.T @ L = cov
        return mu[None, :] + z @ L


def gethalton_default(nrep: int, nobs: int, ndim: int, mu: NDArray, cov: NDArray, runno: int) -> NDArray:
    """Direct port of GAUSS `gethaltonDefault` (normal=1, corbreak=1, rantuff=0, rancolumn=0)."""
    return gethalton(nrep, nobs, ndim, mu, cov, runno, normal=1, corbreak=1, rantuff=0, rancolumn=0)


# ---------------------------------------------------------------------------
# 5. DrawSource wrapper -- drop-in replacement for ScipyHaltonDrawSource.
# ---------------------------------------------------------------------------
class ExplicitHaltonDrawSource:
    """Each individual gets a contiguous block of `n_rep` points from its own
    position in the scrambled Halton sequence (matching GAUSS's
    "first nrep rows = individual 1" layout), reshaped to the DrawSource
    contract `(n_rep, n_ind, n_rnd)`.

    Determinism: with the default ``corbreak=1, rantuff=0, rancolumn=0`` the
    draws are fully deterministic -- identical for every run and every
    ``draw_seed``. The randomized options (``corbreak=2``, ``rantuff=1``,
    ``rancolumn=1``) seed from the GAUSS formula
    ``9_836_506 + (run - 1) * 40_000``, NOT from ``draw_seed``. ``seed`` only
    affects the SciPy fallback used beyond the dimension limit.

    Parameters
    ----------
    corbreak, rantuff, rancolumn : int, default 1/0/0
        Passed straight through to `gethalton` 
    seed : int or None
        Used only by the SciPy fallback when the dimension limit is exceeded.
        The fallback is retained so repeated likelihood evaluations use fixed draws.
    """

    def __init__(self, *, corbreak: int = 1, rantuff: int = 0, rancolumn: int = 0,
                 seed: int | None = None) -> None:
        self.corbreak = corbreak
        self.rantuff = rantuff
        self.rancolumn = rancolumn
        self.seed = seed
        self._fallback = None

    def draws(self, n_ind: int, n_rnd: int, n_rep: int, *, run: int = 1) -> NDArray:
        if run < 1:
            raise ValueError(f"run must be >= 1, got {run}")

        limit = len(_XBRAT) if self.corbreak == 1 else len(_HALT_PRIM)
        if n_rnd > limit:
            if self._fallback is None:
                from pybhatlib.mixed._draws import ScipyHaltonDrawSource
                warnings.warn(
                    f"Explicit Halton supports at most {limit} dimensions with "
                    f"corbreak={self.corbreak}; requested {n_rnd}. "
                    "Reverting to ScipyHaltonDrawSource.",
                    stacklevel=2,
                )
                self._fallback = ScipyHaltonDrawSource(seed=self.seed)
            return self._fallback.draws(n_ind, n_rnd, n_rep, run=run)

        if run > 1 and self.corbreak != 2 and not (self.rantuff or self.rancolumn):
            warnings.warn(
                f"Explicit Halton draws are deterministic with corbreak="
                f"{self.corbreak}, rantuff=0, rancolumn=0; run={run} reuses "
                "exactly the same draws as run 1.",
                stacklevel=2,
            )

        rng = None
        if self.corbreak == 2 or self.rantuff == 1 or self.rancolumn == 1:
            rng = np.random.default_rng(9_836_506 + (run - 1) * 40_000)

        ass_flat = gethalton(
            n_rep, n_ind, n_rnd,
            mu=np.zeros(n_rnd), cov=np.eye(n_rnd),
            runno=run, normal=1,
            corbreak=self.corbreak, rantuff=self.rantuff, rancolumn=self.rancolumn,
            rng=rng,
        )  # (n_ind * n_rep, n_rnd), individual-major blocks of n_rep rows

        # individual-major (n_ind, n_rep, n_rnd) -> replication-major (n_rep, n_ind, n_rnd)
        z = ass_flat.reshape(n_ind, n_rep, n_rnd).transpose(1, 0, 2)
        return np.ascontiguousarray(z, dtype=np.float64)


__all__ = ["gethalton", "gethalton_default", "ExplicitHaltonDrawSource"]
