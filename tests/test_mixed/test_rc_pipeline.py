"""Finite-difference gate for the random-coefficient realization pipeline.

Covers ``mixed/_rc_pipeline.py`` (T0.11). On synthetic observations with
``nrndcoef = 3`` (two normal + one Yeo-Johnson coefficient), under both the
additive and the multiplicative injection masks, every analytic Jacobian block
returned by :meth:`RandomCoefPipeline.realize` is checked against central finite
differences of the corresponding ``realize`` output to ``1e-5``:

  * ``dxmunewdxmu``      -- d xmunew / d xmu
  * ``dxmunewdf1rand``   -- d xmunew / d f1rand
  * ``df1randdwdiagrand``-- d f1rand / d wscalrand
  * ``df1randdxlamrnd``  -- d f1rand / d xlamrnd (through meanyj mu/sig)
  * ``df1randdx11chol``  -- d f1rand / d (correlation-Cholesky off-diagonals)
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.vecup._yj import meanyj, standyjinvnonpgiven

INTORDN1 = 20
SCAL = 1.0
EPS = 1e-6
TOL = 1e-5


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------

def _base_spec() -> MixingSpec:
    """nrndcoef = 3: two normal ('v0','v1') + one Yeo-Johnson ('v3')."""
    return MixingSpec.from_var_names(
        var_names=["v0", "v1", "v2", "v3", "v4"],
        normvar=["v0", "v1"],
        yjvar=["v3"],
    )


def _multiplicative_spec(spec: MixingSpec) -> MixingSpec:
    """Force a mixed additive/multiplicative injection mask.

    The natural MIXMNL spec with no log coefficients is purely additive. To
    exercise the ``xmunew2`` (multiplicative) path we override the injection
    masks so that some random-coefficient positions inject multiplicatively.
    """
    nvarm = spec.n_beta
    negmask = np.zeros(nvarm, dtype=np.float64)
    # mixpos = [0, 1, 3]: make positions 1 and 3 multiplicative, 0 additive.
    negmask[1] = 1.0
    negmask[3] = 1.0
    nonegmask = 1.0 - negmask
    return dataclasses.replace(
        spec,
        indxvarnegposlog=negmask,
        indxvarnonegposlog=nonegmask,
    )


def _layout(spec: MixingSpec) -> ParamLayout:
    return ParamLayout(
        n_beta=spec.n_beta,
        n_rcor=spec.nrndtcor,
        n_scal=spec.nscale,
        n_lam=spec.numlam,
        n_kern=0,
    )


def _theta(spec: MixingSpec, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta = rng.uniform(-1.0, 1.0, size=spec.n_beta)
    rcor = rng.uniform(-1.0, 1.0, size=spec.nrndtcor)
    scal = rng.uniform(-0.5, 0.5, size=spec.nscale)
    lam = rng.uniform(-0.8, 0.8, size=spec.numlam)
    return np.concatenate([beta, rcor, scal, lam])


def _errbeta1(K: int, Q: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 999)
    return rng.normal(size=(Q, K))


# ---------------------------------------------------------------------------
# correlation-Cholesky reconstruction helpers (for the chol FD)
# ---------------------------------------------------------------------------

def _strict_upper(U: np.ndarray, K: int) -> np.ndarray:
    """Off-diagonal upper elements in row-based order {(0,1),(0,2),(1,2),...}."""
    return np.array([U[i, j] for i in range(K) for j in range(i + 1, K)])


def _rebuild_corr_chol(S: np.ndarray, K: int) -> np.ndarray:
    """Upper-tri correlation Cholesky from off-diagonal elements (unit columns).

    Matches the ``cholcov=False`` structure assumed by ``ggradchol`` /
    ``gcholeskycor``: the diagonal is determined by unit column norm,
    ``U[j, j] = sqrt(1 - sum_{k<j} U[k, j]**2)``.
    """
    U = np.zeros((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            U[i, j] = S[idx]
            idx += 1
    for j in range(K):
        U[j, j] = np.sqrt(1.0 - np.sum(U[:j, j] ** 2))
    return U


# ---------------------------------------------------------------------------
# realize-output recomputation helpers (hold everything but one input fixed)
# ---------------------------------------------------------------------------

def _xmunew_from(xmu, f1rand, indxrndvar, nonegmask, negmask):
    B = f1rand @ indxrndvar.T
    xmunew1 = xmu[None, :] + B
    xmunew2 = xmu[None, :] * B
    return xmunew1 * nonegmask[None, :] + xmunew2 * negmask[None, :]


def _central(f, x0, i):
    """Central difference of vector-valued ``f`` w.r.t. scalar entry ``x0[i]``."""
    xp = np.array(x0, dtype=np.float64)
    xm = np.array(x0, dtype=np.float64)
    xp[i] += EPS
    xm[i] -= EPS
    return (f(xp) - f(xm)) / (2.0 * EPS)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mask", ["additive", "multiplicative"])
def test_rc_pipeline_jacobians_vs_fd(mask):
    spec = _base_spec()
    if mask == "multiplicative":
        spec = _multiplicative_spec(spec)

    K = spec.nrndcoef
    nvarm = spec.n_beta
    Q = 6
    layout = _layout(spec)
    pipe = RandomCoefPipeline(spec, layout, scal=SCAL, intordn1=INTORDN1)

    theta = _theta(spec, seed=7)
    errbeta1 = _errbeta1(K, Q, seed=7)

    rc = pipe.prepare(theta, want_grad=True)
    cache = pipe.realize(errbeta1, rc, want_grad=True)
    jac = cache.jac

    xmu = np.asarray(rc.xmu, dtype=np.float64)
    wscalrand = np.asarray(rc.wscalrand, dtype=np.float64)
    xlamrnd = np.asarray(rc.xlamrnd, dtype=np.float64)
    x11chol = np.asarray(rc.x11chol, dtype=np.float64)
    indxrndvar = np.asarray(spec.indxrndvar, dtype=np.float64)
    nonegmask = np.asarray(spec.indxvarnonegposlog, dtype=np.float64)
    negmask = np.asarray(spec.indxvarnegposlog, dtype=np.float64)
    f1rand = cache.f1rand

    # sanity: the reconstruction round-trips the correlation Cholesky.
    S0 = _strict_upper(x11chol, K)
    assert np.allclose(_rebuild_corr_chol(S0, K), x11chol, atol=1e-10)

    # --- dxmunewdxmu[q, v, w] = d xmunew[q, w] / d xmu[v] -----------------
    for v in range(nvarm):
        fd = _central(
            lambda z: _xmunew_from(z, f1rand, indxrndvar, nonegmask, negmask),
            xmu, v,
        )  # (Q, nvarm)
        assert np.allclose(jac.dxmunewdxmu[:, v, :], fd, atol=TOL), (
            f"dxmunewdxmu mismatch at v={v} ({mask})"
        )

    # --- dxmunewdf1rand[q, r, v] = d xmunew[q, v] / d f1rand[q, r] --------
    for r in range(K):
        def f(col_r, r=r):
            f1 = f1rand.copy()
            f1[:, r] = col_r
            return _xmunew_from(xmu, f1, indxrndvar, nonegmask, negmask)
        fd = (f(f1rand[:, r] + EPS) - f(f1rand[:, r] - EPS)) / (2.0 * EPS)
        assert np.allclose(jac.dxmunewdf1rand[:, r, :], fd, atol=TOL), (
            f"dxmunewdf1rand mismatch at r={r} ({mask})"
        )

    # --- df1randdwdiagrand[q, r, s] = d f1rand[q, s] / d wscalrand[r] -----
    ftemprand = cache.ftemprand
    for r in range(K):
        def f(w):
            return ftemprand * w[None, :]
        fd = _central(f, wscalrand, r)  # (Q, K)
        assert np.allclose(jac.df1randdwdiagrand[:, r, :], fd, atol=TOL), (
            f"df1randdwdiagrand mismatch at r={r} ({mask})"
        )

    # --- df1randdxlamrnd[q, r, s] = d f1rand[q, s] / d xlamrnd[r] ---------
    errbeta3 = cache.errbeta3
    for r in range(K):
        def f(lam):
            mu, sig = meanyj(lam, INTORDN1)
            ft = standyjinvnonpgiven(lam, mu, sig, errbeta3.T).T
            return ft * wscalrand[None, :]
        fd = _central(f, xlamrnd, r)  # (Q, K)
        assert np.allclose(jac.df1randdxlamrnd[:, r, :], fd, atol=TOL), (
            f"df1randdxlamrnd mismatch at r={r} ({mask})"
        )

    # --- df1randdx11chol[q, k, r] = d f1rand[q, r] / d chol_offdiag[k] ----
    ncorr = K * (K - 1) // 2
    for k in range(ncorr):
        def f(S):
            U = _rebuild_corr_chol(S, K)
            eb3 = errbeta1 @ U
            ft = standyjinvnonpgiven(
                xlamrnd, rc.mulamrnd, rc.siglamrnd, eb3.T
            ).T
            return ft * wscalrand[None, :]
        fd = _central(f, S0, k)  # (Q, K)
        assert np.allclose(jac.df1randdx11chol[:, k, :], fd, atol=TOL), (
            f"df1randdx11chol mismatch at k={k} ({mask})"
        )
