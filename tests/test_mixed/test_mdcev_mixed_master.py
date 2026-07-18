"""MASTER FD gate for the mixed / panel MDCEV engine (plan Phase 3, G4).

Self-contained (no GAUSS oracle): on a small synthetic MDCEV **panel** with
``nrndcoef = 2`` (one log-normal + one Yeo-Johnson random baseline-utility
coefficient) and a **non-zero** MDCEV kernel scale, the engine's analytic
per-individual score summed over individuals must match a central finite
difference of the summed simulated log-likelihood, block by block, to ``1e-5``.

Because the MDCEV logit likelihood is *exact* (not an OVUS approximation) the
gate is tight. It exercises every gradient path of the shared engine under
``score_convention = "divide"`` (GAUSS ``gcomp ./ Pobs``):

* ``beta``  -- the utility path ``dlogp_dV`` through ``xmunew`` and the sign /
  log / YJ reparameterization chain;
* ``gamma`` -- the kernel-owned satiation gradient (``dlogp_dkparams`` gamma
  columns), striped into the physical ``gamma`` layout block;
* ``kern``  -- the MDCEV kernel-error scale gradient (``dlogp_dkparams`` last
  column);
* ``scal`` / ``lam`` -- the random-coefficient scale and Yeo-Johnson power
  reparameterizations (Path A only; MDCEV has no copula, ``dlogp_drc = 0``);
* ``rcor``  -- the random-coefficient correlation Cholesky chain.

The outside-good satiation ``gamma[0]`` is inert (its design loads only the
outside good, whose satiation is forced to ``-1000``), so its analytic gradient
and FD are both ~0 -- checked, but not required to be load-bearing.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel
from pybhatlib.vecup._panel import PanelIndex

_MASTER_TOL = 1e-5
NC = 5            # outside + 4 inside goods
NVARM = 4         # baseline-utility parameters
NVARGAM = 4       # translation (gamma) parameters


def _build_estimator(seed: int):
    rng = np.random.default_rng(seed)
    n_ind = 6
    occ = rng.integers(1, 4, size=n_ind)             # 1..3 occasions/person
    n_obs = int(occ.sum())
    person_ids = np.repeat(np.arange(n_ind), occ)

    var_names = [f"x{i}" for i in range(NVARM)]
    # x0 log-normal (must be strict-sign) + x1 Yeo-Johnson.
    spec = MixingSpec.from_var_names(
        var_names=var_names, logvar=("x0",), yjvar=("x1",),
        varneg=("x0",), nvargam=NVARGAM,
    )
    assert spec.nrndcoef == 2 and spec.nrndlog == 1 and spec.nrndyj == 1

    layout = ParamLayout(
        n_beta=spec.n_beta, n_gamma=NVARGAM, n_rcor=spec.nrndtcor,
        n_kern=1, n_scal=spec.nscale, n_lam=spec.numlam,
        kern_before_scal=True,
    )

    # baseline design tensor (n_obs, nc, nvarm)
    X = 0.5 * rng.standard_normal((n_obs, NC, NVARM))

    # satiation design (n_obs, nc, nvargam); outside-good row + param 0 inert.
    gamma_design = np.zeros((n_obs, NC, NVARGAM), dtype=np.float64)
    gamma_design[:, 1:, :] = rng.uniform(0.3, 1.2, size=(n_obs, NC - 1, NVARGAM))
    gamma_design[:, :, 0] = 0.0

    # consumption: outside always consumed; a mix of inside goods.
    consumption = np.zeros((n_obs, NC), dtype=np.float64)
    consumption[:, 0] = rng.uniform(1.0, 4.0, size=n_obs)
    for q in range(n_obs):
        k_consumed = rng.integers(1, NC - 1)
        picks = rng.choice(np.arange(1, NC), size=k_consumed, replace=False)
        consumption[q, picks] = rng.uniform(0.5, 3.0, size=k_consumed)
    price = np.tile(rng.uniform(0.8, 1.5, size=NC), (n_obs, 1))
    price[:, 0] = 1.0

    obs = SimpleNamespace(
        consumption=consumption, price=price, gamma_design=gamma_design
    )
    design = DesignData(X=X, obs=obs)

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = LogitJacobianKernel(
        NC, NVARGAM, control=MDCEVControl(utility="trad")
    )
    panel = PanelIndex.from_ids(person_ids)
    cfg = MSLConfig(
        n_rep=4, floor_pcomp=0.0, floor_z=0.0, score_convention="divide"
    )
    ass2d = rng.normal(size=(cfg.n_rep, n_ind * spec.nrndcoef))
    draws = FixtureDrawSource(ass2d)

    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    # non-trivial theta exercising every block; gamma[0] left inert (-1000).
    theta = np.zeros(layout.n_theta, dtype=np.float64)
    sl = layout.slices()
    theta[sl["beta"]] = rng.normal(scale=0.4, size=layout.n_beta)
    gamma = np.zeros(NVARGAM)
    gamma[0] = -1000.0
    gamma[1:] = rng.normal(scale=0.4, size=NVARGAM - 1)
    theta[sl["gamma"]] = gamma
    theta[sl["rcor"]] = rng.normal(scale=0.5, size=layout.n_rcor)
    theta[sl["kern"]] = np.array([rng.normal(scale=0.3)])   # nonzero kernel scale
    theta[sl["scal"]] = rng.normal(scale=0.3, size=layout.n_scal)
    theta[sl["lam"]] = rng.normal(scale=0.4, size=layout.n_lam)
    return est, layout, theta


def _fd_grad(est, theta, *, eps=1e-6):
    g = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        lp, _ = est.simulated_loglik(tp, want_grad=False)
        lm, _ = est.simulated_loglik(tm, want_grad=False)
        g[j] = (lp.sum() - lm.sum()) / (2.0 * eps)
    return g


@pytest.mark.parametrize("seed", [20260717, 909090])
def test_mdcev_mixed_master_fd(seed):
    est, layout, theta = _build_estimator(seed)

    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)

    sl = layout.slices()
    for blk in ("beta", "gamma", "rcor", "kern", "scal", "lam"):
        s = sl[blk]
        if s.stop - s.start == 0:
            continue
        worst = float(np.max(np.abs(g_an[s] - g_fd[s])))
        assert np.allclose(g_an[s], g_fd[s], atol=_MASTER_TOL, rtol=0.0), (
            f"[seed={seed}] block {blk!r} analytic!=FD (max|Δ|={worst:.2e})"
        )

    # sanity: the non-inert blocks are genuinely load-bearing (not 0==0 passes).
    for blk in ("beta", "rcor", "kern", "scal", "lam"):
        assert float(np.max(np.abs(g_fd[sl[blk]]))) > 1e-4, (
            f"block {blk!r} FD is degenerate (~0), not a real gate"
        )
