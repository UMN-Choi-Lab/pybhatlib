"""Master finite-difference gate for the MNP engine with **active correlation**.

Self-contained (no GAUSS oracle). On a small synthetic MNP panel with
``copula = True``, ``nrndcoef = 2`` (one normal + one Yeo-Johnson), **nonzero
``rcor`` correlation parameters** and **nonzero random-coefficient scale**, this
gate asserts the engine's analytic per-individual score summed over individuals
equals a central finite difference of the summed simulated log-likelihood --
including the joint-correlation (``rcor``) block, which is the piece this task
wires through.

Why this is the crux
--------------------
The ``rcor`` score funnels the total ``d lnP / d omegastar`` (the full
``nrndtot = nrndcoef + (nc - 1)`` joint correlation) from three sources -- the
kernel's direct dependence (``dlogp_domega``), the utility path routed through
``f1rand``'s Cholesky (``df1randdx11chol``), and the copula path
``dlogp_drc = d lnP / d errbeta3`` routed through ``errbeta3 = errbeta1 @
x11chol`` (``gerrbeta3dx11chol``) -- then chains once via
``gnewcholparmcorscaled`` to the ``rcor`` parameters. Any mis-wiring of those
three contributions (or the sub-block <-> full-``omegastar`` embedding) shows up
as a ``rcor``-block FD mismatch here. The gate therefore checks the **full**
score vector (all five blocks ``beta``/``rcor``/``scal``/``kern``/``lam``) to
``1e-4`` (the OVUS-approximation tolerance) and asserts the ``rcor`` block is
genuinely load-bearing (FD non-degenerate).
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
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel
from pybhatlib.vecup._panel import PanelIndex

_MASTER_TOL = 1e-4


def _build_estimator(seed: int):
    """Synthetic ``nc=3`` MNP copula panel with all reparam blocks active."""
    rng = np.random.default_rng(seed)
    nc = 3
    nvar = 4
    n_ind = 6
    occ = rng.integers(1, 4, size=n_ind)                 # 1..3 occasions/person
    n_obs = int(occ.sum())
    person_ids = np.repeat(np.arange(n_ind), occ)

    var_names = [f"x{i}" for i in range(nvar)]
    spec = MixingSpec.from_var_names(
        var_names=var_names, normvar=("x0",), yjvar=("x1",), kernel_dim=nc - 1,
    )
    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=spec.nrndtcor, n_scal=spec.nscale,
        n_lam=spec.numlam, n_kern=spec.n_kern, kern_before_lam=True,
    )

    X = rng.normal(size=(n_obs, nc, nvar)) * 0.7
    chosen = np.zeros((n_obs, nc))
    chosen[np.arange(n_obs), rng.integers(0, nc, n_obs)] = 1.0
    avail = np.ones((n_obs, nc))
    design = DesignData(X=X, obs=SimpleNamespace(avail=avail, chosen=chosen))

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = MvncdKernel(nc, spec.nrndcoef, copula=True, scal=1.0)
    panel = PanelIndex.from_ids(person_ids)
    cfg = MSLConfig(n_rep=4, floor_pcomp=1e-4, floor_z=1e-4, score_convention="mask")

    ass2d = rng.normal(size=(cfg.n_rep, n_ind * spec.nrndcoef))
    draws = FixtureDrawSource(ass2d)

    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    # nonzero correlation (rcor) AND nonzero rc scale -> every block active.
    theta = np.zeros(layout.n_theta)
    sl = layout.slices()
    theta[sl["beta"]] = rng.normal(scale=0.4, size=layout.n_beta)
    theta[sl["rcor"]] = rng.normal(scale=0.5, size=layout.n_rcor)      # active corr
    theta[sl["scal"]] = rng.normal(scale=0.3, size=layout.n_scal)      # active scale
    theta[sl["kern"]] = rng.normal(scale=0.4, size=layout.n_kern)
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


@pytest.mark.parametrize("seed", [20260716, 424242, 13579])
def test_mnpkercp_master_corr_fd(seed):
    est, layout, theta = _build_estimator(seed)

    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)

    sl = layout.slices()

    # rcor must be load-bearing: FD is non-degenerate (not a zero==zero pass).
    assert float(np.max(np.abs(g_fd[sl["rcor"]]))) > 1e-3

    # FULL score, including the now-nonzero rcor block, matches FD to 1e-4.
    for blk in ("beta", "rcor", "scal", "kern", "lam"):
        s = sl[blk]
        if s.stop - s.start == 0:
            continue
        worst = float(np.max(np.abs(g_an[s] - g_fd[s])))
        assert np.allclose(g_an[s], g_fd[s], atol=_MASTER_TOL, rtol=0.0), (
            f"[seed={seed}] block {blk!r} analytic!=FD (max|Δ|={worst:.2e})"
        )

    # the assembled rcor block is itself non-zero (the wiring fills it, not 0).
    assert float(np.max(np.abs(g_an[sl["rcor"]]))) > 1e-3
