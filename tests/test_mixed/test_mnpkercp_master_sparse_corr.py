"""Master finite-difference gate for the **sparse** joint-correlation score.

Self-contained (no GAUSS oracle). With the rc<->kernel copula **off**, the
joint correlation is block-sparse: only random-coefficient pairs (unless
``randdiag``) and kernel-error pairs (unless ``iid``) are free, and
``MixingSpec.active_corr_pairs`` lists exactly those. The engine scores that
case through a dedicated branch (kernel block chained through its own radial
parameterization, RC block through the ``x11chol`` utility path) instead of the
full-``nrndtot`` chain used by the copula gate in ``test_mnpkercp_master.py``.

This gate asserts, for every ``(randdiag, iid)`` combination that leaves at
least one active pair, that the analytic per-individual score summed over
individuals matches a central finite difference of the summed simulated
log-likelihood on **all** blocks (``beta``/``rcor``/``scal``/``kern``/``lam``)
to ``1e-4`` (the OVUS-approximation tolerance), and that the ``rcor`` block is
genuinely load-bearing.
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


def _build_estimator(seed: int, *, randdiag: bool, iid: bool):
    """Synthetic ``nc=3`` MNP panel, copula off, sparse joint correlation."""
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
        randdiag=randdiag, copula=False, iid=iid,
    )
    # mirrors MNPKerCPModel._build_spec_layout
    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=len(spec.active_corr_pairs), n_scal=spec.nscale,
        n_lam=spec.numlam, n_kern=0 if iid else spec.n_kern, kern_before_lam=True,
    )

    X = rng.normal(size=(n_obs, nc, nvar)) * 0.7
    chosen = np.zeros((n_obs, nc))
    chosen[np.arange(n_obs), rng.integers(0, nc, n_obs)] = 1.0
    avail = np.ones((n_obs, nc))
    design = DesignData(X=X, obs=SimpleNamespace(avail=avail, chosen=chosen))

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = MvncdKernel(
        nc, spec.nrndcoef, copula=False, scal=1.0, iid=iid,
        active_corr_pairs=spec.active_corr_pairs,
    )
    panel = PanelIndex.from_ids(person_ids)
    cfg = MSLConfig(n_rep=4, floor_pcomp=1e-4, floor_z=1e-4, score_convention="mask")

    ass2d = rng.normal(size=(cfg.n_rep, n_ind * spec.nrndcoef))
    draws = FixtureDrawSource(ass2d)

    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    theta = np.zeros(layout.n_theta)
    sl = layout.slices()
    for blk, scale in (("beta", 0.4), ("rcor", 0.5), ("scal", 0.3),
                       ("kern", 0.4), ("lam", 0.4)):
        s = sl[blk]
        theta[s] = rng.normal(scale=scale, size=s.stop - s.start)
    return est, layout, spec, theta


def _fd_grad(est, theta, *, eps=1e-6):
    g = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        lp, _ = est.simulated_loglik(tp, want_grad=False)
        lm, _ = est.simulated_loglik(tm, want_grad=False)
        g[j] = (lp.sum() - lm.sum()) / (2.0 * eps)
    return g


@pytest.mark.parametrize("randdiag,iid,expect_pairs", [
    (False, False, {(0, 1), (2, 3)}),   # RC pair + kernel pair, no cross pairs
    (True, False, {(2, 3)}),            # kernel pair only
    (False, True, {(0, 1)}),            # RC pair only
])
@pytest.mark.parametrize("seed", [20260908, 424242])
def test_sparse_corr_active_pairs_and_master_fd(seed, randdiag, iid, expect_pairs):
    est, layout, spec, theta = _build_estimator(seed, randdiag=randdiag, iid=iid)

    assert set(spec.active_corr_pairs) == expect_pairs
    assert layout.n_rcor == len(expect_pairs)

    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)

    sl = layout.slices()
    for blk in ("beta", "rcor", "scal", "kern", "lam"):
        s = sl[blk]
        if s.stop - s.start == 0:
            continue
        worst = float(np.max(np.abs(g_an[s] - g_fd[s])))
        assert np.allclose(g_an[s], g_fd[s], atol=_MASTER_TOL, rtol=0.0), (
            f"[seed={seed} randdiag={randdiag} iid={iid}] block {blk!r} "
            f"analytic!=FD (max|Δ|={worst:.2e})"
        )

    # the sparse rcor block must be load-bearing, not a zero==zero pass
    assert float(np.max(np.abs(g_fd[sl["rcor"]]))) > 1e-3
