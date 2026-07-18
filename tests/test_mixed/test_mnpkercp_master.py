"""Master finite-difference gate for the mixed-panel MNP (MNPKerCP) engine.

Self-contained (no GAUSS oracle): on a small synthetic MNP panel with
``nrndcoef = 2`` (one normal + one Yeo-Johnson), ``copula = True``, we check that
the engine's analytic per-individual score summed over individuals matches a
central finite difference of the summed simulated log-likelihood.

Purpose (plan 2.3 / master gate)
--------------------------------
This validates that the copula ``dlogp_drc`` gradient (the kernel error's
sensitivity to the drawn random coefficients) is correctly chained through the
shared random-coefficient reparameterization into the engine score.  Because the
copula draw enters the scale / Yeo-Johnson chain, the ``dlogp_drc`` term appears
in the ``scal`` and ``lam`` blocks (Path B, ``df1_B``) in addition to the
``beta`` block (Path A) and the kernel-scale ``kern`` block; the gate asserts FD
agreement on all four to ``1e-4`` (the OVUS-approximation tolerance).

Joint-correlation block: ``rcor``
---------------------------------
The **joint correlation** analytic gradient is now chained (plan P1.2): the
MVNCD kernel emits ``dlogp_domega`` (its direct dependence on the joint
correlation via the conditional covariance ``xi2subq`` and mean ``B1subq``), and
the engine adds the two ``x11chol``-routed contributions (the utility path via
``df1randdx11chol`` and the copula path ``dlogp_drc`` via ``gerrbeta3dx11chol``)
before chaining the total once through ``gnewcholparmcorscaled`` to the ``rcor``
parameters. This test therefore FD-checks **all five** blocks (``beta``,
``rcor``, ``scal``, ``kern``, ``lam``) to ``1e-4``.
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

    # non-trivial theta exercising corr / scale / kernel-scale / YJ lambda
    theta = np.zeros(layout.n_theta)
    sl = layout.slices()
    theta[sl["beta"]] = rng.normal(scale=0.4, size=layout.n_beta)
    theta[sl["rcor"]] = rng.normal(scale=0.5, size=layout.n_rcor)
    theta[sl["scal"]] = rng.normal(scale=0.3, size=layout.n_scal)
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


@pytest.mark.parametrize("seed", [20260716, 424242])
def test_mnpkercp_master_fd(seed):
    est, layout, theta = _build_estimator(seed)

    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)

    sl = layout.slices()
    # --- all five blocks: analytic must match FD to 1e-4 -------------------- #
    for blk in ("beta", "rcor", "scal", "kern", "lam"):
        s = sl[blk]
        if s.stop - s.start == 0:
            continue
        worst = float(np.max(np.abs(g_an[s] - g_fd[s])))
        assert np.allclose(g_an[s], g_fd[s], atol=_MASTER_TOL, rtol=0.0), (
            f"[seed={seed}] block {blk!r} analytic!=FD (max|Δ|={worst:.2e})"
        )

    # sanity: the joint-correlation block is genuinely load-bearing (the LL
    # depends non-trivially on rcor), i.e. the FD match is not a degenerate
    # zero==zero pass.
    assert float(np.max(np.abs(g_fd[sl["rcor"]]))) > 1e-3
