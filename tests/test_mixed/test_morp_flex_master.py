"""Master finite-difference gate for the full MORP engine + RectMvncdKernel + facade.

Self-contained (no GAUSS oracle). Builds a small synthetic MORP panel with the
copula active, a Yeo-Johnson kernel, one YJ random coefficient, ordered
thresholds, nonzero correlation and scale, and asserts the engine's analytic
per-individual score summed over individuals equals a central finite difference
of the summed simulated log-likelihood -- across ALL blocks
(thresh / beta / rcor / scal / lam / kernlam). This validates that the facade
wires the RectMvncdKernel + shared engine + copula seam correctly end-to-end.
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
from pybhatlib.models.morp_flex._morp_flex_kernel import RectMvncdKernel
from pybhatlib.vecup._panel import PanelIndex

_TOL = 1e-4

NORD = 2
N_CATEGORIES = (3, 4)
N_THRESH = sum(c - 1 for c in N_CATEGORIES)   # 5
NVAR = 3

# observed ordered categories per obs (low/middle/high coverage in both dims).
Y_ORD = np.array([[0, 3], [1, 1], [2, 0], [1, 2], [0, 1], [2, 3]], dtype=np.int64)


def _build_estimator(seed: int):
    rng = np.random.default_rng(seed)
    n_obs = Y_ORD.shape[0]
    # panel: 4 persons, split the 6 obs among them
    person_ids = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    n_ind = 4

    var_names = [f"x{i}" for i in range(NVAR)]
    spec = MixingSpec.from_var_names(
        var_names=var_names, yjvar=("x0",), nord=NORD, n_categories=N_CATEGORIES,
    )
    nrndcoef = spec.nrndcoef                        # 1
    n_kernlam = NORD                                # YJ kernel
    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=spec.nrndtcor, n_scal=spec.nscale,
        n_lam=spec.numlam, n_kern=0, kern_before_lam=True,
        n_thresh=N_THRESH, n_kernlam=n_kernlam,
    )

    X = rng.normal(size=(n_obs, NORD, NVAR)) * 0.6
    design = DesignData(X=X, obs=SimpleNamespace(y_ord=Y_ORD))

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = RectMvncdKernel(
        NORD, nrndcoef, N_CATEGORIES, copula=True, yj_kernel=True, scal=1.0,
    )
    panel = PanelIndex.from_ids(person_ids)
    cfg = MSLConfig(n_rep=4, floor_pcomp=1e-4, floor_z=1e-4, score_convention="mask")
    ass2d = rng.normal(size=(cfg.n_rep, n_ind * nrndcoef))
    draws = FixtureDrawSource(ass2d)

    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    theta = np.zeros(layout.n_theta)
    sl = layout.slices()
    theta[sl["thresh"]] = np.array([-0.5, 0.0, -1.0, 0.0, 0.0]) + rng.normal(scale=0.15, size=N_THRESH)
    theta[sl["beta"]] = rng.normal(scale=0.4, size=layout.n_beta)
    theta[sl["rcor"]] = rng.normal(scale=0.5, size=layout.n_rcor)
    theta[sl["scal"]] = rng.normal(scale=0.3, size=layout.n_scal)
    theta[sl["lam"]] = rng.normal(scale=0.4, size=layout.n_lam)
    theta[sl["kernlam"]] = rng.normal(scale=0.4, size=n_kernlam)
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
def test_morp_flex_master_fd(seed):
    est, layout, theta = _build_estimator(seed)
    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)
    sl = layout.slices()
    # every block load-bearing + FD-correct
    for name in ("thresh", "beta", "rcor", "scal", "lam", "kernlam"):
        block = sl[name]
        assert float(np.max(np.abs(g_fd[block]))) > 1e-4, f"{name} FD degenerate"
        assert np.allclose(g_an[block], g_fd[block], atol=_TOL, rtol=_TOL), (
            f"{name}: analytic {g_an[block]} vs FD {g_fd[block]}"
        )
