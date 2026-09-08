"""Master finite-difference gate for the full MORP engine + RectMvncdKernel + facade.

Self-contained (no GAUSS oracle). Builds a small synthetic MORP panel with a
Yeo-Johnson (or normal) kernel, one YJ random coefficient, ordered thresholds,
nonzero correlation and scale, and asserts the engine's analytic
per-individual score summed over individuals equals a central finite difference
of the summed simulated log-likelihood -- across ALL blocks
(thresh / beta / rcor / scal / lam / kernlam) -- with the rc<->kernel copula
both ON and OFF.

With the copula off (GAUSS ``_nocorrrcker = 1``) the joint correlation block
still carries the random-coefficient correlation and the ordinal-error
correlation; those columns must match FD, while the rc<->kernel slots are
held at zero by the engine (their analytic score is exactly zero, like GAUSS
``_max_active``) even though the likelihood depends on them through the joint
Cholesky parameterization.
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


def _build_estimator(seed: int, *, copula: bool = True, yj_kernel: bool = True):
    rng = np.random.default_rng(seed)
    n_obs = Y_ORD.shape[0]
    # panel: 4 persons, split the 6 obs among them
    person_ids = np.array([0, 0, 1, 1, 2, 3], dtype=np.int64)
    n_ind = 4

    var_names = [f"x{i}" for i in range(NVAR)]
    spec = MixingSpec.from_var_names(
        var_names=var_names, yjvar=("x0",), nord=NORD, n_categories=N_CATEGORIES,
        normker=not yj_kernel,
        copula=copula,   # declares which joint-correlation pairs are estimated
    )
    nrndcoef = spec.nrndcoef                        # 1
    n_kernlam = NORD if yj_kernel else 0
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
        NORD, nrndcoef, N_CATEGORIES, copula=copula, yj_kernel=yj_kernel, scal=1.0,
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
    if n_kernlam:
        theta[sl["kernlam"]] = rng.normal(scale=0.4, size=n_kernlam)
    if not copula:
        # copula off: the rc<->kernel slots are not estimated (held at zero).
        inactive = _inactive_rcor_columns(spec, layout)
        theta[sl["rcor"].start + np.asarray(inactive, dtype=int)] = 0.0
    return est, spec, layout, theta


def _inactive_rcor_columns(spec, layout) -> list[int]:
    nrndtot = spec.nrndtot
    full = [(p, q) for p in range(nrndtot) for q in range(p + 1, nrndtot)]
    assert layout.n_rcor == len(full)
    active = set(spec.active_corr_pairs)
    return [i for i, pair in enumerate(full) if pair not in active]


def _fd_grad(est, theta, *, eps=1e-6):
    g = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        lp, _ = est.simulated_loglik(tp, want_grad=False)
        lm, _ = est.simulated_loglik(tm, want_grad=False)
        g[j] = (lp.sum() - lm.sum()) / (2.0 * eps)
    return g


@pytest.mark.parametrize(
    ("seed", "copula", "yj_kernel"),
    [
        (20260716, True, True),
        (424242, True, True),
        (20260716, False, True),
        (424242, False, False),
    ],
)
def test_morp_flex_master_fd(seed, copula, yj_kernel):
    est, spec, layout, theta = _build_estimator(seed, copula=copula, yj_kernel=yj_kernel)
    _, score = est.simulated_loglik(theta, want_grad=True)
    assert score.shape[1] == layout.n_theta
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)
    sl = layout.slices()
    inactive = [] if copula else _inactive_rcor_columns(spec, layout)
    # every block load-bearing + FD-correct
    for name in ("thresh", "beta", "rcor", "scal", "lam", "kernlam"):
        block = sl.get(name)               # no ``kernlam`` slice for the normal kernel
        if block is None or block.stop - block.start == 0:
            continue
        cols = np.arange(block.start, block.stop)
        if name == "rcor" and inactive:
            masked = block.start + np.asarray(inactive, dtype=int)
            # held-at-zero slots: exactly zero analytic score (GAUSS _max_active)
            assert np.all(g_an[masked] == 0.0), (
                f"rcor inactive slots carry a score: {g_an[masked]}"
            )
            cols = np.array([c for c in cols if c not in set(masked.tolist())])
        assert float(np.max(np.abs(g_fd[cols]))) > 1e-4, f"{name} FD degenerate"
        assert np.allclose(g_an[cols], g_fd[cols], atol=_TOL, rtol=_TOL), (
            f"{name}: analytic {g_an[cols]} vs FD {g_fd[cols]}"
        )
