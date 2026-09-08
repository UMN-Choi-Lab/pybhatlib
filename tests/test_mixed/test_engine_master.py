"""MASTER GATE for the shared mixed MSL engine (plan T0.13).

Self-contained finite-difference gate -- NO GAUSS reference. On a small
synthetic MNL panel (5 persons, 1-3 occasions each, ``nc = 3``,
``nrndcoef = 2`` with one normal + one Yeo-Johnson coefficient, ``nrep = 8``,
:class:`FixtureDrawSource` with fixed draws), at a random ``theta`` the analytic
per-individual score summed over individuals must equal the central finite
difference of the summed log-likelihood::

    score_per_ind.sum(0) == d/dtheta [ ll_per_ind.sum() ]     (atol 1e-5)

This validates the ENTIRE analytic score: the reparameterization Jacobians
(scale / correlation / Yeo-Johnson lambda) chained through the engine's
``_assemble_score`` assembly. If it fails, either the score assembly or a
Jacobian is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig, Tracer
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.vecup._panel import PanelIndex

SCAL = 1.0
INTORDN1 = 20
NREP = 8
# floors well below every realised probability so the mask is 1 everywhere and
# the FD stays smooth (no flooring discontinuity).
FLOOR = 1e-10


class _Obs:
    """Minimal DesignData.obs stand-in exposing ``avail`` and ``chosen``."""

    def __init__(self, avail: np.ndarray, chosen: np.ndarray) -> None:
        self.avail = avail
        self.chosen = chosen


def _build_estimator(*, trace: Tracer | None = None):
    """Assemble the engine over a fixed synthetic MNL panel."""
    rng = np.random.default_rng(20260716)

    # --- spec: nrndcoef = 2 (one normal 'x0' + one Yeo-Johnson 'x1') --------
    var_names = ["x0", "x1", "x2"]
    spec = MixingSpec.from_var_names(
        var_names=var_names, normvar=["x0"], yjvar=["x1"]
    )
    nvarm = spec.n_beta
    nc = 3

    layout = ParamLayout(
        n_beta=spec.n_beta,
        n_rcor=spec.nrndtcor,   # 1
        n_scal=spec.nscale,     # 2
        n_lam=spec.numlam,      # 2
        n_kern=0,
    )

    # --- panel: 5 persons, occasions [3, 1, 2, 1, 3] -> 10 obs --------------
    occ = [3, 1, 2, 1, 3]
    person_ids = np.concatenate([np.full(c, i) for i, c in enumerate(occ)])
    panel = PanelIndex.from_ids(person_ids)
    n_obs = panel.n_obs
    n_ind = panel.n_ind

    # --- design: alt-specific covariates; moderate scale keeps probs high ---
    X = 0.6 * rng.standard_normal((n_obs, nc, nvarm))
    avail = np.ones((n_obs, nc), dtype=np.float64)
    chosen = np.zeros((n_obs, nc), dtype=np.float64)
    chosen[np.arange(n_obs), rng.integers(0, nc, size=n_obs)] = 1.0
    design = DesignData(X=X, obs=_Obs(avail, chosen))

    # --- fixed draws: (nrep, n_ind * n_rnd) --------------------------------
    ass2d = rng.standard_normal((NREP, n_ind * spec.nrndcoef))
    draws = FixtureDrawSource(ass2d)

    space = EstimationSpace(layout, scal=SCAL, intordn1=INTORDN1)
    pipeline = RandomCoefPipeline(spec, layout, scal=SCAL, intordn1=INTORDN1)
    kernel = SoftmaxKernel(nc)
    cfg = MSLConfig(
        n_rep=NREP, scal=SCAL, intordn1=INTORDN1,
        floor_pcomp=FLOOR, floor_z=FLOOR,
    )
    weightind = np.ones(n_ind, dtype=np.float64)

    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design, weightind=weightind,
        config=cfg, trace=trace,
    )
    return est, layout


def _random_theta(layout: ParamLayout, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    beta = 0.4 * rng.standard_normal(layout.n_beta)
    rcor = rng.uniform(-0.6, 0.6, size=layout.n_rcor)
    scal = 0.2 * rng.standard_normal(layout.n_scal)
    lam = 0.4 * rng.standard_normal(layout.n_lam)
    kern = np.zeros(layout.n_kern)
    return np.concatenate([beta, rcor, scal, lam, kern])


def test_master_score_matches_central_fd():
    """Full analytic score == central FD of the summed log-likelihood (1e-5)."""
    est, layout = _build_estimator()
    theta = _random_theta(layout, seed=7)

    ll, score = est.simulated_loglik(theta, want_grad=True)
    assert ll.shape == (est.panel.n_ind,)
    assert score.shape == (est.panel.n_ind, layout.n_theta)
    analytic = score.sum(0)

    eps = 1e-6
    fd = np.zeros(layout.n_theta, dtype=np.float64)
    for i in range(layout.n_theta):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        lp = est.simulated_loglik(tp, want_grad=False)[0].sum()
        lm = est.simulated_loglik(tm, want_grad=False)[0].sum()
        fd[i] = (lp - lm) / (2.0 * eps)

    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-5)


def test_objective_matches_simulated_loglik():
    """``objective`` returns the negated summed LL and gradient."""
    est, layout = _build_estimator()
    theta = _random_theta(layout, seed=11)

    ll, score = est.simulated_loglik(theta, want_grad=True)
    neg_ll, neg_grad = est.objective(theta)

    np.testing.assert_allclose(neg_ll, -ll.sum(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(neg_grad, -score.sum(0), rtol=0, atol=1e-12)


def test_tracer_records_pipeline_intermediates():
    """A supplied Tracer captures flat and per-replication intermediates."""
    tr = Tracer()
    est, layout = _build_estimator(trace=tr)
    theta = _random_theta(layout, seed=3)
    est.simulated_loglik(theta, want_grad=True)

    # flat intermediates present
    for name in ("xmu", "omegastar", "p0", "z", "final", "g0", "grad"):
        assert tr.get(name) is not None, f"missing flat trace {name!r}"
    # per-replication intermediates present for every replication
    for name in ("errbeta3", "ftemprand", "f1rand", "xmunew", "Vsubq",
                 "pcomp", "Pprod", "gcomp"):
        rec = tr.per_rep.get(name)
        assert rec is not None and len(rec) == NREP, (
            f"missing per-rep trace {name!r}"
        )
