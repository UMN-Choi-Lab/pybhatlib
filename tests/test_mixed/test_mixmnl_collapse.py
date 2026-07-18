"""COLLAPSE GATE: mixed engine reduces to the fixed-coef MNL (plan T0.14).

Self-contained -- NO GAUSS reference. With ``nrndcoef = 0`` (no random
coefficients) the shared MSL engine must reproduce the shipped fixed-coefficient
:func:`pybhatlib.models.mnl.mnl_loglik` and :func:`~pybhatlib.models.mnl.mnl_gradient`
value-for-value, both:

* **cross-sectional** -- one observation per person (``Dmask == I``): the
  per-individual LL equals the per-observation MNL LL; the summed score equals
  the summed MNL gradient.
* **panel** -- multiple occasions per person: the summed LL equals
  ``mnl_loglik.sum()`` (the panel product collapses to a sum of logs) and the
  summed score equals ``mnl_gradient.sum(0)``.

The engine floors the person-level product ``z`` while ``mnl_loglik`` floors each
per-observation probability; the gate uses equal floors (``1e-4``) and a design
with no floored observations so ``1e-8`` equality is reachable (plan flooring
note, T0.14).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.models.mnl._mnl_loglik import mnl_gradient, mnl_loglik
from pybhatlib.vecup._panel import PanelIndex

FLOOR = 1e-4          # match mnl_loglik's per-obs floor
NC = 3
NUMUNORD = 3          # variables per alternative == n_beta


class _Obs:
    def __init__(self, avail: np.ndarray, chosen: np.ndarray) -> None:
        self.avail = avail
        self.chosen = chosen


def _build_mnl_problem(occ, seed):
    """Build a synthetic MNL problem shared by the fixed-coef path and engine.

    Returns the flat ``dta`` + index vectors (for ``mnl_loglik`` /
    ``mnl_gradient``), the ``(n_obs, nc, numunord)`` design tensor, the
    availability / chosen bundle, the person ids and the beta vector.
    """
    rng = np.random.default_rng(seed)
    person_ids = np.concatenate([np.full(c, i) for i, c in enumerate(occ)])
    n_obs = person_ids.shape[0]

    # moderate utilities keep every chosen probability well above the 1e-4 floor
    iv = 0.5 * rng.standard_normal((n_obs, NC * NUMUNORD))
    avail = np.ones((n_obs, NC), dtype=np.float64)
    chosen = np.zeros((n_obs, NC), dtype=np.float64)
    chosen[np.arange(n_obs), rng.integers(0, NC, size=n_obs)] = 1.0

    dta = np.hstack([iv, avail, chosen])
    indxivunord = np.arange(NC * NUMUNORD)
    davunord = np.arange(NC * NUMUNORD, NC * NUMUNORD + NC)
    dvunord = np.arange(NC * NUMUNORD + NC, NC * NUMUNORD + 2 * NC)

    # X[q, k, v] = iv[q, k*numunord + v]  ->  V[:, k] = X[:, k, :] @ beta,
    # identical to the softmax utilities mnl_loglik forms internally.
    X = iv.reshape(n_obs, NC, NUMUNORD)
    beta = 0.5 * rng.standard_normal(NUMUNORD)

    return dict(
        dta=dta, indxivunord=indxivunord, davunord=davunord, dvunord=dvunord,
        X=X, avail=avail, chosen=chosen, person_ids=person_ids, beta=beta,
    )


def _build_engine(p, *, n_rep):
    """Assemble the engine with an empty random-coefficient spec (nrndcoef=0)."""
    var_names = [f"v{i}" for i in range(NUMUNORD)]
    spec = MixingSpec.from_var_names(var_names=var_names)     # no rnd coefs
    assert spec.nrndcoef == 0

    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=0, n_scal=0, n_lam=0, n_kern=0
    )
    panel = PanelIndex.from_ids(p["person_ids"])
    design = DesignData(X=p["X"], obs=_Obs(p["avail"], p["chosen"]))

    # n_rnd == 0 -> the draw block has zero columns.
    ass2d = np.zeros((n_rep, panel.n_ind * spec.nrndcoef), dtype=np.float64)
    draws = FixtureDrawSource(ass2d)

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = SoftmaxKernel(NC)
    cfg = MSLConfig(n_rep=n_rep, floor_pcomp=FLOOR, floor_z=FLOOR)
    weightind = np.ones(panel.n_ind, dtype=np.float64)

    return MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design, weightind=weightind,
        config=cfg,
    )


def _mnl_ll(p):
    return mnl_loglik(
        p["beta"], p["dta"], p["indxivunord"], p["davunord"], p["dvunord"],
        NC, NUMUNORD,
    )


def _mnl_grad(p):
    return mnl_gradient(
        p["beta"], p["dta"], p["indxivunord"], p["davunord"], p["dvunord"],
        NC, NUMUNORD,
    )


@pytest.mark.parametrize("n_rep", [1, 5])
def test_collapse_cross_sectional(n_rep):
    """One obs per person (Dmask=I): engine LL/score == fixed-coef MNL (1e-8)."""
    occ = [1, 1, 1, 1, 1, 1]                # cross-sectional
    p = _build_mnl_problem(occ, seed=101)
    est = _build_engine(p, n_rep=n_rep)

    # Dmask is the identity for one-obs-per-person panels.
    assert np.allclose(est.panel.mask(), np.eye(est.panel.n_obs))

    ll, score = est.simulated_loglik(p["beta"], want_grad=True)

    ll_ref = _mnl_ll(p)
    grad_ref = _mnl_grad(p)

    # no floored observations, so per-individual LL == per-obs MNL LL.
    assert np.min(np.exp(ll_ref)) > FLOOR
    np.testing.assert_allclose(ll, ll_ref, rtol=0, atol=1e-8)
    np.testing.assert_allclose(ll.sum(), ll_ref.sum(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(score.sum(0), grad_ref.sum(0), rtol=0, atol=1e-8)


@pytest.mark.parametrize("n_rep", [1, 5])
def test_collapse_panel(n_rep):
    """Multiple occasions per person: summed LL/score == fixed-coef MNL (1e-8)."""
    occ = [3, 1, 2, 1, 3]                   # panel: 1-3 occasions each
    p = _build_mnl_problem(occ, seed=202)
    est = _build_engine(p, n_rep=n_rep)

    ll, score = est.simulated_loglik(p["beta"], want_grad=True)

    ll_ref = _mnl_ll(p)
    grad_ref = _mnl_grad(p)

    # no floored per-obs prob, and each person's product stays above floor_z.
    assert np.min(np.exp(ll_ref)) > FLOOR
    assert np.min(est.panel.logprod(np.log(np.exp(ll_ref)))) > FLOOR

    np.testing.assert_allclose(ll.sum(), ll_ref.sum(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(score.sum(0), grad_ref.sum(0), rtol=0, atol=1e-8)
