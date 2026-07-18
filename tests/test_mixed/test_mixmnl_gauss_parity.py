"""GAUSS <-> Python value-for-value parity for the mixed-MNL engine (plan T0.19).

The fixtures under ``tests/fixtures/mixed/mnl/travelmode_converged/`` are a
fixed-``b`` dump from an instrumented ``MIXMNL.gss`` adapted to TRAVELMODE with
``normvar={IVTT}``, ``yjvar={COST}`` (see ``meta.json``). GAUSS is the oracle:
we feed the Python engine the *identical* draws GAUSS consumed and evaluate the
simulated log-likelihood + analytic score at the same parameter vector, then
assert every quantity matches to (near) machine precision.

This exercises the full mixing stack on a real spec — correlation Cholesky
(``newcholparmscaled``), the exp scale, the Yeo-Johnson lambda chain
(``2*cdlogit`` -> ``standyjinvnonpgiven`` with ``meanyj`` standardisation), the
softmax kernel, and all twelve analytic-gradient blocks.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed import FixtureDrawSource
from pybhatlib.models.mixmnl import MixMNLControl, MixMNLModel
from pybhatlib.vecup import PanelIndex

REPO = pathlib.Path(__file__).resolve().parents[2]
FIX = REPO / "tests" / "fixtures" / "mixed" / "mnl" / "travelmode_converged"
DATA = REPO / "examples" / "data" / "TRAVELMODE.csv"

pytestmark = pytest.mark.skipif(
    not (FIX / "b.csv").exists() or not DATA.exists(),
    reason="GAUSS MixMNL parity fixtures or TRAVELMODE data not present",
)


def _build_estimator_with_gauss_draws():
    meta = json.loads((FIX / "meta.json").read_text())
    data = pd.read_csv(DATA)
    ctrl = MixMNLControl(
        normvar=tuple(meta["normvar"]),
        yjvar=tuple(meta["yjvar"]),
        n_rep=meta["flags"]["n_rep"],
        intordn1=meta["flags"]["intordn1"],
        spher=bool(meta["flags"]["spher"]),
        scal=float(meta["flags"]["scal"]),
        floor_pcomp=0.0,  # GAUSS driver sets w1=0 -> no flooring
        floor_z=0.0,      # GAUSS driver sets w2=0
        person_id=None,   # cross-sectional: Dmask == I
    )
    model = MixMNLModel(
        data=data,
        alternatives=meta["alternatives"],
        availability="none",
        spec=meta["spec"],
        control=ctrl,
    )
    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    est = model._build_estimator(spec, layout, panel)
    # Feed the engine the exact GAUSS draws instead of scipy Halton.
    est.draws = FixtureDrawSource(np.loadtxt(FIX / "ass.csv", delimiter=","))
    return est, spec, layout, panel


def test_gauss_draws_reshape_matches_perrep_dump():
    """FixtureDrawSource(ass) reproduces GAUSS's per-rep errbeta1temp exactly."""
    _, spec, _, panel = _build_estimator_with_gauss_draws()
    ass = np.loadtxt(FIX / "ass.csv", delimiter=",")
    py = FixtureDrawSource(ass).draws(panel.n_ind, spec.nrndcoef, 15)
    ref = np.stack(
        [np.loadtxt(FIX / f"draws_r{r}.csv", delimiter=",") for r in range(1, 16)]
    )
    assert py.shape == (15, panel.n_ind, spec.nrndcoef)
    np.testing.assert_allclose(py, ref, atol=1e-12)


def test_param_layout_matches_gauss():
    _, spec, layout, _ = _build_estimator_with_gauss_draws()
    meta = json.loads((FIX / "meta.json").read_text())["param_layout"]
    assert (spec.n_beta, spec.nrndtcor, spec.nscale, spec.numlam) == (
        meta["n_beta"], meta["n_rcor"], meta["n_scal"], meta["n_lam"],
    )
    assert layout.n_theta == meta["n_theta"]


def test_loglik_matches_gauss_value_for_value():
    est, _, _, _ = _build_estimator_with_gauss_draws()
    b = np.loadtxt(FIX / "b.csv", delimiter=",")
    ll, _ = est.simulated_loglik(b, want_grad=False)
    ll_ref = np.loadtxt(FIX / "ll_obs.csv", delimiter=",")
    ll_total_ref = json.loads((FIX / "meta.json").read_text())["reference"]["ll_total"]

    assert ll.sum() == pytest.approx(ll_total_ref, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(ll, ll_ref, atol=1e-6)


def test_analytic_score_matches_gauss_value_for_value():
    est, _, _, _ = _build_estimator_with_gauss_draws()
    b = np.loadtxt(FIX / "b.csv", delimiter=",")
    _, score = est.simulated_loglik(b, want_grad=True)
    gr_ref = np.loadtxt(FIX / "grad_obs.csv", delimiter=",")
    assert score.shape == gr_ref.shape
    np.testing.assert_allclose(score, gr_ref, atol=1e-5)
    # summed gradient (what the optimizer sees) also matches
    np.testing.assert_allclose(score.sum(0), gr_ref.sum(0), atol=1e-5)
