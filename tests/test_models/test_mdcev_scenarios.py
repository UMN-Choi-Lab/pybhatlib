"""Tests for the MDCEV scenarios= / .comparison() API.

MDCEV predicts consumption-indicator shares over ``nc`` alternatives, so
``shares_per_scenario`` maps scenario name -> NDArray(nc,) and ``comparison()``
returns a single percentage-change array — same shape as MNP.  Each scenario's
design matrices are rebuilt via ``prepare_mdcev_forecast_data``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev import MDCEVControl, MDCEVModel, mdcev_ate, mdcev_predict
from pybhatlib.models.mdcev._mdcev_forecast import mdcev_forecast, prepare_mdcev_forecast_data
from pybhatlib.gradmvn._logit_dist import simmdcev

ALTS = ["alt_out", "alt1", "alt2"]
USPEC = {
    "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
    "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
}
GSPEC = {
    "g1": {"alt1": "uno", "alt2": "sero"},
    "g2": {"alt1": "sero", "alt2": "uno"},
}
SCEN = {"base": {"x1": 0.0}, "treatment": {"x1": 1.0}}


@pytest.fixture(scope="module")
def mdcev_setup():
    rng = np.random.default_rng(123)
    n = 60
    df = pd.DataFrame({
        "ID": np.arange(n),
        "x1": rng.standard_normal(n), "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "alt_out": rng.uniform(0, 2, n),
        "alt1": rng.uniform(0, 2, n), "alt2": rng.uniform(0, 2, n),
    })
    model = MDCEVModel(data=df, alternatives=ALTS, utility_spec=USPEC,
                       gamma_spec=GSPEC, control=MDCEVControl(maxiter=8, verbose=0))
    return model.fit(), model, df


def test_scenarios_match_manual_rebuild(mdcev_setup):
    """Scenario shares equal a manual prepare_mdcev_forecast_data + predict."""
    res, model, df = mdcev_setup
    a = mdcev_ate(res, model=model, data=df, scenarios=SCEN, n_draws=200, seed=7)
    assert set(a.shares_per_scenario) == set(SCEN)
    for name, ov in SCEN.items():
        Xs, Xgs, ps, budget, _ = prepare_mdcev_forecast_data(
            model, df, list(ov.keys()), [float(v) for v in ov.values()]
        )
        ref = (mdcev_forecast(
            res, Xs, Xgs, ps, budget,
            n_replications=200, seed=7,
        ) > 0.0).mean(axis=0)
        np.testing.assert_allclose(a.shares_per_scenario[name], ref, atol=1e-12)


def test_comparison(mdcev_setup):
    res, model, df = mdcev_setup
    a = mdcev_ate(res, model=model, data=df, scenarios=SCEN, n_draws=200, seed=7)
    cmp = a.comparison("base", "treatment")
    b = a.shares_per_scenario["base"]
    t = a.shares_per_scenario["treatment"]
    expect = np.where(b > 0, 100.0 * (t - b) / b, np.nan)
    np.testing.assert_allclose(cmp, expect, equal_nan=True)


def test_scenarios_accept_string_source_columns(mdcev_setup):
    res, model, df = mdcev_setup
    df = df.copy()
    df["uno"] = 1.0
    df["sero"] = 0.0
    scenario_spec = {"base": {"x1": "uno"}, "treatment": {"x1": "sero"}}
    a = mdcev_ate(res, model=model, data=df, scenarios=scenario_spec, n_draws=100, seed=7)
    assert set(a.shares_per_scenario) == set(scenario_spec)
    assert a.shares_per_scenario["base"].shape == (len(ALTS),)
    assert a.shares_per_scenario["treatment"].shape == (len(ALTS),)


def test_scenarios_require_model_and_data(mdcev_setup):
    res, _, _ = mdcev_setup
    with pytest.raises(ValueError, match="model and data are required"):
        mdcev_ate(res, scenarios=SCEN)


def test_legacy_prebuilt_path_still_works(mdcev_setup):
    res, model, df = mdcev_setup
    Xs, Xgs, ps, _, _ = prepare_mdcev_forecast_data(model, df)
    a = mdcev_ate(res, Xs, Xgs, ps, n_draws=200, seed=7)
    assert a.shares_per_scenario is None
    assert a.predicted_shares.shape[0] == len(ALTS)


def test_simmdcev_handles_boundary_extreme_values():
    a = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    seed_out, draws = simmdcev(a=a, m=1, n=5, sig=0.5, seed=7)
    assert seed_out > 7
    assert draws.shape == (5, 4)
    assert np.all(np.isfinite(draws))
