"""Uniform post-estimation object surface across all four models.

Every model (MNP / MORP / MDCEV / MNL) must:

* subclass :class:`BaseModel`,
* expose ``fit`` / ``predict`` / ``ate`` (plus a ``predict_choice`` or
  ``predict_category`` argmax helper),
* cache the fitted results object on ``self.results_`` after ``fit()``, and
* raise a clear error when a post-estimation helper is used before fitting.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._base import BaseModel
from pybhatlib.models.mnp import MNPModel, MNPControl
from pybhatlib.models.morp import MORPModel, MORPControl, morp_ate
from pybhatlib.models.mdcev import MDCEVModel, MDCEVControl, mdcev_ate
from pybhatlib.models.mnl import MNLModel, MNLControl

ALTS = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]
SPEC = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

ALL_MODELS = [MNPModel, MORPModel, MDCEVModel, MNLModel]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_model_subclasses_basemodel(model_cls):
    assert issubclass(model_cls, BaseModel)


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_model_exposes_uniform_facade(model_cls):
    for meth in ("fit", "predict", "ate"):
        assert callable(getattr(model_cls, meth, None)), (model_cls, meth)
    # argmax helper is named per-domain (choice vs category)
    assert hasattr(model_cls, "predict_choice") or hasattr(
        model_cls, "predict_category"
    )


def _mnl(travelmode_path, **ctrl):
    return MNLModel(
        data=travelmode_path,
        alternatives=ALTS,
        availability="none",
        spec=SPEC,
        control=MNLControl(maxiter=ctrl.get("maxiter", 40), verbose=0),
    )


def test_fit_caches_results_and_delegates(travelmode_path):
    model = _mnl(travelmode_path)
    assert model.results_ is None  # not fit yet
    results = model.fit()
    # fit() caches the same object it returns.
    assert model.results_ is results

    # predict() defaults to the training design and returns valid probabilities.
    probs = model.predict()
    assert probs.shape == (model.N, len(ALTS))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    # ate() delegates and auto-fills data/spec/alternatives.
    ate = model.ate()
    assert ate.predicted_shares.shape == (len(ALTS),)
    np.testing.assert_allclose(ate.predicted_shares.sum(), 1.0, atol=1e-6)

    # scenario ATE + comparison via the object surface.
    sc = model.ate(scenarios={"base": {"COST_DA": 0.0}, "hi": {"COST_DA": 5.0}})
    assert set(sc.shares_per_scenario) == {"base", "hi"}
    assert sc.comparison("base", "hi").shape == (len(ALTS),)


def test_predict_before_fit_raises(travelmode_path):
    model = _mnl(travelmode_path)
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.predict()
    with pytest.raises(RuntimeError, match="has not been fit"):
        model.ate()


def test_mnp_fit_kwarg_forwards_through_wrapper(travelmode_path):
    """BaseModel.fit(*args, **kwargs) must forward to the model-specific _fit."""
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTS,
        availability="none",
        spec=SPEC,
        control=MNPControl(iid=True, maxiter=5, verbose=0, seed=0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = model.fit(bounds=None)  # bounds= is an MNP-specific _fit kwarg
    assert model.results_ is results
    assert isinstance(model.predict().sum(axis=1).round(6).tolist()[0], float)


# ----------------------------------------------------------------------
# Uniform scenarios= at the object layer (all four models)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_ate_facade_accepts_scenarios(model_cls):
    """``model.ate(scenarios=...)`` must exist on every model.

    MORP and MDCEV gained the free-function ``scenarios=`` API separately
    (#45, #46); this pins the object-layer half so the four facades cannot
    drift apart again.
    """
    params = inspect.signature(model_cls.ate).parameters
    assert "scenarios" in params, f"{model_cls.__name__}.ate lacks scenarios="
    assert params["scenarios"].kind is inspect.Parameter.KEYWORD_ONLY


MORP_SPEC = {"x1": {"y1": "x1", "y2": "x1"}, "x2": {"y1": "x2", "y2": "x2"}}
MORP_DEP = ["y1", "y2"]
MORP_NCAT = [3, 3]
SCENARIOS = {"base": {"x1": 0.0}, "treatment": {"x1": 1.0}}


@pytest.fixture(scope="module")
def morp_model():
    rng = np.random.default_rng(42)
    n = 120
    x1, x2 = rng.standard_normal(n), rng.standard_normal(n)
    lin = np.column_stack([x1, x2]) @ np.array([0.5, -0.3])
    df = pd.DataFrame({
        "x1": x1, "x2": x2,
        "y1": np.digitize(lin + rng.standard_normal(n), [-0.5, 0.5]),
        "y2": np.digitize(lin + rng.standard_normal(n), [-0.3, 0.7]),
    })
    model = MORPModel(
        data=df, dep_vars=MORP_DEP, spec=MORP_SPEC, n_categories=MORP_NCAT,
        control=MORPControl(
            iid=True, method="scipy", verbose=0, seed=42, maxiter=50
        ),
    )
    model.fit()
    return model


def test_morp_facade_scenarios_matches_free_function(morp_model):
    """MORPModel.ate(scenarios=) auto-fills data/spec/dep_vars."""
    via_facade = morp_model.ate(scenarios=SCENARIOS)
    via_func = morp_ate(
        morp_model.results_, data=morp_model.data, spec=MORP_SPEC,
        dep_vars=MORP_DEP, scenarios=SCENARIOS,
    )
    assert set(via_facade.shares_per_scenario) == set(SCENARIOS)
    for name in SCENARIOS:
        for d in range(len(MORP_DEP)):
            np.testing.assert_allclose(
                via_facade.shares_per_scenario[name][d],
                via_func.shares_per_scenario[name][d],
            )
    # .comparison() returns one percentage-change array per outcome dimension.
    cmp_ = via_facade.comparison("base", "treatment")
    assert len(cmp_) == len(MORP_DEP)
    assert cmp_[0].shape == (MORP_NCAT[0],)


def test_morp_facade_rejects_joint_with_scenarios(morp_model):
    with pytest.raises(ValueError, match="joint=True is not supported"):
        morp_model.ate(scenarios=SCENARIOS, joint=True)


MDCEV_ALTS = ["alt_out", "alt1", "alt2"]
MDCEV_USPEC = {
    "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
    "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
}
MDCEV_GSPEC = {
    "g_out": {"alt_out": "uno", "alt1": "sero", "alt2": "sero"},
    "g1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "g2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
}


@pytest.fixture(scope="module")
def mdcev_model():
    rng = np.random.default_rng(123)
    n = 60
    df = pd.DataFrame({
        "ID": np.arange(n),
        "x1": rng.standard_normal(n), "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "alt_out": rng.uniform(0, 2, n),
        "alt1": rng.uniform(0, 2, n), "alt2": rng.uniform(0, 2, n),
    })
    model = MDCEVModel(
        data=df, alternatives=MDCEV_ALTS, utility_spec=MDCEV_USPEC,
        gamma_spec=MDCEV_GSPEC, control=MDCEVControl(maxiter=8, verbose=0),
    )
    model.fit()
    return model


def test_mdcev_facade_scenarios_matches_free_function(mdcev_model):
    """MDCEVModel.ate(scenarios=) auto-fills model/data and labels the alts."""
    via_facade = mdcev_model.ate(scenarios=SCENARIOS, n_draws=200, seed=7)
    via_func = mdcev_ate(
        mdcev_model.results_, model=mdcev_model, data=mdcev_model.data,
        scenarios=SCENARIOS, n_draws=200, seed=7,
    )
    assert set(via_facade.shares_per_scenario) == set(SCENARIOS)
    for name in SCENARIOS:
        np.testing.assert_allclose(
            via_facade.shares_per_scenario[name],
            via_func.shares_per_scenario[name],
        )
    assert via_facade.alternative_names == MDCEV_ALTS
    assert via_facade.comparison("base", "treatment").shape == (len(MDCEV_ALTS),)
