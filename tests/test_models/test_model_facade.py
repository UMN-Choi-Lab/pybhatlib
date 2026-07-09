"""Uniform post-estimation object surface across all four models.

Every model (MNP / MORP / MDCEV / MNL) must:

* subclass :class:`BaseModel`,
* expose ``fit`` / ``predict`` / ``ate`` (plus a ``predict_choice`` or
  ``predict_category`` argmax helper),
* cache the fitted results object on ``self.results_`` after ``fit()``, and
* raise a clear error when a post-estimation helper is used before fitting.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pybhatlib.models._base import BaseModel
from pybhatlib.models.mnp import MNPModel, MNPControl
from pybhatlib.models.morp import MORPModel
from pybhatlib.models.mdcev import MDCEVModel
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
