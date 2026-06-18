"""Tests for mnl_ate_from_params / MNLResults.from_estimates (UTA issue #34).

Mirrors morp_ate_from_params: compute ATEs from manually entered (e.g. GAUSS)
coefficients without re-fitting.  Round-trip = feeding a fitted model's own
coefficients back through the wrapper reproduces mnl_ate(results, ...) exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mnl import (
    MNLControl,
    MNLModel,
    MNLResults,
    mnl_ate,
    mnl_ate_from_params,
    mnl_predict,
    mnl_predict_choice,
)

_DATA = os.path.join(os.path.dirname(__file__), "..", "..",
                     "examples", "data", "TRAVELMODE.csv")
ALTS = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]
SPEC = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}


@pytest.fixture(scope="module")
def mnl_fit():
    df = pd.read_csv(_DATA)
    model = MNLModel(df, ALTS, "none", SPEC, control=MNLControl(verbose=0))
    return model.fit(), model.X


def test_predict_exports_importable():
    assert callable(mnl_predict) and callable(mnl_predict_choice)


def test_from_estimates_sets_nan_se_and_preserves_b(mnl_fit):
    res, _ = mnl_fit
    r = MNLResults.from_estimates(res.b, param_names=res.param_names)
    assert np.allclose(r.b, res.b)
    assert np.all(np.isnan(r.se))
    assert r.param_names == res.param_names


def test_ate_from_params_roundtrip_base(mnl_fit):
    res, X = mnl_fit
    a_fit = mnl_ate(res, X)
    a_par = mnl_ate_from_params(res.b, X, param_names=res.param_names)
    np.testing.assert_allclose(a_par.predicted_shares, a_fit.predicted_shares, atol=1e-12)


def test_ate_from_params_roundtrip_counterfactual(mnl_fit):
    res, X = mnl_fit
    kw = dict(changevar_idx=(0, 2), base_val=0.0, treatment_val=1.0)
    a_fit = mnl_ate(res, X, **kw)
    a_par = mnl_ate_from_params(res.b, X, **kw)
    np.testing.assert_allclose(a_par.pct_ate, a_fit.pct_ate, atol=1e-9, equal_nan=True)
