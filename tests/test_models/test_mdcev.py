"""Smoke tests for MDCEVModel construction and fitting."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev._mdcev_model import MDCEVModel
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl

@pytest.fixture
def synthetic_mdcev_data():
    """Generate a small synthetic MDCEV dataset."""
    rng = np.random.default_rng(123)
    n = 50
    # 3 alternatives: alt_out (outside good), alt1, alt2
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    # Random consumption, nonnegative
    alt_out = rng.uniform(0, 2, n)
    alt1 = rng.uniform(0, 2, n)
    alt2 = rng.uniform(0, 2, n)
    df = pd.DataFrame({
        "ID": np.arange(n),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "alt_out": alt_out,
        "alt1": alt1,
        "alt2": alt2,
    })
    return df

class TestMDCEVModel:
    def test_model_construction(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        alternatives = ["alt_out", "alt1", "alt2"]
        utility_spec = {
            "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
            "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
            "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
        }
        gamma_spec = {
            "g_out": {"alt_out": "uno", "alt1": "sero", "alt2": "sero"},
            "g1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
            "g2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
        }
        model = MDCEVModel(
            data=df,
            alternatives=alternatives,
            utility_spec=utility_spec,
            gamma_spec=gamma_spec,
            control=MDCEVControl(maxiter=1, verbose=0),
        )
        assert model.n_alts == 3
        assert model.utility_spec.shape[1] == 3
        assert model.gamma_spec.shape[1] == 3

    def test_model_fit_smoke(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        alternatives = ["alt_out", "alt1", "alt2"]
        utility_spec = {
            "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
            "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
            "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
        }
        gamma_spec = {
            "g_out": {"alt_out": "uno", "alt1": "sero", "alt2": "sero"},
            "g1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
            "g2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
        }
        model = MDCEVModel(
            data=df,
            alternatives=alternatives,
            utility_spec=utility_spec,
            gamma_spec=gamma_spec,
            control=MDCEVControl(maxiter=5, verbose=1),
        )
        results = model.fit()
        assert np.isfinite(results.loglik)
        assert results.se is not None
        assert np.all(np.isfinite(results.se))
        assert results.n_obs == len(df)
