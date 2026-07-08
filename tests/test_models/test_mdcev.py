"""Smoke tests for MDCEVModel construction and fitting."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev._mdcev_model import MDCEVModel
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev import mdcev_forecast, mdcev_predict_choice

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

    def test_forecast_function_accepts_results_keyword(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        model = MDCEVModel(
            data=df,
            alternatives=["alt_out", "alt1", "alt2"],
            utility_spec=_USPEC,
            gamma_spec=_GSPEC,
            control=MDCEVControl(maxiter=5, verbose=0),
        )
        results = model.fit()
        X_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        X_gam_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        price_new = np.ones((len(df), 3), dtype=np.float64)
        budget = np.ones(len(df), dtype=np.float64)

        forecasts = mdcev_forecast(
            results=results,
            X_new=X_new,
            X_gam_new=X_gam_new,
            price_new=price_new,
            budget=budget,
            n_replications=3,
            seed=123,
        )

        assert forecasts.shape == (3 * len(df), 3)
        assert np.all(forecasts >= 0)

    def test_forecast_with_raw_parameters(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        model = MDCEVModel(
            data=df,
            alternatives=["alt_out", "alt1", "alt2"],
            utility_spec=_USPEC,
            gamma_spec=_GSPEC,
            control=MDCEVControl(maxiter=5, verbose=0),
        )
        results = model.fit()

        X_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        X_gam_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        price_new = np.ones((len(df), 3), dtype=np.float64)
        budget = np.ones(len(df), dtype=np.float64)

        # Call forecasting by supplying raw parameter vector and sigma
        forecasts = mdcev_forecast(
            results=None,
            X_new=X_new,
            X_gam_new=X_gam_new,
            price_new=price_new,
            budget=budget,
            n_replications=2,
            seed=123,
            b_reported=results.b_reported,
            sigma=results.sigma,
        )

        assert forecasts.shape == (2 * len(df), 3)
        assert np.all(forecasts >= 0)

    def test_forecast_with_full_raw_vector_and_implicit_sigma(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        model = MDCEVModel(
            data=df,
            alternatives=["alt_out", "alt1", "alt2"],
            utility_spec=_USPEC,
            gamma_spec=_GSPEC,
            control=MDCEVControl(maxiter=5, verbose=0),
        )
        results = model.fit()

        X_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        X_gam_new = np.zeros((len(df), 3, 3), dtype=np.float64)
        price_new = np.ones((len(df), 3), dtype=np.float64)
        budget = np.ones(len(df), dtype=np.float64)

        raw_params = results.b_reported
        forecasts = mdcev_forecast(
            results=None,
            b_reported=raw_params,
            sigma=None,
            X_new=X_new,
            X_gam_new=X_gam_new,
            price_new=price_new,
            budget=budget,
            n_replications=2,
            seed=123,
        )

        assert forecasts.shape == (2 * len(df), 3)
        assert np.all(forecasts >= 0)


_ALTS = ["alt_out", "alt1", "alt2"]
_USPEC = {
    "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
    "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
}
_GSPEC = {
    "g_out": {"alt_out": "uno", "alt1": "sero", "alt2": "sero"},
    "g1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "g2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
}


def _fit_mdcev(df, **ctrl_kwargs):
    model = MDCEVModel(
        data=df, alternatives=_ALTS,
        utility_spec=_USPEC, gamma_spec=_GSPEC,
        control=MDCEVControl(maxiter=10, verbose=0, **ctrl_kwargs),
    )
    return model.fit()


class TestMDCEVSeDiagnostic:
    """se_diagnostic computes all three SE estimators (A3 parity with MORP/MNP)."""

    def test_default_only_primary_estimator_populated(self, synthetic_mdcev_data):
        r = _fit_mdcev(synthetic_mdcev_data, se_method="bhhh")
        assert r.se_bhhh is not None
        assert r.se_hessian is None
        assert r.se_sandwich is None

    def test_diagnostic_populates_all_three(self, synthetic_mdcev_data):
        r = _fit_mdcev(synthetic_mdcev_data, se_method="bhhh", se_diagnostic=True)
        for name, se in [
            ("bhhh", r.se_bhhh), ("hessian", r.se_hessian),
            ("sandwich", r.se_sandwich),
        ]:
            assert se is not None, f"{name} SE not computed under se_diagnostic"
            assert np.all(np.isfinite(se)), f"{name} SE has non-finite entries"

    def test_diagnostic_reported_se_matches_primary(self, synthetic_mdcev_data):
        """Reported `se` equals the primary se_method's estimator."""
        r = _fit_mdcev(synthetic_mdcev_data, se_method="hessian", se_diagnostic=True)
        assert np.allclose(r.se, r.se_hessian, equal_nan=True)
        # bhhh and hessian SEs are both finite but generally differ
        assert r.se_bhhh is not None and r.se_sandwich is not None

    def test_diagnostic_off_by_default(self, synthetic_mdcev_data):
        r = _fit_mdcev(synthetic_mdcev_data, se_method="sandwich")
        assert r.se_sandwich is not None
        assert r.se_bhhh is None  # not requested, diagnostic off
