"""Smoke tests for MDCEVModel construction and fitting."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev._mdcev_model import MDCEVModel
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev._mdcev_forecast import (
    _mdcev_linear_forecast_allocation,
    mdcev_forecast,
    mdcev_predict,
    mdcev_predict_choice,
)

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
            "g1": {"alt1": "uno", "alt2": "sero"},
            "g2": {"alt1": "sero", "alt2": "uno"},
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
        assert model.gamma_spec.shape == (2, 2)

    def test_model_fit_smoke(self, synthetic_mdcev_data):
        df = synthetic_mdcev_data
        alternatives = ["alt_out", "alt1", "alt2"]
        utility_spec = {
            "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
            "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
            "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
        }
        gamma_spec = {
            "g1": {"alt1": "uno", "alt2": "sero"},
            "g2": {"alt1": "sero", "alt2": "uno"},
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

    def test_forecast_extreme_utilities_remain_finite(self):
        forecasts = mdcev_forecast(
            results=None,
            X_new=np.ones((1, 3, 1), dtype=np.float64),
            X_gam_new=np.ones((1, 3, 1), dtype=np.float64),
            price_new=np.ones((1, 3), dtype=np.float64),
            budget=np.ones(1, dtype=np.float64),
            n_replications=1,
            seed=123,
            b_reported=np.array([1e6, 0.0]),
            sigma=1.0,
        )

        assert np.all(np.isfinite(forecasts))

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
            sigma=results.sigma,
            X_new=X_new,
            X_gam_new=X_gam_new,
            price_new=price_new,
            budget=budget,
            n_replications=2,
            seed=123,
        )

        assert forecasts.shape == (2 * len(df), 3)
        assert np.all(forecasts >= 0)

    def test_linear_allocation_solves_the_kkt_conditions(self):
        """Independent check of the linear-MDCEV allocation (GAUSS ``forec``).

        For the linear outside-good utility the KKT conditions give, with the
        outside good as numeraire (``lambda = v_out``),
        ``x_k = gamma_k * (v_k / lambda - 1)`` for every inside good with
        ``v_k > lambda`` and ``x_k = 0`` otherwise; the outside good takes the
        budget residual. The expected allocation is computed here from that
        closed form, not from the implementation under test.
        """
        v = np.array([2.0, 5.0, 1.0, 3.0, 2.0])      # outside good first
        f1 = np.array([0.0, 0.5, 2.0, 1.5, 0.7])     # gamma_k = exp(u_k)
        prices = np.ones(5)
        budget = 40.0

        lam = v[0]
        expected = np.zeros(5)
        for k in range(1, 5):
            if v[k] > lam:
                expected[k] = f1[k] * (v[k] / lam - 1.0)
        expected[0] = budget - expected[1:].sum()

        out = _mdcev_linear_forecast_allocation(v, prices, f1, budget=budget, num_outside=1)
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)
        assert out[2] == 0.0 and out[4] == 0.0       # v_k <= lambda: not consumed
        assert np.isclose(out.sum(), budget)

        # No inside good beats the outside good -> the whole budget is outside.
        out_none = _mdcev_linear_forecast_allocation(
            np.array([9.0, 1.0, 2.0]), np.ones(3), np.array([0.0, 1.0, 1.0]), budget=7.0,
        )
        np.testing.assert_allclose(out_none, [7.0, 0.0, 0.0])

        # Output is in the original good order whatever the internal sort by v.
        perm = np.array([0, 3, 1, 4, 2])
        out_perm = _mdcev_linear_forecast_allocation(v[perm], prices[perm], f1[perm], budget=budget)
        np.testing.assert_allclose(out_perm, expected[perm], rtol=1e-12, atol=1e-12)

    def test_linear_predict_uses_observation_budget(self):
        b_reported = np.array([0.2, -0.4, 0.3, 0.5, -0.2, 1.0], dtype=np.float64)
        X_new = np.array(
            [[[1.0, 0.2, 0.1], [0.8, -0.1, 0.4], [0.7, 0.3, -0.2]]],
            dtype=np.float64,
        )
        X_gam_new = np.array(
            [[[0.1, 0.2], [0.5, -0.1], [0.4, 0.2]]],
            dtype=np.float64,
        )
        price_new = np.ones((1, 3), dtype=np.float64)

        results_lin = __import__("pybhatlib.models.mdcev._mdcev_results", fromlist=["MDCEVResults"]).MDCEVResults.from_estimates(
            b_reported=b_reported,
            sigma=1.0,
            control=MDCEVControl(utility="linear"),
            param_names=["b0", "b1", "b2", "g1", "g2", "sigma"],
        )

        pred_low = mdcev_predict(results_lin, X_new, X_gam_new, price_new, n_draws=200, seed=7, budget=np.array([10.0]))
        pred_high = mdcev_predict(results_lin, X_new, X_gam_new, price_new, n_draws=200, seed=7, budget=np.array([20.0]))

        assert pred_low.shape == (1, 3)
        assert np.all((pred_low >= 0.0) & (pred_low <= 1.0))
        assert np.all((pred_high >= 0.0) & (pred_high <= 1.0))
        assert np.any(pred_low < 1.0)
        assert not np.allclose(pred_low, pred_high)

    def test_linear_and_traditional_predict_paths_diverge(self):
        b_reported = np.array([0.2, -0.4, 0.3, 0.2, -0.1, 1.0], dtype=np.float64)
        X_new = np.array(
            [[[1.0, 0.2, 0.1], [0.8, -0.1, 0.4], [0.7, 0.3, -0.2]]],
            dtype=np.float64,
        )
        X_gam_new = np.array(
            [[[0.1, 0.2], [0.5, -0.1], [0.4, 0.2]]],
            dtype=np.float64,
        )
        price_new = np.ones((1, 3), dtype=np.float64)

        results_trad = __import__("pybhatlib.models.mdcev._mdcev_results", fromlist=["MDCEVResults"]).MDCEVResults.from_estimates(
            b_reported=b_reported,
            sigma=1.0,
            control=MDCEVControl(utility="trad"),
            param_names=["b0", "b1", "b2", "g1", "g2", "sigma"],
        )
        results_lin = __import__("pybhatlib.models.mdcev._mdcev_results", fromlist=["MDCEVResults"]).MDCEVResults.from_estimates(
            b_reported=b_reported,
            sigma=1.0,
            control=MDCEVControl(utility="linear"),
            param_names=["b0", "b1", "b2", "g1", "g2", "sigma"],
        )

        trad_pred = mdcev_predict(results_trad, X_new, X_gam_new, price_new, n_draws=500, seed=7)
        lin_pred = mdcev_predict(results_lin, X_new, X_gam_new, price_new, n_draws=500, seed=7)

        assert np.isfinite(lin_pred).all()
        assert not np.allclose(trad_pred.mean(axis=0), lin_pred.mean(axis=0), atol=1e-10)


_ALTS = ["alt_out", "alt1", "alt2"]
_USPEC = {
    "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
    "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
    "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
}
_GSPEC = {
    "g1": {"alt1": "uno", "alt2": "sero"},
    "g2": {"alt1": "sero", "alt2": "uno"},
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
