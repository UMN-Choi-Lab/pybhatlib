"""Smoke tests for MNLModel construction and fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mnl import MNLControl, MNLModel

_ALTS = ["Alt1_ch", "Alt2_ch"]
_SPEC = {
    "X1": {"Alt1_ch": "x1_a1", "Alt2_ch": "x1_a2"},
    "X2": {"Alt1_ch": "x2_a1", "Alt2_ch": "x2_a2"},
}


@pytest.fixture
def synthetic_mnl_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_obs = 200

    x = rng.standard_normal(n_obs)
    z = rng.standard_normal(n_obs)

    x_a1 = x + rng.standard_normal(n_obs) * 0.1
    x_a2 = x + rng.standard_normal(n_obs) * 0.1
    z_a1 = z + rng.standard_normal(n_obs) * 0.1
    z_a2 = z + rng.standard_normal(n_obs) * 0.1

    beta_true = np.array([1.0, -1.0, 0.5, -0.5])
    V1 = beta_true[0] * x_a1 + beta_true[2] * z_a1
    V2 = beta_true[1] * x_a2 + beta_true[3] * z_a2
    probs = np.exp(np.column_stack([V1, V2]))
    probs = probs / probs.sum(axis=1, keepdims=True)

    draws = rng.uniform(size=n_obs)
    choices = (draws > probs[:, 0]).astype(int)

    return pd.DataFrame(
        {
            "Alt1_ch": 1 - choices,
            "Alt2_ch": choices,
            "x1_a1": x_a1,
            "x1_a2": x_a2,
            "x2_a1": z_a1,
            "x2_a2": z_a2,
        }
    )


def _fit_mnl(df: pd.DataFrame, **ctrl_kwargs):
    model = MNLModel(
        data=df,
        alternatives=_ALTS,
        spec=_SPEC,
        control=MNLControl(verbose=0, maxiter=50, **ctrl_kwargs),
    )
    return model.fit()


class TestMNLModel:
    def test_model_construction(self, synthetic_mnl_data: pd.DataFrame):
        model = MNLModel(
            data=synthetic_mnl_data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=MNLControl(verbose=0, maxiter=1),
        )

        assert model.n_alts == 2
        assert model.n_beta == len(_SPEC)
        assert model.var_names == ["X1", "X2"]
        assert hasattr(model, "y")
        assert model.N == len(synthetic_mnl_data)

    def test_model_fit_smoke(self, synthetic_mnl_data: pd.DataFrame):
        results = _fit_mnl(synthetic_mnl_data)

        assert np.isfinite(results.ll)
        assert np.isfinite(results.ll_total)
        assert results.n_obs == len(synthetic_mnl_data)
        assert results.se.shape == (len(_SPEC),)
        assert np.all(np.isfinite(results.se))
        assert results.control is not None
        assert results.control.want_covariance is True


class TestMNLSeMethod:
    def test_se_method_selection(self, synthetic_mnl_data: pd.DataFrame):
        for method in ("bhhh", "hessian", "sandwich"):
            results = _fit_mnl(
                synthetic_mnl_data,
                se_method=method,
                optimizer="newton",
                analytic_grad=True,
                analytic_hess=True,
            )

            assert results.control.se_method == method
            assert results.se.shape == (len(_SPEC),)
            assert np.all(np.isfinite(results.se))
            assert np.all(results.se >= 0.0)
            assert np.isfinite(results.ll)
