"""Tests for MORP (Multivariate Ordered Response Probit) model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.models.morp import MORPControl, MORPModel
from pybhatlib.models.morp._morp_loglik import (
    _unpack_morp_params,
    count_morp_params,
    morp_loglik,
)


@pytest.fixture
def xp():
    return get_backend("numpy")


@pytest.fixture
def synthetic_morp_data():
    """Generate small synthetic MORP dataset."""
    rng = np.random.default_rng(42)
    n = 100

    beta_true = np.array([0.5, -0.3])
    tau_true = [np.array([-0.5, 0.5]), np.array([-0.3, 0.7])]
    rho_true = 0.3
    sigma_true = np.array([[1.0, rho_true], [rho_true, 1.0]])

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X_vars = np.column_stack([x1, x2])

    eps = rng.multivariate_normal(np.zeros(2), sigma_true, size=n)
    Y_star = np.column_stack([
        X_vars @ beta_true + eps[:, 0],
        X_vars @ beta_true + eps[:, 1],
    ])

    y1 = np.digitize(Y_star[:, 0], tau_true[0])
    y2 = np.digitize(Y_star[:, 1], tau_true[1])

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
    })

    return df, beta_true, tau_true, sigma_true


class TestMORPControl:
    def test_defaults(self):
        ctrl = MORPControl()
        assert ctrl.method == "ovus"
        assert ctrl.indep is False
        assert ctrl.spherical is True

    def test_independent(self):
        ctrl = MORPControl(indep=True)
        assert ctrl.indep is True


class TestCountMORPParams:
    def test_independent_2d_3cat(self):
        ctrl = MORPControl(indep=True)
        n = count_morp_params(3, 2, [3, 3], ctrl)
        # 3 betas + 2*2 thresholds = 7
        assert n == 7

    def test_full_cov_2d_3cat(self):
        ctrl = MORPControl(indep=False)
        n = count_morp_params(3, 2, [3, 3], ctrl)
        # 3 betas + 2*2 thresholds + 1 scale + 1 corr = 9
        assert n == 9

    def test_heteronly_3d(self):
        ctrl = MORPControl(heteronly=True)
        n = count_morp_params(2, 3, [3, 4, 3], ctrl)
        # 2 betas + (2+3+2) thresholds + 2 scales = 11
        assert n == 11


class TestUnpackMORPParams:
    def test_threshold_ordering(self):
        """Thresholds should be strictly increasing."""
        ctrl = MORPControl(indep=True)
        # 2 betas + 2 thresholds for dim1 + 2 thresholds for dim2
        theta = np.array([0.5, -0.3, -0.5, 0.2, -0.3, 0.3])
        beta, thresholds, sigma = _unpack_morp_params(
            theta, 2, 2, [3, 3], ctrl
        )
        # tau[0] = theta[2] = -0.5
        # tau[1] = tau[0] + exp(theta[3]) = -0.5 + exp(0.2)
        assert thresholds[0][1] > thresholds[0][0]
        assert thresholds[1][1] > thresholds[1][0]

    def test_sigma_identity_for_indep(self):
        ctrl = MORPControl(indep=True)
        theta = np.zeros(7)
        _, _, sigma = _unpack_morp_params(theta, 3, 2, [3, 3], ctrl)
        np.testing.assert_allclose(sigma, np.eye(2))


class TestMORPLoglik:
    def test_loglik_finite(self, xp):
        """Log-likelihood should be finite for valid parameters."""
        ctrl = MORPControl(indep=True, method="scipy", verbose=0)

        X = np.random.randn(20, 2, 2)
        y = np.random.randint(0, 3, size=(20, 2))

        # 2 betas + 2*2 thresholds = 6 params
        theta = np.array([0.1, -0.1, -0.5, 0.2, -0.3, 0.3])

        nll = morp_loglik(theta, X, y, 2, [3, 3], 2, ctrl, xp=xp)
        assert np.isfinite(nll)
        assert nll > 0  # negative mean log-likelihood should be positive

    def test_loglik_with_gradient(self, xp):
        """Gradient computation should work."""
        ctrl = MORPControl(indep=True, method="scipy", verbose=0)

        X = np.random.randn(10, 2, 2)
        y = np.random.randint(0, 3, size=(10, 2))
        theta = np.array([0.1, -0.1, -0.5, 0.2, -0.3, 0.3])

        nll, grad = morp_loglik(
            theta, X, y, 2, [3, 3], 2, ctrl,
            return_gradient=True, xp=xp,
        )
        assert np.isfinite(nll)
        assert len(grad) == len(theta)
        assert np.all(np.isfinite(grad))


class TestMORPModel:
    def test_model_construction(self, synthetic_morp_data):
        df, _, _, _ = synthetic_morp_data

        model = MORPModel(
            data=df,
            dep_vars=["y1", "y2"],
            indep_vars=["x1", "x2"],
            n_categories=[3, 3],
            control=MORPControl(indep=True, verbose=0),
        )

        assert model.N == 100
        assert model.n_dims == 2
        assert model.n_beta == 2

    def test_model_fit_indep(self, synthetic_morp_data):
        """Test that independent MORP model converges."""
        df, _, _, _ = synthetic_morp_data

        model = MORPModel(
            data=df,
            dep_vars=["y1", "y2"],
            indep_vars=["x1", "x2"],
            n_categories=[3, 3],
            control=MORPControl(
                indep=True,
                method="scipy",
                verbose=0,
                seed=42,
                maxiter=50,
            ),
        )

        results = model.fit()
        assert np.isfinite(results.loglik)
        assert len(results.params) == model.n_params
        assert len(results.thresholds) == 2
        # Thresholds should be ordered
        for tau_d in results.thresholds:
            if len(tau_d) > 1:
                assert np.all(np.diff(tau_d) > 0)

    def test_mismatched_categories_raises(self):
        df = pd.DataFrame({
            "x1": [1.0, 2.0],
            "y1": [0, 1],
            "y2": [0, 2],
        })
        with pytest.raises(ValueError, match="n_categories"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                indep_vars=["x1"],
                n_categories=[3],  # only 1 but 2 dep_vars
                control=MORPControl(verbose=0),
            )

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x1": [1.0], "y1": [0], "y2": [1]})
        with pytest.raises(ValueError, match="not found"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                indep_vars=["x1", "missing_col"],
                n_categories=[3, 3],
            )
