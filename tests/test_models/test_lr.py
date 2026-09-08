"""Analytical LR estimation and post-estimation contracts."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import t

from pybhatlib import LRControl, LRModel, LRResults, lr_ate_from_params, lr_predict
from pybhatlib.models.lr._lr_loglik import lr_gradient, lr_hessian, lr_loglik


@pytest.fixture
def data():
    rng = np.random.default_rng(731)
    x = rng.normal(size=80)
    return pd.DataFrame({"x": x, "y": 2 + 3 * x + rng.normal(size=80)})


def model(data, **kwargs):
    return LRModel(data, "y", {"CON": "uno", "B_X": "x"},
                   control=LRControl(verbose=0, **kwargs))


def test_classical_reference(data, capsys):
    m = model(data)
    r = m.fit()
    # Independent simple-regression formulas.
    x, y = data.x.to_numpy(), data.y.to_numpy()
    slope = ((x - x.mean()) @ (y - y.mean())) / np.sum((x - x.mean())**2)
    intercept = y.mean() - slope * x.mean()
    assert_allclose(r.params, [intercept, slope])
    residual = y - intercept - slope * x
    variance = residual @ residual / (len(x) - 2)
    sxx = np.sum((x - x.mean())**2)
    covariance = variance * np.array([[1 / len(x) + x.mean()**2 / sxx, -x.mean() / sxx],
                                      [-x.mean() / sxx, 1 / sxx]])
    assert_allclose(r.cov_matrix, covariance)
    assert_allclose(r.p_value, 2 * t.sf(abs(r.params / r.se), 78))
    assert_allclose(r.gradient, 0, atol=1e-12)
    assert_allclose(m.predict(), y - residual)
    assert_allclose(m.predict(data[["x"]]), lr_predict(r, m.X))
    assert r is m.results_
    assert r.n_iter == 0 and r.converged
    assert "LR Estimation" in r.summary()
    assert list(r.to_dataframe().columns) == ["Estimate", "Std.Error", "t-stat", "p-value", "Gradient"]


@pytest.mark.parametrize("method", ["hessian", "sandwich", "bhhh"])
def test_covariances(data, method):
    m = model(data, se_method=method, df_correction=False)
    r = m.fit()
    bread = np.linalg.inv(m.X.T @ m.X)
    xr = m.X * r.residuals[:, None]
    expected = {"hessian": r.sigma2 * bread,
                "sandwich": bread @ xr.T @ xr @ bread,
                "bhhh": np.linalg.inv(xr.T @ xr) * r.sigma2**2}[method]
    assert_allclose(r.cov_matrix, expected)
    corrected = model(data, se_method=method).fit()
    assert_allclose(corrected.cov_matrix, expected * 80 / 78)


def test_likelihood_derivatives(data):
    X = np.column_stack([np.ones(len(data)), data.x])
    beta = np.array([1.2, 2.1])
    y = data.y.to_numpy()
    assert_allclose(lr_gradient(beta, X, y, 1.7),
                    approx_derivative(lambda b: lr_loglik(b, X, y, 1.7), beta), atol=1e-8)
    assert_allclose(lr_hessian(beta, X, y, 1.7),
                    approx_derivative(lambda b: lr_gradient(b, X, y, 1.7).sum(axis=0), beta))


def test_scenarios_external_and_csv(data, tmp_path):
    original = data.copy()
    scenarios = {"base": {"x": 0}, "treatment": {"x": 2}, "observed": {"x": "x"}}
    m = model(data)
    with pytest.raises(RuntimeError, match="fit"):
        m.predict()
    r = m.fit()
    a = m.ate(scenarios=scenarios)
    assert_allclose(a.comparison("base", "treatment", percent=False), 2 * r.params[1])
    assert_allclose(a.comparison("base", "treatment"), 200 * r.params[1] / r.params[0])
    assert_allclose(a.means_per_scenario["observed"], a.predicted_mean)
    external = lr_ate_from_params(r.params, data=data, spec=m.spec_dict, dep_var="y",
                                  scenarios=pd.DataFrame.from_dict(scenarios, orient="index"))
    assert external.means_per_scenario == a.means_per_scenario
    pd.testing.assert_frame_equal(data, original)
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    csv_model = LRModel(path, "y", {"CON": {"y": "uno"}, "B": {"y": "x"}},
                        var_names=["Intercept", "Slope"], control=LRControl(verbose=0))
    assert_allclose(csv_model.fit().params, r.params)
    assert csv_model.var_names == ["Intercept", "Slope"]
    with pytest.raises(ValueError, match="target"):
        m.ate(scenarios={"bad": {"typo": 1}})


def test_validation_and_no_covariance(data):
    with pytest.raises(ValueError, match="rank deficient"):
        LRModel(data, "y", {"a": "uno", "b": "uno"}).fit()
    with pytest.raises(ValueError, match="n_obs"):
        model(data.iloc[:2])
    with pytest.raises(ValueError, match="nonfinite"):
        model(data.assign(x=np.nan))
    with pytest.raises(ValueError, match="finite"):
        model(data.assign(y=np.inf))
    with pytest.raises(ValueError, match="se_method"):
        LRControl(se_method="invalid")
    r = model(data, want_covariance=False).fit()
    assert np.isnan(r.se).all() and np.isnan(r.p_value).all()
    with pytest.raises(ValueError, match="design matrix"):
        lr_predict(r, np.ones((2, 3)))
    with pytest.raises(ValueError, match="param_names"):
        LRResults.from_estimates([1, 2], param_names=["a"])


def test_exact_fit_and_no_intercept(data):
    r = model(data.assign(y=2 + 3 * data.x)).fit()
    assert_allclose(r.params, [2, 3])
    assert r.sigma2 == 0 and np.isposinf(r.loglik)
    assert_allclose(r.se, 0)
    m = LRModel(data, "y", {"B_X": "x"}, control=LRControl(verbose=0))
    r = m.fit()
    assert_allclose(r.params, [data.x @ data.y / (data.x @ data.x)])
    assert_allclose(r.r_squared, 1 - r.residuals @ r.residuals / (data.y @ data.y))
