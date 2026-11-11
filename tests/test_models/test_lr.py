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
    residual = y - intercept - slope * x
    # params = [beta..., sigma] with sigma the Gaussian MLE sqrt(SSE / N).
    assert_allclose(r.params, [intercept, slope, np.sqrt(residual @ residual / 80)])
    assert r.param_names == ["CON", "B_X", "sigma"]
    variance = residual @ residual / (len(x) - 2)
    sxx = np.sum((x - x.mean())**2)
    covariance = variance * np.array([[1 / len(x) + x.mean()**2 / sxx, -x.mean() / sxx],
                                      [-x.mean() / sxx, 1 / sxx]])
    assert_allclose(r.cov_matrix[:2, :2], covariance)
    # sigma is block-diagonal to the coefficients; SE = sigma / sqrt(2 N) times
    # the N / (N - K) df correction.
    assert_allclose(r.cov_matrix[2, :2], 0, atol=1e-14)
    assert_allclose(r.se[2], r.params[2] / np.sqrt(2 * 78))
    assert_allclose(r.p_value, 2 * t.sf(abs(r.params / r.se), 78))
    assert_allclose(r.gradient, 0, atol=1e-12)
    assert_allclose(m.predict(), y - residual)
    assert_allclose(m.predict(data[["x"]]), lr_predict(r, m.X))
    assert r is m.results_
    assert r.n_iter == 0 and r.converged
    assert "LR Estimation" in r.summary()
    assert list(r.to_dataframe().columns) == ["Estimate", "Std.Error", "t-stat", "p-value", "Gradient"]


@pytest.mark.parametrize("intercept", [True, False])
def test_analytical_baseline_loglikelihoods(data, intercept, monkeypatch):
    from scipy.stats import norm
    from pybhatlib.models.lr import _lr_model as module

    calls = []
    likelihood = module.lr_loglik

    def counted(*args, **kwargs):
        calls.append(1)
        return likelihood(*args, **kwargs)

    monkeypatch.setattr(module, "lr_loglik", counted)
    spec = {"CON": "uno", "B_X": "x"} if intercept else {"B_X": "x"}
    result = LRModel(data, "y", spec, control=LRControl(verbose=0)).fit()
    y = data.y.to_numpy()
    expected_constants = norm.logpdf(y, loc=y.mean(), scale=y.std(ddof=0)).mean()
    expected_zero = norm.logpdf(y, loc=0, scale=np.sqrt(np.mean(y**2))).mean()
    assert_allclose(result.loglik_constants, expected_constants)
    assert_allclose(result.loglik_zero, expected_zero)
    text = result.summary()
    assert "Mean LL(C)" in text and "Mean LL(0)" in text
    assert f"{expected_constants:.6f}" in text
    assert f"{expected_zero:.6f}" in text
    assert len(calls) == 1  # Only the existing fitted-model likelihood call.


@pytest.mark.parametrize("outcome", [0.0, 3.0])
def test_baseline_zero_variance_boundary(outcome):
    frame = pd.DataFrame({"x": np.arange(8.0), "y": np.full(8, outcome)})
    with np.errstate(divide="raise", invalid="raise"):
        result = model(frame).fit()
        text = result.summary()
    assert result.loglik_constants == np.inf
    if outcome == 0:
        assert result.loglik_zero == np.inf
    else:
        assert_allclose(result.loglik_zero, -0.5 * (np.log(2 * np.pi * outcome**2) + 1))
    assert "Mean LL(C)" in text and "inf" in text


def test_external_estimates_have_unavailable_baselines():
    result = LRResults.from_estimates([1.0, 2.0])
    assert np.isnan(result.loglik_constants) and np.isnan(result.loglik_zero)
    for line in result.summary().splitlines():
        if "Mean LL(C)" in line or "Mean LL(0)" in line:
            assert "N/A" in line


@pytest.mark.parametrize("method", ["hessian", "sandwich", "bhhh"])
def test_covariances(data, method):
    m = model(data, se_method=method, df_correction=False)
    r = m.fit()
    xtx_inv = np.linalg.inv(m.X.T @ m.X)
    sigma = np.sqrt(r.sigma2)
    bread = np.zeros((3, 3))
    bread[:2, :2], bread[2, 2] = r.sigma2 * xtx_inv, r.sigma2 / 160
    scores = np.column_stack([m.X * r.residuals[:, None] / r.sigma2,
                              (r.residuals**2 / r.sigma2 - 1) / sigma])
    expected = {"hessian": bread,
                "sandwich": bread @ scores.T @ scores @ bread,
                "bhhh": np.linalg.inv(scores.T @ scores)}[method]
    assert_allclose(r.cov_matrix, expected)
    # Coefficient block keeps the OLS / White forms (bhhh picks up the
    # third-moment cross term with sigma, so it is not exactly separable).
    xr = m.X * r.residuals[:, None]
    if method != "bhhh":
        assert_allclose(r.cov_matrix[:2, :2], {"hessian": r.sigma2 * xtx_inv,
                                               "sandwich": xtx_inv @ xr.T @ xr @ xtx_inv}[method])
    # Student-t p-values with N - K df for every se_method.
    assert_allclose(r.p_value, 2 * t.sf(np.abs(r.t_stat), 78))
    corrected = model(data, se_method=method).fit()
    assert_allclose(corrected.cov_matrix, expected * 80 / 78)


def test_likelihood_derivatives(data):
    X = np.column_stack([np.ones(len(data)), data.x])
    theta = np.array([1.2, 2.1, 1.3])  # [beta..., sigma]
    y = data.y.to_numpy()
    assert_allclose(lr_gradient(theta, X, y),
                    approx_derivative(lambda b: lr_loglik(b, X, y), theta), atol=1e-8)
    assert_allclose(lr_hessian(theta, X, y),
                    approx_derivative(lambda b: lr_gradient(b, X, y).sum(axis=0), theta))
    with pytest.raises(ValueError, match="sigma"):
        lr_loglik([1.2, 2.1, -1.0], X, y)


def test_scenarios_external_and_csv(data, tmp_path):
    original = data.copy()
    scenarios = {"base": {"x": 0}, "treatment": {"x": 2}, "observed": {"x": "x"}}
    m = model(data)
    with pytest.raises(RuntimeError, match="fit"):
        m.predict()
    r = m.fit()
    a = m.ate(scenarios=scenarios)
    assert_allclose(a.comparison("base", "treatment"), 2 * r.params[1])
    assert_allclose(a.comparison("base", "treatment", percent=True),
                    200 * r.params[1] / r.params[0])
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
    with pytest.raises(ValueError, match="sigma must be positive"):
        LRResults.from_estimates([1, 2, -0.5])
    assert LRResults.from_estimates([1, 2], sigma=0.5).param_names == ["b1", "sigma"]
    with pytest.raises(ValueError, match="reserved"):
        LRModel(data, "y", {"sigma": "x"})


def test_exact_fit_and_no_intercept(data):
    r = model(data.assign(y=2 + 3 * data.x)).fit()
    assert_allclose(r.params, [2, 3, 0])
    assert r.sigma2 == 0 and np.isposinf(r.loglik)
    assert_allclose(r.se, 0)
    m = LRModel(data, "y", {"B_X": "x"}, control=LRControl(verbose=0))
    r = m.fit()
    assert_allclose(r.params[:-1], [data.x @ data.y / (data.x @ data.x)])
    assert_allclose(r.r_squared, 1 - r.residuals @ r.residuals / (data.y @ data.y))
