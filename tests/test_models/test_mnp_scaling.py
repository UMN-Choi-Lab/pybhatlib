"""Identification changes must preserve MNP probabilities and score chains."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._mnp_scaling import (
    kernel_scales, kernel_scale_jacobian, encode_kernel_scales,
)
from pybhatlib.models.mnp import MNPControl, MNPModel, MNPResults
from pybhatlib.models.mnp._mnp_ate import mnp_ate_from_params
from pybhatlib.models.mnp._mnp_forecast import mnp_predict
from pybhatlib.models.mnp._mnp_loglik import (
    _build_lambda, _build_lambda_direct, _lambda_par_to_unpar,
    _param_to_unpar, _per_obs_loglik_unpar, count_params, mnp_loglik,
)
from pybhatlib.models.mnpkercp import MNPKerCPControl, MNPKerCPModel
from pybhatlib.models.mnpkercp._mnpkercp_ate import (
    _prepare_at_params, mnpkercp_ate_from_params,
)
from pybhatlib.models.mnpkercp._mnpkercp_forecast import mnpkercp_predict


def _data(n_alts=3):
    rng = np.random.default_rng(18)
    n = 12
    alts = [f"a{k}" for k in range(n_alts)]
    y = np.arange(n) % n_alts
    data = pd.DataFrame({a: (y == k).astype(float) for k, a in enumerate(alts)})
    for k in range(n_alts):
        data[f"x{k}"] = rng.normal(scale=0.4, size=n)
    data["person"] = np.repeat(np.arange(n // 2), 2)
    spec = {"x": {a: f"x{k}" for k, a in enumerate(alts)}}
    return data, alts, spec


def _fd(function, theta, eps=1e-6):
    return np.array([
        (function(theta + eps * row) - function(theta - eps * row)) / (2 * eps)
        for row in np.eye(len(theta))
    ])


@pytest.mark.parametrize("control,default", [(MNPControl, "sumsq"), (MNPKerCPControl, "sumsq")])
def test_defaults_and_validation(control, default):
    assert control().scaling == default
    with pytest.raises(ValueError, match="scaling"):
        control(scaling="invalid")


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
@pytest.mark.parametrize("dim", [1, 2, 4])
def test_scale_coordinates_and_jacobian(scaling, dim):
    raw = np.linspace(-0.4, 0.7, dim - 1)
    scales = kernel_scales(raw, scaling)
    np.testing.assert_allclose(encode_kernel_scales(scales, scaling), raw, atol=1e-14)
    if scaling == "first":
        assert scales[0] == 1
    else:
        assert scales @ scales == pytest.approx(1)
    fd = np.array([
        (kernel_scales(raw + 1e-6 * row, scaling)
         - kernel_scales(raw - 1e-6 * row, scaling)) / 2e-6
        for row in np.eye(len(raw))
    ]).reshape(dim - 1, dim)
    np.testing.assert_allclose(kernel_scale_jacobian(scales, scaling), fd, atol=1e-9)


@pytest.mark.parametrize("iid,heteronly,mix,nseg", [
    (True, False, False, 1), (False, True, False, 1),
    (False, False, False, 1), (False, False, True, 1),
    (False, False, True, 2),
])
@pytest.mark.parametrize("n_alts", [3, 4])
@pytest.mark.parametrize("varying_availability", [False, True])
def test_basic_sumsq_likelihood_and_gradient(iid, heteronly, mix, nseg, n_alts, varying_availability):
    rng = np.random.default_rng(27)
    x = rng.normal(scale=0.3, size=(9, n_alts, 2))
    y = np.arange(9) % n_alts
    control = MNPControl(scaling="sumsq", iid=iid, heteronly=heteronly,
                         mix=mix, nseg=nseg, randdiag=True, method="me", verbose=0)
    ranvars = [0] if mix else None
    theta = np.full(count_params(2, n_alts, control, ranvars), 0.15)
    avail = np.ones((9, n_alts))
    if varying_availability:
        avail[0, 1] = 0
        avail[1, -1] = 0
    args = (x, y, avail, n_alts, 2, control, ranvars)
    value, score = mnp_loglik(theta, *args, return_gradient=True)
    fd = _fd(lambda t: mnp_loglik(t, *args), theta)
    np.testing.assert_allclose(score, fd, atol=2e-6)
    natural = _param_to_unpar(theta, 2, n_alts, control, ranvars)
    assert -_per_obs_loglik_unpar(natural, *args).mean() == pytest.approx(value, abs=1e-10)


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
@pytest.mark.parametrize("n_alts", [2, 3, 4])
def test_basic_external_covariance_prediction_and_ate(scaling, n_alts):
    data, alts, spec = _data(n_alts)
    model = MNPModel(data=data, alternatives=alts, spec=spec, control=MNPControl(verbose=0))
    dim = n_alts - 1
    scales = np.linspace(0.8, 1.3, dim)
    corr = 0.8 * np.eye(dim) + 0.2 * np.ones((dim, dim))
    cov = np.outer(scales, scales) * corr
    normalized = scales[0] if scaling == "first" else np.linalg.norm(scales)
    c = MNPControl(scaling=scaling, method="me", verbose=0)
    external = MNPResults.from_estimates([0.6], kernel_cov=cov, control=c)
    matching = MNPResults.from_estimates([0.6 / normalized], kernel_cov=cov / normalized**2, control=c)
    k = _build_lambda(external.params[1:], n_alts, c)
    np.testing.assert_allclose(k, cov / normalized**2, atol=1e-12)
    np.testing.assert_allclose(mnp_predict(external, model.X), mnp_predict(matching, model.X))
    ate = mnp_ate_from_params([0.6], kernel_cov=cov, control=c, data=data,
                             alternatives=alts, spec=spec, scenarios={"zero": {"x1": 0}})
    np.testing.assert_allclose(ate.predicted_shares, mnp_predict(external, model.X).mean(0), atol=1e-7)


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
@pytest.mark.parametrize("copula", [False, True])
def test_mixed_panel_gradient_and_external_reporting_roundtrip(scaling, copula):
    data, alts, spec = _data()
    ctrl = MNPKerCPControl(scaling=scaling, normvar=["x"], copula=copula,
                           person_id=None if copula else "person", n_rep=3,
                           method="me", verbose=0, floor_pcomp=0, floor_z=0)
    model = MNPKerCPModel(data=data, alternatives=alts, spec=spec, control=ctrl)
    mixing, layout = model._build_spec_layout()
    from pybhatlib.vecup._panel import PanelIndex
    est = model._build_estimator(mixing, layout, PanelIndex.from_ids(model.person_ids))
    theta = np.linspace(-0.2, 0.2, layout.n_theta)
    ll, score = est.simulated_loglik(theta, want_grad=True)
    fd = _fd(lambda t: est.simulated_loglik(t, want_grad=False)[0].sum(), theta)
    np.testing.assert_allclose(score.sum(0), fd, atol=2e-6)
    # Changing only identification and consistently rescaling utility means
    # and random-coefficient SDs must leave every panel likelihood unchanged.
    other = "first" if scaling == "sumsq" else "sumsq"
    sl = layout.slices()
    scales = kernel_scales(theta[sl["kern"]], scaling)
    factor = scales[0] if other == "first" else np.linalg.norm(scales)
    equivalent = theta.copy()
    equivalent[sl["kern"]] = encode_kernel_scales(scales / factor, other)
    equivalent[sl["beta"]] /= factor
    equivalent[sl["scal"]] -= np.log(factor)
    est.kernel.scaling = other
    other_ll, _ = est.simulated_loglik(equivalent, want_grad=False)
    np.testing.assert_allclose(other_ll, ll, atol=1e-10)
    est.kernel.scaling = scaling
    # Isolate scale entry from the existing direct-correlation entry path.
    theta[layout.slices()["rcor"]] = 0.0
    ll, _ = est.simulated_loglik(theta, want_grad=False)
    reported = model._to_reporting(theta, mixing, layout, est.space)
    original = reported.copy()
    _prepare_at_params(model, reported)
    np.testing.assert_array_equal(reported, original)
    ll_external, _ = model._est.simulated_loglik(model._theta_hat, want_grad=False)
    np.testing.assert_allclose(ll_external, ll, atol=1e-10)
    forecast = mnpkercp_predict(model)
    ate = mnpkercp_ate_from_params(reported, data=data, alternatives=alts,
                                 spec=spec, control=ctrl, scenarios={"zero": {"x1": 0}})
    np.testing.assert_allclose(ate.predicted_shares, forecast, atol=1e-10)


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
def test_regular_and_mixed_kernel_normalization_agree(scaling):
    data, alts, spec = _data(4)
    ctrl = MNPKerCPControl(scaling=scaling, verbose=0, n_rep=1)
    model = MNPKerCPModel(data=data, alternatives=alts, spec=spec, control=ctrl)
    mixing, layout = model._build_spec_layout()
    theta = np.zeros(layout.n_theta)
    raw = np.array([-0.3, 0.6])
    theta[layout.slices()["kern"]] = raw
    from pybhatlib.vecup._panel import PanelIndex
    est = model._build_estimator(mixing, layout, PanelIndex.from_ids(model.person_ids))
    state = est.kernel.prepare(theta, layout)
    scales = np.diag(state.wdiagker)
    basic = MNPControl(scaling=scaling)
    params = np.r_[raw, np.zeros(3)]
    cov = _build_lambda(params, 4, basic)
    np.testing.assert_allclose(np.diag(cov), scales**2, atol=1e-14)
    direct = _lambda_par_to_unpar(params, 4, basic)
    np.testing.assert_allclose(_build_lambda_direct(direct, 4, basic), cov, atol=1e-14)


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
def test_basic_reporting_marks_only_fixed_reference_scale(scaling):
    data, alts, spec = _data()
    ctrl = MNPControl(scaling=scaling, method="me", maxiter=0, verbose=0)
    model = MNPModel(data=data, alternatives=alts, spec=spec, control=ctrl)
    theta = np.array([0.4, 0.2, 0.1])
    report, se, *_ = model._normalize_for_reporting(theta, None, np.zeros(3))
    if scaling == "first":
        assert report[2] == 1 and np.isnan(se[2])
    else:
        assert np.square(report[2:]).sum() == pytest.approx(1)
        assert np.all(np.isfinite(se[2:]))
    assert ("(fixed)" in " ".join(model._build_report_names())) == (scaling == "first")


def test_sumsq_fixed_logit_does_not_fix_individual_scales():
    data, alts, spec = _data(4)
    # Freeze just the first scale logit; all three natural scales still depend
    # on the remaining active logit through the sum-of-squares constraint.
    active = np.array([True, False, True, True, True, True])
    ctrl = MNPControl(scaling="sumsq", method="me", verbose=0, active_mask=active)
    model = MNPModel(data=data, alternatives=alts, spec=spec, control=ctrl)
    theta = np.array([0.4, -0.2, 0.3, 0.1, -0.1, 0.2])
    _, se, *_ = model._normalize_for_reporting(theta, None, np.zeros(6), active_mask=active)
    assert np.all(np.isfinite(se[4:]))


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
def test_mixed_fit_reports_selected_kernel_scales(scaling):
    data, alts, spec = _data()
    ctrl = MNPKerCPControl(scaling=scaling, method="me", verbose=0, n_rep=1,
                           maxiter=0, want_covariance=False)
    model = MNPKerCPModel(data=data, alternatives=alts, spec=spec, control=ctrl)
    _, layout = model._build_spec_layout()
    start = np.array([0.4, 0.1, -0.3])
    assert len(start) == layout.n_theta
    ctrl.startb = start.copy()
    result = model.fit()
    np.testing.assert_array_equal(model._theta_hat, start)
    scales = result.mixing_summary.scales
    np.testing.assert_allclose(scales, kernel_scales([-0.3], scaling))
    expected_free = kernel_scales([-0.3], scaling)[1:]
    np.testing.assert_allclose(result.params[layout.slices()["kern"]], expected_free)
