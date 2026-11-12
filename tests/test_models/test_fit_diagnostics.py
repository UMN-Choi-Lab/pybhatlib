"""Null benchmarks and arithmetic observed-choice probabilities."""

import importlib

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._fit_diagnostics import (
    ProbabilityCache, baseline_loglikelihoods, record_probabilities,
)
from pybhatlib.mixed._engine import MixedMSLEstimator, Tracer


def _data():
    rng = np.random.default_rng(2026)
    chosen = np.array([0] * 6 + [1] * 4 + [2] * 2)
    df = pd.DataFrame({f"a{j}": (chosen == j).astype(int) for j in range(3)})
    for j in range(3):
        df[f"x{j}"] = rng.normal(size=len(df))
    df["y1"] = chosen
    df["y2"] = [0, 1, 0, 1, 0, 2, 2, 0, 1, 1, 1, 2]
    df["pid"] = [0, 0, 0, 1, 2, 2, 3, 3, 3, 3, 4, 5]
    return df


def _model(family, *, analytic=True, iid=True):
    df = _data()
    if family in ("morp", "morp_flex"):
        module = importlib.import_module(f"pybhatlib.models.{family}")
        cls, control = ((module.MORPModel, module.MORPControl) if family == "morp"
                        else (module.MORPFlexModel, module.MORPFlexControl))
        kwargs = dict(maxiter=0, verbose=0, iid=iid)
        if family == "morp_flex":
            kwargs.update(n_rep=3, normvar=("x",), person_id="pid", want_covariance=False)
        else:
            kwargs.update(analytic_grad=analytic)
        return cls(df, dep_vars=["y1", "y2"], n_categories=[3, 3],
                   spec={"x": {"y1": "x0", "y2": "x1"}}, control=control(**kwargs))
    module = importlib.import_module(f"pybhatlib.models.{family}")
    names = {"mnl": ("MNLModel", "MNLControl"), "mnp": ("MNPModel", "MNPControl"),
             "mixmnl": ("MixMNLModel", "MixMNLControl"),
             "mnpkercp": ("MNPKerCPModel", "MNPKerCPControl")}
    cls, control = (getattr(module, name) for name in names[family])
    kwargs = dict(maxiter=0, verbose=0, want_covariance=False)
    if family in ("mixmnl", "mnpkercp"):
        kwargs.update(n_rep=3, normvar=("x",), person_id="pid")
    if family == "mnp":
        kwargs.update(iid=True, analytic_grad=analytic)
    return cls(df, alternatives=["a0", "a1", "a2"],
               spec={"x": {f"a{j}": f"x{j}" for j in range(3)}}, control=control(**kwargs))


def test_independent_ordinal_nulls_and_unobserved_categories():
    y = np.array([[0, 0], [0, 1], [0, 1], [1, 2]])
    ll_c, ll_0 = baseline_loglikelihoods(y, [3, 4])
    expected = (np.log([0.75, 0.75, 0.75, 0.25])
                + np.log([0.25, 0.5, 0.5, 0.25])).mean()
    assert ll_c == pytest.approx(expected)
    assert ll_0 == pytest.approx(-np.log(12))


def test_availability_specific_shares_and_panel_weight_normalization():
    y = np.array([0, 0, 1, 1])
    avail = np.array([[1, 0], [1, 1], [1, 1], [1, 1]])
    persons = np.array([0, 0, 1, 1])
    weights = np.array([2.0, 1.0])
    ll_c, ll_0 = baseline_loglikelihoods(
        y, [2], availability=avail, row_to_person=persons, person_weights=weights,
    )
    assert ll_c == pytest.approx(4 * np.log(0.5) / 2)
    assert ll_0 == pytest.approx(-4 * np.log(2) / 2)


@pytest.mark.parametrize("family", ["mnl", "mnp", "morp", "mixmnl", "mnpkercp", "morp_flex"])
def test_fit_stores_nulls_and_actual_observed_probabilities(family, monkeypatch):
    collected = []
    original = MixedMSLEstimator.simulated_loglik

    def traced(self, theta, **kwargs):
        if kwargs.get("collect_probabilities"):
            self.trace = Tracer()
        result = original(self, theta, **kwargs)
        if kwargs.get("collect_probabilities"):
            per_case = np.mean(list(self.trace.per_rep["pcomp"].values()), axis=0)
            per_person = self.panel.scatter_sum(per_case) / np.bincount(self.panel.row_to_person)
            collected.append(float(per_person.mean()))
        return result

    monkeypatch.setattr(MixedMSLEstimator, "simulated_loglik", traced)
    model = _model(family)
    result = model.fit()
    y = model.y_ord if family == "morp_flex" else model.y
    is_ordinal = family in ("morp", "morp_flex")
    categories = [3, 3] if is_ordinal else [3]
    y = np.asarray(y).reshape(len(y), -1)
    expected_c = sum(np.log(np.bincount(y[:, d], minlength=k)[y[:, d]] / len(y)).sum()
                     for d, k in enumerate(categories))
    n_ind = 6 if family in ("mixmnl", "mnpkercp", "morp_flex") else len(y)
    assert result.loglik_constants == pytest.approx(expected_c / n_ind)
    assert result.loglik_zero == pytest.approx(-len(y) / n_ind * np.log(np.prod(categories)))
    assert np.isfinite(result.average_correct_probability)
    assert 0 <= result.average_correct_probability <= 1
    if collected:
        assert len(collected) == 1  # One existing final pass, not an extra evaluation.
        assert result.average_correct_probability == pytest.approx(collected[0])
    elif family == "morp":
        from pybhatlib.models.morp._morp_loglik import _per_obs_loglik
        reference = np.exp(_per_obs_loglik(
            result.params, model.X, model.y, model.n_dims, model.n_categories,
            model.n_beta, model.control,
        )).mean()
        assert result.average_correct_probability == pytest.approx(reference)
    else:
        probabilities = model.predict()
        reference = probabilities[np.arange(len(y)), y[:, 0]].mean()
        assert result.average_correct_probability == pytest.approx(reference, abs=1e-7)
    summary = result.summary()
    assert "Mean LL(C)" in summary and "Mean LL(0)" in summary
    assert "APCP" in summary
    if family in ("mnpkercp", "morp_flex"):
        # The distribution blocks and null/prediction diagnostics must coexist
        # after combining the previously conflicting fit-result constructors.
        mixing = result.mixing_summary
        assert mixing is not None and mixing.n_random == 1
        assert len(mixing.names) == 3
        assert np.all(mixing.scales > 0)
        np.testing.assert_allclose(np.diag(mixing.corr_matrix), 1.0)
        np.testing.assert_allclose(np.diag(mixing.cov_matrix), mixing.scales**2)
        assert "Joint random-coefficient / kernel-error" in summary


@pytest.mark.parametrize("family", ["mnl", "mnp", "morp"])
@pytest.mark.parametrize("analytic", [True, False])
def test_diagnostics_add_no_likelihood_calls(family, analytic, monkeypatch):
    module = importlib.import_module(f"pybhatlib.models.{family}._{family}_model")
    likelihood = getattr(module, f"{family}_loglik")
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return likelihood(*args, **kwargs)

    monkeypatch.setattr(module, f"{family}_loglik", counted)
    result = _model(family, analytic=analytic).fit()
    with_collection = len(calls)
    assert np.isfinite(result.average_correct_probability)

    class WithoutCollection(ProbabilityCache):
        def evaluate(self, theta, function):
            return function(theta)

    monkeypatch.setattr(module, "ProbabilityCache", WithoutCollection)
    calls.clear()
    _model(family, analytic=analytic).fit()
    assert len(calls) == with_collection


def test_cache_reuses_evaluation_and_restores_observer_after_errors():
    def objective(theta):
        record_probabilities([0.1, 0.9])
        return 2.0, np.zeros(1)

    cache = ProbabilityCache(objective, 2)
    cache(np.zeros(1))
    assert cache.get(np.zeros(1)) == pytest.approx(0.5)
    assert np.isnan(cache.get(np.ones(1)))
    with pytest.raises(ValueError):
        cache.evaluate(np.zeros(1), lambda theta: (_ for _ in ()).throw(ValueError()))
    record_probabilities([0.0])
    assert cache.get(np.zeros(1)) == pytest.approx(0.5)


@pytest.mark.parametrize("segments,varying_availability", [(1, True), (2, False), (2, True)])
@pytest.mark.parametrize("analytic", [True, False])
def test_mnp_probability_capture_includes_mixtures_and_single_available_choices(
    segments, varying_availability, analytic,
):
    from pybhatlib.models.mnp import MNPControl
    from pybhatlib.models.mnp._mnp_loglik import mnp_loglik, _per_obs_loglik, count_params

    rng = np.random.default_rng(12)
    x = rng.normal(scale=0.2, size=(6, 3, 2))
    y = np.array([0, 1, 2, 0, 1, 2])
    avail = np.ones((6, 3))
    if varying_availability:
        avail[0] = [1, 0, 0]
        avail[1, 2] = 0
    ctrl = MNPControl(iid=True, mix=True, nseg=segments, randdiag=True,
                      analytic_grad=analytic, method="me", verbose=0)
    n = count_params(2, 3, ctrl, [0])
    theta = np.full(n, 0.15)
    objective = ProbabilityCache(
        lambda p: mnp_loglik(p, x, y, avail, 3, 2, ctrl, [0], return_gradient=True), 6,
    )
    objective(theta)
    reference = np.exp(_per_obs_loglik(theta, x, y, avail, 3, 2, ctrl, [0])).mean()
    assert objective.get(theta) == pytest.approx(reference, abs=1e-7)


def test_morp_final_exact_evaluation_supplies_joint_probability():
    from pybhatlib.models.morp._morp_control import morp_control_replace
    from pybhatlib.models.morp._morp_loglik import _per_obs_loglik

    model = _model("morp", iid=False)
    result = model.fit()
    exact = morp_control_replace(model.control, method="scipy", analytic_grad=False)
    ll = _per_obs_loglik(
        result.params, model.X, model.y, model.n_dims, model.n_categories, model.n_beta, exact,
    )
    assert result.average_correct_probability == pytest.approx(np.exp(ll).mean(), abs=1e-8)


def test_rho_squared_hand_computed_with_adjustment():
    from pybhatlib.models._fit_diagnostics import rho_squared

    # n=10 cases: totals LL=-6, LL(C)=-8, LL(0)=-10, K=2 (means /10).
    r0, rc, radj = rho_squared(-0.6, -0.8, -1.0, 2, 10)
    assert r0 == pytest.approx(0.4)
    assert rc == pytest.approx(0.25)
    assert radj == pytest.approx(1 - (-6 - 2) / -10)  # 0.2
    assert np.isnan(rho_squared(-0.6, np.nan, np.nan, 2, 10)[1])
    assert np.isnan(rho_squared(-0.6, -0.8, -1.0, -1, 10)[2])


def test_results_rho2_uses_panel_normalization_and_free_params():
    from pybhatlib.models._fit_diagnostics import FitMetricsMixin
    from pybhatlib.models.mixmnl import MixMNLResults

    assert issubclass(MixMNLResults, FitMetricsMixin)

    class Res(FitMetricsMixin):
        params = np.zeros(5)
        loglik, loglik_constants, loglik_zero = -0.6, -0.8, -1.0
        n_obs, n_ind, n_free_params = 100, 10, 2

    res = Res()
    assert res.rho2_zero == pytest.approx(0.4)
    assert res.rho2_adjusted == pytest.approx(0.2)  # K/n_ind, not K/n_obs
    assert "Adj. rho2" in "\n".join(res.rho2_summary_lines())


def test_lazy_log_recording_equals_eager_and_skips_exp_without_recorder(monkeypatch):
    calls = []
    real_exp = np.exp
    monkeypatch.setattr(np, "exp", lambda *a, **k: calls.append(1) or real_exp(*a, **k))
    logp = np.log(np.array([0.2, 0.5, 0.8]))
    record_probabilities(logp, log=True)
    assert not calls  # no recorder: never exponentiated

    cache = ProbabilityCache(lambda t: record_probabilities(logp, log=True), 3)
    cache(np.zeros(1))
    assert cache.get(np.zeros(1)) == pytest.approx(0.5)


def test_cache_is_bounded_and_keeps_recent_entries():
    cache = ProbabilityCache(lambda t: record_probabilities(np.full(2, float(t[0]))), 2,
                             max_entries=3)
    for v in range(10):
        cache(np.array([float(v)]))
    assert len(cache.values) == 3
    assert cache.get(np.array([9.0])) == 9.0
    assert np.isnan(cache.get(np.array([0.0])))
