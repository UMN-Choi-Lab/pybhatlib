"""MDCEV constants-only benchmarks use the basic, unmixed engine."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._fit_diagnostics import mdcev_constants_loglik
from pybhatlib.models.mdcev import MDCEVControl, MDCEVModel, MDCEVResults
from pybhatlib.models.mdcev_mixed import MDCEVMixedControl, MDCEVMixedModel, MDCEVMixedResults


def _data():
    rng = np.random.default_rng(43)
    n = 30
    frame = pd.DataFrame({"outside": rng.uniform(1, 4, n),
                          "a": rng.uniform(.2, 2, n), "b": rng.uniform(.2, 2, n),
                          "p0": rng.uniform(.8, 1.2, n),
                          "p1": rng.uniform(.8, 1.2, n),
                          "p2": rng.uniform(.8, 1.2, n),
                          "x": rng.normal(size=n), "w": rng.uniform(.5, 2, n)})
    frame.loc[::3, "a"] = 0.
    frame.loc[1::3, "b"] = 0.
    frame["pid"] = np.repeat(np.arange(n // 2), 2)
    return frame


@pytest.mark.parametrize("utility", ["trad", "linear"])
def test_mdcev_baseline_matches_explicit_constants_fit_and_mixed_normalization(utility):
    frame = _data()
    alts = ["outside", "a", "b"]
    prices = ["p0", "p1", "p2"]
    uspec = {"Ca": {"a": "uno"}, "Cb": {"b": "uno"}}
    gspec = {"Ga": {"a": "uno"}, "Gb": {"b": "uno"}}
    ctrl = MDCEVControl(utility=utility, weight_var="w", verbose=0,
                        want_covariance=False, tol=1e-6, maxiter=1000)
    reference_model = MDCEVModel(frame, alts, availability=prices,
                                 utility_spec=uspec, gamma_spec=gspec, control=ctrl)
    reference_model._skip_fit_diagnostics = True
    reference = reference_model.fit()
    assert reference.converged
    ll = mdcev_constants_loglik(frame[alts].to_numpy(), frame[prices].to_numpy(),
                                utility=utility, outside_good_gamma=-1000.,
                                observation_weights=frame.w.to_numpy())
    assert ll == pytest.approx(reference.loglik, abs=1e-8)
    uspec["x"] = {"a": "x", "b": "x"}
    gspec["gx"] = {"a": "x"}
    fitted = MDCEVModel(frame, alts, availability=prices, utility_spec=uspec,
                        gamma_spec=gspec, control=ctrl).fit()
    assert fitted.loglik_constants == pytest.approx(ll, abs=1e-8)
    assert fitted.n_constants_params == 5  # 2 ASCs + 2 gamma + sigma
    assert fitted.n_free_params == len(fitted.params)
    assert fitted.rho2_adjusted_constants == pytest.approx(
        1 - (fitted.loglik - (fitted.n_free_params - 5) / len(frame)) / ll,
    )
    assert "Mean LL(C)" in fitted.summary()
    assert "Adj. rho2 vs LL(C)" in fitted.summary()
    assert "rho2 vs LL(0)" not in fitted.summary()

    # With no mixing, person weights are still averaged within each person.
    mixed_ctrl = MDCEVMixedControl(utility=utility, verbose=0, maxiter=0,
                                   n_rep=1, person_id="pid", weight_var="w",
                                   want_covariance=False)
    mixed = MDCEVMixedModel(frame, alts, price=prices, utility_spec=uspec,
                            gamma_spec=gspec, control=mixed_ctrl).fit()
    persons = frame.pid.to_numpy()
    person_weights = frame.groupby("pid").w.mean().to_numpy()
    direct = mdcev_constants_loglik(frame[alts].to_numpy(), frame[prices].to_numpy(),
                                    utility=utility, outside_good_gamma=-1000.,
                                    observation_weights=person_weights[persons])
    assert mixed.loglik_constants == pytest.approx(direct * 2, abs=1e-8)
    assert mixed.n_constants_params == 5
    assert mixed.rho2_adjusted_constants == pytest.approx(
        1 - (mixed.loglik - (mixed.n_free_params - 5) / mixed.n_ind)
        / mixed.loglik_constants,
    )
    assert "Mean LL(C)" in mixed.summary()


def test_external_mdcev_results_have_no_training_baseline():
    fixed = MDCEVResults.from_estimates(np.array([0., 0., 1.]))
    mixed = MDCEVMixedResults.from_estimates(np.array([0., 0., 1.]))
    assert np.isnan(fixed.loglik_constants) and np.isnan(mixed.loglik_constants)
    assert np.isnan(fixed.rho2_adjusted_constants)
    assert np.isnan(mixed.rho2_adjusted_constants)
    for result in (fixed, mixed):
        line = next(line for line in result.summary().splitlines() if "Mean LL(C)" in line)
        assert "N/A" in line
