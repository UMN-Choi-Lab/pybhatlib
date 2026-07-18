"""GATE: mixed MDCEV ``.predict()`` / ``.ate()`` / ``.forecast()`` (Phase 4).

Self-contained (no GAUSS reference). Validates that the mixed MDCEV family wires
the shared mixed-prediction machinery (:mod:`pybhatlib.mixed._predict`) by
lifting the shipped fixed-coefficient MDCEV participation / allocation
formulation (:mod:`pybhatlib.models.mdcev._mdcev_forecast`,
:mod:`pybhatlib.models.mdcev._mdcev_ate`) over the mixing draws:

1. **COLLAPSE** -- with an empty random-coefficient spec (``nrndcoef == 0``) the
   drawn baseline utility equals the fixed ``beta`` for every draw, so the mixed
   ``predict`` / ``ate`` must reproduce the shipped fixed-coefficient
   ``mdcev_predict`` / ``mdcev_ate`` value-for-value on the same data (the
   Monte-Carlo seed is shared, so the RNG path is identical) -- ``1e-6``.
2. **MIXING ACTIVE** -- with a normal random coefficient, ``predict`` returns
   valid participation shares (finite, non-negative, summing to one),
   ``forecast`` returns a finite per-observation allocation, and
   ``ate(scenarios=...)`` returns a well-formed result.
3. **INTERFACE CONFORMANCE** -- the ATE result is an
   :class:`~pybhatlib.models._ate_common.ATEResultMixin` exposing the harmonized
   ``predicted_shares`` / ``shares_per_scenario`` fields plus ``.comparison()``
   and ``.summary()``; ``scenarios=`` accepts both dict and DataFrame form (via
   :func:`~pybhatlib.models._ate_common.scenarios_to_dict`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._predict import MixedATEResult
from pybhatlib.models._ate_common import ATEResultMixin
from pybhatlib.models.mdcev._mdcev_ate import mdcev_ate, mdcev_ate_from_params
from pybhatlib.models.mdcev._mdcev_forecast import mdcev_predict
from pybhatlib.models.mdcev._mdcev_model import MDCEVModel
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults
from pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast import mdcev_mixed_predict
from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_model import MDCEVMixedModel
from pybhatlib.models.mdcev_mixed._mdcev_mixed_ate import (
    mdcev_mixed_ate,
    mdcev_mixed_ate_from_params,
)
from pybhatlib.vecup._panel import PanelIndex

NC = 5            # outside + 4 inside goods
NVARM = 4         # baseline-utility parameters
NVARGAM = 4       # translation (gamma) parameters


def _make_frame(seed: int = 7):
    """Build a small synthetic MDCEV DataFrame + matching spec arrays.

    Returns the DataFrame, the alternative / price / utility / gamma spec
    objects, and the natural-space (``beta``, ``gamma_raw``, ``log_sigma``)
    parameters used to drive both the mixed and the shipped evaluators.
    """
    rng = np.random.default_rng(seed)
    n_obs = 15

    df = pd.DataFrame()
    # consumption: outside good always consumed; inside a mixed pattern.
    cons = np.zeros((n_obs, NC))
    cons[:, 0] = rng.uniform(1.0, 4.0, n_obs)
    for q in range(n_obs):
        k = rng.integers(1, NC - 1)
        picks = rng.choice(np.arange(1, NC), size=k, replace=False)
        cons[q, picks] = rng.uniform(0.5, 3.0, k)
    alt_cols = [f"q{k}" for k in range(NC)]
    for k in range(NC):
        df[alt_cols[k]] = cons[:, k]

    price = np.tile(rng.uniform(0.8, 1.5, NC), (n_obs, 1))
    price[:, 0] = 1.0
    price_cols = [f"p{k}" for k in range(NC)]
    for k in range(NC):
        df[price_cols[k]] = price[:, k]

    utility_spec = np.empty((NC, NVARM), dtype=object)
    for k in range(NC):
        for v in range(NVARM):
            nm = f"x{k}_{v}"
            df[nm] = 0.5 * rng.standard_normal(n_obs)
            utility_spec[k, v] = nm

    # gamma param 0 loads the outside good only (zero inside design).
    gamma_spec = np.empty((NC, NVARGAM), dtype=object)
    for k in range(NC):
        for j in range(NVARGAM):
            if j == 0:
                gamma_spec[k, j] = "sero"
            else:
                nm = f"g{k}_{j}"
                df[nm] = rng.uniform(0.3, 1.2, n_obs) if k > 0 else 0.0
                gamma_spec[k, j] = nm

    df["uno"] = 1.0
    df["sero"] = 0.0
    df["ID"] = np.arange(n_obs)

    beta = 0.4 * rng.standard_normal(NVARM)
    gamma_raw = np.zeros(NVARGAM, dtype=np.float64)
    gamma_raw[0] = -1000.0
    gamma_raw[1:] = 0.3 * rng.standard_normal(NVARGAM - 1)
    log_sigma = 0.15

    return dict(
        df=df, alt_cols=alt_cols, price_cols=price_cols,
        utility_spec=utility_spec, gamma_spec=gamma_spec,
        beta=beta, gamma_raw=gamma_raw, log_sigma=log_sigma,
    )


def _param_names():
    return (
        [f"b{v}" for v in range(NVARM)]
        + [f"g{j}" for j in range(NVARGAM)]
    )


def _prime_fitted(model: MDCEVMixedModel, theta_setter) -> None:
    """Assemble the engine and set the fitted state at a chosen ``theta``.

    Bypasses the optimizer (the collapse / interface gates need a *known*
    parameter vector, not an estimated one). ``theta_setter(sl, layout)`` returns
    the estimation-space ``theta``.
    """
    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    est = model.build_estimator(spec, layout, panel)
    model._fitted_spec = spec
    model._fitted_layout = layout
    model._fitted_est = est
    model._fitted_theta = theta_setter(layout.slices(), layout)
    model.results_ = object()   # sentinel so model.predict/ate _require_results passes


def _make_mixed(fx, control):
    return MDCEVMixedModel(
        fx["df"], alternatives=fx["alt_cols"], price=fx["price_cols"],
        utility_spec=fx["utility_spec"], gamma_spec=fx["gamma_spec"],
        param_names=[f"b{v}" for v in range(NVARM)],
        gamma_names=[f"g{j}" for j in range(NVARGAM)],
        control=control,
    )


# ---------------------------------------------------------------------------
# 1. COLLAPSE
# ---------------------------------------------------------------------------

def test_collapse_predict_equals_shipped_mdcev_predict():
    """nrndcoef=0 mixed predict == shipped mdcev_predict (sample mean), 1e-6."""
    fx = _make_frame()
    beta, gamma_raw, log_sigma = fx["beta"], fx["gamma_raw"], fx["log_sigma"]

    ctrl = MDCEVMixedControl(utility="trad", n_rep=3, draw_seed=0)
    m = _make_mixed(fx, ctrl)

    def setter(sl, layout):
        theta = np.zeros(layout.n_theta, dtype=np.float64)
        theta[sl["beta"]] = beta
        theta[sl["gamma"]] = gamma_raw
        theta[sl["kern"]] = log_sigma
        return theta

    _prime_fitted(m, setter)
    assert m._fitted_spec.nrndcoef == 0

    shares_mixed = m.predict()

    ref = mdcev_predict(
        None, m.X, m.gamma_design, m.price,
        n_draws=1000, seed=1234,
        b_reported=np.concatenate([beta, gamma_raw]),
        sigma=float(np.exp(log_sigma)), outside_good_gamma=-1000.0,
    ).mean(axis=0)

    assert shares_mixed.shape == (NC,)
    np.testing.assert_allclose(shares_mixed, ref, rtol=0.0, atol=1e-6)


def test_randdiag_removes_random_correlation_block():
    fx = _make_frame()
    model = _make_mixed(
        fx, MDCEVMixedControl(normvar=("b0", "b1"), randdiag=True)
    )
    spec, layout = model._build_spec_layout()
    state = model.build_estimator(spec, layout).space.unpack(
        np.zeros(layout.n_theta), spec
    )

    assert layout.n_rcor == 0
    np.testing.assert_array_equal(state.omegastar, np.eye(2))


def test_fix_location_zero_is_wired_to_shared_spec():
    fx = _make_frame()
    model = _make_mixed(
        fx,
        MDCEVMixedControl(
            normvar=("b0",), fix_location_zero=("b0",)
        ),
    )
    spec, _ = model._build_spec_layout()

    assert spec.fix_location_zero_mask[0] == 1.0


def test_collapse_ate_equals_shipped_mdcev_ate():
    """nrndcoef=0 mixed ate == shipped mdcev_ate (base + per scenario), 1e-6."""
    fx = _make_frame()
    beta, gamma_raw, log_sigma = fx["beta"], fx["gamma_raw"], fx["log_sigma"]
    sigma = float(np.exp(log_sigma))

    ctrl = MDCEVMixedControl(utility="trad", n_rep=3, draw_seed=0)
    m = _make_mixed(fx, ctrl)

    def setter(sl, layout):
        theta = np.zeros(layout.n_theta, dtype=np.float64)
        theta[sl["beta"]] = beta
        theta[sl["gamma"]] = gamma_raw
        theta[sl["kern"]] = log_sigma
        return theta

    _prime_fitted(m, setter)

    scenarios = {"lo": {"x1_0": -0.5}, "hi": {"x1_0": 0.5}}
    mixed = m.ate(scenarios=scenarios)

    # shipped reference: a fixed-coefficient MDCEVModel + mdcev_ate scenario mode.
    # The harmonized model .ate() uses the underlying default MC draw count /
    # seed (n_draws=1000, seed=1234), so the reference must match.
    ship_model = MDCEVModel(
        fx["df"], alternatives=fx["alt_cols"], availability=fx["price_cols"],
        utility_spec=fx["utility_spec"], gamma_spec=fx["gamma_spec"],
        param_names=[f"b{v}" for v in range(NVARM)],
        gamma_names=[f"g{j}" for j in range(NVARGAM)],
    )
    res = MDCEVResults.from_estimates(
        np.concatenate([beta, gamma_raw, [sigma]]), sigma,
        param_names=_param_names() + ["sigma"],
    )
    ship = mdcev_ate(
        res, model=ship_model, data=fx["df"], scenarios=scenarios,
        n_draws=1000, seed=1234, alternative_names=fx["alt_cols"],
    )

    np.testing.assert_allclose(
        mixed.predicted_shares, ship.predicted_shares, rtol=0.0, atol=1e-6
    )
    for name in scenarios:
        np.testing.assert_allclose(
            mixed.shares_per_scenario[name],
            ship.shares_per_scenario[name],
            rtol=0.0, atol=1e-6,
        )
    # the derived pairwise ATE also collapses.
    np.testing.assert_allclose(
        mixed.comparison("lo", "hi"),
        ship.comparison("lo", "hi"),
        rtol=0.0, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# 2. MIXING ACTIVE
# ---------------------------------------------------------------------------

def _mixed_active():
    fx = _make_frame()
    beta, gamma_raw, log_sigma = fx["beta"], fx["gamma_raw"], fx["log_sigma"]
    ctrl = MDCEVMixedControl(
        utility="trad", n_rep=4, draw_seed=0, normvar=("b1",),
    )
    m = _make_mixed(fx, ctrl)

    def setter(sl, layout):
        theta = np.zeros(layout.n_theta, dtype=np.float64)
        theta[sl["beta"]] = beta
        theta[sl["gamma"]] = gamma_raw
        theta[sl["kern"]] = log_sigma
        theta[sl["scal"]] = 0.0     # exp(0) -> sd = 1 (active heterogeneity)
        theta[sl["lam"]] = 0.0      # identity Yeo-Johnson
        return theta

    _prime_fitted(m, setter)
    assert m._fitted_spec.nrndcoef == 1
    return m, fx


def test_mixing_active_predict_valid_shares():
    m, _ = _mixed_active()
    shares = m.predict()
    assert shares.shape == (NC,)
    assert np.all(np.isfinite(shares))
    assert np.all(shares >= 0.0)
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-9)


def test_mixing_active_forecast_valid_allocation():
    m, fx = _mixed_active()
    fc = m.forecast(n_replications=30, seed=5)
    n_obs = len(fx["df"])
    assert fc.shape == (n_obs, NC)
    assert np.all(np.isfinite(fc))
    assert np.all(fc >= 0.0)


def test_mixing_active_ate_well_formed():
    m, _ = _mixed_active()
    scenarios = {"lo": {"x1_0": -0.5}, "hi": {"x1_0": 0.5}}
    res = m.ate(scenarios=scenarios)

    assert isinstance(res, MixedATEResult)
    assert res.predicted_shares.shape == (NC,)
    assert np.all(np.isfinite(res.predicted_shares))
    assert set(res.shares_per_scenario) == {"lo", "hi"}
    for s in res.shares_per_scenario.values():
        assert s.shape == (NC,)
        assert np.all(np.isfinite(s))


# ---------------------------------------------------------------------------
# 3. INTERFACE CONFORMANCE
# ---------------------------------------------------------------------------

def test_interface_conformance_ate_result():
    m, _ = _mixed_active()
    scenarios = {"base": {"x1_0": 0.0}, "treat": {"x1_0": 1.0}}
    res = m.ate(scenarios=scenarios)

    # harmonized ATEResultMixin surface.
    assert isinstance(res, ATEResultMixin)
    assert hasattr(res, "predicted_shares")
    assert hasattr(res, "shares_per_scenario")
    assert res.n_obs == len(m.data)

    # .comparison() -> percentage change vector.
    pct = res.comparison("base", "treat")
    assert pct.shape == (NC,)
    assert np.all(np.isfinite(pct))

    # .summary() -> a printable string.
    txt = res.summary()
    assert isinstance(txt, str) and "ATE" in txt


def test_interface_scenarios_dataframe_form():
    """scenarios= accepts DataFrame form (scenarios_to_dict), same as dict."""
    m, _ = _mixed_active()
    scen_df = pd.DataFrame(
        {"x1_0": [-0.5, 0.5]}, index=["lo", "hi"]
    )
    res_df = m.ate(scenarios=scen_df)
    res_dict = m.ate(
        scenarios={"lo": {"x1_0": -0.5}, "hi": {"x1_0": 0.5}},
    )
    assert set(res_df.shares_per_scenario) == {"lo", "hi"}
    for name in ("lo", "hi"):
        np.testing.assert_allclose(
            res_df.shares_per_scenario[name],
            res_dict.shares_per_scenario[name],
            rtol=0.0, atol=1e-12,
        )


@pytest.mark.parametrize("bad_scenarios", [None])
def test_ate_requires_scenarios(bad_scenarios):
    m, _ = _mixed_active()
    with pytest.raises(ValueError):
        m.ate(scenarios=bad_scenarios)


# ---------------------------------------------------------------------------
# 4. ate_from_params ACCEPTS REPORTING-SPACE (NATURAL) PARAMS
# ---------------------------------------------------------------------------

def test_ate_from_params_roundtrip_matches_fitted_ate():
    """Fit a small model; feed fitted results.params (reporting space) to
    mdcev_mixed_ate_from_params; its ATE must equal the fitted model's .ate()
    to 1e-6 (no reporting->estimation inversion needed -- ReportingSpace
    consumes the natural params directly)."""
    fx = _make_frame()
    ctrl = MDCEVMixedControl(
        utility="trad", n_rep=5, draw_seed=0, normvar=("b1",),
        maxiter=10, verbose=0, want_covariance=False,
    )
    m = _make_mixed(fx, ctrl)
    res = m.fit()
    assert m._fitted_spec.nrndcoef == 1

    scenarios = {"lo": {"x1_0": -0.5}, "hi": {"x1_0": 0.5}}
    fitted = m.ate(scenarios=scenarios)

    # fresh, UNfitted model primed at the reporting-space params.
    m2 = _make_mixed(fx, ctrl)
    from_params = mdcev_mixed_ate_from_params(m2, res.params, scenarios=scenarios)

    np.testing.assert_allclose(
        from_params.predicted_shares, fitted.predicted_shares,
        rtol=0.0, atol=1e-6,
    )
    for name in scenarios:
        np.testing.assert_allclose(
            from_params.shares_per_scenario[name],
            fitted.shares_per_scenario[name],
            rtol=0.0, atol=1e-6,
        )


def test_ate_from_params_collapse_matches_fixed_coef():
    """nrndcoef=0: the reporting params reduce to [beta | gamma | log_sigma]
    (== beta with the pass-through gamma / kernel scale); feeding them to
    mdcev_mixed_ate_from_params matches the shipped fixed-coefficient
    mdcev_ate_from_params (which likewise accepts reported coefficients), 1e-6."""
    fx = _make_frame()
    beta, gamma_raw, log_sigma = fx["beta"], fx["gamma_raw"], fx["log_sigma"]
    sigma = float(np.exp(log_sigma))

    ctrl = MDCEVMixedControl(utility="trad", n_rep=3, draw_seed=0)
    m = _make_mixed(fx, ctrl)

    spec, layout = m._build_spec_layout()
    assert spec.nrndcoef == 0
    sl = layout.slices()
    params = np.zeros(layout.n_theta, dtype=np.float64)
    params[sl["beta"]] = beta
    params[sl["gamma"]] = gamma_raw
    params[sl["kern"]] = log_sigma

    scenarios = {"lo": {"x1_0": -0.5}, "hi": {"x1_0": 0.5}}
    mixed = mdcev_mixed_ate_from_params(m, params, scenarios=scenarios)

    ship_model = MDCEVModel(
        fx["df"], alternatives=fx["alt_cols"], availability=fx["price_cols"],
        utility_spec=fx["utility_spec"], gamma_spec=fx["gamma_spec"],
        param_names=[f"b{v}" for v in range(NVARM)],
        gamma_names=[f"g{j}" for j in range(NVARGAM)],
    )
    fixed = mdcev_ate_from_params(
        np.concatenate([beta, gamma_raw]), sigma,
        model=ship_model, data=fx["df"], scenarios=scenarios,
        alternative_names=fx["alt_cols"], n_draws=1000, seed=1234,
        param_names=_param_names(),
    )

    np.testing.assert_allclose(
        mixed.predicted_shares, fixed.predicted_shares, rtol=0.0, atol=1e-6
    )
    for name in scenarios:
        np.testing.assert_allclose(
            mixed.shares_per_scenario[name],
            fixed.shares_per_scenario[name],
            rtol=0.0, atol=1e-6,
        )
