"""Gate for MORPFlex (mixed-panel MORP) .predict() / .ate() wiring (Phase 4).

Validated by COLLAPSE + interface conformance, NOT GAUSS parity (no GAUSS
forecast oracle exists for the mixed drivers -- Dale's guidance):

1. COLLAPSE -- with an empty random-coefficient spec (``nrndcoef = 0``) the
   mixed family's draw-integrated ``predict`` / ``ate`` reduce *exactly* (to
   1e-6) to the shipped fixed-coefficient MORP marginal
   (:func:`pybhatlib.models.morp.morp_predict` /
   :func:`pybhatlib.models.morp.morp_ate`) evaluated at the same coefficients on
   the same data.
2. MIXING ACTIVE -- with a random coefficient, ``predict`` returns valid
   per-dimension category distributions and ``ate(scenarios=...)`` returns a
   well-formed per-outcome result with a working ``.comparison``.
3. INTERFACE CONFORMANCE -- the result exposes the harmonized *per-outcome*
   fields (a list of per-dimension arrays) and is **not** an
   :class:`~pybhatlib.models._ate_common.ATEResultMixin`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._ate_common import ATEResultMixin
from pybhatlib.models.morp import MORPResults, morp_ate, morp_predict
from pybhatlib.models.morp_flex import (
    MORPFlexATEResult,
    MORPFlexControl,
    MORPFlexModel,
    morp_flex_ate,
    morp_flex_ate_from_params,
    morp_flex_predict,
)
from pybhatlib.vecup._panel import PanelIndex

SPEC = {"x1": {"y1": "x1", "y2": "x1"}, "x2": {"y1": "x2", "y2": "x2"}}
DEP = ["y1", "y2"]
NCAT = [3, 3]
SCEN = {"base": {"x1": 0.0}, "treatment": {"x1": 1.0}}


def _make_data(seed: int = 20260717, n: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    beta = np.array([0.7, -0.4])
    lin = np.column_stack([x1, x2]) @ beta
    y1 = np.digitize(lin + rng.standard_normal(n), [-0.5, 0.5])
    y2 = np.digitize(lin + rng.standard_normal(n), [-0.3, 0.7])
    return pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})


def _fixed_results_from_mixed(model: MORPFlexModel) -> MORPResults:
    """Build a fixed MORPResults from the mixed model's fitted coefficients.

    Reads the reporting-space ``params`` block layout
    ``[thresh | beta | rcor]`` (empty rc + normal kernel), rebuilding the
    per-dimension ordered thresholds, the fixed betas and the 2x2 error
    correlation.
    """
    p = np.asarray(model.results_.params, dtype=np.float64)
    n_cut = [c - 1 for c in NCAT]                      # [2, 2]
    n_thresh = sum(n_cut)                              # 4
    n_beta = model.n_beta
    thresh = p[:n_thresh]
    thresholds = [thresh[0:n_cut[0]], thresh[n_cut[0]:n_thresh]]
    beta = p[n_thresh:n_thresh + n_beta]
    r = float(p[n_thresh + n_beta])                    # single off-diagonal
    corr = np.array([[1.0, r], [r, 1.0]], dtype=np.float64)
    return MORPResults.from_estimates(
        beta, thresholds, corr, dep_vars=DEP, n_categories=NCAT,
    )


@pytest.fixture(scope="module")
def collapse_fit():
    df = _make_data()
    ctrl = MORPFlexControl(
        copula=False, yj_kernel=False, n_rep=1, verbose=0,
        want_covariance=False, maxiter=200, seed=7,
    )
    model = MORPFlexModel(
        data=df, dep_vars=DEP, spec=SPEC, n_categories=NCAT, control=ctrl,
    )
    model.fit()
    return model, df


# ---------------------------------------------------------------------------
# (1) COLLAPSE
# ---------------------------------------------------------------------------

def test_collapse_predict_equals_fixed(collapse_fit):
    model, _ = collapse_fit
    fixed = _fixed_results_from_mixed(model)

    mixed_probs = model.predict()
    fixed_probs = morp_predict(fixed, model.X)

    assert len(mixed_probs) == len(fixed_probs) == 2
    for d in range(2):
        assert mixed_probs[d].shape == (model.N, NCAT[d])
        np.testing.assert_allclose(mixed_probs[d], fixed_probs[d], atol=1e-6)


def test_randdiag_removes_joint_correlation_block():
    model = MORPFlexModel(
        data=_make_data(n=20), dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPFlexControl(normvar=("x1",), randdiag=True, verbose=0),
    )
    spec, layout = model._build_spec_layout()
    state = model._build_estimator(
        spec, layout, PanelIndex.from_ids(model.person_ids)
    ).kernel.prepare(np.zeros(layout.n_theta), layout)

    assert layout.n_rcor == 0
    np.testing.assert_array_equal(state.omegastar, np.eye(spec.nrndtot))


def test_observation_weights_are_averaged_by_person():
    data = _make_data(n=5)
    data["pid"] = [0, 0, 1, 1, 1]
    data["weight"] = [1.0, 3.0, 2.0, 4.0, 6.0]
    model = MORPFlexModel(
        data=data, dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPFlexControl(
            person_id="pid", weight_var="weight", n_rep=1, verbose=0
        ),
    )
    spec, layout = model._build_spec_layout()
    estimator = model._build_estimator(
        spec, layout, PanelIndex.from_ids(model.person_ids)
    )

    np.testing.assert_allclose(estimator.weightind, [2.0, 4.0])


def test_iid_forces_ordinal_independence():
    model = MORPFlexModel(
        data=_make_data(n=20), dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPFlexControl(iid=True),
    )
    spec, layout = model._build_spec_layout()
    theta = np.zeros(layout.n_theta)
    theta[layout.slices()["rcor"]] = 0.8
    state = model._build_estimator(
        spec, layout, PanelIndex.from_ids(model.person_ids)
    ).kernel.prepare(theta, layout)

    np.testing.assert_array_equal(state.xi2subq, np.eye(2))


def test_correst_masks_selected_ordinal_correlations():
    model = MORPFlexModel(
        data=_make_data(n=20), dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPFlexControl(correst=np.eye(2)),
    )
    spec, layout = model._build_spec_layout()
    theta = np.zeros(layout.n_theta)
    theta[layout.slices()["rcor"]] = 0.8
    state = model._build_estimator(
        spec, layout, PanelIndex.from_ids(model.person_ids)
    ).kernel.prepare(theta, layout)

    np.testing.assert_array_equal(state.xi2subq, np.eye(2))


def test_fix_location_zero_is_wired_to_shared_spec():
    model = MORPFlexModel(
        data=_make_data(n=20), dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPFlexControl(
            normvar=("x1",), fix_location_zero=("x1",), verbose=0
        ),
    )
    spec, _ = model._build_spec_layout()

    np.testing.assert_array_equal(spec.fix_location_zero_mask, [1.0, 0.0])


def test_collapse_ate_equals_fixed(collapse_fit):
    model, _ = collapse_fit
    fixed = _fixed_results_from_mixed(model)

    mixed_ate = model.ate()
    fixed_ate = morp_ate(fixed, model.X, 2, NCAT, model.n_beta)

    assert len(mixed_ate.predicted_probs) == 2
    for d in range(2):
        np.testing.assert_allclose(
            mixed_ate.predicted_probs[d], fixed_ate.predicted_probs[d], atol=1e-6
        )


def test_collapse_scenarios_equal_fixed(collapse_fit):
    model, df = collapse_fit
    fixed = _fixed_results_from_mixed(model)

    mixed = model.ate(scenarios=SCEN)
    ref = morp_ate(
        fixed, data=df, spec=SPEC, dep_vars=DEP, n_dims=2,
        n_categories=NCAT, n_beta=model.n_beta, scenarios=SCEN,
    )
    assert set(mixed.shares_per_scenario) == set(SCEN)
    for name in SCEN:
        for d in range(2):
            np.testing.assert_allclose(
                mixed.shares_per_scenario[name][d],
                ref.shares_per_scenario[name][d],
                atol=1e-6,
            )


# ---------------------------------------------------------------------------
# (2) MIXING ACTIVE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mixed_fit():
    df = _make_data()
    ctrl = MORPFlexControl(
        normvar=("x1",), copula=False, yj_kernel=False, n_rep=8, verbose=0,
        want_covariance=False, maxiter=15, draw_seed=11, seed=11,
    )
    model = MORPFlexModel(
        data=df, dep_vars=DEP, spec=SPEC, n_categories=NCAT, control=ctrl,
    )
    model.fit()
    return model, df


def test_mixing_predict_valid(mixed_fit):
    model, _ = mixed_fit
    probs = model.predict()
    assert len(probs) == 2
    for d in range(2):
        assert probs[d].shape == (model.N, NCAT[d])
        assert np.all(probs[d] >= -1e-12)
        assert np.all(probs[d] <= 1.0 + 1e-9)
        np.testing.assert_allclose(probs[d].sum(axis=1), 1.0, atol=1e-9)


def test_mixing_ate_scenarios_wellformed(mixed_fit):
    model, _ = mixed_fit
    res = model.ate(scenarios=SCEN)
    assert set(res.shares_per_scenario) == set(SCEN)
    for name in SCEN:
        for d in range(2):
            s = res.shares_per_scenario[name][d]
            assert s.shape == (NCAT[d],)
            np.testing.assert_allclose(s.sum(), 1.0, atol=1e-9)
    cmp = res.comparison("base", "treatment")
    assert isinstance(cmp, list) and len(cmp) == 2
    for d in range(2):
        b = res.shares_per_scenario["base"][d]
        t = res.shares_per_scenario["treatment"][d]
        expect = np.where(b > 0, 100.0 * (t - b) / b, np.nan)
        np.testing.assert_allclose(cmp[d], expect, equal_nan=True)


# ---------------------------------------------------------------------------
# (3) INTERFACE CONFORMANCE
# ---------------------------------------------------------------------------

def test_result_is_per_outcome_not_ate_mixin(mixed_fit):
    model, _ = mixed_fit
    res = model.ate(scenarios=SCEN)
    # per-outcome: a list with one array per ordinal dimension.
    assert isinstance(res, MORPFlexATEResult)
    assert isinstance(res.predicted_probs, list)
    assert len(res.predicted_probs) == 2
    for d in range(2):
        assert res.predicted_probs[d].shape == (NCAT[d],)
    # opts OUT of the single-vector ATE mixin.
    assert not isinstance(res, ATEResultMixin)


def test_comparison_requires_scenarios(mixed_fit):
    model, _ = mixed_fit
    res = model.ate()          # single-design: no scenarios
    assert res.shares_per_scenario is None
    with pytest.raises(ValueError):
        res.comparison("base", "treatment")


# ---------------------------------------------------------------------------
# (4) ate_from_params — externally supplied REPORTING-space (natural) params
# ---------------------------------------------------------------------------

def test_ate_from_params_matches_fitted_ate(mixed_fit):
    """Round-trip: reporting-space results.params reproduce the fitted .ate().

    Take the fitted ``results.params`` (reporting/natural space) and feed them to
    :func:`morp_flex_ate_from_params`; the reporting-space engine
    (ReportingSpace + reporting-mode kernel, no estimation<->reporting inversion)
    must reproduce the fitted model's own ``.ate()`` to 1e-6.
    """
    model, df = mixed_fit
    params = np.asarray(model.results_.params, dtype=np.float64)

    ctrl = MORPFlexControl(
        normvar=("x1",), copula=False, yj_kernel=False, n_rep=8, verbose=0,
        want_covariance=False, maxiter=15, draw_seed=11, seed=11,
    )
    res = morp_flex_ate_from_params(
        params, data=df, dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=ctrl, scenarios=SCEN,
    )
    ref = model.ate(scenarios=SCEN)

    assert isinstance(res, MORPFlexATEResult)
    assert set(res.shares_per_scenario) == set(SCEN)
    for name in SCEN:
        for d in range(2):
            np.testing.assert_allclose(
                res.shares_per_scenario[name][d],
                ref.shares_per_scenario[name][d],
                atol=1e-6,
            )


def test_ate_from_params_collapse_matches_fixed(collapse_fit):
    """Collapse round-trip: nrndcoef=0 reporting params == the fixed-coef case.

    With no random coefficients the reporting-space ``params`` are just
    ``[thresh | beta | rcor]``; feeding them to
    :func:`morp_flex_ate_from_params` must match the shipped fixed-coefficient
    :func:`pybhatlib.models.morp.morp_ate_from_params` at the same natural
    coefficients on the same data.
    """
    from pybhatlib.models.morp import morp_ate_from_params

    model, df = collapse_fit
    params = np.asarray(model.results_.params, dtype=np.float64)

    ctrl = MORPFlexControl(
        copula=False, yj_kernel=False, n_rep=1, verbose=0,
        want_covariance=False, maxiter=200, seed=7,
    )
    mixed = morp_flex_ate_from_params(
        params, data=df, dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=ctrl, scenarios=SCEN,
    )

    # Decompose the reporting params into the fixed-coef natural coefficients.
    n_cut = [c - 1 for c in NCAT]                      # [2, 2]
    n_thresh = sum(n_cut)                              # 4
    n_beta = model.n_beta
    thresholds = [params[0:n_cut[0]], params[n_cut[0]:n_thresh]]
    beta = params[n_thresh:n_thresh + n_beta]
    r = float(params[n_thresh + n_beta])
    corr = np.array([[1.0, r], [r, 1.0]], dtype=np.float64)

    ref = morp_ate_from_params(
        beta, thresholds, corr, data=df, spec=SPEC, dep_vars=DEP,
        n_dims=2, n_categories=NCAT, n_beta=n_beta, scenarios=SCEN,
    )

    assert set(mixed.shares_per_scenario) == set(SCEN)
    for name in SCEN:
        for d in range(2):
            np.testing.assert_allclose(
                mixed.shares_per_scenario[name][d],
                ref.shares_per_scenario[name][d],
                atol=1e-6,
            )


def test_ate_from_params_wrong_length_raises():
    df = _make_data()
    ctrl = MORPFlexControl(
        normvar=("x1",), copula=False, yj_kernel=False, n_rep=4, verbose=0,
        want_covariance=False, draw_seed=3, seed=3,
    )
    with pytest.raises(ValueError):
        morp_flex_ate_from_params(
            np.zeros(3), data=df, dep_vars=DEP, spec=SPEC,
            n_categories=NCAT, control=ctrl,
        )


def test_package_exports_harmonized_surface():
    """The family package exposes the harmonized module functions."""
    assert callable(morp_flex_ate)
    assert callable(morp_flex_ate_from_params)
    assert callable(morp_flex_predict)
