"""Regression tests pinning MNP Table 2 log-likelihoods against the BHATLIB paper.

Gates MNP-001a (unified likelihood kernel refactor) and any later code changes
that touch the MNP log-likelihood / probability paths. Values come from
`tests/fixtures/bhatlib_table2_targets.json`, which is derived from the
BHATLIB paper Table 2 (TRAVELMODE, 1125 observations, 3 alternatives).

Model (d) (mixture-of-normals) has a documented 2.0-unit tolerance due to
local-optima sensitivity. MNP-006 (M1) landed the ergonomic ranvars
auto-expansion across segments; the remaining gap to the paper LL is
structural (shared/varying coefficient handling, deferred to a separate
phase — see ``docs/plans/MIXTURE_SHARED_COEFFICIENTS_PLAN.md``).
"""

from __future__ import annotations

import json
import os

import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fixtures",
    "bhatlib_table2_targets.json",
)

ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

SPEC_BASE = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

AGE45_ROWS = {
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero", "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
}

SPEC_WITH_AGE45 = {
    "CON_SR": SPEC_BASE["CON_SR"],
    "CON_TR": SPEC_BASE["CON_TR"],
    **AGE45_ROWS,
    "IVTT": SPEC_BASE["IVTT"],
    "OVTT": SPEC_BASE["OVTT"],
    "COST": SPEC_BASE["COST"],
}


@pytest.fixture(scope="module")
def table2_targets():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


def _fit_model(spec, control_kwargs, ranvars, travelmode_path):
    ctrl = MNPControl(maxiter=200, verbose=0, seed=42, **control_kwargs)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=spec,
        control=ctrl,
        ranvars=ranvars,
    )
    return model.fit()


@pytest.mark.slow
def test_table2_model_ai_iid(table2_targets, travelmode_path):
    target = table2_targets["models"]["a_i_iid"]
    results = _fit_model(SPEC_BASE, {"iid": True}, None, travelmode_path)
    assert abs(results.loglik * results.n_obs - target["ll"]) < 0.5, (
        f"Model (a)(i) IID LL = {results.loglik * results.n_obs:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_aii_flexible(table2_targets, travelmode_path):
    target = table2_targets["models"]["a_ii_flexible"]
    results = _fit_model(SPEC_BASE, {"iid": False}, None, travelmode_path)
    assert abs(results.loglik * results.n_obs - target["ll"]) < 0.5, (
        f"Model (a)(ii) flexible LL = {results.loglik * results.n_obs:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_b_age45(table2_targets, travelmode_path):
    target = table2_targets["models"]["b_age45"]
    results = _fit_model(SPEC_WITH_AGE45, {"iid": False}, None, travelmode_path)
    assert abs(results.loglik * results.n_obs - target["ll"]) < 0.5, (
        f"Model (b) +AGE45 LL = {results.loglik * results.n_obs:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_c_random_coef(table2_targets, travelmode_path):
    target = table2_targets["models"]["c_random_coef"]
    results = _fit_model(
        SPEC_WITH_AGE45, {"iid": False, "mix": True}, ["OVTT"], travelmode_path
    )
    assert abs(results.loglik * results.n_obs - target["ll"]) < 1.0, (
        f"Model (c) random coef LL = {results.loglik * results.n_obs:.3f}, expected {target['ll']}"
    )


# Current Python baseline for Model (d) mixture — pins refactor drift.
# Paper target is -634.975 (±2); current Python gives ~-627.885 with the
# MNP-006 (M1) ranvars auto-expansion (replicates each base ranvar across
# segments). The residual gap to the paper target (~7 LL units) is
# structural — the GAUSS run has a 1-D random coefficient per segment with
# segment-specific OVTT columns (one zeroed per segment in ivunord), while
# pybhatlib's shared X forces a 2-D random coefficient over duplicate
# columns. Closing this requires the shared/varying refactor in
# ``docs/plans/MIXTURE_SHARED_COEFFICIENTS_PLAN.md`` (a separate phase).
_MODEL_D_BASELINE_LL = -627.885


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "Mixture shared/varying coefficient refactor pending; current LL "
        "~-627.9 vs paper -634.975. M1 ergonomic auto-expansion landed; "
        "the residual gap is structural."
    ),
    strict=False,
)
def test_table2_model_d_mixture_paper_target(table2_targets, travelmode_path):
    """Asserts BHATLIB paper Model (d) LL. Expected to fail until the
    shared/varying coefficient refactor lands.
    """
    target = table2_targets["models"]["d_mixture"]
    tol = target.get("ll_tolerance", 2.0)
    results = _fit_model(
        SPEC_WITH_AGE45,
        {"iid": False, "mix": True, "nseg": 2},
        ["OVTT"],
        travelmode_path,
    )
    assert abs(results.loglik * results.n_obs - target["ll"]) < tol, (
        f"Model (d) mixture LL = {results.loglik * results.n_obs:.3f}, expected {target['ll']} (tol={tol})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("se_method", ["bhhh", "hessian", "sandwich"])
def test_se_methods_produce_valid_errors(se_method, travelmode_path):
    """Each SE method must produce positive finite standard errors on Model (b)."""
    ctrl = MNPControl(
        iid=False, maxiter=200, verbose=0, seed=42, se_method=se_method,
    )
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
    )
    results = model.fit()
    import numpy as np
    assert (results.se > 0).all(), f"{se_method}: non-positive SE entries"
    assert np.isfinite(results.se).all(), f"{se_method}: non-finite SE entries"


def _assert_bhhh_se_parity(
    target, model_label, spec, control_kwargs, ranvars, travelmode_path,
    tol=0.02,
):
    """Shared assertion: fit `model_label` with BHHH SE, compare to paper SE."""
    ctrl = MNPControl(
        maxiter=200, verbose=0, seed=42, se_method="bhhh", **control_kwargs,
    )
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=spec,
        control=ctrl,
        ranvars=ranvars,
    )
    results = model.fit()

    name_to_se = dict(zip(results.param_names, results.se))
    for param, tdata in target["params"].items():
        if tdata["status"] != "estimated":
            continue
        expected = tdata["se"]
        got = name_to_se.get(param)
        assert got is not None, f"{model_label}: param {param} missing from results"
        assert abs(got - expected) < tol, (
            f"{model_label} BHHH SE mismatch for {param}: got {got:.4f}, "
            f"paper {expected:.4f}"
        )


@pytest.mark.slow
def test_table2_model_b_bhhh_se_parity(table2_targets, travelmode_path):
    """Model (b) BHHH standard errors should match paper (within 2 decimals).

    Paper values are derived from t-stats via SE = |coef| / |t|, so they
    carry ~2 decimals of effective precision. Parameter names follow the
    BHATLIB reporting convention (normalized b_report).
    """
    _assert_bhhh_se_parity(
        table2_targets["models"]["b_age45"],
        "Model (b)",
        SPEC_WITH_AGE45,
        {"iid": False},
        None,
        travelmode_path,
    )


@pytest.mark.slow
def test_table2_model_aii_bhhh_se_parity(table2_targets, travelmode_path):
    """Model (a)(ii) flexible — broadens BHHH SE coverage beyond Model (b).

    Same 0.02 tolerance against paper SE. Catches scaling/conditioning bugs
    that a single-model parity test could mask.
    """
    _assert_bhhh_se_parity(
        table2_targets["models"]["a_ii_flexible"],
        "Model (a)(ii)",
        SPEC_BASE,
        {"iid": False},
        None,
        travelmode_path,
    )


@pytest.mark.slow
@pytest.mark.xfail(
    reason=(
        "MNP-002b — BHHH SE underestimates the random-coefficient variance "
        "(CovCOv01) on Model (c) by ~50% (got 0.218 vs paper 0.453). The "
        "score outer-product is near-singular (cond ~1e18), forcing pinv. "
        "Likely fixed by switching to unparameterized scoring (lpr1/lgd1) "
        "and a noise-aware FD step. Tracked in MNP-002b PR scope."
    ),
    strict=False,
)
def test_table2_model_c_bhhh_se_parity(table2_targets, travelmode_path):
    """Model (c) random coefficient — exercises the mixture/Omega_L SE path.

    Looser 0.10 tolerance: Model (c)'s OVTT t-stat is ~5, but CovCOv01's
    is ~2.4 with SE ~0.45, so a 0.02 absolute tolerance is too tight given
    the paper-rounding precision. Currently xfails — see decorator.
    """
    _assert_bhhh_se_parity(
        table2_targets["models"]["c_random_coef"],
        "Model (c)",
        SPEC_WITH_AGE45,
        {"iid": False, "mix": True},
        ["OVTT"],
        travelmode_path,
        tol=0.10,
    )


@pytest.mark.slow
def test_ate_mixed_mnp_uses_ranvar_indices(travelmode_path):
    """Guards that mnp_ate actually propagates random-coefficient information
    for mixed MNP (Model (c)). Before MNPResults.ranvar_indices was added,
    mnp_ate silently dropped the mixture component because the attribute was
    absent, so Omega_L was never constructed. This test fits Model (c) and
    runs ATE, asserting shares are valid and randvar_indices flow through.
    """
    import pandas as pd

    from pybhatlib.models.mnp._mnp_ate import mnp_ate

    results = _fit_model(
        SPEC_WITH_AGE45, {"iid": False, "mix": True}, ["OVTT"], travelmode_path
    )
    assert results.ranvar_indices is not None, "ranvar_indices not propagated to results"
    assert len(results.ranvar_indices) == 1, "expected 1 random coefficient (OVTT)"

    data = pd.read_csv(travelmode_path)
    ate = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
    )
    assert abs(ate.predicted_shares.sum() - 1.0) < 1e-6
    assert (ate.predicted_shares >= 0).all()
    assert (ate.predicted_shares <= 1).all()


@pytest.mark.slow
def test_table2_model_d_mixture_baseline_no_regression(travelmode_path):
    """Pins current Python LL for Model (d) so the MNP-001a refactor cannot
    silently drift the mixture path. Separate from the paper-target test above.
    """
    results = _fit_model(
        SPEC_WITH_AGE45,
        {"iid": False, "mix": True, "nseg": 2},
        ["OVTT"],
        travelmode_path,
    )
    assert abs(results.loglik * results.n_obs - _MODEL_D_BASELINE_LL) < 0.5, (
        f"Model (d) LL drifted from baseline: got {results.loglik * results.n_obs:.3f}, "
        f"baseline {_MODEL_D_BASELINE_LL}"
    )
