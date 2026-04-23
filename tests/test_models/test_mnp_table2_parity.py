"""Regression tests pinning MNP Table 2 log-likelihoods against the BHATLIB paper.

Gates MNP-001a (unified likelihood kernel refactor) and any later code changes
that touch the MNP log-likelihood / probability paths. Values come from
`tests/fixtures/bhatlib_table2_targets.json`, which is derived from the
BHATLIB paper Table 2 (TRAVELMODE, 1125 observations, 3 alternatives).

Model (d) (mixture-of-normals) has a documented 2.0-unit tolerance due to
local-optima sensitivity and the pending mixture-spec propagation work
tracked as P6 in the UTA feedback plan.
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
    assert abs(results.ll_total - target["ll"]) < 0.5, (
        f"Model (a)(i) IID LL = {results.ll_total:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_aii_flexible(table2_targets, travelmode_path):
    target = table2_targets["models"]["a_ii_flexible"]
    results = _fit_model(SPEC_BASE, {"iid": False}, None, travelmode_path)
    assert abs(results.ll_total - target["ll"]) < 0.5, (
        f"Model (a)(ii) flexible LL = {results.ll_total:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_b_age45(table2_targets, travelmode_path):
    target = table2_targets["models"]["b_age45"]
    results = _fit_model(SPEC_WITH_AGE45, {"iid": False}, None, travelmode_path)
    assert abs(results.ll_total - target["ll"]) < 0.5, (
        f"Model (b) +AGE45 LL = {results.ll_total:.3f}, expected {target['ll']}"
    )


@pytest.mark.slow
def test_table2_model_c_random_coef(table2_targets, travelmode_path):
    target = table2_targets["models"]["c_random_coef"]
    results = _fit_model(
        SPEC_WITH_AGE45, {"iid": False, "mix": True}, ["OVTT"], travelmode_path
    )
    assert abs(results.ll_total - target["ll"]) < 1.0, (
        f"Model (c) random coef LL = {results.ll_total:.3f}, expected {target['ll']}"
    )


# Current Python baseline for Model (d) mixture — pins refactor drift.
# Paper target is -634.975 (±2); current Python gives ~-629.684 due to the
# known mixture ranvars-propagation gap tracked as P6 in the UTA feedback
# plan (and documented in docs/plans/MIXTURE_SHARED_COEFFICIENTS_PLAN.md).
_MODEL_D_BASELINE_LL = -629.684


@pytest.mark.slow
@pytest.mark.xfail(
    reason="P6 mixture ranvars propagation pending; current LL ~-629.7 vs paper -634.975",
    strict=False,
)
def test_table2_model_d_mixture_paper_target(table2_targets, travelmode_path):
    """Asserts BHATLIB paper Model (d) LL. Expected to fail until P6 lands."""
    target = table2_targets["models"]["d_mixture"]
    tol = target.get("ll_tolerance", 2.0)
    results = _fit_model(
        SPEC_WITH_AGE45,
        {"iid": False, "mix": True, "nseg": 2},
        ["OVTT"],
        travelmode_path,
    )
    assert abs(results.ll_total - target["ll"]) < tol, (
        f"Model (d) mixture LL = {results.ll_total:.3f}, expected {target['ll']} (tol={tol})"
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


@pytest.mark.slow
def test_table2_model_b_bhhh_se_parity(table2_targets, travelmode_path):
    """Model (b) BHHH standard errors should match paper (within 2 decimals).

    Paper values are derived from t-stats via SE = |coef| / |t|, so they
    carry ~2 decimals of effective precision. Parameter names follow the
    BHATLIB reporting convention (normalized b_report).
    """
    target = table2_targets["models"]["b_age45"]
    ctrl = MNPControl(
        iid=False, maxiter=200, verbose=0, seed=42, se_method="bhhh",
    )
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
    )
    results = model.fit()

    name_to_se = dict(zip(results.param_names, results.se))
    for param, tdata in target["params"].items():
        if tdata["status"] != "estimated":
            continue
        expected = tdata["se"]
        got = name_to_se.get(param)
        assert got is not None, f"param {param} missing from results"
        assert abs(got - expected) < 0.02, (
            f"Model (b) BHHH SE mismatch for {param}: got {got:.4f}, "
            f"paper {expected:.4f}"
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
    assert abs(results.ll_total - _MODEL_D_BASELINE_LL) < 0.5, (
        f"Model (d) LL drifted from baseline: got {results.ll_total:.3f}, "
        f"baseline {_MODEL_D_BASELINE_LL}"
    )
