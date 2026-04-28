"""Tests for MNP-004: ATE scenario matrix.

TDD-first suite. Tests are written against the new ``scenarios`` API
before implementation exists. All slow tests require a TRAVELMODE fixture
and full model fitting.

Spec rows: scenario dict / DataFrame, column-name resolution, legacy compat,
mutual-exclusion guard, comparison helper, mixed MNP path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mnp._mnp_ate import ATEResult, mnp_ate

# ---------------------------------------------------------------------------
# Shared spec constants (same as test_mnp_table2_parity for Model (b))
# ---------------------------------------------------------------------------
ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

SPEC_WITH_AGE45 = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero", "Alt3_ch": "sero"},
    "AGE45_TR": {"Alt1_ch": "sero", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

SPEC_RANDOM = {
    **SPEC_WITH_AGE45,
}


def _fit_model_b(travelmode_path):
    """Fit Model (b) +AGE45 (flexible covariance)."""
    from pybhatlib.models.mnp import MNPControl, MNPModel

    ctrl = MNPControl(maxiter=200, verbose=0, seed=42, iid=False)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
        ranvars=None,
    )
    return model.fit()


def _fit_model_c(travelmode_path):
    """Fit Model (c) random coefficients."""
    from pybhatlib.models.mnp import MNPControl, MNPModel

    ctrl = MNPControl(maxiter=200, verbose=0, seed=42, iid=False, mix=True)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_RANDOM,
        control=ctrl,
        ranvars=["OVTT"],
    )
    return model.fit()


# ---------------------------------------------------------------------------
# Test 1: dict scenarios produce valid, distinct shares
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dict_scenarios_two_scenarios_distinct_shares(travelmode_path):
    """Two AGE45 scenarios (0 vs 1) must each sum to 1 and differ from each other."""
    results = _fit_model_b(travelmode_path)
    data = pd.read_csv(travelmode_path)

    scenarios = {
        "AGE45_lo": {"AGE45": 0},
        "AGE45_hi": {"AGE45": 1},
    }
    ate = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        scenarios=scenarios,
    )

    assert ate.shares_per_scenario is not None
    assert set(ate.shares_per_scenario.keys()) == {"AGE45_lo", "AGE45_hi"}

    for name, shares in ate.shares_per_scenario.items():
        assert shares.shape == (len(ALTERNATIVES),), f"shares shape wrong for {name}"
        assert abs(shares.sum() - 1.0) < 1e-6, f"shares don't sum to 1 for {name}"
        assert (shares >= 0).all(), f"negative share in {name}"

    # AGE45 has nonzero coefficient — shares must differ
    lo = ate.shares_per_scenario["AGE45_lo"]
    hi = ate.shares_per_scenario["AGE45_hi"]
    assert not np.allclose(lo, hi), "AGE45=0 and AGE45=1 shares are identical (unexpected)"


# ---------------------------------------------------------------------------
# Test 2: DataFrame scenarios match dict scenarios exactly
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_dataframe_scenarios_equivalent_to_dict(travelmode_path):
    """DataFrame form of scenarios must produce identical shares as dict form."""
    results = _fit_model_b(travelmode_path)
    data = pd.read_csv(travelmode_path)

    scenarios_dict = {
        "AGE45_lo": {"AGE45": 0},
        "AGE45_hi": {"AGE45": 1},
    }
    ate_dict = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        scenarios=scenarios_dict,
    )

    # Build equivalent DataFrame
    df_scenarios = pd.DataFrame(
        {"AGE45": [0, 1]},
        index=["AGE45_lo", "AGE45_hi"],
    )
    ate_df = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        scenarios=df_scenarios,
    )

    assert ate_df.shares_per_scenario is not None
    for name in scenarios_dict:
        np.testing.assert_allclose(
            ate_dict.shares_per_scenario[name],
            ate_df.shares_per_scenario[name],
            atol=1e-12,
            err_msg=f"Mismatch for scenario {name} between dict and DataFrame form",
        )


# ---------------------------------------------------------------------------
# Test 3: string column reference resolves to the original column values
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_column_name_value_resolves(travelmode_path):
    """Passing 'AGE45' as the value (override AGE45 with itself) should match
    the overall predicted shares computed from unmodified data."""
    results = _fit_model_b(travelmode_path)
    data = pd.read_csv(travelmode_path)

    # Overall predicted shares (no scenarios)
    ate_baseline = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
    )

    # Scenario: override AGE45 with the AGE45 column itself (identity override)
    ate_identity = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        scenarios={"by_age": {"AGE45": "AGE45"}},
    )

    np.testing.assert_allclose(
        ate_identity.shares_per_scenario["by_age"],
        ate_baseline.predicted_shares,
        atol=1e-12,
        err_msg="Identity column override doesn't match baseline shares",
    )


# ---------------------------------------------------------------------------
# Test 4: unknown column string raises ValueError
# ---------------------------------------------------------------------------
def test_unknown_column_string_raises(travelmode_path):
    """Passing a string value that is not a column name must raise ValueError."""
    from pybhatlib.models.mnp import MNPControl, MNPModel

    results_stub = None
    # We need a fitted result — use a minimal fit or mock; use full fit
    data = pd.read_csv(travelmode_path)
    ctrl = MNPControl(maxiter=5, verbose=0, seed=42, iid=True)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
        ranvars=None,
    )
    results_stub = model.fit()

    with pytest.raises(ValueError, match="NOT_A_COLUMN"):
        mnp_ate(
            results_stub,
            data=data,
            spec=SPEC_WITH_AGE45,
            alternatives=ALTERNATIVES,
            scenarios={"x": {"AGE45": "NOT_A_COLUMN"}},
        )


# ---------------------------------------------------------------------------
# Test 5: legacy changevar/changeval path still works
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_changevar_legacy_path_still_works(travelmode_path):
    """changevar/changeval legacy API must still populate treatment_shares and pct_ate."""
    results = _fit_model_b(travelmode_path)
    data = pd.read_csv(travelmode_path)

    ate = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        changevar="AGE45",
        changeval=1,
    )

    assert ate.treatment_shares is not None, "treatment_shares not populated"
    assert ate.base_shares is not None, "base_shares not populated"
    assert ate.pct_ate is not None, "pct_ate not populated"
    assert np.all(np.isfinite(ate.pct_ate)), "pct_ate has non-finite values"
    assert abs(ate.treatment_shares.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Test 6: scenarios and changevar are mutually exclusive
# ---------------------------------------------------------------------------
def test_scenarios_and_changevar_mutually_exclusive(travelmode_path):
    """Passing both scenarios and changevar must raise ValueError."""
    from pybhatlib.models.mnp import MNPControl, MNPModel

    data = pd.read_csv(travelmode_path)
    ctrl = MNPControl(maxiter=5, verbose=0, seed=42, iid=True)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
        ranvars=None,
    )
    results = model.fit()

    with pytest.raises(ValueError, match="mutually exclusive"):
        mnp_ate(
            results,
            data=data,
            spec=SPEC_WITH_AGE45,
            alternatives=ALTERNATIVES,
            scenarios={"s1": {"AGE45": 0}},
            changevar="AGE45",
            changeval=1,
        )


# ---------------------------------------------------------------------------
# Test 7: ATEResult.comparison method
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_comparison_method(travelmode_path):
    """result.comparison('base', 'treatment') == 100*(t-b)/b element-wise."""
    results = _fit_model_b(travelmode_path)
    data = pd.read_csv(travelmode_path)

    scenarios = {
        "base": {"AGE45": 0},
        "treatment": {"AGE45": 1},
    }
    ate = mnp_ate(
        results,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
        scenarios=scenarios,
    )

    b = ate.shares_per_scenario["base"]
    t = ate.shares_per_scenario["treatment"]
    expected = 100.0 * (t - b) / b

    result_cmp = ate.comparison("base", "treatment")
    np.testing.assert_allclose(
        result_cmp, expected, atol=1e-12,
        err_msg="comparison() does not match 100*(t-b)/b",
    )


# ---------------------------------------------------------------------------
# Test 8: mixed MNP (random coefficients) scenario path
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_mixed_mnp_scenarios(travelmode_path):
    """Model (c) with random coeff: scenario shares must be valid."""
    results = _fit_model_c(travelmode_path)
    assert results.ranvar_indices is not None, "ranvar_indices not propagated"

    data = pd.read_csv(travelmode_path)
    scenarios = {
        "AGE45_lo": {"AGE45": 0},
        "AGE45_hi": {"AGE45": 1},
    }
    ate = mnp_ate(
        results,
        data=data,
        spec=SPEC_RANDOM,
        alternatives=ALTERNATIVES,
        scenarios=scenarios,
    )

    assert ate.shares_per_scenario is not None
    for name, shares in ate.shares_per_scenario.items():
        assert abs(shares.sum() - 1.0) < 1e-6, f"shares don't sum to 1 for {name}"
        assert (shares >= 0).all(), f"negative share in {name}"
