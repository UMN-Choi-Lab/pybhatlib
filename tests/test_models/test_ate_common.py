"""Tests for the shared ATE utilities in ``models/_ate_common.py``.

These guard the pieces that MNP, MORP, MDCEV and MNL now share rather than
copy-paste, so a regression here would surface in all four models at once.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models._ate_common import (
    ATEComparisonMixin,
    apply_scenario_overrides,
    scenarios_to_dict,
)


def _frame(n_cols: int) -> pd.DataFrame:
    return pd.DataFrame({f"c{i}": np.arange(3, dtype=float) for i in range(n_cols)})


@pytest.mark.parametrize("n_cols", [3, 25])
def test_override_target_error_names_the_column(n_cols):
    """The message must identify the offending column at any column count.

    Regression: the branches were built with implicit string concatenation and
    an unparenthesised ``if/else``, which binds so that the ``"... is not a
    column in data."`` prefix is part of the >20-column branch only.  With <= 20
    columns the user saw a bare ``Available columns: [...]`` and no clue which
    override was wrong.
    """
    data = _frame(n_cols)
    with pytest.raises(ValueError, match=r"Scenario override target 'NOPE'"):
        apply_scenario_overrides(data, {"NOPE": 0.0})


def test_override_scalar_broadcasts_and_leaves_source_untouched():
    data = _frame(3)
    out = apply_scenario_overrides(data, {"c1": 7.0})
    np.testing.assert_allclose(out["c1"].to_numpy(), 7.0)
    np.testing.assert_allclose(data["c1"].to_numpy(), np.arange(3))  # no mutation


def test_override_string_resolves_a_source_column():
    data = _frame(3)
    out = apply_scenario_overrides(data, {"c0": "c2"})
    np.testing.assert_allclose(out["c0"].to_numpy(), data["c2"].to_numpy())


def test_override_string_rejects_a_non_column():
    data = _frame(3)
    with pytest.raises(ValueError, match=r"references column 'ghost'"):
        apply_scenario_overrides(data, {"c0": "ghost"})


def test_scenarios_to_dict_accepts_dict_and_frame():
    as_dict = {"base": {"c0": 0.0}, "treat": {"c0": 1.0}}
    assert scenarios_to_dict(as_dict) == as_dict

    frame = pd.DataFrame({"c0": [0.0, 1.0]}, index=["base", "treat"])
    assert scenarios_to_dict(frame) == as_dict


def test_comparison_pct_change_and_zero_base():
    class _R(ATEComparisonMixin):
        _ate_func_name = "demo_ate"

        def __init__(self, shares):
            self.shares_per_scenario = shares

    r = _R({"base": np.array([2.0, 0.0]), "treat": np.array([3.0, 5.0])})
    got = r.comparison("base", "treat")
    np.testing.assert_allclose(got[0], 50.0)
    assert np.isnan(got[1])  # zero base -> NaN, not a divide-by-zero blow-up


def test_comparison_requires_scenarios_and_names_the_function():
    class _R(ATEComparisonMixin):
        _ate_func_name = "demo_ate"
        shares_per_scenario = None

    with pytest.raises(ValueError, match=r"demo_ate"):
        _R().comparison("base", "treat")


def test_mdcev_result_reuses_the_shared_comparison():
    """MDCEV must not re-grow its own ``comparison()``.

    PR #50 removed the copy; #46 (merged after that branch was cut) had added
    an identical one.  Git auto-merges the two cleanly and silently restores
    the duplicate, and every behavioural test still passes.  Only an identity
    check catches it.
    """
    from pybhatlib.models.mdcev._mdcev_ate import MDCEVATEResult

    assert MDCEVATEResult.comparison is ATEComparisonMixin.comparison
    assert MDCEVATEResult._ate_func_name == "mdcev_ate"
