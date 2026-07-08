"""Tests for the MORP scenarios= / .comparison() API (mirrors MNP-004).

MORP predicts one ordinal-category distribution per outcome dimension, so
``shares_per_scenario`` maps scenario name -> list[NDArray] (one array per
dimension) and ``comparison()`` returns one percentage-change array per
dimension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models.morp import (
    MORPControl,
    MORPModel,
    MORPResults,
    morp_ate,
    morp_ate_from_params,
)

SPEC = {"x1": {"y1": "x1", "y2": "x1"}, "x2": {"y1": "x2", "y2": "x2"}}
DEP = ["y1", "y2"]
NCAT = [3, 3]
SCEN = {"base": {"x1": 0.0}, "treatment": {"x1": 1.0}}


@pytest.fixture(scope="module")
def morp_fit():
    rng = np.random.default_rng(42)
    n = 120
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    beta = np.array([0.5, -0.3])
    lin = np.column_stack([x1, x2]) @ beta
    y1 = np.digitize(lin + rng.standard_normal(n), [-0.5, 0.5])
    y2 = np.digitize(lin + rng.standard_normal(n), [-0.3, 0.7])
    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    model = MORPModel(
        data=df, dep_vars=DEP, spec=SPEC, n_categories=NCAT,
        control=MORPControl(iid=True, method="scipy", verbose=0, seed=42, maxiter=50),
    )
    return model.fit(), df, model.n_beta


def test_scenarios_match_single_design(morp_fit):
    """Each scenario's shares equal the single-design result on overridden data."""
    res, df, n_beta = morp_fit
    a = morp_ate(res, data=df, spec=SPEC, dep_vars=DEP, n_dims=2,
                 n_categories=NCAT, n_beta=n_beta, scenarios=SCEN)
    assert set(a.shares_per_scenario) == set(SCEN)
    for name, ov in SCEN.items():
        d_mod = df.copy()
        d_mod["x1"] = ov["x1"]
        X_mod, _ = parse_spec(SPEC, d_mod, DEP, nseg=1)
        ref = morp_ate(res, np.asarray(X_mod, float), 2, NCAT, n_beta)
        for d in range(2):
            np.testing.assert_allclose(
                a.shares_per_scenario[name][d], ref.predicted_probs[d], atol=1e-12
            )
        # each dimension's probabilities are a valid distribution
        for d in range(2):
            assert a.shares_per_scenario[name][d].shape == (NCAT[d],)
            np.testing.assert_allclose(a.shares_per_scenario[name][d].sum(), 1.0, atol=1e-9)


def test_comparison_per_dimension(morp_fit):
    res, df, n_beta = morp_fit
    a = morp_ate(res, data=df, spec=SPEC, dep_vars=DEP, n_dims=2,
                 n_categories=NCAT, n_beta=n_beta, scenarios=SCEN)
    cmp = a.comparison("base", "treatment")
    assert isinstance(cmp, list) and len(cmp) == 2
    for d in range(2):
        b = a.shares_per_scenario["base"][d]
        t = a.shares_per_scenario["treatment"][d]
        expect = np.where(b > 0, 100.0 * (t - b) / b, np.nan)
        np.testing.assert_allclose(cmp[d], expect, equal_nan=True)


def test_comparison_requires_scenarios(morp_fit):
    res, df, n_beta = morp_fit
    X, _ = parse_spec(SPEC, df, DEP, nseg=1)
    a = morp_ate(res, np.asarray(X, float), 2, NCAT, n_beta)
    assert a.shares_per_scenario is None
    with pytest.raises(ValueError, match="shares_per_scenario"):
        a.comparison("base", "treatment")


def test_positional_backcompat(morp_fit):
    """Existing positional call morp_ate(res, X, n_dims, n_categories, n_beta)."""
    res, df, n_beta = morp_fit
    X, _ = parse_spec(SPEC, df, DEP, nseg=1)
    a = morp_ate(res, np.asarray(X, float), 2, NCAT, n_beta)
    assert len(a.predicted_probs) == 2


def test_ate_from_params_scenarios(morp_fit):
    """morp_ate_from_params forwards scenarios and matches from_estimates+morp_ate."""
    res, df, n_beta = morp_fit
    beta = np.array([0.5, -0.3])
    thresholds = [np.array([-0.5, 0.5]), np.array([-0.3, 0.7])]
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])

    a1 = morp_ate_from_params(
        beta, thresholds, corr, n_dims=2, n_categories=NCAT, n_beta=n_beta,
        dep_vars=DEP, data=df, spec=SPEC, scenarios=SCEN,
    )
    r = MORPResults.from_estimates(beta, thresholds, corr, dep_vars=DEP,
                                   n_categories=NCAT)
    a2 = morp_ate(r, data=df, spec=SPEC, dep_vars=DEP, n_dims=2,
                  n_categories=NCAT, n_beta=n_beta, scenarios=SCEN)
    for name in SCEN:
        for d in range(2):
            np.testing.assert_allclose(
                a1.shares_per_scenario[name][d],
                a2.shares_per_scenario[name][d], atol=1e-12,
            )
    assert len(a1.comparison("base", "treatment")) == 2


def test_ate_from_params_joint_scenarios_rejected(morp_fit):
    _, df, n_beta = morp_fit
    beta = np.array([0.5, -0.3])
    thresholds = [np.array([-0.5, 0.5]), np.array([-0.3, 0.7])]
    with pytest.raises(ValueError, match="joint=True is not supported"):
        morp_ate_from_params(
            beta, thresholds, None, n_dims=2, n_categories=NCAT, n_beta=n_beta,
            dep_vars=DEP, data=df, spec=SPEC, scenarios=SCEN, joint=True,
        )
