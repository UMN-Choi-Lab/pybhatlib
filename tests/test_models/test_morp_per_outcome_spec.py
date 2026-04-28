"""Tests for MORP per-outcome spec (MORP-001).

Covers:
  - New spec dict API for MORPModel (breaking change)
  - MORPControl.indep → iid rename with deprecation
  - Path input for MORPModel.data
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.morp import (
    MORPControl,
    MORPModel,
    morp_control_asdict,
    morp_control_replace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    """Synthetic dataset with 2 dep_vars and columns a, b, c."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.standard_normal(n),
            "b": rng.standard_normal(n),
            "c": rng.standard_normal(n),
            "y1": rng.integers(0, 3, size=n),
            "y2": rng.integers(0, 4, size=n),
        }
    )
    return df


def _make_spec(dep_vars: list[str]) -> dict:
    """Build a minimal 3-coef spec for 2 dep_vars."""
    # coef_a: uses col 'a' for y1, 'b' for y2
    # coef_b: uses col 'b' for y1, 'sero' for y2
    # coef_c: 'sero' for y1, col 'c' for y2
    return {
        "coef_a": {dep_vars[0]: "a", dep_vars[1]: "b"},
        "coef_b": {dep_vars[0]: "b", dep_vars[1]: "sero"},
        "coef_c": {dep_vars[0]: "sero", dep_vars[1]: "c"},
    }


def _make_model(df: pd.DataFrame | None = None, **kwargs) -> MORPModel:
    if df is None:
        df = _make_small_df()
    spec = _make_spec(["y1", "y2"])
    defaults = dict(
        data=df,
        dep_vars=["y1", "y2"],
        spec=spec,
        n_categories=[3, 4],
        control=MORPControl(iid=True, verbose=0),
    )
    defaults.update(kwargs)
    return MORPModel(**defaults)


# ---------------------------------------------------------------------------
# (1) Spec parsing
# ---------------------------------------------------------------------------


class TestSpecBasicParse:
    def test_shape(self):
        """X must be (N, n_outcomes, n_coefs)."""
        df = _make_small_df(n=50)
        model = _make_model(df)
        assert model.X.shape == (50, 2, 3)

    def test_col_values(self):
        """X[:, d, v] must match the specified data column."""
        df = _make_small_df(n=50, seed=7)
        model = _make_model(df)
        # coef_a (v=0): y1 (d=0) → col 'a', y2 (d=1) → col 'b'
        np.testing.assert_allclose(model.X[:, 0, 0], df["a"].values)
        np.testing.assert_allclose(model.X[:, 1, 0], df["b"].values)

    def test_sero_is_zeros(self):
        """'sero' cells must produce a zero column slice."""
        df = _make_small_df(n=50, seed=3)
        model = _make_model(df)
        # coef_b (v=1): y2 (d=1) → sero
        np.testing.assert_allclose(model.X[:, 1, 1], 0.0)
        # coef_c (v=2): y1 (d=0) → sero
        np.testing.assert_allclose(model.X[:, 0, 2], 0.0)

    def test_param_names_from_spec(self):
        """Beta parameter names should come from spec outer keys."""
        df = _make_small_df()
        model = _make_model(df)
        # n_beta = 3 → first 3 param_names are spec keys
        param_names = model._build_param_names()
        assert param_names[:3] == ["coef_a", "coef_b", "coef_c"]

    def test_n_beta_equals_n_spec_keys(self):
        df = _make_small_df()
        model = _make_model(df)
        assert model.n_beta == 3


class TestSpecSeroStringZerosColumn:
    def test_explicit(self):
        """Explicit 'sero' entries produce zero slices."""
        df = _make_small_df(n=20, seed=11)
        spec = {
            "v1": {"y1": "a", "y2": "sero"},
            "v2": {"y1": "sero", "y2": "b"},
        }
        model = MORPModel(
            data=df,
            dep_vars=["y1", "y2"],
            spec=spec,
            n_categories=[3, 4],
            control=MORPControl(iid=True, verbose=0),
        )
        np.testing.assert_allclose(model.X[:, 1, 0], 0.0)  # v1, y2 → sero
        np.testing.assert_allclose(model.X[:, 0, 1], 0.0)  # v2, y1 → sero


class TestSpecUnknownColumnRaises:
    def test_raises(self):
        """Spec value referring to a missing column should raise ValueError."""
        df = _make_small_df()
        spec = {
            "v1": {"y1": "does_not_exist", "y2": "a"},
        }
        with pytest.raises(ValueError, match="does_not_exist"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec=spec,
                n_categories=[3, 4],
                control=MORPControl(iid=True, verbose=0),
            )


class TestSpecUnknownOutcomeRaises:
    def test_raises(self):
        """Spec inner key not in dep_vars should raise ValueError."""
        df = _make_small_df()
        spec = {
            "v1": {"y1": "a", "not_a_dep_var": "b"},
        }
        with pytest.raises(ValueError, match="not_a_dep_var"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec=spec,
                n_categories=[3, 4],
                control=MORPControl(iid=True, verbose=0),
            )


class TestIndepVarsKwargRemoved:
    def test_raises_type_error(self):
        """Passing indep_vars= must raise TypeError (breaking change)."""
        df = _make_small_df()
        spec = _make_spec(["y1", "y2"])
        with pytest.raises(TypeError):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                indep_vars=["a", "b"],  # old API — must raise
                n_categories=[3, 4],
                control=MORPControl(iid=True, verbose=0),
            )

    def test_friendly_message_mentions_spec(self):
        """The TypeError message should mention 'spec=' and 'MORP-001' to guide migration."""
        df = _make_small_df()
        spec = _make_spec(["y1", "y2"])
        with pytest.raises(TypeError, match="spec="):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec=spec,
                n_categories=[3, 4],
                indep_vars=["a", "b"],  # old API: triggers friendly message
            )

    def test_other_unexpected_kwargs_raise(self):
        """Any other unexpected kwarg should raise TypeError with the key name."""
        df = _make_small_df()
        spec = _make_spec(["y1", "y2"])
        with pytest.raises(TypeError, match="totally_unknown"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec=spec,
                n_categories=[3, 4],
                totally_unknown=True,
            )


# ---------------------------------------------------------------------------
# (2) MORPControl.indep → iid rename
# ---------------------------------------------------------------------------


class TestIidReplacesIndep:
    def test_iid_default_false(self):
        ctrl = MORPControl()
        assert ctrl.iid is False

    def test_iid_true(self):
        ctrl = MORPControl(iid=True)
        assert ctrl.iid is True

    def test_iid_attribute_canonical(self):
        """iid is the real attribute; no more indep dataclass field."""
        ctrl = MORPControl(iid=True)
        assert hasattr(ctrl, "iid")
        # Internal count_morp_params and loglik reference ctrl.iid
        from pybhatlib.models.morp._morp_loglik import count_morp_params
        n = count_morp_params(2, 2, [3, 3], ctrl)
        # iid=True → no cov params → 2 + 4 = 6
        assert n == 6


class TestIndepKwargEmitsDeprecationWarning:
    def test_deprecation_warning_emitted(self):
        """MORPControl(indep=True) should emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match="indep"):
            ctrl = MORPControl(indep=True)
        assert ctrl.iid is True

    def test_indep_getter_warns(self):
        """Accessing ctrl.indep should emit DeprecationWarning."""
        ctrl = MORPControl(iid=True)
        with pytest.warns(DeprecationWarning, match="indep"):
            val = ctrl.indep
        assert val is True

    def test_indep_setter_warns(self):
        """Setting ctrl.indep should emit DeprecationWarning and update iid."""
        ctrl = MORPControl()
        with pytest.warns(DeprecationWarning, match="indep"):
            ctrl.indep = True
        assert ctrl.iid is True


# ---------------------------------------------------------------------------
# (3) Path input
# ---------------------------------------------------------------------------


class TestPathInputDispatch:
    def test_csv_path(self, tmp_path: "pathlib.Path"):
        """Passing a CSV path should load data and build X correctly."""
        df = _make_small_df(n=30, seed=5)
        csv_file = tmp_path / "test_data.csv"
        df.to_csv(csv_file, index=False)

        spec = _make_spec(["y1", "y2"])
        model = MORPModel(
            data=str(csv_file),
            dep_vars=["y1", "y2"],
            spec=spec,
            n_categories=[3, 4],
            control=MORPControl(iid=True, verbose=0),
        )

        assert isinstance(model.data, pd.DataFrame)
        assert model.X.shape == (30, 2, 3)
        assert model.N == 30

    def test_pathlib_path(self, tmp_path: "pathlib.Path"):
        """Passing a pathlib.Path should also work."""
        from pathlib import Path

        df = _make_small_df(n=20, seed=9)
        csv_file = tmp_path / "test_data2.csv"
        df.to_csv(csv_file, index=False)

        spec = _make_spec(["y1", "y2"])
        model = MORPModel(
            data=Path(csv_file),
            dep_vars=["y1", "y2"],
            spec=spec,
            n_categories=[3, 4],
            control=MORPControl(iid=True, verbose=0),
        )
        assert isinstance(model.data, pd.DataFrame)


# ---------------------------------------------------------------------------
# (4) Smoke test with MORP Dining dataset
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_morp_dining_spec_smoke():
    """MORP Dining end-to-end: path input + per-outcome spec + model.fit()."""
    import os
    from pathlib import Path

    # Locate the CSV
    repo_root = Path(__file__).parents[3]
    csv_path = repo_root / "Gauss Files and Comparison" / "MORP" / "Example_Dining.csv"
    if not csv_path.exists():
        pytest.skip(f"MORP Dining CSV not found at {csv_path}")

    # ivord from MORP_DINING.gss lines 105-107 translated to spec dict:
    #
    #   let ivord = { resta20   in150    sero      sero    sero   sero ,
    #                 sero      sero     resta20   urb     sero   sero ,
    #                 sero      sero     sero      sero    wrk_H  urb  };
    #
    # var_ordnames = { E_rest20  E_in150  P_rest20  P_urb   D_wrk_h  D_urb };
    # Outer keys = coefficient names; inner keys = dep_var (outcome) names.
    #
    # Column name capitalisation in the CSV:
    dep_vars = ["NeatoutO", "Npickupo", "Ndelivo"]
    spec = {
        "E_rest20": {"NeatoutO": "resta20",  "Npickupo": "sero",    "Ndelivo": "sero"},
        "E_in150":  {"NeatoutO": "in150",    "Npickupo": "sero",    "Ndelivo": "sero"},
        "P_rest20": {"NeatoutO": "sero",     "Npickupo": "resta20", "Ndelivo": "sero"},
        "P_urb":    {"NeatoutO": "sero",     "Npickupo": "urb",     "Ndelivo": "sero"},
        "D_wrk_h":  {"NeatoutO": "sero",     "Npickupo": "sero",    "Ndelivo": "wrk_H"},
        "D_urb":    {"NeatoutO": "sero",     "Npickupo": "sero",    "Ndelivo": "urb"},
    }

    # Load and inspect the CSV to verify column names
    df_check = pd.read_csv(csv_path)
    required_cols = {"resta20", "in150", "urb", "wrk_H"} | set(dep_vars)
    missing = required_cols - set(df_check.columns)
    if missing:
        # Try lowercase column names
        df_check.columns = [c.lower() for c in df_check.columns]
        if missing - set(df_check.columns):
            pytest.skip(f"CSV missing columns: {missing}")

    # Determine n_categories from the dep_var max values
    n_categories = [
        int(df_check[dv].max()) + 1 for dv in dep_vars
    ]

    model = MORPModel(
        data=csv_path,
        dep_vars=dep_vars,
        spec=spec,
        n_categories=n_categories,
        control=MORPControl(
            iid=True,
            method="ovus",
            verbose=0,
            maxiter=200,
            seed=42,
        ),
    )

    results = model.fit()
    assert results.converged or results.return_code in (0, 1, 2)
    assert np.isfinite(results.loglik)


# ---------------------------------------------------------------------------
# (5) dataclasses.replace / asdict compatibility shims (MORP-001 Fix #2)
# ---------------------------------------------------------------------------


class TestMORPControlReplace:
    def test_replace_single_field(self):
        """morp_control_replace should return a new instance with the changed field."""
        ctrl = MORPControl(maxiter=100, verbose=1)
        ctrl2 = morp_control_replace(ctrl, maxiter=500)
        assert ctrl2.maxiter == 500
        # Original must be unchanged
        assert ctrl.maxiter == 100

    def test_replace_preserves_other_fields(self):
        """Unchanged fields should carry over from the original."""
        ctrl = MORPControl(iid=True, verbose=2, seed=42)
        ctrl2 = morp_control_replace(ctrl, verbose=0)
        assert ctrl2.iid is True
        assert ctrl2.seed == 42
        assert ctrl2.verbose == 0

    def test_replace_returns_morp_control_instance(self):
        ctrl = MORPControl()
        ctrl2 = morp_control_replace(ctrl, tol=1e-8)
        assert isinstance(ctrl2, MORPControl)

    def test_replace_unknown_field_raises(self):
        ctrl = MORPControl()
        with pytest.raises(TypeError, match="bad_field"):
            morp_control_replace(ctrl, bad_field=True)

    def test_replace_multiple_fields(self):
        ctrl = MORPControl(maxiter=200, tol=1e-5, optimizer="bfgs")
        ctrl2 = morp_control_replace(ctrl, maxiter=50, optimizer="lbfgsb")
        assert ctrl2.maxiter == 50
        assert ctrl2.optimizer == "lbfgsb"
        assert ctrl2.tol == 1e-5


class TestMORPControlAsDict:
    def test_returns_dict(self):
        ctrl = MORPControl()
        d = morp_control_asdict(ctrl)
        assert isinstance(d, dict)

    def test_contains_all_fields(self):
        ctrl = MORPControl(iid=True, verbose=0, seed=7)
        d = morp_control_asdict(ctrl)
        assert d["iid"] is True
        assert d["verbose"] == 0
        assert d["seed"] == 7

    def test_roundtrip(self):
        """asdict → MORPControl(**d) should produce an equivalent instance."""
        ctrl = MORPControl(iid=True, method="scipy", maxiter=50, verbose=0)
        d = morp_control_asdict(ctrl)
        ctrl2 = MORPControl(**d)
        assert ctrl2.iid == ctrl.iid
        assert ctrl2.method == ctrl.method
        assert ctrl2.maxiter == ctrl.maxiter

    def test_deprecated_indep_not_in_dict(self):
        """The deprecated 'indep' alias must NOT appear in asdict output."""
        ctrl = MORPControl(iid=True)
        d = morp_control_asdict(ctrl)
        assert "indep" not in d
