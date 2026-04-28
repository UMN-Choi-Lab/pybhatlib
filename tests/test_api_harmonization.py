"""Tests for PR A — MNP/MORP API harmonization.

Covers:

1. ``MNPResults`` exposes canonical field names ``params``, ``loglik``,
   and ``n_iter``.
2. The legacy attribute names (``b``, ``ll``, ``ll_total``,
   ``n_iterations``) still work as deprecated aliases that emit
   ``DeprecationWarning`` and forward to the canonical fields.
3. ``MNPModel`` accepts a ``pathlib.Path`` for ``data=``.
4. ``mnp_predict`` is importable from the public package path.
5. The package-level convenience imports (``from pybhatlib import
   MNPModel, MORPModel, ...``) succeed.
6. ``MNPControl(indep=True)`` raises ``TypeError`` (the dead field
   has been removed).
"""

from __future__ import annotations

import os
import pathlib
import warnings

import numpy as np
import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel

ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]
SPEC_BASE = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}


@pytest.fixture
def fitted_iid(travelmode_path):
    """Fit a tiny IID MNP once per test function.

    Function-scope inherited from ``travelmode_path``; cheap enough
    given the tiny ``maxiter`` budget below (20 IID iterations on
    1125 obs is well under a second).
    """
    ctrl = MNPControl(iid=True, maxiter=20, verbose=0, seed=42)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_BASE,
        control=ctrl,
    )
    return model.fit()


# ----------------------------------------------------------------------
# 1. Canonical field names
# ----------------------------------------------------------------------


def test_mnp_results_canonical_field_names(fitted_iid):
    results = fitted_iid

    # Each canonical attribute must exist and have the expected type.
    assert isinstance(results.params, np.ndarray)
    assert results.params.ndim == 1
    assert results.params.size > 0

    assert isinstance(results.loglik, float)
    assert np.isfinite(results.loglik)

    assert isinstance(results.n_iter, int)
    assert results.n_iter >= 1


# ----------------------------------------------------------------------
# 2. Deprecated aliases
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_name, canonical_name",
    [
        ("b", "params"),
        ("ll", "loglik"),
        ("n_iterations", "n_iter"),
    ],
)
def test_mnp_results_deprecated_aliases_warn(fitted_iid, old_name, canonical_name):
    """Reading a legacy attribute warns and returns the canonical value."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        old_value = getattr(fitted_iid, old_name)

    # A DeprecationWarning must have been emitted, mentioning the old name
    # and the canonical replacement.
    matching = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning)
        and old_name in str(w.message)
        and canonical_name in str(w.message)
    ]
    assert matching, (
        f"Expected DeprecationWarning mentioning {old_name!r} -> "
        f"{canonical_name!r}, got {[str(w.message) for w in captured]}"
    )

    canonical_value = getattr(fitted_iid, canonical_name)
    if isinstance(canonical_value, np.ndarray):
        np.testing.assert_array_equal(old_value, canonical_value)
    else:
        assert old_value == canonical_value


def test_mnp_results_ll_total_alias_returns_total(fitted_iid):
    """``ll_total`` keeps its semantic meaning (total = mean * n_obs)."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        total = fitted_iid.ll_total

    matching = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning)
        and "ll_total" in str(w.message)
    ]
    assert matching, (
        f"Expected DeprecationWarning for ll_total, got "
        f"{[str(w.message) for w in captured]}"
    )

    expected = fitted_iid.loglik * fitted_iid.n_obs
    assert total == pytest.approx(expected)


# ----------------------------------------------------------------------
# 3. PathLike support
# ----------------------------------------------------------------------


def test_mnp_model_pathlike(travelmode_path):
    p = pathlib.Path(travelmode_path)
    assert p.exists()

    model = MNPModel(
        data=p,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_BASE,
        control=MNPControl(iid=True, maxiter=1, verbose=0, seed=0),
    )

    # Loaded the data and recorded the path as a plain string.
    assert model.N > 0
    assert isinstance(model.data_path, str)
    assert model.data_path == os.fspath(p)


def test_mnp_model_rejects_unknown_data_type():
    with pytest.raises(TypeError):
        MNPModel(
            data=12345,  # not a DataFrame, str, or PathLike
            alternatives=ALTERNATIVES,
            availability="none",
            spec=SPEC_BASE,
        )


# ----------------------------------------------------------------------
# 4. Public mnp_predict export
# ----------------------------------------------------------------------


def test_mnp_predict_exported():
    from pybhatlib.models.mnp import mnp_predict, mnp_predict_choice  # noqa: F401

    assert callable(mnp_predict)
    assert callable(mnp_predict_choice)


# ----------------------------------------------------------------------
# 5. Top-level imports
# ----------------------------------------------------------------------


def test_top_level_imports():
    from pybhatlib import (  # noqa: F401
        MNPControl,
        MNPModel,
        MNPResults,
        MORPControl,
        MORPModel,
        MORPResults,
        mnp_ate,
        mnp_predict,
        mnp_predict_choice,
        morp_ate,
        morp_predict,
        morp_predict_category,
    )


# ----------------------------------------------------------------------
# 6. MNPControl.indep removed
# ----------------------------------------------------------------------


def test_mnp_control_no_indep_field():
    """``MNPControl(indep=True)`` should now fail — the field is dead."""
    with pytest.raises(TypeError):
        MNPControl(indep=True)

    # And the attribute should not exist on a default-constructed
    # control either.
    ctrl = MNPControl()
    assert not hasattr(ctrl, "indep")
