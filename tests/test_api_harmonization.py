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
from pybhatlib.models.mnp._mnp_results import MNPResults

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


# ----------------------------------------------------------------------
# 7. MNPResults legacy-kwarg deprecation shim (construction-time)
# ----------------------------------------------------------------------

# Minimal set of values for all required (no-default) MNPResults fields.
_CORR = np.eye(2)
_REQUIRED = dict(
    b_original=np.array([0.5, -0.5]),
    se=np.array([0.1, 0.1]),
    t_stat=np.array([5.0, -5.0]),
    p_value=np.array([0.01, 0.01]),
    gradient=np.array([0.0, 0.0]),
    loglik=-1.5,
    n_obs=100,
    param_names=["x1", "x2"],
    corr_matrix=_CORR,
    cov_matrix=_CORR * 0.01,
    n_iter=10,
    convergence_time=0.01,
    converged=True,
    return_code=0,
)


_LEGACY_TO_CANONICAL = {"b": "params", "ll": "loglik", "n_iterations": "n_iter"}


def _make_results(**overrides):
    """Construct MNPResults with all required fields, allowing overrides.

    Automatically drops the canonical field when a legacy alias is provided
    (so callers can say ``_make_results(b=arr)`` without also passing params).
    """
    kw = dict(_REQUIRED)
    # params is required — supply it unless the caller provides b= (legacy)
    if "params" not in overrides and "b" not in overrides:
        kw["params"] = np.array([0.5, -0.5])
    # Drop canonical names for any legacy alias the caller is providing.
    for legacy, canonical in _LEGACY_TO_CANONICAL.items():
        if legacy in overrides:
            kw.pop(canonical, None)
    kw.update(overrides)
    return MNPResults(**kw)


def test_legacy_b_kwarg_emits_deprecation_warning():
    """``b=`` at construction time warns and maps to ``params``."""
    arr = np.array([1.0, 2.0])
    with pytest.warns(DeprecationWarning, match=r"MNPResults\(b=\.\.\.\) is deprecated"):
        r = _make_results(b=arr)
    np.testing.assert_array_equal(r.params, arr)
    # The read alias should also still work (and point to the same value).
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        np.testing.assert_array_equal(r.b, arr)


def test_legacy_ll_kwarg_emits_deprecation_warning():
    """``ll=`` at construction time warns and maps to ``loglik``."""
    with pytest.warns(DeprecationWarning, match=r"MNPResults\(ll=\.\.\.\) is deprecated"):
        r = _make_results(ll=-2.5)
    assert r.loglik == pytest.approx(-2.5)


def test_legacy_n_iterations_kwarg_emits_deprecation_warning():
    """``n_iterations=`` at construction time warns and maps to ``n_iter``."""
    with pytest.warns(DeprecationWarning, match=r"MNPResults\(n_iterations=\.\.\.\) is deprecated"):
        r = _make_results(n_iterations=42)
    assert r.n_iter == 42


def test_legacy_ll_total_kwarg_warns_and_discards():
    """``ll_total=`` warns and is discarded; reading it returns loglik*n_obs."""
    with pytest.warns(DeprecationWarning, match="ll_total"):
        r = _make_results(ll_total=-999.0)
    # ll_total was discarded; canonical loglik unchanged
    assert r.loglik == pytest.approx(_REQUIRED["loglik"])
    # The read property still computes total on demand
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always", DeprecationWarning)
        assert r.ll_total == pytest.approx(r.loglik * r.n_obs)


def test_passing_both_legacy_and_canonical_raises():
    """Supplying both ``b=`` and ``params=`` is ambiguous and must raise."""
    with pytest.raises(TypeError, match="received both legacy"):
        _make_results(b=np.array([1.0, 2.0]), params=np.array([3.0, 4.0]))


def test_unknown_kwarg_still_raises():
    """Unknown kwargs should raise ``TypeError`` as before."""
    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        _make_results(xyz="bad")


def test_canonical_construction_unaffected():
    """Constructing with canonical kwargs must not emit any warning."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        r = _make_results()
    dep_warnings = [w for w in captured if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings == [], f"Unexpected DeprecationWarning(s): {dep_warnings}"
    assert isinstance(r.params, np.ndarray)


def test_read_aliases_still_work_after_init_change(fitted_iid):
    """Existing read-alias tests must pass with the new __init__ in place."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        _ = fitted_iid.b
        _ = fitted_iid.ll
        _ = fitted_iid.n_iterations

    names = [str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)]
    assert any("b" in m for m in names)
    assert any("ll" in m for m in names)
    assert any("n_iterations" in m for m in names)
