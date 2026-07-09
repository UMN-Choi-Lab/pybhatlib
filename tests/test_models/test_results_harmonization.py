"""Cross-model results-field harmonization (MNL, MDCEV).

Companion to ``tests/test_api_harmonization.py`` (which covers MNP).  All four
model results objects must expose the canonical estimation fields ``params``
(estimate vector), ``loglik`` (mean log-likelihood) and ``n_iter``, while the
per-model legacy names continue to work as deprecated, warning-emitting
aliases.

Field renames covered here:

* ``MNLResults``:  ``b`` → ``params``, ``ll`` → ``loglik``,
  ``n_iterations`` → ``n_iter``; ``ll_total`` is now computed (``loglik*n_obs``).
* ``MDCEVResults``: ``b`` → ``params`` (``loglik`` / ``n_iter`` were already
  canonical; ``ll_total`` remains a stored field).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pybhatlib.models.mnl._mnl_results import MNLResults
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults


# ----------------------------------------------------------------------
# Canonical field names present on every model
# ----------------------------------------------------------------------


def _mnl_results() -> MNLResults:
    return MNLResults.from_estimates(np.array([0.5, -0.5]), param_names=["x1", "x2"])


def _mdcev_results() -> MDCEVResults:
    return MDCEVResults.from_estimates(
        np.array([0.5, -0.3, 1.0]), param_names=["b0", "b1", "sigma"]
    )


def test_mnl_canonical_fields():
    r = _mnl_results()
    assert isinstance(r.params, np.ndarray) and r.params.ndim == 1
    assert isinstance(r.loglik, float)
    assert isinstance(r.n_iter, int)


def test_mdcev_canonical_fields():
    r = _mdcev_results()
    assert isinstance(r.params, np.ndarray) and r.params.ndim == 1
    assert isinstance(r.loglik, float)
    assert isinstance(r.n_iter, int)


# ----------------------------------------------------------------------
# Deprecated read-aliases warn and forward to the canonical field
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "old_name, canonical_name",
    [("b", "params"), ("ll", "loglik"), ("n_iterations", "n_iter")],
)
def test_mnl_read_aliases_warn(old_name, canonical_name):
    r = _mnl_results()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", DeprecationWarning)
        old_value = getattr(r, old_name)

    matching = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning)
        and old_name in str(w.message)
        and canonical_name in str(w.message)
    ]
    assert matching, [str(w.message) for w in captured]

    canonical_value = getattr(r, canonical_name)
    if isinstance(canonical_value, np.ndarray):
        np.testing.assert_array_equal(old_value, canonical_value)
    else:
        assert old_value == canonical_value or (
            np.isnan(old_value) and np.isnan(canonical_value)
        )


def test_mnl_ll_total_alias_returns_total():
    r = MNLResults(
        params=np.array([1.0]),
        se=np.array([0.1]),
        t_stat=np.array([10.0]),
        p_value=np.array([0.0]),
        gradient=np.array([0.0]),
        loglik=-1.5,
        n_obs=100,
        param_names=["x"],
        corr_matrix=np.eye(1),
        cov_matrix=np.eye(1) * 0.01,
        n_iter=5,
        convergence_time=0.0,
        converged=True,
        return_code=0,
    )
    with pytest.warns(DeprecationWarning, match="ll_total"):
        total = r.ll_total
    assert total == pytest.approx(r.loglik * r.n_obs)


def test_mdcev_b_alias_warns():
    r = _mdcev_results()
    with pytest.warns(DeprecationWarning, match=r"MDCEVResults\.b is deprecated"):
        old = r.b
    np.testing.assert_array_equal(old, r.params)


# ----------------------------------------------------------------------
# Legacy construction kwargs warn and map to canonical fields
# ----------------------------------------------------------------------


def test_mnl_legacy_b_kwarg_warns_and_maps():
    # Canonical construction (via from_estimates) must NOT warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        MNLResults.from_estimates(np.array([1.0, 2.0]))
    # Direct legacy construction warns and maps b= → params.
    with pytest.warns(DeprecationWarning, match=r"MNLResults\(b=\.\.\.\)"):
        r2 = _mnl_with_legacy(b=np.array([3.0, 4.0]))
    np.testing.assert_array_equal(r2.params, np.array([3.0, 4.0]))


def test_mnl_both_legacy_and_canonical_raises():
    with pytest.raises(TypeError, match="received both legacy"):
        _mnl_with_legacy(b=np.array([1.0]), params=np.array([2.0]))


def test_mdcev_legacy_b_kwarg_warns():
    with pytest.warns(DeprecationWarning, match=r"MDCEVResults\(b=\.\.\.\) is deprecated"):
        r = _mdcev_with_legacy(b=np.array([0.5, 1.0]))
    np.testing.assert_array_equal(r.params, np.array([0.5, 1.0]))


# --- construction helpers with all required fields --------------------

_MNL_REQ = dict(
    se=np.array([0.1]),
    t_stat=np.array([10.0]),
    p_value=np.array([0.0]),
    gradient=np.array([0.0]),
    loglik=-1.5,
    n_obs=100,
    param_names=["x"],
    corr_matrix=np.eye(1),
    cov_matrix=np.eye(1) * 0.01,
    n_iter=5,
    convergence_time=0.0,
    converged=True,
    return_code=0,
)

_MDCEV_REQ = dict(
    b_reported=np.array([0.5, 1.0]),
    se=np.array([0.1, 0.1]),
    t_stat=np.array([1.0, 1.0]),
    p_value=np.array([0.1, 0.1]),
    gradient=np.array([0.0, 0.0]),
    loglik=-1.5,
    ll_total=-150.0,
    n_obs=100,
    param_names=["b0", "sigma"],
    corr_matrix=np.eye(2),
    cov_matrix=np.eye(2) * 0.01,
    n_iter=5,
    convergence_time=0.0,
    converged=True,
    return_code=0,
    sigma=1.0,
)


def _mnl_with_legacy(**overrides):
    kw = dict(_MNL_REQ)
    n = len(overrides.get("b", overrides.get("params", [0.0])))
    kw["param_names"] = [f"x{i}" for i in range(n)]
    kw["se"] = np.full(n, 0.1)
    kw["t_stat"] = np.full(n, 1.0)
    kw["p_value"] = np.full(n, 0.1)
    kw["gradient"] = np.zeros(n)
    kw["corr_matrix"] = np.eye(n)
    kw["cov_matrix"] = np.eye(n) * 0.01
    kw.update(overrides)
    return MNLResults(**kw)


def _mdcev_with_legacy(**overrides):
    kw = dict(_MDCEV_REQ)
    n = len(overrides.get("b", overrides.get("params", [0.0])))
    kw["b_reported"] = np.full(n, 0.5)
    kw["param_names"] = [f"b{i}" for i in range(n)]
    kw["se"] = np.full(n, 0.1)
    kw["t_stat"] = np.full(n, 1.0)
    kw["p_value"] = np.full(n, 0.1)
    kw["gradient"] = np.zeros(n)
    kw["corr_matrix"] = np.eye(n)
    kw["cov_matrix"] = np.eye(n) * 0.01
    kw.update(overrides)
    return MDCEVResults(**kw)
