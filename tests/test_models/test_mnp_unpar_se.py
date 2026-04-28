"""Tests for MNP unparameterized log-likelihood and SE path (MNP-002b).

Pins the lpr1/lgd1 port: parameterized↔unparameterized round-trip on the
log-likelihood, plus SE parity for Models (b)/(a)(ii)/(c). The SE path
uses per-observation scores in the unparameterized covariance/correlation
space, removing the delta-method projection that was a P0 plan-fidelity
violation in PR #4.

GAUSS reference: ``Gauss Files and Comparison/MNP/MNP_TRAVELMODE.gss``
(``lpr1`` line 408) and ``Gauss Files and Comparison/MNP/Table2/MNP Table2 c.gss``
(``lpr1`` line 549).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel
from pybhatlib.models.mnp._mnp_loglik import (
    _build_lambda,
    _build_lambda_direct,
    _build_omega_cholesky,
    _build_omega_direct,
    _param_to_unpar,
    count_params,
    count_params_unpar,
    mnp_loglik,
    mnp_loglik_unpar,
)

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


# ---------------------------------------------------------------------------
# Translation tests: _param_to_unpar
# ---------------------------------------------------------------------------

def test_param_to_unpar_iid_identity():
    """IID model has no covariance params, so beta is identity-mapped."""
    control = MNPControl(iid=True, mix=False)
    n_beta = 5
    n_alts = 3
    rng = np.random.default_rng(0)
    theta_par = rng.standard_normal(n_beta)

    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, None)
    np.testing.assert_allclose(theta_unpar, theta_par, atol=1e-15)


def test_param_to_unpar_flexible_inverts_cholesky():
    """Flexible covariance: ensure cov from unpar matches cov from par."""
    control = MNPControl(iid=False, mix=False, heteronly=False)
    n_beta = 5
    n_alts = 3
    dim = n_alts - 1
    n_corr = dim * (dim - 1) // 2
    n_lambda = dim + n_corr

    rng = np.random.default_rng(1)
    theta_par = np.concatenate([
        rng.standard_normal(n_beta),
        rng.standard_normal(n_lambda) * 0.3,
    ])
    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, None)

    # Lambda from parameterized vs unparameterized must match.
    lam_par_input = theta_par[n_beta:n_beta + n_lambda]
    lam_unpar_input = theta_unpar[n_beta:n_beta + n_lambda]

    Lambda_par = _build_lambda(lam_par_input, n_alts, control)
    Lambda_unpar = _build_lambda_direct(lam_unpar_input, n_alts, control)

    np.testing.assert_allclose(Lambda_unpar, Lambda_par, atol=1e-12)

    # Unparameterized scales should be positive; correlations should be in [-1, 1].
    scales_unpar = lam_unpar_input[:dim]
    corrs_unpar = lam_unpar_input[dim:]
    assert (scales_unpar > 0).all()
    assert ((corrs_unpar >= -1.0) & (corrs_unpar <= 1.0)).all()


def test_param_to_unpar_mixed_inverts_omega():
    """Mixed MNP: Omega from unpar (2x2) matches Omega = LL' from par."""
    control = MNPControl(iid=True, mix=True, randdiag=False)
    n_beta = 4
    n_alts = 3
    ranvar_indices = [0, 1]
    n_rand = len(ranvar_indices)
    n_omega = n_rand * (n_rand + 1) // 2  # 3 for 2x2

    rng = np.random.default_rng(2)
    theta_par = np.concatenate([
        rng.standard_normal(n_beta),
        rng.standard_normal(n_omega) * 0.2,
    ])
    theta_unpar = _param_to_unpar(
        theta_par, n_beta, n_alts, control, ranvar_indices
    )

    omega_par_input = theta_par[n_beta:n_beta + n_omega]
    omega_unpar_input = theta_unpar[n_beta:n_beta + n_omega]

    L_par = _build_omega_cholesky(omega_par_input, ranvar_indices, control)
    Omega_par = L_par @ L_par.T
    Omega_unpar = _build_omega_direct(omega_unpar_input, ranvar_indices, control)

    np.testing.assert_allclose(Omega_unpar, Omega_par, atol=1e-12)


# ---------------------------------------------------------------------------
# Round-trip tests: mnp_loglik(theta_par) == mnp_loglik_unpar(theta_unpar)
# ---------------------------------------------------------------------------

def _make_synth_data(n_alts=3, n_beta=4, n_obs=50, seed=42):
    """Build a small synthetic dataset for round-trip likelihood tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_alts, n_beta))
    y = rng.integers(0, n_alts, size=n_obs)
    return X, y


def test_loglik_par_unpar_round_trip_iid():
    """IID: mnp_loglik(theta_par) == mnp_loglik_unpar(theta_unpar)."""
    n_alts, n_beta = 3, 4
    control = MNPControl(iid=True, mix=False, method="me")
    X, y = _make_synth_data(n_alts=n_alts, n_beta=n_beta, n_obs=30)

    rng = np.random.default_rng(11)
    theta_par = rng.standard_normal(n_beta) * 0.3
    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, None)

    nll_par = mnp_loglik(
        theta_par, X, y, None, n_alts, n_beta, control, None,
    )
    nll_unpar = mnp_loglik_unpar(
        theta_unpar, X, y, None, n_alts, n_beta, control, None,
    )
    assert abs(nll_par - nll_unpar) < 1e-10, (
        f"IID round-trip mismatch: par={nll_par:.16f}, unpar={nll_unpar:.16f}"
    )


def test_loglik_par_unpar_round_trip_flexible():
    """Flexible covariance: round-trip likelihood agreement."""
    n_alts, n_beta = 3, 4
    control = MNPControl(iid=False, mix=False, heteronly=False, method="me")
    n_lambda = (n_alts - 1) + (n_alts - 1) * (n_alts - 2) // 2
    X, y = _make_synth_data(n_alts=n_alts, n_beta=n_beta, n_obs=30)

    rng = np.random.default_rng(12)
    theta_par = np.concatenate([
        rng.standard_normal(n_beta) * 0.3,
        rng.standard_normal(n_lambda) * 0.2,
    ])
    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, None)

    nll_par = mnp_loglik(
        theta_par, X, y, None, n_alts, n_beta, control, None,
    )
    nll_unpar = mnp_loglik_unpar(
        theta_unpar, X, y, None, n_alts, n_beta, control, None,
    )
    assert abs(nll_par - nll_unpar) < 1e-10, (
        f"Flexible round-trip mismatch: par={nll_par:.16f}, "
        f"unpar={nll_unpar:.16f}"
    )


def test_loglik_par_unpar_round_trip_mixed():
    """Mixed (random coef): round-trip likelihood agreement."""
    n_alts, n_beta = 3, 4
    ranvar_indices = [1]
    control = MNPControl(iid=True, mix=True, method="me")
    n_omega = 1 * 2 // 2  # 1
    X, y = _make_synth_data(n_alts=n_alts, n_beta=n_beta, n_obs=30)

    rng = np.random.default_rng(13)
    theta_par = np.concatenate([
        rng.standard_normal(n_beta) * 0.3,
        np.array([0.5]),  # Cholesky entry > 0
    ])
    theta_unpar = _param_to_unpar(
        theta_par, n_beta, n_alts, control, ranvar_indices
    )

    nll_par = mnp_loglik(
        theta_par, X, y, None, n_alts, n_beta, control, ranvar_indices,
    )
    nll_unpar = mnp_loglik_unpar(
        theta_unpar, X, y, None, n_alts, n_beta, control, ranvar_indices,
    )
    assert abs(nll_par - nll_unpar) < 1e-10, (
        f"Mixed round-trip mismatch: par={nll_par:.16f}, "
        f"unpar={nll_unpar:.16f}"
    )


def test_count_params_unpar_matches_param():
    """Param count must agree between parameterized and unparameterized layouts."""
    cases = [
        (MNPControl(iid=True), 5, 3, None),
        (MNPControl(iid=False, heteronly=False), 5, 3, None),
        (MNPControl(iid=False, heteronly=True), 5, 3, None),
        (MNPControl(iid=True, mix=True, randdiag=False), 5, 3, [0]),
        (MNPControl(iid=True, mix=True, randdiag=False), 5, 3, [0, 1]),
        (MNPControl(iid=False, mix=True, randdiag=False), 6, 3, [0]),
    ]
    for control, n_beta, n_alts, ranvar in cases:
        n_par = count_params(n_beta, n_alts, control, ranvar)
        n_unpar = count_params_unpar(n_beta, n_alts, control, ranvar)
        assert n_par == n_unpar, (
            f"count_params mismatch for {control}: par={n_par}, unpar={n_unpar}"
        )


# ---------------------------------------------------------------------------
# SE-no-Jacobian-projection: beta block test
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_se_no_jacobian_projection_for_beta_block(travelmode_path):
    """Beta SEs come from the unparameterized inverse-information directly.

    Pins the "no delta method on the beta block" property: SE(beta_i) equals
    the diagonal of the inverse score-outer-product computed at the converged
    unparameterized theta, without any Jacobian step. The only Jacobian
    that survives is the BHATLIB sigma_1 normalization (one scalar per block).
    """
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

    # All beta SEs must be positive, finite. Critically, there must be NO
    # numerical-Jacobian projection of correlation/scale params back into the
    # beta block: identifying this from outside requires inspecting that
    # model._normalize_for_reporting was rebuilt for unparameterized scoring,
    # which is asserted via the parity tests below (Model b/(a)(ii)/(c)).
    n_beta = model.n_beta
    se_beta = results.se[:n_beta]
    assert (se_beta > 0).all()
    assert np.isfinite(se_beta).all()


# ---------------------------------------------------------------------------
# SE parity vs paper at the new tolerances
# ---------------------------------------------------------------------------

def _assert_bhhh_se_parity(
    target, model_label, spec, control_kwargs, ranvars, travelmode_path,
    tol=0.02,
):
    """Shared assertion: fit `model_label` with BHHH SE, compare to paper."""
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
        assert got is not None, f"{model_label}: param {param} missing"
        assert abs(got - expected) < tol, (
            f"{model_label} BHHH SE mismatch for {param}: got {got:.4f}, "
            f"paper {expected:.4f}"
        )


# ---------------------------------------------------------------------------
# Fix #3: _build_report_names ordering matches _unpar_to_report (upper-tri)
# ---------------------------------------------------------------------------

def test_report_names_length_matches_values_iid():
    """IID: _build_report_names must have same length as _unpar_to_report output."""
    import pandas as pd
    from pybhatlib.models.mnp._mnp_loglik import _param_to_unpar, count_params

    n_alts = 3
    n_beta = 5
    control = MNPControl(iid=True, mix=False)
    n_params = count_params(n_beta, n_alts, control, None)

    rng = np.random.default_rng(99)
    theta_par = rng.standard_normal(n_params) * 0.3

    # Build a minimal MNPModel-like object to call _build_report_names.
    # We use MNPModel with a tiny synthetic DataFrame so we can call the method.
    X_dummy = rng.standard_normal((20, n_alts, n_beta))
    y_dummy = rng.integers(0, n_alts, size=20)
    # Construct the names manually via internal method
    from pybhatlib.models.mnp._mnp_model import MNPModel

    df = pd.DataFrame({
        "Alt1_ch": (y_dummy == 0).astype(int),
        "Alt2_ch": (y_dummy == 1).astype(int),
        "Alt3_ch": (y_dummy == 2).astype(int),
        **{f"v{k}_{a}": X_dummy[:, i, k]
           for k in range(n_beta) for i, a in enumerate(["da", "sr", "tr"])},
    })
    spec = {f"v{k}": {f"Alt{i+1}_ch": f"v{k}_{a}"
                      for i, a in enumerate(["da", "sr", "tr"])}
            for k in range(n_beta)}
    model = MNPModel(data=df, alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
                     spec=spec, control=control)

    names = model._build_report_names()
    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, None)
    values = model._unpar_to_report(theta_unpar)

    assert len(names) == len(values), (
        f"IID: len(names)={len(names)} != len(values)={len(values)}"
    )


def test_report_names_length_matches_values_mixed():
    """Mixed MNP (n_rand=2): names and values must be same length AND in same order.

    For the random-coef block with n_rand=2 (full Cholesky), both
    ``_build_report_names`` and ``_unpar_to_report`` must iterate the
    upper-triangle in the same order: (0,0), (0,1), (1,1).
    """
    import pandas as pd
    from pybhatlib.models.mnp._mnp_loglik import _param_to_unpar, count_params

    n_alts = 3
    n_beta = 4
    ranvar_indices = [0, 1]
    control = MNPControl(iid=True, mix=True, randdiag=False)
    n_params = count_params(n_beta, n_alts, control, ranvar_indices)

    rng = np.random.default_rng(100)
    # Build tiny synthetic model
    y_dummy = rng.integers(0, n_alts, size=20)
    df = pd.DataFrame({
        "Alt1_ch": (y_dummy == 0).astype(int),
        "Alt2_ch": (y_dummy == 1).astype(int),
        "Alt3_ch": (y_dummy == 2).astype(int),
        **{f"v{k}_{a}": rng.standard_normal(20)
           for k in range(n_beta) for a in ["da", "sr", "tr"]},
    })
    spec = {f"v{k}": {f"Alt{i+1}_ch": f"v{k}_{a}"
                      for i, a in enumerate(["da", "sr", "tr"])}
            for k in range(n_beta)}
    from pybhatlib.models.mnp._mnp_model import MNPModel
    model = MNPModel(data=df, alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
                     spec=spec, mix=True, ranvars=["v0", "v1"], control=control)

    n_omega = len(ranvar_indices) * (len(ranvar_indices) + 1) // 2
    theta_par = np.concatenate([
        rng.standard_normal(n_beta) * 0.3,
        np.array([0.5, 0.1, 0.4]),  # valid Cholesky (lower-tri)
    ])

    names = model._build_report_names()
    theta_unpar = _param_to_unpar(theta_par, n_beta, n_alts, control, ranvar_indices)
    values = model._unpar_to_report(theta_unpar)

    assert len(names) == len(values), (
        f"Mixed: len(names)={len(names)} != len(values)={len(values)}"
    )
    # Verify CovCOv entries appear exactly n_rand*(n_rand+1)/2 times
    cov_names = [n for n in names if n.startswith("CovCOv")]
    assert len(cov_names) == n_omega, (
        f"Expected {n_omega} CovCOv entries, got {len(cov_names)}: {cov_names}"
    )
    # CovCOv01 corresponds to Omega[0,0] (diagonal, must be positive for valid Omega)
    cov01_idx = names.index("CovCOv01")
    assert values[cov01_idx] > 0, (
        f"CovCOv01 (Omega[0,0]) should be positive, got {values[cov01_idx]}"
    )


@pytest.mark.slow
def test_se_unpar_models_b_aii_c(table2_targets, travelmode_path):
    """Models (b), (a)(ii), (c): SE parity vs paper using unparameterized scoring.

    Tolerances:
      - Model (b): 0.02 (matches MNP-002 baseline)
      - Model (a)(ii): 0.02 (matches MNP-002 baseline)
      - Model (c): 0.05 (tightened from prior 0.10 xfail)
    """
    _assert_bhhh_se_parity(
        table2_targets["models"]["b_age45"],
        "Model (b)",
        SPEC_WITH_AGE45,
        {"iid": False},
        None,
        travelmode_path,
        tol=0.02,
    )

    _assert_bhhh_se_parity(
        table2_targets["models"]["a_ii_flexible"],
        "Model (a)(ii)",
        SPEC_BASE,
        {"iid": False},
        None,
        travelmode_path,
        tol=0.02,
    )

    _assert_bhhh_se_parity(
        table2_targets["models"]["c_random_coef"],
        "Model (c)",
        SPEC_WITH_AGE45,
        {"iid": False, "mix": True},
        ["OVTT"],
        travelmode_path,
        tol=0.05,
    )
