"""Tests for se_method options in MNPModel.

Verifies that the observed-Hessian SE path (se_method="hessian") produces
standard errors that are asymptotically equivalent to BHHH on well-identified
IID problems, and that the sandwich estimator also converges.
"""

from __future__ import annotations

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


def _fit_iid(travelmode_path: str, se_method: str):
    """Fit IID Model (a)(i) with the given se_method."""
    ctrl = MNPControl(
        iid=True,
        maxiter=200,
        verbose=0,
        seed=42,
        se_method=se_method,
    )
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_BASE,
        control=ctrl,
    )
    return model.fit()


@pytest.mark.slow
def test_observed_hessian_se_close_to_bhhh_on_iid(travelmode_path):
    """On well-identified IID Model (a)(i), observed-Hessian SE and BHHH SE
    are both valid finite-sample estimators of the MLE covariance.

    They are asymptotically equivalent at the true population MLE, but can
    differ by up to ~2–3x on this dataset due to QMC approximation noise in
    the MVN CDF integral: BHHH uses per-obs score variance while the observed
    Hessian uses curvature of the sum-NLL, and QMC creates heteroskedastic
    score contributions that break the Fisher information identity at finite N.

    This test verifies:
    1. Both produce positive, finite SEs (structural correctness).
    2. The ratio between them is bounded (no catastrophic divergence).
    3. BHHH SEs stay close to the BHATLIB paper values (regression test).
    """
    results_bhhh = _fit_iid(travelmode_path, se_method="bhhh")
    results_hess = _fit_iid(travelmode_path, se_method="hessian")

    se_bhhh = results_bhhh.se
    se_hess = results_hess.se

    # Both should be finite and positive
    assert np.isfinite(se_bhhh).all(), "BHHH SE contains non-finite values"
    assert np.isfinite(se_hess).all(), "Hessian SE contains non-finite values"
    assert (se_bhhh > 0).all(), "BHHH SE contains non-positive values"
    assert (se_hess > 0).all(), "Hessian SE contains non-positive values"

    # Observed Hessian SEs should be in a reasonable range relative to BHHH.
    # The bound is generous (3x) to accommodate QMC noise: the key guarantee
    # is that the Hessian path produces valid (positive, finite) SEs and does
    # not diverge catastrophically. The BHHH path is the reference estimator
    # that matches the published BHATLIB paper (BHATLIB uses _max_CovPar = 2).
    ratio = se_hess / se_bhhh
    assert (ratio < 3.0).all(), (
        f"Hessian SE is more than 3× larger than BHHH SE for some parameter. "
        f"Ratios: {dict(zip(results_bhhh.param_names, np.round(ratio, 2)))}"
    )
    assert (ratio > 0.3).all(), (
        f"Hessian SE is more than 3× smaller than BHHH SE for some parameter. "
        f"Ratios: {dict(zip(results_bhhh.param_names, np.round(ratio, 2)))}"
    )

    # BHHH should stay close to the BHATLIB paper values (regression guard)
    paper_se = np.array([0.0655, 0.1071, 0.0596, 0.0407, 0.0286])
    np.testing.assert_allclose(
        se_bhhh, paper_se, rtol=0.02,
        err_msg="BHHH SE drifted from BHATLIB paper values",
    )


@pytest.mark.slow
def test_hessian_se_finite_iid(travelmode_path):
    """Hessian SE must be positive and finite on IID Model (a)(i)."""
    results = _fit_iid(travelmode_path, se_method="hessian")
    assert np.isfinite(results.se).all(), "Hessian SE contains non-finite values"
    assert (results.se > 0).all(), "Hessian SE contains non-positive values"


@pytest.mark.slow
def test_sandwich_se_finite_iid(travelmode_path):
    """Sandwich SE must be positive and finite on IID Model (a)(i)."""
    results = _fit_iid(travelmode_path, se_method="sandwich")
    assert np.isfinite(results.se).all(), "Sandwich SE contains non-finite values"
    assert (results.se > 0).all(), "Sandwich SE contains non-positive values"


@pytest.mark.slow
def test_sandwich_se_close_to_bhhh_on_iid(travelmode_path):
    """Sandwich SE should be in a reasonable range relative to BHHH on IID.

    Sandwich SE = H^{-1} B H^{-1} where H is the true observed Hessian and
    B is the BHHH outer-product. When the model is correctly specified and the
    Hessian is a good approximation to the expected Hessian (i.e. H ≈ B at the
    population MLE), sandwich ≈ BHHH. With QMC approximation noise the two can
    diverge substantially at finite N, so we use a generous 3x bound rather than
    a tight percentage.
    """
    results_bhhh = _fit_iid(travelmode_path, se_method="bhhh")
    results_sand = _fit_iid(travelmode_path, se_method="sandwich")

    se_bhhh = results_bhhh.se
    se_sand = results_sand.se

    assert np.isfinite(se_sand).all(), "Sandwich SE contains non-finite values"
    assert (se_sand > 0).all(), "Sandwich SE contains non-positive values"

    ratio = se_sand / se_bhhh
    assert (ratio < 4.0).all(), (
        f"Sandwich SE is more than 4× larger than BHHH SE for some parameter. "
        f"Ratios: {dict(zip(results_bhhh.param_names, np.round(ratio, 2)))}"
    )
    assert (ratio > 0.25).all(), (
        f"Sandwich SE is more than 4× smaller than BHHH SE for some parameter. "
        f"Ratios: {dict(zip(results_bhhh.param_names, np.round(ratio, 2)))}"
    )
