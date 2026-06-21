"""GAUSS BHATLIB <-> pybhatlib parity test for MNP (non-circular ATE check).

Mirrors the MORP GAUSS-parity approach (``test_morp_gauss_parity.py``): it
hardcodes the converged ``est`` block of the GAUSS flexible + AGE45 MNP model
from ``Gauss Files and Comparison/MNP/MNP_TRAVELMODE_ATE.gss`` and verifies that
feeding those reported estimates VERBATIM into :func:`mnp_ate_from_params`
reproduces the predicted shares of a freshly fitted flexible + AGE45 model.

This is a NON-circular check: the GAUSS numbers are transcribed constants, not
derived from the pybhatlib fit. If pybhatlib's reporting convention drifted away
from the GAUSS first-differenced-variance=1 kernel (e.g. a stray scale factor on
the betas, or a different kernel parameterization), the two share vectors would
diverge. Reproducing them to ~1e-3 confirms the reported GAUSS estimates plug in
directly.

GAUSS reference (``MNP_TRAVELMODE_ATE.gss`` lines 86-99, ``est`` block in
``var_unordnames`` order ``IVTT OVTT COST AGE45_DA AGE45_SR CON_SR CON_TR``,
then the kernel correlation ``parker`` and the relative alt-3 scale)::

    IVTT     = -0.8859
    OVTT     = -1.0367
    COST     = -0.5972
    AGE45_DA =  0.4375
    AGE45_SR =  0.3038
    CON_SR   = -0.9416
    CON_TR   = -0.4136
    parker   =  0.478
    scale    =  2.0075

Under the GAUSS homogeneous kernel the first differenced error variance is
pinned to 1, so the differenced (I-1)x(I-1) kernel is::

    K = [[1,            parker * scale],
         [parker * scale, scale ** 2  ]]

i.e. K[0,0] == 1 (alt-2's differenced variance), K[1,1] == scale**2 (alt-3's
relative differenced variance), and the off-diagonal is the correlation times
the geometric mean of the scales (scale01 = 1).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel
from pybhatlib.models.mnp._mnp_ate import mnp_ate, mnp_ate_from_params

ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# Flexible kernel + AGE45 alternative-specific covariates. Key order defines the
# coefficient order consumed by ``mnp_ate_from_params``.
SPEC_WITH_AGE45 = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero", "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# GAUSS converged estimates, transcribed verbatim from MNP_TRAVELMODE_ATE.gss.
_GAUSS = {
    "IVTT": -0.8859, "OVTT": -1.0367, "COST": -0.5972,
    "AGE45_DA": 0.4375, "AGE45_SR": 0.3038,
    "CON_SR": -0.9416, "CON_TR": -0.4136,
}
_GAUSS_PARKER = 0.478
_GAUSS_SCALE = 2.0075

# Betas in the SPEC_WITH_AGE45 key order.
_GAUSS_BETA = np.array([
    _GAUSS["CON_SR"], _GAUSS["CON_TR"],
    _GAUSS["AGE45_DA"], _GAUSS["AGE45_SR"],
    _GAUSS["IVTT"], _GAUSS["OVTT"], _GAUSS["COST"],
])

# Differenced kernel under first-diff-var=1 (K[0,0] pinned to 1).
_GAUSS_KERNEL = np.array([
    [1.0,                          _GAUSS_PARKER * _GAUSS_SCALE],
    [_GAUSS_PARKER * _GAUSS_SCALE, _GAUSS_SCALE ** 2],
])


@pytest.mark.slow
def test_mnp_gauss_flex_age45_ate_from_reported_estimates(travelmode_path):
    """GAUSS flexible+AGE45 reported estimates plug into mnp_ate_from_params and
    reproduce a freshly fitted model's predicted shares (non-circular).
    """
    data = pd.read_csv(travelmode_path)

    # 1) Fresh fit -> ATE predicted shares (the reference).
    ctrl = MNPControl(iid=False, maxiter=200, verbose=0, seed=42)
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_WITH_AGE45,
        control=ctrl,
    )
    res = model.fit()
    # Sanity: the fresh fit recovers the published flexible+AGE45 LL anchor.
    assert abs(res.loglik * res.n_obs - (-659.285)) < 0.5, (
        f"flexible+AGE45 LL drifted: got {res.loglik * res.n_obs:.3f}"
    )
    ate_fit = mnp_ate(res, data=data, spec=SPEC_WITH_AGE45, alternatives=ALTERNATIVES)

    # 2) GAUSS reported estimates -> ATE predicted shares (no re-fit).
    ate_gauss = mnp_ate_from_params(
        _GAUSS_BETA,
        kernel_cov=_GAUSS_KERNEL,
        control=MNPControl(iid=False),
        n_alts=3,
        data=data,
        spec=SPEC_WITH_AGE45,
        alternatives=ALTERNATIVES,
    )

    # Shares are probabilities and sum to 1.
    assert ate_gauss.predicted_shares.sum() == pytest.approx(1.0, abs=1e-6)

    # The reported GAUSS estimates reproduce the fresh fit's shares. Tolerance
    # 5e-3 comfortably bounds the ~2.5e-5 observed gap (the GAUSS est block is
    # rounded to 4 d.p., which is the dominant source of the tiny mismatch).
    max_diff = np.max(np.abs(ate_gauss.predicted_shares - ate_fit.predicted_shares))
    assert max_diff < 5e-3, (
        "GAUSS reported estimates do not reproduce the fresh-fit shares: "
        f"GAUSS {ate_gauss.predicted_shares}, fit {ate_fit.predicted_shares}, "
        f"max|Δ| = {max_diff:.2e}"
    )
