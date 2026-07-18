"""GAUSS-LL parity gate for the mixed-panel MNP (MNPKerCP) engine.

Runs the Python MSL engine at the GAUSS fixed parameter vector ``b`` with the
byte-identical GAUSS draws (``FixtureDrawSource``) and asserts the total
simulated log-likelihood reproduces the GAUSS ``lpr`` dump
(``-659.284470695413``) to a relative tolerance of ``1e-4``.

The fixture (``tests/fixtures/mixed/mnp/travelmode_nocopula/``) was dumped from
``gauss_harness/mnpkercp/mnpkercp_travelmode_dump.gss`` with ``_nocorrrcker = 1``
(copula off), cross-sectional (``persid = CASE``), on the TRAVELMODE data with
the spec ``normvar = {IVTT}``, ``yjvar = {COST}``, ``nc = 3``, ``nrep = 5``.

Note on tolerance
-----------------
GAUSS ``cdfmvnanl`` is an OVUS (one-variate-conditioned-upon-successive) MVNCD
approximation carrying an internal seed; the Python ``mvncd`` OVUS path uses an
independent internal approximation, so the two agree to ~1e-6 on the total LL
rather than to machine precision.  The observed relative delta is ~3e-9 (see the
report), well inside the 1e-4 target; the gate is *not* loosened beyond the
plan's stated ``1e-4`` OVUS tolerance.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel
from pybhatlib.vecup._panel import PanelIndex

# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(__file__)
_FIX = os.path.join(_HERE, "..", "fixtures", "mixed", "mnp", "travelmode_nocopula")
_FIX_COP = os.path.join(_HERE, "..", "fixtures", "mixed", "mnp", "travelmode_copula")
_DATA = os.path.join(_HERE, "..", "..", "examples", "data", "TRAVELMODE.csv")

_GAUSS_TOTAL_LL = -659.284470695413
_GAUSS_TOTAL_LL_COP = -736.182691701131

# GAUSS spec (mnpkercp_travelmode_estimate.gss)
_VAR_NAMES = ["IVTT", "OVTT", "COST", "AGE45_DA", "AGE45_SR", "CON_SR", "CON_TR"]
_IVUNORD = [
    ["IVTT_DA", "OVTT_DA", "COST_DA", "AGE45", "SERO", "SERO", "SERO"],
    ["IVTT_SR", "OVTT_SR", "COST_SR", "SERO", "AGE45", "UNO", "SERO"],
    ["IVTT_TR", "OVTT_TR", "COST_TR", "SERO", "SERO", "SERO", "UNO"],
]
_CHOICE = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]


def _build_design(df: pd.DataFrame):
    n_obs = len(df)
    nc = len(_IVUNORD)
    nvar = len(_VAR_NAMES)
    X = np.zeros((n_obs, nc, nvar), dtype=np.float64)
    for a in range(nc):
        for v in range(nvar):
            X[:, a, v] = df[_IVUNORD[a][v]].to_numpy(dtype=np.float64)
    chosen = df[_CHOICE].to_numpy(dtype=np.float64)
    avail = np.ones((n_obs, nc), dtype=np.float64)
    return X, chosen, avail


class _Obs:
    def __init__(self, avail, chosen):
        self.avail = avail
        self.chosen = chosen


@pytest.mark.skipif(
    not os.path.exists(_DATA) or not os.path.exists(os.path.join(_FIX, "b.csv")),
    reason="TRAVELMODE data or MNP fixture missing",
)
def test_mnpkercp_gauss_ll_parity():
    df = pd.read_csv(_DATA)
    X, chosen, avail = _build_design(df)
    nc = len(_IVUNORD)

    b = np.loadtxt(os.path.join(_FIX, "b.csv"))
    ll_obs_ref = np.loadtxt(os.path.join(_FIX, "ll_obs.csv"))
    assert b.shape[0] == 18, f"expected 18 params, got {b.shape[0]}"

    spec = MixingSpec.from_var_names(
        var_names=_VAR_NAMES, normvar=("IVTT",), yjvar=("COST",),
        kernel_dim=nc - 1,
    )
    # spec sanity vs GAUSS globals (nrndtot=4, nrndtcor=6, n_kern=1)
    assert (spec.nrndcoef, spec.nrndtot, spec.nrndtcor, spec.nscale,
            spec.numlam, spec.n_kern) == (2, 4, 6, 2, 2, 1)

    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=spec.nrndtcor, n_scal=spec.nscale,
        n_lam=spec.numlam, n_kern=spec.n_kern, kern_before_lam=True,
    )
    assert layout.n_theta == b.shape[0]

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    # fixture is _nocorrrcker=1 -> copula=False
    kernel = MvncdKernel(nc, spec.nrndcoef, copula=False, scal=1.0)

    design = DesignData(X=X, obs=_Obs(avail, chosen))
    draws = FixtureDrawSource(os.path.join(_FIX, "ass.csv"))
    panel = PanelIndex.from_ids(df["CASE"].to_numpy())
    assert panel.n_ind == len(df)  # cross-sectional (Dmask = I)

    cfg = MSLConfig(
        n_rep=5, scal=1.0, intordn1=20,
        floor_pcomp=1e-4, floor_z=1e-4, score_convention="mask",
    )
    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    ll, _ = est.simulated_loglik(b, want_grad=False)
    total = float(ll.sum())

    rel = abs(total - _GAUSS_TOTAL_LL) / abs(_GAUSS_TOTAL_LL)
    assert rel < 1e-4, (
        f"MNPKerCP total LL {total:.9f} vs GAUSS {_GAUSS_TOTAL_LL:.9f} "
        f"(rel={rel:.2e}) exceeds 1e-4"
    )
    # per-observation LL also tracks the GAUSS dump within OVUS noise
    assert np.max(np.abs(ll - ll_obs_ref)) < 1e-3


@pytest.mark.skipif(
    not os.path.exists(_DATA) or not os.path.exists(os.path.join(_FIX_COP, "b.csv")),
    reason="TRAVELMODE data or MNP copula fixture missing",
)
def test_mnpkercp_gauss_ll_parity_copula():
    """GAUSS-LL parity for the **copula-active** kernel (``_nocorrrcker = 0``).

    Same TRAVELMODE recipe as the no-copula gate, cross-sectional
    (``persid = CASE``), but with the rc<->kernel copula switched on so the
    kernel error is conditioned on the drawn (correlated, pre-YJ) random
    coefficients ``errbeta3 = errbeta1 @ x11chol``. The fixture is dumped from
    ``gauss_harness/mnpkercp/mnpkercp_travelmode_copula_dump.gss`` with
    ``_nocorrrcker = 0``; the GAUSS ``lpr`` total is ``-736.182691701131``.
    """
    df = pd.read_csv(_DATA)
    X, chosen, avail = _build_design(df)
    nc = len(_IVUNORD)

    b = np.loadtxt(os.path.join(_FIX_COP, "b.csv"))
    ll_obs_ref = np.loadtxt(os.path.join(_FIX_COP, "ll_obs.csv"))
    assert b.shape[0] == 18, f"expected 18 params, got {b.shape[0]}"

    spec = MixingSpec.from_var_names(
        var_names=_VAR_NAMES, normvar=("IVTT",), yjvar=("COST",),
        kernel_dim=nc - 1,
    )
    layout = ParamLayout(
        n_beta=spec.n_beta, n_rcor=spec.nrndtcor, n_scal=spec.nscale,
        n_lam=spec.numlam, n_kern=spec.n_kern, kern_before_lam=True,
    )
    assert layout.n_theta == b.shape[0]

    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    # copula ON (_nocorrrcker = 0)
    kernel = MvncdKernel(nc, spec.nrndcoef, copula=True, scal=1.0)

    design = DesignData(X=X, obs=_Obs(avail, chosen))
    draws = FixtureDrawSource(os.path.join(_FIX_COP, "ass.csv"))
    panel = PanelIndex.from_ids(df["CASE"].to_numpy())
    assert panel.n_ind == len(df)  # cross-sectional (Dmask = I)

    cfg = MSLConfig(
        n_rep=5, scal=1.0, intordn1=20,
        floor_pcomp=1e-4, floor_z=1e-4, score_convention="mask",
    )
    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )

    ll, _ = est.simulated_loglik(b, want_grad=False)
    total = float(ll.sum())

    rel = abs(total - _GAUSS_TOTAL_LL_COP) / abs(_GAUSS_TOTAL_LL_COP)
    assert rel < 1e-4, (
        f"MNPKerCP copula total LL {total:.9f} vs GAUSS "
        f"{_GAUSS_TOTAL_LL_COP:.9f} (rel={rel:.2e}) exceeds 1e-4"
    )
    # per-observation LL also tracks the GAUSS dump within OVUS noise
    assert np.max(np.abs(ll - ll_obs_ref)) < 1e-2
