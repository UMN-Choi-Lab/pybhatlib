"""GAUSS value-for-value LL parity for the mixed / PANEL MORP facade (morp_flex).

At the converged GAUSS ``b`` with the byte-identical GAUSS draws
(``FixtureDrawSource``), the Python MORP engine must reproduce the GAUSS total
log-likelihood.  This is the first PANEL parity check (2323 individuals over
10816 observations, so the ``Dmask`` panel product is genuinely exercised), and
the first ``pdfrectn`` (rectangle-MVNCD) kernel check.  ``pdfrectn`` is an
OVUS-style approximation like MNP's ``cdfmvnanl``, so the target is
OVUS-level (``rtol=1e-4``), not machine precision.

The GAUSS driver is ``gauss_harness/morp_flex/morp_panelcommute_dump.gss``
(spec: nord=2 binary outcomes M_stop_b/E_stop_b; normvar={UNO_M,UNO_E};
_normker=1 normal kernel; _nocorrrcker=1 so no rc<->kernel copula).
Skipped when the (gitignored, 87 MB) panel_commute dataset is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.models.morp_flex import MORPFlexControl, MORPFlexModel
from pybhatlib.vecup._panel import PanelIndex

_FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures", "mixed", "morp", "panelcommute_converged")
_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "0714_UTA_request",
    "panel_commute_dataset_with_exog.csv",
)
_GAUSS_LL = -7400.626799592934

# ivord (from the GAUSS driver) -> per-coefficient {outcome: data column}, in
# GAUSS var_ordnames order (this fixes the beta block order to match b.csv).
_SPEC = {
    "UNO_M":  {"M_stop_b": "UNO",          "E_stop_b": "SERO"},
    "UNO_E":  {"M_stop_b": "SERO",         "E_stop_b": "UNO"},
    "AGE65M": {"M_stop_b": "age_above_64", "E_stop_b": "SERO"},
    "AGE45E": {"M_stop_b": "SERO",         "E_stop_b": "age_45_64"},
    "AGE65E": {"M_stop_b": "SERO",         "E_stop_b": "age_above_64"},
    "BLACKM": {"M_stop_b": "black_only",   "E_stop_b": "SERO"},
    "CHILD":  {"M_stop_b": "pres_child",   "E_stop_b": "pres_child"},
    "HYBRID": {"M_stop_b": "hybrid",       "E_stop_b": "hybrid"},
    "TT":     {"M_stop_b": "M_TT",         "E_stop_b": "E_TT"},
}


@pytest.mark.skipif(not os.path.exists(_DATA), reason="panel_commute dataset absent (gitignored 87MB)")
def test_morp_flex_gauss_ll_parity():
    import pandas as pd

    df = pd.read_csv(_DATA, low_memory=False)
    # sort by person id so first-appearance order == sorted-unique order == GAUSS nuu
    df = df.sort_values("INDID", kind="mergesort").reset_index(drop=True)

    ctrl = MORPFlexControl(
        person_id="INDID",
        normvar=("UNO_M", "UNO_E"),
        copula=False,      # _nocorrrcker = 1
        yj_kernel=False,   # _normker = 1
        n_rep=10,
        spher=False,
        scal=1.0,
        floor_pcomp=0.0,   # GAUSS sets w1 = w2 = 0
        floor_z=0.0,
    )
    model = MORPFlexModel(
        data=df,
        dep_vars=["M_stop_b", "E_stop_b"],
        spec=_SPEC,
        n_categories=[2, 2],
        control=ctrl,
    )
    spec, layout = model._build_spec_layout()
    assert layout.n_theta == 21, f"expected 21 params, got {layout.n_theta}"

    panel = PanelIndex.from_ids(model.person_ids)
    assert panel.n_ind == 2323 and panel.n_obs == 10816

    draws = FixtureDrawSource(os.path.join(_FIX, "ass.csv"))
    est = model._build_estimator(spec, layout, panel, draws=draws)

    b = np.loadtxt(os.path.join(_FIX, "b.csv"))
    assert b.shape == (21,)

    ll, _ = est.simulated_loglik(b, want_grad=False)
    total = float(np.asarray(ll).sum())
    assert np.isclose(total, _GAUSS_LL, rtol=1e-4, atol=0.0), (
        f"MORP total LL {total!r} vs GAUSS {_GAUSS_LL!r} "
        f"(rel {abs(total - _GAUSS_LL) / abs(_GAUSS_LL):.3e})"
    )

    # per-individual LL alignment (nuu order)
    ll_obs = np.loadtxt(os.path.join(_FIX, "ll_obs.csv"))
    assert ll_obs.shape == (2323,)
    worst = float(np.max(np.abs(np.asarray(ll) - ll_obs)))
    assert worst < 1e-3, f"per-individual LL max|delta| = {worst:.3e}"


def test_morp_parity_meta_present():
    with open(os.path.join(_FIX, "meta.json")) as fh:
        meta = json.load(fh)
    assert meta["reference"]["ll_total"] == _GAUSS_LL
    assert meta["n_ind"] == 2323
