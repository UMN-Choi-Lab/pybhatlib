"""Gamma index-vector layouts accepted by the MDCEV likelihood helpers.

The fixed-coefficient model builds ``ivg`` with ``nc - 1`` rows per gamma
parameter (inside goods only: the outside good's satiation is pinned to
``MDCEVControl.outside_good_gamma`` and is not a parameter). The mixed MDCEV
kernel keeps the full GAUSS ``ivgt`` layout (``nc`` rows per parameter, outside
good first, its slot a pinned placeholder). Both layouts must give the same
log-likelihood and gradient, and any other length must raise instead of
silently reading the wrong columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev import MDCEVControl, MDCEVModel
from pybhatlib.models.mdcev._mdcev_loglik import (
    _gamma_rows_per_param,
    mdcev_gradient,
    mdcev_loglik,
)
from pybhatlib.models.mdcev._mdcev_model import _build_data_arrays

ALTS = ["alt_out", "alt1", "alt2", "alt3"]
USPEC = {
    "c1": {"alt1": "uno"},
    "c2": {"alt2": "uno"},
    "c3": {"alt3": "uno"},
    "x": {"alt1": "x1", "alt2": "x2", "alt3": "x1"},
}
GSPEC = {
    "g1": {"alt1": "uno"},
    "g2": {"alt2": "uno"},
    "g3": {"alt3": "uno"},
    "gz": {"alt1": "z", "alt2": "z"},
}


def _frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "z": rng.uniform(0.2, 1.0, n),
    })
    for k, a in enumerate(ALTS):
        on = np.ones(n, dtype=bool) if k == 0 else rng.uniform(size=n) < 0.6
        df[a] = np.where(on, rng.uniform(0.5, 5.0, n), 0.0)
    df["uno"] = 1.0
    df["sero"] = 0.0
    df["ID"] = np.arange(n)
    return df


def _arrays(utility: str):
    ctrl = MDCEVControl(utility=utility, verbose=0)
    m = MDCEVModel(
        data=_frame(), alternatives=ALTS, availability=None,
        utility_spec=USPEC, gamma_spec=GSPEC, control=ctrl,
    )
    dta, ivm, ivg, flagchm, flagprcm, wtind = _build_data_arrays(
        m.data, m.alternatives, m.availability, m.utility_spec, m.gamma_spec,
        ctrl.weight_var,
    )
    nc, nvarm, nvargam = m.n_alts, m.utility_spec.shape[1], m.gamma_spec.shape[1]
    return ctrl, dta, ivm, ivg, flagchm, flagprcm, wtind, nc, nvarm, nvargam


@pytest.mark.parametrize("utility", ["trad", "linear"])
def test_full_and_inside_only_layouts_agree(utility):
    ctrl, dta, ivm, ivg, flagchm, flagprcm, wtind, nc, nvarm, nvargam = _arrays(utility)
    assert ivg.size == nvargam * (nc - 1)
    assert _gamma_rows_per_param(ivg, nvargam, nc) == nc - 1

    # Full GAUSS layout: prepend an outside-good row to every parameter. Point
    # it at a column of ones to prove the row is ignored (the outside-good
    # gamma is pinned, so its data must not leak into the likelihood).
    ones_col = int(np.flatnonzero(np.all(dta == 1.0, axis=0))[0])
    ivg_full = np.concatenate([
        np.concatenate([[ones_col], ivg[j * (nc - 1): (j + 1) * (nc - 1)]])
        for j in range(nvargam)
    ]).astype(ivg.dtype)
    assert ivg_full.size == nvargam * nc
    assert _gamma_rows_per_param(ivg_full, nvargam, nc) == nc

    rng = np.random.default_rng(0)
    theta = np.concatenate([
        0.3 * rng.standard_normal(nvarm),
        np.log(rng.uniform(0.5, 3.0, nvargam)),
        [np.log(0.8)],
    ])
    eq = np.eye(nvargam)
    args = (flagchm, flagprcm, wtind, nvarm, nvargam, nc, eq, ctrl)

    ll_inside = mdcev_loglik(theta, dta, ivm, ivg, *args)
    ll_full = mdcev_loglik(theta, dta, ivm, ivg_full, *args)
    np.testing.assert_allclose(ll_full, ll_inside, rtol=0.0, atol=1e-13)

    g_inside = mdcev_gradient(theta, dta, ivm, ivg, *args)
    g_full = mdcev_gradient(theta, dta, ivm, ivg_full, *args)
    np.testing.assert_allclose(g_full, g_inside, rtol=0.0, atol=1e-12)


def test_malformed_gamma_layout_raises():
    ctrl, dta, ivm, ivg, flagchm, flagprcm, wtind, nc, nvarm, nvargam = _arrays("trad")
    theta = np.zeros(nvarm + nvargam + 1)
    eq = np.eye(nvargam)
    bad = ivg[:-1]                       # neither nvargam*(nc-1) nor nvargam*nc
    with pytest.raises(ValueError, match="ivg has"):
        mdcev_loglik(theta, dta, ivm, bad, flagchm, flagprcm, wtind, nvarm, nvargam, nc, eq, ctrl)
    with pytest.raises(ValueError, match="ivg has"):
        _gamma_rows_per_param(bad, nvargam, nc)
