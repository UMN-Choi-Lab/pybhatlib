"""GAUSS LL parity gate for the mixed / panel MDCEV facade (plan Phase 3, G5).

At the converged GAUSS ``b`` (27 params) with the **exact** post-``createhalt``
draws GAUSS consumed (``FixtureDrawSource``), the shared MSL engine driving the
:class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel.LogitJacobianKernel`
must reproduce the GAUSS total log-likelihood **value-for-value**. The MDCEV
logit likelihood is *exact* (not an OVUS approximation), so the tolerance is far
tighter than the copula (MNP/MORP) models.

Fixture (``tests/fixtures/mixed/mdcev/scag_converged/``)
    * ``b.csv``      -- 27-param converged GAUSS vector,
      ``[beta(12) | gamma(9) | rcor(1) | kern(1) | scal(2) | lam(2)]``.
    * ``ass.csv``    -- ``(nrep=10, nii*nrndcoef=3000)`` post-``createhalt`` draws.
    * ``ll_obs.csv`` -- per-individual GAUSS log-likelihood (1500 cross-sectional
      individuals); its sum is the GAUSS total.

GAUSS total LL = ``-14979.9098285287`` on ``Workshop_SCAG_Est.csv`` (``nc=6``,
``nrndcoef=2``, ``logvar={male_ho}``, ``yjvar={male_Soc}``, ``varneg={male_ho}``).

The engine matches GAUSS ``w1 = w2 = 0`` (no probability flooring) via
``floor_pcomp = floor_z = 0`` (the MDCEV logit probability is strictly positive).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_model import MDCEVMixedModel
from pybhatlib.vecup._panel import PanelIndex

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_FIXTURE = os.path.join(
    _REPO, "tests", "fixtures", "mixed", "mdcev", "scag_converged"
)
_DATA_CANDIDATES = [
    os.path.join(_REPO, "0714_UTA_request", "Workshop_SCAG_Est.csv"),
    os.path.join(_FIXTURE, "Workshop_SCAG_Est.csv"),
]

_GAUSS_LL = -14979.9098285287


# SCAG baseline-utility design (GAUSS ivmt: nc=6 rows x nvarm=12 cols).
_IVMT = np.array(
    [
        ["sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "Lin",  "sero", "sero", "sero"],
        ["sero", "uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "Lin",  "sero", "sero"],
        ["sero", "sero", "uno",  "sero", "sero", "sero", "sero", "gend1", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero", "sero", "Lin",  "Min"],
    ],
    dtype=str,
)
_VARNAM = [
    "ASC_Esc", "ASC_ho", "ASC_Soc", "ASC_AR", "ASC_Eo",
    "male_Esc", "male_ho", "male_Soc", "Lin_Esc", "Lin_ho", "Lin_Eo", "Min_Eo",
]

# SCAG satiation design (GAUSS ivgt: nc=6 rows x nvargam=9 cols).
_IVGT = np.array(
    [
        ["uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "uno",  "sero", "sero", "sero", "sero", "sero", "sero", "sero"],
        ["sero", "sero", "uno",  "sero", "sero", "sero", "gend1", "sero", "sero"],
        ["sero", "sero", "sero", "uno",  "sero", "sero", "sero", "gend1", "sero"],
        ["sero", "sero", "sero", "sero", "uno",  "sero", "sero", "sero", "sero"],
        ["sero", "sero", "sero", "sero", "sero", "uno",  "sero", "sero", "gend1"],
    ],
    dtype=str,
)
_VARNGAM = [
    "G_Out", "G_Esc", "G_ho", "G_Soc", "G_AR", "G_Eo",
    "male_ho", "male_Soc", "male_Eo",
]

_ALTS = ["alt_out", "Esc", "Ho", "Soc", "AR", "Eo"]


def _data_path() -> str:
    for p in _DATA_CANDIDATES:
        if os.path.exists(p):
            return p
    pytest.skip(
        "Workshop_SCAG_Est.csv not found (searched "
        f"{_DATA_CANDIDATES}); GAUSS-parity fixture data unavailable."
    )


def _build_model() -> MDCEVMixedModel:
    ctrl = MDCEVMixedControl(
        utility="trad",
        logvar=("male_ho",),
        yjvar=("male_Soc",),
        varneg=("male_ho",),
        n_rep=10,
        intordn1=20,
        floor_pcomp=0.0,       # GAUSS w1 = 0
        floor_z=0.0,           # GAUSS w2 = 0
        verbose=0,
    )
    return MDCEVMixedModel(
        data=_data_path(),
        alternatives=_ALTS,
        price=["uno"] * len(_ALTS),
        utility_spec=_IVMT,
        gamma_spec=_IVGT,
        param_names=_VARNAM,
        gamma_names=_VARNGAM,
        control=ctrl,
        obs_id_var="ID",
    )


def test_mdcev_mixed_gauss_ll_parity():
    """Engine total LL at the fixture ``b`` == GAUSS -14979.9098285287."""
    model = _build_model()
    spec, layout = model._build_spec_layout()

    # layout must reproduce the GAUSS b partition exactly.
    assert layout.n_theta == 27
    sl = layout.slices()
    assert (sl["beta"].start, sl["beta"].stop) == (0, 12)
    assert (sl["gamma"].start, sl["gamma"].stop) == (12, 21)
    assert (sl["rcor"].start, sl["rcor"].stop) == (21, 22)
    assert (sl["kern"].start, sl["kern"].stop) == (22, 23)
    assert (sl["scal"].start, sl["scal"].stop) == (23, 25)
    assert (sl["lam"].start, sl["lam"].stop) == (25, 27)

    panel = PanelIndex.from_ids(model.person_ids)
    assert panel.n_ind == 1500 and panel.n_obs == 1500      # cross-sectional

    draws = FixtureDrawSource(os.path.join(_FIXTURE, "ass.csv"))
    est = model.build_estimator(spec, layout, panel, draws=draws)

    b = np.loadtxt(os.path.join(_FIXTURE, "b.csv"))
    assert b.shape == (27,)

    ll, _ = est.simulated_loglik(b, want_grad=False)
    total = float(ll.sum())

    # value-for-value (MDCEV is exact): expect far tighter than 1e-5.
    assert np.isclose(total, _GAUSS_LL, rtol=1e-5, atol=0.0), (
        f"total LL {total!r} != GAUSS {_GAUSS_LL!r} (delta {total - _GAUSS_LL:.3e})"
    )

    # per-individual parity against the dumped ll_obs.csv.
    ll_obs = np.loadtxt(os.path.join(_FIXTURE, "ll_obs.csv"))
    assert ll_obs.shape == (1500,)
    worst = float(np.max(np.abs(ll - ll_obs)))
    assert worst < 1e-8, f"per-individual LL max|delta| = {worst:.3e}"
