"""MDCEV forecast parity against the GAUSS BHATLIB forecasting drivers.

The references are the ``fout1.xlsx`` files written by
``Gauss Files and Comparison/MDCEV Traditional/Forecasting TradMDCEV.gss`` and
``Gauss Files and Comparison/MDCEV Linear/Forecasting_LinMDCEV.gss`` (UTA drop,
gitignored -- the tests skip when it is absent). Both drivers hard-code their
converged ``bmdcev`` vectors, which are reproduced here with the GAUSS ``-1000``
outside-good placeholder removed (pybhatlib has no such parameter).

GAUSS and pybhatlib use different Gumbel draw streams, so the comparison is on
aggregate participation rates and mean allocations over ``reps x obs``
simulated allocations, with tolerances set by the Monte Carlo noise of the
GAUSS reference itself. The budget identity (row sums == budget) is exact.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mdcev import MDCEVControl, MDCEVModel
from pybhatlib.models.mdcev._mdcev_forecast import (
    mdcev_forecast,
    prepare_mdcev_forecast_data,
)
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GAUSS = os.path.join(_REPO, "Gauss Files and Comparison")
_LIN = os.path.join(_GAUSS, "MDCEV Linear")
_TRAD = os.path.join(_GAUSS, "MDCEV Traditional")

pytestmark = pytest.mark.slow


def _require(*paths: str) -> None:
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"GAUSS reference not available: {p}")
    pytest.importorskip("openpyxl")


def _reference(xlsx: str) -> np.ndarray:
    """GAUSS ``forec`` output: columns are ID, 1, then one allocation per good."""
    return pd.read_excel(xlsx, header=None).to_numpy(dtype=float)[:, 2:]


def _row(alts, **kw):
    r = {a: "sero" for a in alts}
    r.update(kw)
    return r


def _check(fc: np.ndarray, ref: np.ndarray, budget: np.ndarray, *,
           part_tol_pp: float, inside_mean_rtol: float) -> None:
    assert fc.shape == ref.shape
    n_obs = budget.shape[0]
    # exact budget identity for every simulated allocation
    np.testing.assert_allclose(fc.sum(axis=1), np.tile(budget, fc.shape[0] // n_obs),
                               rtol=1e-10, atol=1e-8)
    part_py = (fc > 0).mean(axis=0) * 100.0
    part_gs = (ref > 0).mean(axis=0) * 100.0
    assert np.all(np.abs(part_py - part_gs) < part_tol_pp), (part_py, part_gs)
    mean_py = fc.mean(axis=0)
    mean_gs = ref.mean(axis=0)
    # outside good: dominant share of the budget, tight
    assert abs(mean_py[0] / mean_gs[0] - 1.0) < 2e-3, (mean_py[0], mean_gs[0])
    # inside goods: heavy-tailed allocations, Monte Carlo noise dominates
    rel = np.abs(mean_py[1:] / mean_gs[1:] - 1.0)
    assert np.all(rel < inside_mean_rtol), (mean_py, mean_gs)


def test_linear_forecast_matches_gauss_fout1():
    data = os.path.join(_LIN, "WorkshopData_ToursimExp_Vali.csv")
    xlsx = os.path.join(_LIN, "fout1.xlsx")
    _require(data, xlsx)

    alts = ["Transp", "Accomod", "FandB", "Shp", "Recr"]
    uspec = {
        "ASCAcc": _row(alts, Accomod="uno"), "ASCFnB": _row(alts, FandB="uno"),
        "ASCShp": _row(alts, Shp="uno"), "ASCRec": _row(alts, Recr="uno"),
        "urbAcc": _row(alts, Accomod="urban"), "urbFnB": _row(alts, FandB="urban"),
        "urbshp": _row(alts, Shp="urban"), "urbRec": _row(alts, Recr="urban"),
        "stl3Acc": _row(alts, Accomod="stlt3"), "st410acc": _row(alts, Accomod="st410"),
    }
    gspec = {
        "GAcc": _row(alts, Accomod="uno"), "GFnB": _row(alts, FandB="uno"),
        "GShp": _row(alts, Shp="uno"), "GRec": _row(alts, Recr="uno"),
        "urbAcc": _row(alts, Accomod="urban"), "urbFnB": _row(alts, FandB="urban"),
        "urbshp": _row(alts, Shp="urban"),
        "stl3Acc": _row(alts, Accomod="stlt3"), "st410acc": _row(alts, Accomod="st410"),
        "trlFnB": _row(alts, FandB="b51q11"), "trlShp": _row(alts, Shp="b51q11"),
        "trlRec": _row(alts, Recr="b51q11"),
    }
    # Forecasting_LinMDCEV.gss ``bmdcev`` (minus the -1000 placeholder).
    psi = [-0.808202714251692, 0.731234459824711, 1.06054820199217, -0.611599523507261,
           0.369888842845592, 0.181422387925194, -0.326089727282693, 0.0797315296162311,
           0.793299599509931, 0.447620582542694]
    gam = [9.23016054313031, 6.00682426848757, 6.03739098374529, 6.69388399872308,
           0.207089920449732, 0.360784692370941, 0.667134608295719, -1.88769730534571,
           -1.09326625172878, 0.0129648764666295, 0.0251824728822728, 0.0321997832805945]
    sigma = 0.392863578838737

    vali = pd.read_csv(data).head(500)                    # GAUSS: nobs = 500
    model = MDCEVModel(data=vali, alternatives=alts, availability=None,
                       utility_spec=uspec, gamma_spec=gspec,
                       control=MDCEVControl(utility="linear", verbose=0))
    names = list(model.param_names) + list(model.gamma_names) + ["sigma"]
    b = np.array(psi + gam + [sigma])
    assert len(names) == b.size
    res = MDCEVResults.from_estimates(b_reported=b, sigma=sigma,
                                      control=MDCEVControl(utility="linear"),
                                      param_names=names)
    X, Xg, P, budget, _ = prepare_mdcev_forecast_data(model, vali, None, None, budget_col="bud")
    fc = mdcev_forecast(res, X, Xg, P, budget, n_replications=50, seed=232445, num_outside=1)
    _check(fc, _reference(xlsx), budget, part_tol_pp=1.0, inside_mean_rtol=0.15)


def test_traditional_forecast_matches_gauss_fout1():
    data = os.path.join(_TRAD, "Workshop_SCAG_Vali.csv")
    xlsx = os.path.join(_TRAD, "fout1.xlsx")
    _require(data, xlsx)

    alts = ["alt_out", "Esc", "Ho", "Soc", "AR", "Eo"]
    uspec = {
        "ASC_Esc": _row(alts, Esc="uno"), "ASC_ho": _row(alts, Ho="uno"),
        "ASC_Soc": _row(alts, Soc="uno"), "ASC_AR": _row(alts, AR="uno"),
        "ASC_Eo": _row(alts, Eo="uno"),
        "male_Esc": _row(alts, Esc="gend1"), "male_ho": _row(alts, Ho="gend1"),
        "male_Soc": _row(alts, Soc="gend1"),
        "Lin_Esc": _row(alts, Esc="Lin"), "Lin_ho": _row(alts, Ho="Lin"),
        "Lin_Eo": _row(alts, Eo="Lin"), "Min_Eo": _row(alts, Eo="Min"),
    }
    gspec = {
        "G_Esc": _row(alts, Esc="uno"), "G_ho": _row(alts, Ho="uno"),
        "G_Soc": _row(alts, Soc="uno"), "G_AR": _row(alts, AR="uno"),
        "G_Eo": _row(alts, Eo="uno"),
        "male_ho": _row(alts, Ho="gend1"), "male_Soc": _row(alts, Soc="gend1"),
        "male_Eo": _row(alts, Eo="gend1"),
    }
    # Forecasting TradMDCEV.gss ``bmdcev`` (minus the -1000 placeholder).
    psi = [-7.39239, -6.42058, -7.44907, -7.85216, -7.65862, -0.16493,
           -0.40753, 0.30726, -0.21468, -0.47719, 0.15036, 0.19853]
    gam = [3.21384, 5.77560, 5.17502, 2.71833, 3.35770, -0.76811, 0.30558, 0.21766]
    sigma = 0.60838

    vali = pd.read_csv(data).head(500)
    model = MDCEVModel(data=vali, alternatives=alts, availability=None,
                       utility_spec=uspec, gamma_spec=gspec,
                       control=MDCEVControl(utility="trad", verbose=0))
    names = list(model.param_names) + list(model.gamma_names) + ["sigma"]
    b = np.array(psi + gam + [sigma])
    assert len(names) == b.size
    res = MDCEVResults.from_estimates(b_reported=b, sigma=sigma,
                                      control=MDCEVControl(utility="trad"),
                                      param_names=names)
    # GAUSS driver scenario: gender dummy fixed to 0, time budget "tot".
    X, Xg, P, budget, _ = prepare_mdcev_forecast_data(
        model, vali, changevar=["gend1"], changeval=[0], budget_col="tot",
    )
    fc = mdcev_forecast(res, X, Xg, P, budget, n_replications=200, seed=232445, num_outside=1)
    _check(fc, _reference(xlsx), budget, part_tol_pp=0.5, inside_mean_rtol=0.10)
