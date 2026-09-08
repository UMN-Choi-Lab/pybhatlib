"""COLLAPSE GATE: mixed MDCEV engine reduces to the shipped fixed-coef MDCEV.

Self-contained (no GAUSS reference). With ``nrndcoef = 0`` (empty mixing spec:
no random coefficients) and one observation per person (cross-sectional,
``Dmask == I``), the shared MSL engine driving the
:class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel.LogitJacobianKernel`
must reproduce the shipped fixed-coefficient
:func:`pybhatlib.models.mdcev.mdcev_loglik` **value-for-value** on the same data.

With ``nrndcoef = 0`` the drawn baseline utility ``xmunew`` equals the fixed
``beta`` for every draw, so the per-draw probability is constant, the average
over draws is exact, and (cross-sectional) the per-person simulated LL equals
the per-observation MDCEV LL. The engine estimation-space ``theta`` layout
``[beta | gamma | kern]`` (the ``rcor`` / ``scal`` / ``lam`` blocks vanish)
coincides with the shipped MDCEV parameter vector ``[beta | gamma | log_sigma]``,
so the *same* vector drives both. Floors are set to ``0`` to match the strictly
positive MDCEV logit probability (no flooring), giving ``1e-8`` equality.

This validates the identity-baseline-design wrapper of the shipped
``mdcev_loglik`` / ``mdcev_gradient`` inside the mixed kernel.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev._mdcev_loglik import mdcev_loglik
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel
from pybhatlib.vecup._panel import PanelIndex

NC = 5            # outside + 4 inside goods
NVARM = 4         # baseline-utility parameters
NVARGAM = 4       # translation (gamma) parameters


def _make_data(seed: int):
    """Synthetic cross-sectional MDCEV problem shared by both paths.

    Returns the design tensors consumed by the mixed engine plus the flat
    ``dta`` / ``ivm`` / ``ivg`` arrays consumed by the shipped ``mdcev_loglik``,
    all describing the *same* baseline / satiation design and consumption.
    """
    rng = np.random.default_rng(seed)
    n_obs = 12

    # baseline design tensor (n_obs, nc, nvarm); random covariates.
    X = 0.5 * rng.standard_normal((n_obs, NC, NVARM))
    # satiation design tensor (n_obs, nc, nvargam); outside-good row inert.
    gamma_design = np.zeros((n_obs, NC, NVARGAM), dtype=np.float64)
    gamma_design[:, 1:, :] = rng.uniform(0.3, 1.2, size=(n_obs, NC - 1, NVARGAM))
    gamma_design[:, :, 0] = 0.0                # param 0 loads only outside good

    # consumption: outside good always consumed; inside goods a mixed pattern.
    consumption = np.zeros((n_obs, NC), dtype=np.float64)
    consumption[:, 0] = rng.uniform(1.0, 4.0, size=n_obs)
    for q in range(n_obs):
        # ensure at least one and at most nc-2 inside goods consumed (a mix).
        k_consumed = rng.integers(1, NC - 1)
        picks = rng.choice(np.arange(1, NC), size=k_consumed, replace=False)
        consumption[q, picks] = rng.uniform(0.5, 3.0, size=k_consumed)
    price = np.tile(rng.uniform(0.8, 1.5, size=NC), (n_obs, 1))
    price[:, 0] = 1.0                          # numeraire

    beta = 0.4 * rng.standard_normal(NVARM)
    gamma_raw = np.zeros(NVARGAM, dtype=np.float64)
    gamma_raw[0] = -1000.0                     # outside-good satiation (fixed)
    gamma_raw[1:] = 0.5 * rng.standard_normal(NVARGAM - 1)
    log_sigma = 0.15

    # --- flat arrays for the shipped mdcev_loglik ------------------------- #
    # dta layout: [consumption(nc) | price(nc) | wt(1) | baseline(nc*nvarm) |
    #              gamma(nc*nvargam)]; ivm/ivg param-major, alt-minor.
    base_ivm = 2 * NC + 1
    base_ivg = base_ivm + NC * NVARM
    ncols = base_ivg + NC * NVARGAM
    dta = np.zeros((n_obs, ncols), dtype=np.float64)
    dta[:, 0:NC] = consumption
    dta[:, NC:2 * NC] = price
    dta[:, 2 * NC] = 1.0                        # weight
    ivm = np.zeros(NC * NVARM, dtype=np.intp)
    for j in range(NVARM):
        for k in range(NC):
            c = base_ivm + j * NC + k
            dta[:, c] = X[:, k, j]
            ivm[j * NC + k] = c
    ivg = np.zeros(NC * NVARGAM, dtype=np.intp)
    for j in range(NVARGAM):
        for k in range(NC):
            c = base_ivg + j * NC + k
            dta[:, c] = gamma_design[:, k, j]
            ivg[j * NC + k] = c
    flagchm = np.arange(0, NC, dtype=np.intp)
    flagprcm = np.arange(NC, 2 * NC, dtype=np.intp)
    wtind = 2 * NC

    return SimpleNamespace(
        n_obs=n_obs, X=X, gamma_design=gamma_design, consumption=consumption,
        price=price, beta=beta, gamma_raw=gamma_raw, log_sigma=log_sigma,
        dta=dta, ivm=ivm, ivg=ivg, flagchm=flagchm, flagprcm=flagprcm,
        wtind=wtind,
    )


def _build_engine(p, *, utility: str, n_rep: int):
    """Assemble the mixed engine with an empty (nrndcoef=0) mixing spec."""
    var_names = [f"b{i}" for i in range(NVARM)]
    spec = MixingSpec.from_var_names(var_names=var_names, nvargam=NVARGAM)
    assert spec.nrndcoef == 0

    layout = ParamLayout(
        n_beta=spec.n_beta, n_gamma=NVARGAM, n_rcor=0, n_kern=1,
        n_scal=0, n_lam=0, kern_before_scal=True,
    )
    person_ids = np.arange(p.n_obs)            # cross-sectional
    panel = PanelIndex.from_ids(person_ids)
    obs = SimpleNamespace(
        consumption=p.consumption, price=p.price, gamma_design=p.gamma_design
    )
    design = DesignData(X=p.X, obs=obs)
    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = LogitJacobianKernel(
        NC, NVARGAM, control=MDCEVControl(utility=utility)
    )
    cfg = MSLConfig(
        n_rep=n_rep, floor_pcomp=0.0, floor_z=0.0, score_convention="divide"
    )
    draws = FixtureDrawSource(np.zeros((n_rep, panel.n_ind * spec.nrndcoef)))
    est = MixedMSLEstimator(
        panel=panel, draws=draws, pipeline=pipeline, kernel=kernel,
        layout=layout, space=space, design=design,
        weightind=np.ones(panel.n_ind), config=cfg,
    )
    return est, layout


@pytest.mark.parametrize("utility", ["trad", "linear"])
@pytest.mark.parametrize("n_rep", [1, 4])
def test_collapse_to_fixed_mdcev(utility, n_rep):
    p = _make_data(seed=4040)
    est, layout = _build_engine(p, utility=utility, n_rep=n_rep)

    # theta == shipped x == [beta | gamma_raw | log_sigma]
    theta = np.zeros(layout.n_theta, dtype=np.float64)
    sl = layout.slices()
    theta[sl["beta"]] = p.beta
    theta[sl["gamma"]] = p.gamma_raw
    theta[sl["kern"]] = np.array([p.log_sigma])

    ll_mixed, _ = est.simulated_loglik(theta, want_grad=True)

    x = np.concatenate([p.beta, p.gamma_raw, np.array([p.log_sigma])])
    ll_ref = mdcev_loglik(
        x, p.dta, p.ivm, p.ivg, p.flagchm, p.flagprcm, p.wtind,
        NVARM, NVARGAM, NC, np.eye(NVARGAM),
        MDCEVControl(utility=utility),
    )

    assert np.all(np.isfinite(ll_mixed))
    np.testing.assert_allclose(ll_mixed, ll_ref, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        ll_mixed.sum(), ll_ref.sum(), rtol=0.0, atol=1e-8
    )
