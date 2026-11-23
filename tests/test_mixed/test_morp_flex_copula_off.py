"""Regression gates for the mixed MORP facade with the rc<->kernel copula OFF.

Background: with ``copula=False`` (GAUSS ``_nocorrrcker = 1``) the MORP kernel
used to emit no joint-correlation gradient, and the shared engine then dropped
the *entire* ``rcor`` block from the analytic score. The likelihood still
depended on the random-coefficient correlation (through the RC Cholesky) and
on the ordinal-error correlation (through the unconditional ordinal block), so
a gradient-based fit could never move those correlations off their starting
values -- they were reported as exactly ``0.0`` with ``SE = 0.0`` while the fit
still declared convergence. The GAUSS oracle estimates both.

These tests go through the public facade (``MORPFlexModel``), so they also
cover the spec/layout wiring (``MixingSpec.active_corr_pairs``) that tells the
engine which joint-correlation slots are estimated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource
from pybhatlib.models.morp_flex import MORPFlexControl, MORPFlexModel
from pybhatlib.vecup._panel import PanelIndex

_TOL = 1e-4
_DEP = ["y1", "y2"]
_NCAT = [3, 2]
_SPEC = {
    "C1": {"y1": "uno", "y2": "sero"},
    "C2": {"y1": "sero", "y2": "uno"},
    "B1": {"y1": "x1", "y2": "sero"},
    "B2": {"y1": "sero", "y2": "x2"},
}


def _panel(seed: int, n_ind: int, n_per: int, *, corr_rc: float, corr_ord: float) -> pd.DataFrame:
    """Two ordinal outcomes; correlated random intercepts C1/C2 and correlated
    ordinal errors, so both estimable correlations are load-bearing."""
    rng = np.random.default_rng(seed)
    n = n_ind * n_per
    pid = np.repeat(np.arange(n_ind), n_per)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    L_rc = np.linalg.cholesky([[1.0, corr_rc], [corr_rc, 1.0]])
    ri = (rng.normal(size=(n_ind, 2)) @ L_rc.T)[pid]
    L_k = np.linalg.cholesky([[1.0, corr_ord], [corr_ord, 1.0]])
    ek = rng.normal(size=(n, 2)) @ L_k.T
    u1 = 0.6 * x1 + ri[:, 0] + ek[:, 0]
    u2 = -0.5 * x2 + ri[:, 1] + ek[:, 1]
    y1 = np.digitize(u1, [-0.7, 0.6])            # 3 categories
    y2 = (u2 > -0.2).astype(int)                 # 2 categories
    return pd.DataFrame({
        "pid": pid, "x1": x1, "x2": x2, "y1": y1, "y2": y2,
        "uno": 1.0, "sero": 0.0,
    })


def _fd_grad(est, theta, *, eps=1e-6):
    g = np.zeros_like(theta)
    for j in range(theta.shape[0]):
        tp = theta.copy(); tp[j] += eps
        tm = theta.copy(); tm[j] -= eps
        lp, _ = est.simulated_loglik(tp, want_grad=False)
        lm, _ = est.simulated_loglik(tm, want_grad=False)
        g[j] = (lp.sum() - lm.sum()) / (2.0 * eps)
    return g


@pytest.mark.parametrize("yj_kernel", [False, True])
def test_copula_off_facade_score_matches_fd(yj_kernel):
    """Facade-built estimator, copula OFF: analytic score == central FD on every
    estimated slot; the rc<->kernel slots carry exactly zero score."""
    df = _panel(11, n_ind=12, n_per=2, corr_rc=0.6, corr_ord=0.4)
    ctrl = MORPFlexControl(
        person_id="pid", normvar=("C1", "C2"), copula=False, yj_kernel=yj_kernel,
        n_rep=3, floor_pcomp=1e-4, floor_z=1e-4, verbose=0,
    )
    model = MORPFlexModel(data=df, dep_vars=_DEP, spec=_SPEC, n_categories=_NCAT, control=ctrl)
    spec, layout = model._build_spec_layout()
    assert spec.copula is False
    assert set(spec.active_corr_pairs) == {(0, 1), (2, 3)}      # rc-rc, ord-ord
    assert layout.n_rcor == 6                                   # full joint block
    panel = PanelIndex.from_ids(model.person_ids)
    rng = np.random.default_rng(5)
    draws = FixtureDrawSource(rng.normal(size=(ctrl.n_rep, panel.n_ind * spec.nrndcoef)))
    est = model._build_estimator(spec, layout, panel, draws=draws)

    sl = layout.slices()
    theta = np.zeros(layout.n_theta)
    theta[sl["thresh"]] = rng.normal(scale=0.2, size=sl["thresh"].stop - sl["thresh"].start)
    theta[sl["beta"]] = rng.normal(scale=0.3, size=layout.n_beta)
    theta[sl["rcor"].start + 0] = 0.7          # rc-rc
    theta[sl["rcor"].start + 5] = -0.4         # ord-ord
    theta[sl["scal"]] = rng.normal(scale=0.2, size=layout.n_scal)
    kl = sl.get("kernlam")                 # absent for the normal kernel
    if kl is not None and kl.stop > kl.start:
        theta[kl] = rng.normal(scale=0.3, size=kl.stop - kl.start)

    _, score = est.simulated_loglik(theta, want_grad=True)
    g_an = np.asarray(score).sum(0)
    g_fd = _fd_grad(est, theta)

    rc0 = sl["rcor"].start
    active = [rc0 + 0, rc0 + 5]
    masked = [rc0 + 1, rc0 + 2, rc0 + 3, rc0 + 4]
    assert np.all(g_an[masked] == 0.0)
    assert float(np.min(np.abs(g_fd[active]))) > 1e-4, "correlation slots not load-bearing"
    assert np.allclose(g_an[active], g_fd[active], atol=_TOL, rtol=_TOL), (
        f"rcor active: analytic {g_an[active]} vs FD {g_fd[active]}"
    )
    others = [j for j in range(layout.n_theta) if j not in set(active) | set(masked)]
    assert np.allclose(g_an[others], g_fd[others], atol=_TOL, rtol=_TOL)


def test_copula_off_fit_estimates_correlations():
    """``fit(copula=False)`` must move both estimable correlations off zero and
    report positive SEs for them, while the rc<->kernel slots stay at 0/0."""
    df = _panel(3, n_ind=150, n_per=2, corr_rc=0.7, corr_ord=0.5)
    ctrl = MORPFlexControl(
        person_id="pid", normvar=("C1", "C2"), copula=False, yj_kernel=False,
        n_rep=4, draw_seed=1, maxiter=60, verbose=0,
    )
    model = MORPFlexModel(data=df, dep_vars=_DEP, spec=_SPEC, n_categories=_NCAT, control=ctrl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit()
    est = dict(zip(res.param_names, res.params))
    se = dict(zip(res.param_names, res.se))
    assert abs(est["corr[C1,C2]"]) > 0.05 and se["corr[C1,C2]"] > 0.0
    assert abs(est["corr[ord1,ord2]"]) > 0.05 and se["corr[ord1,ord2]"] > 0.0
    for name in ("corr[C1,ord1]", "corr[C1,ord2]", "corr[C2,ord1]", "corr[C2,ord2]"):
        # held at zero; the SE is zero up to Jacobian-projection round-off
        assert est[name] == 0.0 and se[name] < 1e-8
