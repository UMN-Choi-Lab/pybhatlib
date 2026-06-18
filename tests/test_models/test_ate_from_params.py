"""Tests for the ``*_ate_from_params`` / ``Results.from_estimates`` interface.

Issue #34: mirror ``morp_ate_from_params`` across the other models so ATEs can
be computed from manually entered (e.g. GAUSS) coefficient estimates without
re-fitting.  These tests verify the round-trip property — feeding a model's own
reported coefficients back through ``from_estimates`` reproduces the ATE
computed from the fitted results object — plus the export/naming parity fixes.

MNL is not covered here: it lives on a separate (unmerged) branch; its
equivalent stacks on that PR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# MDCEV
# ---------------------------------------------------------------------------

@pytest.fixture
def mdcev_fit():
    """Fast synthetic MDCEV fit + correctly shaped design arrays."""
    from pybhatlib.models.mdcev import MDCEVModel, MDCEVControl

    rng = np.random.default_rng(123)
    n = 50
    df = pd.DataFrame({
        "ID": np.arange(n),
        "x1": rng.standard_normal(n), "x2": rng.standard_normal(n),
        "x3": rng.standard_normal(n),
        "alt_out": rng.uniform(0, 2, n),
        "alt1": rng.uniform(0, 2, n), "alt2": rng.uniform(0, 2, n),
    })
    alts = ["alt_out", "alt1", "alt2"]
    uspec = {
        "ASC_alt1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
        "ASC_alt2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
        "x": {"alt_out": "x1", "alt1": "x2", "alt2": "x3"},
    }
    gspec = {
        "g_out": {"alt_out": "uno", "alt1": "sero", "alt2": "sero"},
        "g1": {"alt_out": "sero", "alt1": "uno", "alt2": "sero"},
        "g2": {"alt_out": "sero", "alt1": "sero", "alt2": "uno"},
    }
    model = MDCEVModel(data=df, alternatives=alts, utility_spec=uspec,
                       gamma_spec=gspec, control=MDCEVControl(maxiter=8, verbose=0))
    res = model.fit()
    nc, nvarm, nvargam = 3, model.utility_spec.shape[1], model.gamma_spec.shape[1]
    N = 20
    X = rng.standard_normal((N, nc, nvarm))
    Xg = rng.standard_normal((N, nc, nvargam))
    price = np.abs(rng.standard_normal((N, nc))) + 0.5
    return res, X, Xg, price


def test_mdcev_predict_import_is_fixed():
    """Regression: mdcev_predict imported simtradmdcev from a nonexistent module."""
    from pybhatlib.models.mdcev import mdcev_predict, mdcev_predict_choice
    assert callable(mdcev_predict) and callable(mdcev_predict_choice)


def test_mdcev_from_estimates_roundtrip(mdcev_fit):
    from pybhatlib.models.mdcev import mdcev_ate, mdcev_ate_from_params

    res, X, Xg, price = mdcev_fit
    a_fit = mdcev_ate(res, X, Xg, price, n_draws=300, seed=7)
    a_par = mdcev_ate_from_params(
        res.b_reported, res.sigma, X, Xg, price,
        control=res.control, param_names=res.param_names, n_draws=300, seed=7,
    )
    # Same coefficients + same seed -> identical Monte Carlo draws -> exact match.
    np.testing.assert_allclose(a_par.predicted_shares, a_fit.predicted_shares, atol=1e-12)


def test_mdcev_from_estimates_counterfactual(mdcev_fit):
    from pybhatlib.models.mdcev import mdcev_ate, mdcev_ate_from_params

    res, X, Xg, price = mdcev_fit
    kw = dict(changevar_idx=(1, 2), base_val=0.0, treatment_val=1.0, n_draws=300, seed=7)
    a_fit = mdcev_ate(res, X, Xg, price, **kw)
    a_par = mdcev_ate_from_params(res.b_reported, res.sigma, X, Xg, price,
                                  control=res.control, **kw)
    np.testing.assert_allclose(a_par.pct_ate, a_fit.pct_ate, atol=1e-9, equal_nan=True)


def test_mdcev_from_estimates_sigma_default(mdcev_fit):
    """sigma defaults to b_reported[-1]; SEs are NaN for fixed inputs."""
    from pybhatlib.models.mdcev import MDCEVResults

    res, *_ = mdcev_fit
    r = MDCEVResults.from_estimates(res.b_reported, control=res.control)
    assert r.sigma == pytest.approx(res.b_reported[-1])
    assert np.all(np.isnan(r.se))


# ---------------------------------------------------------------------------
# MNP
# ---------------------------------------------------------------------------

I, N_BETA = 3, 4
BETA = np.array([0.5, -0.3, 0.8, -1.2])


def _mk_mnp_results(theta, control):
    """Construct an MNPResults directly from a raw theta (no fitting)."""
    from pybhatlib.models.mnp import MNPResults

    n = theta.shape[0]
    nan, nanb = np.full(n, np.nan), np.full(N_BETA, np.nan)
    return MNPResults(
        params=theta, b_original=theta[:N_BETA], se=nanb, t_stat=nanb, p_value=nanb,
        gradient=nan, loglik=float("nan"), n_obs=0,
        param_names=[f"b{i}" for i in range(N_BETA)],
        corr_matrix=np.full((n, n), np.nan), cov_matrix=np.full((n, n), np.nan),
        n_iter=0, convergence_time=float("nan"), converged=True,
        return_code=0, control=control,
    )


def test_mnp_atresult_alias():
    from pybhatlib.models.mnp import ATEResult, MNPATEResult
    assert ATEResult is MNPATEResult


def test_mnp_from_estimates_iid_roundtrip():
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    ctrl = MNPControl(iid=True)
    X = np.random.default_rng(0).standard_normal((40, I, N_BETA))
    a_ref = mnp_ate(_mk_mnp_results(BETA.copy(), ctrl), X=X)
    a_par = mnp_ate_from_params(BETA, control=ctrl, n_alts=I, X=X)
    np.testing.assert_allclose(a_par.predicted_shares, a_ref.predicted_shares, atol=1e-12)


def test_mnp_from_estimates_flexible_roundtrip():
    from pybhatlib.models.mnp import MNPControl, MNPResults, mnp_ate, mnp_ate_from_params
    from pybhatlib.models.mnp._mnp_loglik import _build_lambda

    ctrl = MNPControl(iid=False)  # full covariance
    lambda_raw = np.array([np.log(1.3), np.log(0.9), 0.7])
    theta_ref = np.concatenate([BETA, lambda_raw])
    Lambda = _build_lambda(theta_ref[N_BETA:], I, ctrl)

    # from_estimates encodes the natural covariance back to raw theta exactly.
    res_fe = MNPResults.from_estimates(BETA, kernel_cov=Lambda, control=ctrl, n_alts=I)
    Lambda_fe = _build_lambda(res_fe.params[N_BETA:], I, ctrl)
    np.testing.assert_allclose(Lambda_fe, Lambda, atol=1e-10)
    np.testing.assert_allclose(res_fe.params[:N_BETA], BETA, atol=1e-14)

    X = np.random.default_rng(0).standard_normal((40, I, N_BETA))
    a_ref = mnp_ate(_mk_mnp_results(theta_ref, ctrl), X=X)
    a_par = mnp_ate_from_params(BETA, kernel_cov=Lambda, control=ctrl, n_alts=I, X=X)
    np.testing.assert_allclose(a_par.predicted_shares, a_ref.predicted_shares, atol=1e-9)


def test_mnp_from_estimates_heteronly_roundtrip():
    from pybhatlib.models.mnp import MNPControl, MNPResults
    from pybhatlib.models.mnp._mnp_loglik import _build_lambda

    ctrl = MNPControl(iid=False, heteronly=True)
    Lambda = np.diag([1.3 ** 2, 0.9 ** 2])
    res = MNPResults.from_estimates(np.array([0.1, 0.2]), kernel_cov=Lambda,
                                    control=ctrl, n_alts=I)
    np.testing.assert_allclose(_build_lambda(res.params[2:], I, ctrl), Lambda, atol=1e-10)


def test_mnp_from_estimates_requires_kernel_cov_for_noniid():
    from pybhatlib.models.mnp import MNPControl, MNPResults
    with pytest.raises(ValueError, match="kernel_cov is required"):
        MNPResults.from_estimates(BETA, control=MNPControl(iid=False))


def test_mnp_from_estimates_rejects_mixture_and_randcoef():
    from pybhatlib.models.mnp import MNPControl, MNPResults
    Lambda = np.eye(2)
    with pytest.raises(NotImplementedError):
        MNPResults.from_estimates(BETA, kernel_cov=Lambda,
                                  control=MNPControl(iid=False, mix=True), n_alts=I)
    with pytest.raises(NotImplementedError):
        MNPResults.from_estimates(BETA, kernel_cov=Lambda,
                                  control=MNPControl(iid=False, nseg=2), n_alts=I)
