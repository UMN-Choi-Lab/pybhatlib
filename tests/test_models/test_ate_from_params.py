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


def _reported_kernel_from_theta(lambda_raw, ctrl):
    """Build the reported differenced kernel (``Sigma_norm``, sum(scale**2)=1)
    and the beta scale factor from theta-space kernel params — mirroring what
    the model reports.  Returns ``(Sigma_norm, sqrt(trace(Sigma_diff)))``."""
    from pybhatlib.models.mnp._mnp_loglik import _build_lambda

    Lam = _build_lambda(lambda_raw, I, ctrl)
    dim = I - 1
    Sig_diff = np.ones((dim, dim)) + Lam + np.eye(dim)
    trace = float(np.trace(Sig_diff))
    return Sig_diff / trace, np.sqrt(trace)


def test_mnp_from_estimates_iid_roundtrip():
    """IID: feeding the *reported* slopes reproduces the fit's shares.

    Reported IID slopes are ``beta_theta / sqrt(2*(I-1))``; ``from_estimates``
    inverts that factor, so the reference model has raw theta betas equal to
    ``BETA * sqrt(2*(I-1))`` while the reported input is ``BETA`` (issue #44).
    """
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    ctrl = MNPControl(iid=True)
    X = np.random.default_rng(0).standard_normal((40, I, N_BETA))
    theta_beta = BETA * np.sqrt(2.0 * (I - 1))
    a_ref = mnp_ate(_mk_mnp_results(theta_beta, ctrl), X=X)
    a_par = mnp_ate_from_params(BETA, control=ctrl, n_alts=I, X=X)
    np.testing.assert_allclose(a_par.predicted_shares, a_ref.predicted_shares, atol=1e-12)


def test_mnp_from_estimates_iid_requires_n_alts():
    from pybhatlib.models.mnp import MNPControl, MNPResults
    with pytest.raises(ValueError, match="n_alts is required"):
        MNPResults.from_estimates(BETA, control=MNPControl(iid=True))


def test_mnp_from_estimates_flexible_roundtrip():
    """Full covariance: reported ``(beta, Sigma_norm)`` reproduce the fit's
    shares via the K-reconstruction (issue #44, correlated kernel)."""
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    ctrl = MNPControl(iid=False)  # full covariance
    lambda_raw = np.array([np.log(1.3), np.log(0.9), 0.7])
    theta_ref = np.concatenate([BETA, lambda_raw])

    X = np.random.default_rng(0).standard_normal((40, I, N_BETA))
    a_ref = mnp_ate(_mk_mnp_results(theta_ref, ctrl), X=X)

    Sig_norm, scale = _reported_kernel_from_theta(lambda_raw, ctrl)
    beta_rep = BETA / scale
    a_par = mnp_ate_from_params(beta_rep, kernel_cov=Sig_norm, control=ctrl,
                                n_alts=I, X=X)
    np.testing.assert_allclose(a_par.predicted_shares, a_ref.predicted_shares, atol=1e-9)


def test_mnp_from_estimates_heteronly_roundtrip():
    """Heteroscedastic-only: reported kernel reproduces the fit's shares
    (reconstructed as full covariance internally)."""
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    ctrl = MNPControl(iid=False, heteronly=True)
    lambda_raw = np.array([np.log(1.3), np.log(0.9)])  # scales only, no corr
    theta_ref = np.concatenate([BETA, lambda_raw])

    X = np.random.default_rng(1).standard_normal((40, I, N_BETA))
    a_ref = mnp_ate(_mk_mnp_results(theta_ref, ctrl), X=X)

    Sig_norm, scale = _reported_kernel_from_theta(lambda_raw, ctrl)
    beta_rep = BETA / scale
    a_par = mnp_ate_from_params(beta_rep, kernel_cov=Sig_norm, control=ctrl,
                                n_alts=I, X=X)
    np.testing.assert_allclose(a_par.predicted_shares, a_ref.predicted_shares, atol=1e-9)


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


# ---------------------------------------------------------------------------
# Issue #44: the real user path — feed a FITTED model's own reported
# coefficients (results.b_original + reported scale*/parker* rows) back through
# mnp_ate_from_params and reproduce mnp_ate(results).  The synthetic tests
# above never exercised a real fit's b_original (which differs from theta by
# the sum-of-squared-scales scale factor), so they masked the scaling bug.
# ---------------------------------------------------------------------------

_MNP_ALT = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]
_MNP_SPEC = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}


def _fit_and_design(travelmode_path, iid):
    from pybhatlib.models.mnp import MNPModel, MNPControl
    from pybhatlib.io._spec_parser import parse_spec

    ctrl = MNPControl(iid=iid, maxiter=200, verbose=0, seed=42)
    model = MNPModel(data=travelmode_path, alternatives=_MNP_ALT,
                     availability="none", spec=_MNP_SPEC, control=ctrl)
    res = model.fit()
    df = pd.read_csv(travelmode_path)
    X, _ = parse_spec(_MNP_SPEC, df, _MNP_ALT, nseg=1)
    return res, np.asarray(X, dtype=float), model.n_beta


@pytest.mark.slow
def test_mnp_ate_from_params_matches_fit_iid(travelmode_path):
    """IID: feeding the fitted model's reported slopes reproduces mnp_ate(res)."""
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    res, X, nb = _fit_and_design(travelmode_path, iid=True)
    n_alts = X.shape[1]
    A_fit = mnp_ate(res, X=X).predicted_shares
    A_par = mnp_ate_from_params(
        res.b_original[:nb], control=MNPControl(iid=True), n_alts=n_alts, X=X,
    ).predicted_shares
    np.testing.assert_allclose(A_par, A_fit, atol=1e-9)


@pytest.mark.slow
def test_mnp_ate_from_params_matches_fit_flexible(travelmode_path):
    """Correlated kernel: reported betas + scale*/parker* rows reproduce
    mnp_ate(res).  This is the exact case in issue #44."""
    from pybhatlib.models.mnp import MNPControl, mnp_ate, mnp_ate_from_params

    res, X, nb = _fit_and_design(travelmode_path, iid=False)
    n_alts = X.shape[1]
    dim = n_alts - 1
    coef = dict(zip(res.param_names, res.b_original))

    # Rebuild the reported differenced covariance from the reported rows,
    # exactly as a user copying the output would.
    wker = np.array([coef[f"scale{i + 1:02d}"] for i in range(dim)])
    corr = np.eye(dim)
    k = 0
    for i in range(dim):
        for j in range(i + 1, dim):
            k += 1
            corr[i, j] = corr[j, i] = coef[f"parker{k:02d}"]
    Sig_norm = np.diag(wker) @ corr @ np.diag(wker)

    A_fit = mnp_ate(res, X=X).predicted_shares
    A_par = mnp_ate_from_params(
        res.b_original[:nb], kernel_cov=Sig_norm, control=MNPControl(iid=False),
        n_alts=n_alts, X=X,
    ).predicted_shares
    np.testing.assert_allclose(A_par, A_fit, atol=1e-9)
