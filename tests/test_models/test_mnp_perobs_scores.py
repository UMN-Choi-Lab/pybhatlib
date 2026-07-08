"""Analytic per-observation BHHH scores for MNP (A3-style fix, ported from MORP).

The BHHH / sandwich standard errors need the per-observation score matrix
``S[q, k] = d log P_q / d theta_k``. Historically MNP built this by central
finite differences = ``2 * n_params`` full-data per-observation log-likelihood
passes (``_mnp_model.py`` BHHH path). ``mnp_per_obs_scores`` computes the same
matrix in a SINGLE analytic pass, reusing the analytic-gradient machinery
(method ``me`` / ``ovus``).

These tests verify the analytic per-obs scores match central finite differences
on the parameterized per-obs log-likelihood (``_per_obs_loglik``), and that they
sum to the full analytic gradient (``sum_q S[q] == -N * grad``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel
from pybhatlib.models.mnp._mnp_grad_analytic import (
    mnp_analytic_gradient,
    mnp_per_obs_scores,
)
from pybhatlib.models.mnp._mnp_loglik import _per_obs_loglik


def _make_data(N, I, n_vars, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, I, n_vars))
    beta_true = rng.standard_normal(n_vars) * 0.5
    V = X @ beta_true
    y = (V + rng.standard_normal((N, I))).argmax(axis=1)
    avail = np.ones((N, I), dtype=np.float64)
    return X, y, avail


def _fd_perobs_scores(theta, X, y, avail, I, n_beta, control, ranvar=None, eps=1e-6):
    n = len(theta)
    N = X.shape[0]
    G = np.zeros((N, n), dtype=np.float64)
    for k in range(n):
        tp = theta.copy(); tp[k] += eps
        tm = theta.copy(); tm[k] -= eps
        lp = _per_obs_loglik(tp, X, y, avail, I, n_beta, control, ranvar)
        lm = _per_obs_loglik(tm, X, y, avail, I, n_beta, control, ranvar)
        G[:, k] = (lp - lm) / (2.0 * eps)
    return G


# (label, I, n_vars, theta, control_kwargs, ranvar_indices)
CASES = [
    ("iid_3alt", 3, 2, np.array([0.3, -0.2]),
     dict(iid=True), None),
    ("heteronly_4alt", 4, 2, np.array([0.3, -0.2, 0.1, 0.2, -0.1]),
     dict(iid=False, heteronly=True), None),
    ("full_3alt", 3, 2, np.array([0.3, -0.2, 0.1, 0.2, 0.3]),
     dict(iid=False, heteronly=False), None),
    ("full_4alt", 4, 3, np.array([0.3, -0.2, 0.1, 0.2, -0.1, 0.0, 0.15, -0.2, 0.1]),
     dict(iid=False, heteronly=False), None),
    ("randdiag_3alt", 3, 2, np.array([0.3, -0.2, -0.5]),
     dict(iid=True, mix=True, randdiag=True), [0]),
    ("randfull_3alt", 3, 2, np.array([0.3, -0.2, 0.5, 0.1, 0.4]),
     dict(iid=True, mix=True, randdiag=False), [0, 1]),
]


@pytest.mark.parametrize("label,I,n_vars,theta,ckw,ranvar", CASES)
def test_analytic_perobs_scores_match_fd(label, I, n_vars, theta, ckw, ranvar):
    N = 60
    X, y, avail = _make_data(N, I, n_vars, seed=hash(label) % 2**31)
    control = MNPControl(method="me", **ckw)
    S = mnp_per_obs_scores(theta, X, y, avail, I, n_vars, control, ranvar)
    G = _fd_perobs_scores(theta, X, y, avail, I, n_vars, control, ranvar)
    assert S.shape == (N, len(theta))
    assert np.allclose(S, G, atol=1e-5, rtol=1e-4), (
        f"{label}: max|Δ| = {np.abs(S - G).max():.2e}"
    )


@pytest.mark.parametrize("label,I,n_vars,theta,ckw,ranvar", CASES)
def test_scores_sum_to_gradient(label, I, n_vars, theta, ckw, ranvar):
    """sum_q d log P_q / d theta == -N * grad(neg mean LL)."""
    N = 60
    X, y, avail = _make_data(N, I, n_vars, seed=hash(label) % 2**31)
    control = MNPControl(method="me", **ckw)
    S = mnp_per_obs_scores(theta, X, y, avail, I, n_vars, control, ranvar)
    nll, grad = mnp_analytic_gradient(theta, X, y, avail, I, n_vars, control, ranvar)
    assert np.allclose(S.sum(axis=0), -N * grad, atol=1e-8), (
        f"{label}: max|Δ| = {np.abs(S.sum(0) + N * grad).max():.2e}"
    )


def test_ovus_method_also_supported():
    N = 50
    X, y, avail = _make_data(N, 3, 2, seed=7)
    theta = np.array([0.3, -0.2, 0.1, 0.2, 0.3])
    control = MNPControl(iid=False, heteronly=False, method="ovus")
    S = mnp_per_obs_scores(theta, X, y, avail, 3, 2, control, None)
    G = _fd_perobs_scores(theta, X, y, avail, 3, 2, control, None)
    assert np.allclose(S, G, atol=1e-4, rtol=1e-3), (
        f"ovus: max|Δ| = {np.abs(S - G).max():.2e}"
    )


# ---------------------------------------------------------------------------
# Integration: reported BHHH SEs are unchanged by the analytic-score switch.
# (MNP analog of MORP's test_bhhh_se_unchanged.)
# ---------------------------------------------------------------------------

_SE_ALTS = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]
_SE_SPEC = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}


def _fit(travelmode_path, iid):
    ctrl = MNPControl(
        iid=iid, heteronly=False, maxiter=200, verbose=0, seed=42,
        se_method="bhhh", method="ovus",
    )
    model = MNPModel(
        data=travelmode_path, alternatives=_SE_ALTS,
        availability="none", spec=_SE_SPEC, control=ctrl,
    )
    return model.fit()


@pytest.mark.slow
@pytest.mark.parametrize("iid", [True, False])
def test_bhhh_se_unchanged_by_analytic_scores(travelmode_path, monkeypatch, iid):
    """Reported BHHH SEs from the analytic per-obs scores match the finite-
    difference path. The fit (analytic gradient) is identical either way; only
    the post-convergence score matrix differs, so SEs must agree to ~1e-5.
    """
    # Analytic-score path (default).
    res_analytic = _fit(travelmode_path, iid)

    # Force the finite-difference fallback by making the analytic scorer raise;
    # _bhhh_scores_unpar catches and falls back to FD on the unparameterized LL.
    import pybhatlib.models.mnp._mnp_grad_analytic as ga

    def _raise(*a, **k):
        raise NotImplementedError("forced FD fallback for test")

    monkeypatch.setattr(ga, "mnp_per_obs_scores", _raise)
    res_fd = _fit(travelmode_path, iid)

    se_a = res_analytic.se
    se_f = res_fd.se

    # Under the GAUSS first-diff-var=1 kernel the flexible model reports a FIXED
    # scale01 row (pinned to 1.0) whose SE is NaN by design. Both the analytic
    # and FD score paths must agree on which entries are fixed (same NaN mask),
    # and the SEs of all *estimated* parameters must match. We compare the
    # finite entries only; the NaN pattern itself is asserted to be identical.
    nan_a = np.isnan(se_a)
    nan_f = np.isnan(se_f)
    assert np.array_equal(nan_a, nan_f), (
        f"iid={iid}: analytic/FD disagree on fixed (NaN-SE) parameters: "
        f"analytic NaN at {np.where(nan_a)[0]}, FD NaN at {np.where(nan_f)[0]}"
    )
    finite = ~nan_a
    assert finite.any(), f"iid={iid}: no finite SEs to compare"
    assert np.all(np.isfinite(se_a[finite])) and np.all(np.isfinite(se_f[finite]))
    assert np.allclose(se_a[finite], se_f[finite], rtol=1e-4, atol=1e-6), (
        f"iid={iid}: max rel Δ = "
        f"{np.max(np.abs(se_a[finite] - se_f[finite]) / np.maximum(np.abs(se_f[finite]), 1e-12)):.2e}"
    )
