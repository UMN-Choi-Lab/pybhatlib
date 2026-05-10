"""Tests for MORP analytic gradient (MORP-002).

Covers the analytic gradient implementation in
``pybhatlib.models.morp._morp_grad_analytic`` against central finite
differences, plus the integration with ``morp_loglik`` (default fit path,
fallback for unsupported MVNCD methods).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from pybhatlib.backend import get_backend
from pybhatlib.models.morp import MORPControl, MORPModel
from pybhatlib.models.morp._morp_loglik import (
    count_morp_params,
    morp_loglik,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _central_fd_grad(
    theta: np.ndarray,
    fn,
    eps: float = 1e-6,
) -> np.ndarray:
    """Central-difference gradient of a scalar-valued function."""
    g = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps
        tm[i] -= eps
        g[i] = (fn(tp) - fn(tm)) / (2.0 * eps)
    return g


def _synth_morp_data(
    n: int = 30,
    n_dims: int = 2,
    n_categories: tuple[int, ...] = (3, 3),
    n_beta: int = 2,
    seed: int = 17,
):
    """Build a tiny synthetic (X, y) dataset for MORP tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_dims, n_beta))
    # Outcomes drawn at random within each dim's category count.
    y = np.zeros((n, n_dims), dtype=np.int64)
    for d in range(n_dims):
        y[:, d] = rng.integers(0, n_categories[d], size=n)
    return X, y


def _make_theta(
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
    seed: int = 0,
    perturb_scale: float = 0.0,
) -> np.ndarray:
    """Build a feasible parameter vector with strictly increasing thresholds."""
    rng = np.random.default_rng(seed)
    n = count_morp_params(n_beta, n_dims, n_categories, control)
    theta = np.zeros(n, dtype=np.float64)

    idx = 0
    # Betas
    theta[idx: idx + n_beta] = rng.standard_normal(n_beta) * 0.3
    idx += n_beta

    # Thresholds: first is raw, subsequent are log-spacings (exp -> positive).
    for d in range(n_dims):
        n_thresh = n_categories[d] - 1
        if n_thresh <= 0:
            continue
        theta[idx] = -0.4
        idx += 1
        for _ in range(1, n_thresh):
            theta[idx] = np.log(0.6)  # spacing ~0.6
            idx += 1

    # Covariance params (scales + correlation)
    if not control.iid:
        if control.heteronly:
            for _ in range(n_dims - 1):
                theta[idx] = 0.1
                idx += 1
        else:
            if not getattr(control, "fix_scales", False):
                for _ in range(n_dims - 1):
                    theta[idx] = 0.1
                    idx += 1
            n_corr = n_dims * (n_dims - 1) // 2
            for k in range(n_corr):
                theta[idx + k] = 0.0  # corr ~ 0 (mid spherical/tanh)
            idx += n_corr

    if perturb_scale > 0:
        theta = theta + rng.standard_normal(len(theta)) * perturb_scale

    return theta


# ---------------------------------------------------------------------------
# (1) Synthetic IID: beta + threshold gradient blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_analytic_grad_matches_fd_synthetic_iid(seed):
    """Analytic gradient must match central FD for an IID MORP at multiple thetas."""
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 2
    n_categories = [3, 3]
    X, y = _synth_morp_data(
        n=30, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=11,
    )

    ctrl = MORPControl(
        iid=True, method="ovus", analytic_grad=True, verbose=0,
    )

    perturb = [0.0, 0.05, 0.2][seed]
    theta = _make_theta(
        n_beta, n_dims, n_categories, ctrl,
        seed=100 + seed, perturb_scale=perturb,
    )

    nll_a, grad_a = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )

    def fn(t):
        return morp_loglik(
            t, X, y, n_dims, n_categories, n_beta, ctrl, xp=xp,
        )

    grad_fd = _central_fd_grad(theta, fn, eps=1e-6)

    np.testing.assert_allclose(grad_a, grad_fd, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# (2) Synthetic flexible (correlation + scale gradient blocks)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_analytic_grad_matches_fd_synthetic_flexible(seed):
    """Analytic gradient must match FD when correlation + scale params are active."""
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 2
    n_categories = [3, 3]
    X, y = _synth_morp_data(
        n=30, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=22,
    )

    ctrl = MORPControl(
        iid=False, method="ovus", spherical=True, analytic_grad=True, verbose=0,
    )
    perturb = [0.0, 0.1][seed]
    theta = _make_theta(
        n_beta, n_dims, n_categories, ctrl,
        seed=200 + seed, perturb_scale=perturb,
    )

    nll_a, grad_a = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )

    def fn(t):
        return morp_loglik(
            t, X, y, n_dims, n_categories, n_beta, ctrl, xp=xp,
        )

    grad_fd = _central_fd_grad(theta, fn, eps=1e-6)

    np.testing.assert_allclose(grad_a, grad_fd, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# (2a) fix_scales=True (MORP-104) — analytic grad on the unit-variance layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spherical", [False, True])
def test_analytic_grad_matches_fd_synthetic_fix_scales(spherical):
    """Analytic gradient must match FD when scales are locked at 1.

    Under ``fix_scales=True`` the parameter vector drops the K-1 log-scale
    slots, leaving only ``[beta | thresholds | corr_theta]``. This test
    pins the analytic Jacobian against central FD for both spherical and
    tanh correlation parameterizations. ``spherical=False`` matches the
    GAUSS BHATLIB MORP_WALK driver (``_Spher = 0``).
    """
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 2
    n_categories = [3, 3]
    X, y = _synth_morp_data(
        n=30, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=44,
    )

    ctrl = MORPControl(
        iid=False, method="ovus", spherical=spherical, fix_scales=True,
        analytic_grad=True, verbose=0,
    )
    theta = _make_theta(
        n_beta, n_dims, n_categories, ctrl, seed=400, perturb_scale=0.05,
    )

    # Sanity: count_morp_params returned the fix_scales-aware length and
    # _make_theta wrote exactly that many slots. Catches a layout drift
    # before the gradient comparison reports a mysterious failure.
    assert len(theta) == count_morp_params(
        n_beta, n_dims, n_categories, ctrl,
    )

    nll_a, grad_a = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )

    def fn(t):
        return morp_loglik(
            t, X, y, n_dims, n_categories, n_beta, ctrl, xp=xp,
        )

    grad_fd = _central_fd_grad(theta, fn, eps=1e-6)
    np.testing.assert_allclose(grad_a, grad_fd, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# (2b) K=3 synthetic — exercises the +inf-collapse path and inclusion-exclusion
# fan-out across multiple boundary categories (PR #8 review P1).
# ---------------------------------------------------------------------------


def test_analytic_grad_matches_fd_synthetic_k3_mixed_boundaries():
    """K=3 synthetic FD-vs-analytic with mixed boundary categories.

    Pre-fix, the K=3 branch of ``_rect_prob_and_grad`` (where the +inf
    collapse runs across multiple dims and ``_collapsed_vertex_grad`` is
    actually exercised) was only verified by the slow DINING test.  This
    fast test triggers the same code paths with synthetic data so a
    regression is caught in CI.
    """
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 3
    # Mixed J_d ensures some y_d hit the bottom category (lower=-inf) and
    # some hit the top (upper=+inf), so both boundary branches run.
    n_categories = [3, 4, 3]
    X, y = _synth_morp_data(
        n=24, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=33,
    )

    ctrl = MORPControl(
        iid=False, method="ovus", spherical=True, analytic_grad=True, verbose=0,
    )
    theta = _make_theta(
        n_beta, n_dims, n_categories, ctrl, seed=303, perturb_scale=0.05,
    )

    _, grad_a = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )

    def fn(t):
        return morp_loglik(
            t, X, y, n_dims, n_categories, n_beta, ctrl, xp=xp,
        )

    grad_fd = _central_fd_grad(theta, fn, eps=1e-6)
    np.testing.assert_allclose(grad_a, grad_fd, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# (3) Threshold-block focused regression
# ---------------------------------------------------------------------------


def test_analytic_grad_matches_fd_threshold_block():
    """Perturb only threshold params; remaining slots untouched.

    Verifies the threshold gradient slice (which encodes the cumulative
    log-spacing chain) agrees with FD on those exact slots.
    """
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 2
    n_categories = [3, 4]  # asymmetric categories
    X, y = _synth_morp_data(
        n=30, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=31,
    )
    ctrl = MORPControl(
        iid=True, method="ovus", analytic_grad=True, verbose=0,
    )
    theta = _make_theta(
        n_beta, n_dims, n_categories, ctrl, seed=300, perturb_scale=0.0,
    )

    # Threshold slot range
    thr_start = n_beta
    thr_end = thr_start + sum(c - 1 for c in n_categories)

    nll_a, grad_a = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )

    def fn(t):
        return morp_loglik(
            t, X, y, n_dims, n_categories, n_beta, ctrl, xp=xp,
        )

    grad_fd = _central_fd_grad(theta, fn, eps=1e-6)

    np.testing.assert_allclose(
        grad_a[thr_start:thr_end],
        grad_fd[thr_start:thr_end],
        atol=1e-5, rtol=0,
    )


# ---------------------------------------------------------------------------
# (4) End-to-end fit: analytic-default converges to a reasonable optimum
# ---------------------------------------------------------------------------


def test_morp_fit_with_analytic_default_converges():
    """Fit a synthetic dataset with default (analytic) gradient and check
    the recovered parameters are close to the simulation truth."""
    rng = np.random.default_rng(123)
    n = 400
    beta_true = np.array([0.6, -0.4])
    tau_true = [np.array([-0.5, 0.5]), np.array([-0.3, 0.6])]

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    Xv = np.column_stack([x1, x2])
    eps = rng.standard_normal((n, 2))  # iid
    Y_star = np.column_stack([Xv @ beta_true + eps[:, 0], Xv @ beta_true + eps[:, 1]])
    y1 = np.digitize(Y_star[:, 0], tau_true[0])
    y2 = np.digitize(Y_star[:, 1], tau_true[1])

    df = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    spec = {
        "x1": {"y1": "x1", "y2": "x1"},
        "x2": {"y1": "x2", "y2": "x2"},
    }
    model = MORPModel(
        data=df,
        dep_vars=["y1", "y2"],
        spec=spec,
        n_categories=[3, 3],
        control=MORPControl(
            iid=True, method="ovus", analytic_grad=True,
            verbose=0, seed=42, maxiter=100,
        ),
    )
    res = model.fit()
    assert res.converged, "Fit did not converge"
    # First two params are the betas
    np.testing.assert_allclose(res.params[:2], beta_true, atol=0.15)


# ---------------------------------------------------------------------------
# (5) Fallback path when MVNCD method is unsupported by analytic grad
# ---------------------------------------------------------------------------


def test_analytic_grad_fallback_when_method_unsupported():
    """``analytic_grad=True`` with method='scipy' must silently fall back
    to numerical FD (no error) and return a finite gradient."""
    xp = get_backend("numpy")
    n_beta = 2
    n_dims = 2
    n_categories = [3, 3]
    X, y = _synth_morp_data(
        n=20, n_dims=n_dims, n_categories=tuple(n_categories), n_beta=n_beta,
        seed=51,
    )
    ctrl = MORPControl(
        iid=True, method="scipy", analytic_grad=True, verbose=0,
    )
    theta = _make_theta(n_beta, n_dims, n_categories, ctrl, seed=500)

    nll, grad = morp_loglik(
        theta, X, y, n_dims, n_categories, n_beta, ctrl,
        return_gradient=True, xp=xp,
    )
    assert np.isfinite(nll)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))


# ---------------------------------------------------------------------------
# (6) DINING dataset gradient check (slow)
# ---------------------------------------------------------------------------


_DINING_CSV = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "Gauss Files and Comparison", "MORP", "Example_Dining.csv",
)
_DINING_CSV = os.path.abspath(_DINING_CSV)


def _dining_spec_and_args():
    spec = {
        "E_rest20": {"NeatoutO": "resta20", "Npickupo": "sero", "Ndelivo": "sero"},
        "E_in150":  {"NeatoutO": "in150",   "Npickupo": "sero", "Ndelivo": "sero"},
        "P_rest20": {"NeatoutO": "sero",    "Npickupo": "resta20", "Ndelivo": "sero"},
        "P_urb":    {"NeatoutO": "sero",    "Npickupo": "urb",  "Ndelivo": "sero"},
        "D_wrk_h":  {"NeatoutO": "sero",    "Npickupo": "sero", "Ndelivo": "wrk_H"},
        "D_urb":    {"NeatoutO": "sero",    "Npickupo": "sero", "Ndelivo": "urb"},
    }
    return spec


@pytest.mark.slow
def test_morp_dining_analytic_matches_fd():
    """Analytic gradient matches central FD on the DINING dataset start point."""
    if not os.path.exists(_DINING_CSV):
        pytest.skip(f"DINING CSV missing at {_DINING_CSV}")

    xp = get_backend("numpy")
    df = pd.read_csv(_DINING_CSV)
    spec = _dining_spec_and_args()
    n_categories = [11, 7, 7]

    ctrl_a = MORPControl(
        iid=False, method="ovus", analytic_grad=True, verbose=0,
    )
    model = MORPModel(
        data=df,
        dep_vars=["NeatoutO", "Npickupo", "Ndelivo"],
        spec=spec,
        n_categories=n_categories,
        control=ctrl_a,
    )

    theta0 = model._default_start_values()

    nll_a, grad_a = morp_loglik(
        theta0, model.X, model.y, model.n_dims, model.n_categories,
        model.n_beta, ctrl_a, return_gradient=True, xp=xp,
    )

    # Use the same control with analytic_grad=False for the FD reference,
    # but evaluate the forward-only loglik (which doesn't dispatch through
    # the analytic path) for the FD computation.
    def fn(t):
        return morp_loglik(
            t, model.X, model.y, model.n_dims, model.n_categories,
            model.n_beta, ctrl_a, xp=xp,
        )

    grad_fd = _central_fd_grad(theta0, fn, eps=1e-5)

    # DINING uses the OVUS approximation which carries some numerical noise
    # at K=3; loosen the tolerance from 1e-5 used on the tiny synthetic case.
    np.testing.assert_allclose(grad_a, grad_fd, atol=1e-3, rtol=0)


@pytest.mark.slow
def test_morp_dining_fit_runtime_improves():
    """Analytic-gradient fit on DINING should be faster than FD-gradient fit."""
    if not os.path.exists(_DINING_CSV):
        pytest.skip(f"DINING CSV missing at {_DINING_CSV}")

    df = pd.read_csv(_DINING_CSV)
    spec = _dining_spec_and_args()
    n_categories = [11, 7, 7]

    def _build(analytic: bool) -> MORPModel:
        ctrl = MORPControl(
            iid=False, method="ovus", analytic_grad=analytic,
            verbose=0, seed=42, maxiter=15,  # cap to keep test fast
        )
        return MORPModel(
            data=df,
            dep_vars=["NeatoutO", "Npickupo", "Ndelivo"],
            spec=spec,
            n_categories=n_categories,
            control=ctrl,
        )

    t0 = time.time()
    _build(analytic=True).fit()
    t_analytic = time.time() - t0

    t0 = time.time()
    _build(analytic=False).fit()
    t_fd = time.time() - t0

    # The PR's empirical claim is a ~20x speedup on DINING; tighten the
    # assertion so a regression that loses 90% of the gain still trips.
    # Headroom: factor 2 (instead of 20) is conservative — a noisy slow
    # run still passes, but a flat or actually-slower analytic path would
    # fail loudly (PR #8 review P1).
    assert t_analytic < 0.5 * t_fd, (
        f"Analytic ({t_analytic:.2f}s) lost the FD speedup; "
        f"expected at least 2x faster, got {t_fd / max(t_analytic, 1e-9):.2f}x"
    )
