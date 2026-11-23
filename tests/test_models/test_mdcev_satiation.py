"""Independent Jacobian and score checks for the shared MDCEV satiation block."""

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.models.mdcev import MDCEVControl
from pybhatlib.models.mdcev._mdcev_loglik import (
    _compute_satiation_block, mdcev_gradient, mdcev_loglik,
)
from pybhatlib.models.mdcev._mdcev_satiation import _log_satiation_terms
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel


def _case(full_layout, outside_gamma):
    # Mixed, all-inside, and outside-only observations; non-unit prices/weights.
    qty = np.array([[2., 1., 0., 3.], [1., 2., 3., 4.], [3., 0., 0., 0.]])
    price = np.array([[1.3, .8, 2., 1.1], [.7, 2., 1.1, .9], [2., 1., .8, 3.]])
    weights = np.array([.5, 2., 0.])
    rng = np.random.default_rng(430)
    design = rng.uniform(.1, 1.5, (3, 4, 2))
    # The outside row deliberately loads gamma: both adapters must override it.
    eq = np.array([[1., .3], [-.2, .8]])
    theta = np.array([.4, -.3, .6, np.log(.9)])  # beta, two gamma, log sigma
    baseline = rng.normal(size=(3, 4))
    gamma_columns = design if full_layout else design[:, 1:, :]
    flat_gamma = gamma_columns.transpose(0, 2, 1).reshape(3, -1)
    dta = np.column_stack([qty, price, weights, baseline, flat_gamma])
    ivg = np.arange(13, dta.shape[1])
    args = (dta, np.arange(9, 13), ivg, np.arange(4), np.arange(4, 8),
            8, 1, 2, 4, eq)
    u = np.einsum("qkj,j->qk", design, eq.T @ theta[1:3])
    u[:, 0] = outside_gamma
    return qty, price, design, theta, eq, args, u


@pytest.mark.parametrize("utility", ["trad", "linear"])
@pytest.mark.parametrize("full_layout", [False, True])
@pytest.mark.parametrize("outside_gamma", [-1000., -1.])
def test_satiation_preserves_legacy_jacobian_and_weighted_gradient(
    utility, full_layout, outside_gamma,
):
    qty, price, design, theta, eq, args, u = _case(full_layout, outside_gamma)
    ctrl = MDCEVControl(utility=utility, outside_good_gamma=outside_gamma)
    dta, ivm, ivg, flagchm, flagprcm, wtind, nvarm, ng, nc, _ = args
    fixed = _compute_satiation_block(
        theta, dta, ivg, flagchm, flagprcm, wtind, nvarm, ng, nc, eq, ctrl,
    )
    kernel = LogitJacobianKernel(nc, ng, control=ctrl, eqmatgam=eq)
    obs = SimpleNamespace(consumption=qty, price=price, gamma_design=design)
    state = SimpleNamespace(gamma_raw=theta[1:3])
    mixed = kernel._get_sat_all(state, obs)
    assert kernel._get_sat_all(state, obs) is mixed  # cached across draws

    # Original probability-space Jacobian, independent of the shared helper.
    totals = qty + np.exp(u)
    consumed = qty > 0
    consumed[:, 0] = True
    if utility == "trad":
        product = np.where(consumed, 1 / totals, 1).prod(axis=1)
        expected_c = product * np.where(consumed, price * totals, 0).sum(axis=1)
        expected_c /= price[:, 0]
        np.testing.assert_allclose(fixed["c2"], product / price[:, 0], rtol=2e-14)
        np.testing.assert_allclose(fixed["c3"], 1 / totals[:, 1:], rtol=2e-14)
    else:
        expected_c = np.where(
            consumed[:, 1:], 1 / (totals[:, 1:] * price[:, 1:]), 1,
        ).prod(axis=1)
        assert fixed["c2"] is None and fixed["c3"] is None
    np.testing.assert_allclose(fixed["c"], expected_c, rtol=2e-14)
    np.testing.assert_allclose(mixed["log_c"], np.log(expected_c), atol=2e-14)
    np.testing.assert_allclose(fixed["adjustment"], mixed["adjustment"], atol=2e-14)

    # Pin the equality-matrix chain rule, non-unit observation weights, and
    # optimizer log-sigma derivative against the actual fixed-model objective.
    gradient = mdcev_gradient(theta, *args, ctrl)
    numerical = np.zeros_like(gradient)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        numerical[:, j] = (
            mdcev_loglik(plus, *args, ctrl) - mdcev_loglik(minus, *args, ctrl)
        ) / 2e-6
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-6)


@pytest.mark.parametrize("utility", ["trad", "linear"])
def test_log_satiation_handles_extreme_indices_without_exponentiating(utility):
    qty = np.array([[2., 1., 0.], [1., 0., 2.]])
    price = np.array([[1., 2., .5], [.7, 1., 3.]])
    u = np.array([[-1000., 1000., -1000.], [-1000., -1000., 1000.]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        sat = _log_satiation_terms(qty, price, u, utility=utility)
    for key in ("log_c", "adjustment", "frac_qty", "jac_gamma"):
        assert np.isfinite(sat[key]).all(), key
