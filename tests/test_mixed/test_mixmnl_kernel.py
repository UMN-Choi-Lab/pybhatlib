"""Gate for the MixMNL softmax kernel (plan T0.12).

Deterministic finite-difference gate — no GAUSS reference needed, because the
softmax has an exact analytic gradient:

1. ``p_obs`` equals ``exp`` of the per-observation MNL log-probability computed
   by the shipped fixed-coef ``mnl_loglik`` on the same random problem (1e-10).
2. ``dlogp_dV`` equals the central finite difference of ``log p_obs`` w.r.t.
   ``Vsub`` (1e-6).

Also pins the structural contract: the four-field :class:`KernelObsResult`
shapes, empty ``dlogp_dkparams``, and zero ``dlogp_drc``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._kernel import KernelObsResult
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.models.mnl._mnl_loglik import _compute_probabilities, mnl_loglik


class _Obs:
    """Minimal DesignData stand-in exposing ``avail`` and ``chosen``."""

    def __init__(self, avail: np.ndarray, chosen: np.ndarray) -> None:
        self.avail = avail
        self.chosen = chosen


def _random_mnl_problem(seed: int, *, mask: bool = False):
    """Build a random MNL problem in the fixed-coef ``mnl_loglik`` data layout.

    Returns everything both paths need: the flat ``dta`` + index vectors for
    ``mnl_loglik``, plus the ``(n_obs, nc)`` ``Vsub`` and an ``_Obs`` bundle for
    the kernel.
    """
    rng = np.random.default_rng(seed)
    n_obs, nc, numunord = 7, 4, 3

    iv = rng.standard_normal((n_obs, nc * numunord))          # IV design block
    if mask:
        avail = (rng.random((n_obs, nc)) > 0.3).astype(np.float64)
        avail[:, 0] = 1.0                                      # keep >=1 available
    else:
        avail = np.ones((n_obs, nc), dtype=np.float64)

    # Chosen alternative: uniformly among the available ones.
    chosen = np.zeros((n_obs, nc), dtype=np.float64)
    for o in range(n_obs):
        avail_alts = np.flatnonzero(avail[o] > 0)
        chosen[o, rng.choice(avail_alts)] = 1.0

    dta = np.hstack([iv, avail, chosen])                       # column blocks
    indxivunord = np.arange(nc * numunord)                     # IV cols first
    davunord = np.arange(nc * numunord, nc * numunord + nc)    # then avail
    dvunord = np.arange(nc * numunord + nc, nc * numunord + 2 * nc)  # then choice

    x = rng.standard_normal(numunord)                          # beta

    # Vsub identical to the softmax utilities the fixed-coef path forms.
    _, v = _compute_probabilities(x, dta, indxivunord, davunord, nc, numunord)

    obs = _Obs(avail, chosen)
    return dict(
        x=x, dta=dta, indxivunord=indxivunord, davunord=davunord,
        dvunord=dvunord, nc=nc, numunord=numunord, Vsub=v, obs=obs,
    )


@pytest.mark.parametrize("mask", [False, True])
def test_p_obs_matches_mnl_loglik(mask):
    """``p_obs`` == ``exp(mnl_loglik)`` per observation (1e-10)."""
    p = _random_mnl_problem(seed=0, mask=mask)
    kernel = SoftmaxKernel(p["nc"])
    rc_draw = np.zeros((p["Vsub"].shape[0], 2), dtype=np.float64)

    res = kernel.probability(
        p["Vsub"], p["obs"], None, rc_draw=rc_draw, want_grad=True
    )

    ll_ref = mnl_loglik(
        p["x"], p["dta"], p["indxivunord"], p["davunord"], p["dvunord"],
        p["nc"], p["numunord"],
    )
    np.testing.assert_allclose(res.p_obs, np.exp(ll_ref), rtol=0, atol=1e-10)


def test_dlogp_dV_matches_central_fd():
    """``dlogp_dV`` == central FD of ``log p_obs`` w.r.t. ``Vsub`` (1e-6)."""
    p = _random_mnl_problem(seed=1, mask=True)
    kernel = SoftmaxKernel(p["nc"])
    Vsub = p["Vsub"]
    obs = p["obs"]
    n_obs, nc = Vsub.shape
    rc_draw = np.zeros((n_obs, 2), dtype=np.float64)

    analytic = kernel.probability(
        Vsub, obs, None, rc_draw=rc_draw, want_grad=True
    ).dlogp_dV

    def logp(V: np.ndarray) -> np.ndarray:
        r = kernel.probability(V, obs, None, rc_draw=rc_draw, want_grad=False)
        return np.log(r.p_obs)

    eps = 1e-6
    fd = np.zeros((n_obs, nc), dtype=np.float64)
    for k in range(nc):
        Vp = Vsub.copy(); Vp[:, k] += eps
        Vm = Vsub.copy(); Vm[:, k] -= eps
        fd[:, k] = (logp(Vp) - logp(Vm)) / (2.0 * eps)

    np.testing.assert_allclose(analytic, fd, rtol=0, atol=1e-6)


def test_result_contract_shapes():
    """Four-field result: shapes, empty kparams, zero copula path."""
    p = _random_mnl_problem(seed=2)
    kernel = SoftmaxKernel(p["nc"])
    n_obs = p["Vsub"].shape[0]
    n_rnd = 3
    rc_draw = np.random.default_rng(3).standard_normal((n_obs, n_rnd))

    res = kernel.probability(
        p["Vsub"], p["obs"], None, rc_draw=rc_draw, want_grad=True
    )

    assert isinstance(res, KernelObsResult)
    assert res.p_obs.shape == (n_obs,)
    assert res.dlogp_dV.shape == (n_obs, p["nc"])
    assert res.dlogp_dkparams.shape == (n_obs, 0)
    assert res.dlogp_drc.shape == (n_obs, n_rnd)
    assert np.all(res.dlogp_drc == 0.0)
    assert kernel.kernel_param_names() == []
