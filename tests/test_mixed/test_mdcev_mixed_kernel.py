"""Self-contained finite-difference gate for the MDCEV LogitJacobianKernel.

No GAUSS oracle: on a small synthetic MDCEV observation set (``nc=4`` = outside +
3 inside goods; consumption patterns covering the *mixed*, *all-inside* and
*outside-only* discrete/continuous routing branches; a real satiation ``gamma``
design; ``nrndcoef=2`` -> one log-normal + one Yeo-Johnson coefficient) we check
that the two load-bearing gradient fields of
:class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel.LogitJacobianKernel`
each track a *central finite difference* of ``log(p_obs)`` of the **same** Python
MDCEV logit probability:

* ``dlogp_dV``       vs central-FD wrt the drawn baseline utilities ``Vsub``;
* ``dlogp_dkparams`` vs central-FD wrt the kernel-owned ``[gamma | kern-scale]``
  parameters (``theta[gamma]`` / ``theta[kern]``).

Convention: the MDCEV kernel runs under ``score_convention="divide"``, so its
``dlogp_*`` fields carry **raw** probability derivatives ``d p_obs / d(.)`` (GAUSS
``gcomp``); the engine divides the assembled ``gcomp`` by ``p_obs`` once. This
gate therefore divides each field by ``p_obs`` before comparing to the finite
difference of ``log(p_obs)`` (exactly the divide the engine performs).

The MDCEV logit likelihood is **exact** (not an OVUS approximation), so the
tolerance is tight (``1e-6``): any bug in the identity-baseline-design reuse of
the shipped ``mdcev_loglik`` / ``mdcev_gradient``, the raw<->log conversion, or
the ``gamma`` / kernel-scale slice extraction is caught. ``dlogp_drc`` is
asserted exactly zero (no copula) and ``p_obs`` must not depend on ``rc_draw``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel


# --------------------------------------------------------------------------- #
# synthetic fixture
# --------------------------------------------------------------------------- #
def _make_case(seed: int, *, utility: str = "trad"):
    """Build a small synthetic mixed-MDCEV kernel scenario.

    Returns the kernel, its ParamLayout, a random ``theta`` (only the ``gamma``
    and ``kern`` blocks are load-bearing), the ``Vsub`` / ``rc_draw`` inputs and
    the ``obs`` bundle (consumption / price / gamma design).
    """
    rng = np.random.default_rng(seed)
    nc = 4                          # outside + 3 inside goods
    nvargam = 3                     # translation / satiation parameters
    nrndcoef = 2                    # 1 log-normal + 1 YJ (only sizes rc_draw here)
    n_rcor = nrndcoef * (nrndcoef - 1) // 2   # 1

    kernel = LogitJacobianKernel(
        nc, nvargam, control=MDCEVControl(utility=utility)
    )

    # MDCEV physical order: [beta | gamma | rcor | kern | scal | lam].
    layout = ParamLayout(
        n_beta=0,                   # beta enters only via Vsub (a direct input)
        n_gamma=nvargam,
        n_rcor=n_rcor,
        n_kern=1,                   # single MDCEV kernel (log) scale
        n_scal=nrndcoef,
        n_lam=nrndcoef,
        kern_before_scal=True,
    )

    theta = np.zeros(layout.n_theta, dtype=np.float64)
    sl = layout.slices()
    gamma_raw = rng.normal(scale=0.4, size=nvargam)
    gamma_raw[0] = 0.0              # outside-good satiation (forced -1000; inert)
    theta[sl["gamma"]] = gamma_raw
    theta[sl["kern"]] = np.array([0.2])   # log_sigma -> sigma ~ 1.22
    # rcor / scal / lam are irrelevant to the kernel; leave them as-is.

    # Drawn baseline utilities (X @ xmunew), moderate so gradients are O(1).
    Vsub = rng.normal(scale=0.7, size=(n_obs := 3, nc))

    # Consumption patterns exercising the three routing branches:
    #   obs0 mixed (some inside consumed, some not) -> nonpdfcdfmvlogit
    #   obs1 all inside consumed (h == 0)           -> nonpdfmvlogit
    #   obs2 outside only (m - 1 == 0)              -> noncdfmvlogit
    consumption = np.array(
        [
            [2.0, 1.5, 0.8, 0.0],
            [1.0, 0.5, 0.7, 1.2],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    price = np.tile(np.array([1.0, 1.2, 0.9, 1.5]), (n_obs, 1))

    # Satiation design: outside-good row (k == 0) is inert; inside goods load
    # positive covariates on gamma params 1 and 2 (param 0 is the inert outside).
    gamma_design = np.zeros((n_obs, nc, nvargam), dtype=np.float64)
    gamma_design[:, 1:, :] = rng.uniform(0.5, 1.5, size=(n_obs, nc - 1, nvargam))
    gamma_design[:, :, 0] = 0.0    # param 0 loads nothing (outside good only)

    obs = SimpleNamespace(consumption=consumption, price=price,
                          gamma_design=gamma_design)

    rc_draw = rng.normal(scale=1.0, size=(n_obs, nrndcoef))
    return kernel, layout, theta, Vsub, rc_draw, obs


def _logp(kernel, layout, theta, Vsub, rc_draw, obs):
    """log(p_obs) vector at the given inputs (value-only, no gradients)."""
    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=False)
    return np.log(kev.p_obs)


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("utility", ["trad", "linear"])
def test_logit_jacobian_kernel_fd_gate(utility):
    kernel, layout, theta, Vsub, rc_draw, obs = _make_case(
        20260717, utility=utility
    )
    n_obs, nc = Vsub.shape

    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=True)

    # sanity: probabilities strictly positive
    assert np.all(kev.p_obs > 0.0)
    # no copula -> dlogp_drc is exactly zero and correctly shaped
    assert kev.dlogp_drc.shape == rc_draw.shape
    assert np.allclose(kev.dlogp_drc, 0.0)
    assert kev.dlogp_domega is None

    # raw derivatives -> log-derivatives via the engine's "divide" convention.
    inv_p = 1.0 / kev.p_obs[:, None]
    score_dV = kev.dlogp_dV * inv_p                     # (n_obs, nc)
    score_dk = kev.dlogp_dkparams * inv_p               # (n_obs, n_gamma + 1)

    eps = 1e-6

    # --- dlogp_dV : central FD wrt each Vsub[i, c] ------------------------- #
    fd_dV = np.zeros_like(Vsub)
    for i in range(n_obs):
        for c in range(nc):
            Vp = Vsub.copy(); Vp[i, c] += eps
            Vm = Vsub.copy(); Vm[i, c] -= eps
            lp = _logp(kernel, layout, theta, Vp, rc_draw, obs)
            lm = _logp(kernel, layout, theta, Vm, rc_draw, obs)
            fd_dV[i, c] = (lp[i] - lm[i]) / (2.0 * eps)
    worst_V = float(np.max(np.abs(fd_dV - score_dV)))
    assert np.allclose(score_dV, fd_dV, atol=1e-6, rtol=1e-6), (
        f"dlogp_dV FD mismatch (max |Δ|={worst_V:.2e})"
    )

    # --- dlogp_dkparams : central FD wrt theta[gamma] and theta[kern] ------ #
    sl = layout.slices()
    kparam_idx = list(range(sl["gamma"].start, sl["gamma"].stop)) + list(
        range(sl["kern"].start, sl["kern"].stop)
    )
    assert len(kparam_idx) == score_dk.shape[1] == kernel.nvargam + 1
    fd_dk = np.zeros((n_obs, len(kparam_idx)))
    for jj, p in enumerate(kparam_idx):
        tp = theta.copy(); tp[p] += eps
        tm = theta.copy(); tm[p] -= eps
        lp = _logp(kernel, layout, tp, Vsub, rc_draw, obs)
        lm = _logp(kernel, layout, tm, Vsub, rc_draw, obs)
        fd_dk[:, jj] = (lp - lm) / (2.0 * eps)
    worst_k = float(np.max(np.abs(fd_dk - score_dk)))
    assert np.allclose(score_dk, fd_dk, atol=1e-6, rtol=1e-6), (
        f"dlogp_dkparams FD mismatch (max |Δ|={worst_k:.2e})"
    )

    # the outside-good gamma param (column 0) is inert -> zero gradient / FD.
    assert np.allclose(score_dk[:, 0], 0.0, atol=1e-9)

    print(
        f"utility={utility}  max|Δ| dlogp_dV={worst_V:.2e} "
        f"dlogp_dkparams={worst_k:.2e}"
    )


def test_logit_jacobian_kernel_pobs_independent_of_rc():
    """``p_obs`` must not depend on the drawn coefficients (no copula)."""
    kernel, layout, theta, Vsub, rc_draw, obs = _make_case(20260717)
    kst = kernel.prepare(theta, layout)
    p1 = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=False).p_obs
    p2 = kernel.probability(
        Vsub, obs, kst, rc_draw=rc_draw * 3.0 + 1.0, want_grad=False
    ).p_obs
    assert np.allclose(p1, p2, atol=0.0, rtol=0.0)


def test_want_grad_false_returns_zeros():
    """``want_grad=False`` returns correctly-shaped zero gradient fields."""
    kernel, layout, theta, Vsub, rc_draw, obs = _make_case(20260717)
    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=False)
    n_obs, nc = Vsub.shape
    assert kev.dlogp_dV.shape == (n_obs, nc)
    assert kev.dlogp_dkparams.shape == (n_obs, kernel.nvargam + 1)
    assert np.all(kev.dlogp_dV == 0.0)
    assert np.all(kev.dlogp_dkparams == 0.0)
    assert np.all(kev.p_obs > 0.0)
