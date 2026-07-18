"""Self-contained finite-difference gate for the MNP MvncdKernel.

No GAUSS oracle: on a small synthetic MNP observation set (``nc=3``, ``A=2``,
random valid joint correlation + kernel scales) we check that the three analytic
gradient fields of :class:`~pybhatlib.models.mnpkercp._mnpkercp_kernel.MvncdKernel`
each track a *central finite difference* of ``log(p_obs)`` of the **same** Python
``mvncd`` (OVUS) probability:

* ``dlogp_dV``       vs central-FD wrt the systematic utilities ``Vsub``;
* ``dlogp_dkparams`` vs central-FD wrt the kernel-scale parameters ``theta[kern]``;
* ``dlogp_drc``      vs central-FD wrt the drawn random coefficients ``rc_draw``
  (copula path only).

The tolerance is loose (1e-4) because the OVUS MVNCD is an analytic
*approximation*; what the gate enforces is that the analytic gradient matches the
finite difference of the identical approximated function, so any bug in the
gradient chain (differencing map, kernel-scale ``logitmod`` reparam, or the
copula conditional-mean sensitivity) is caught.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel


# --------------------------------------------------------------------------- #
# synthetic fixture
# --------------------------------------------------------------------------- #
def _make_case(seed: int, *, copula: bool):
    """Build a small synthetic MNP kernel scenario.

    Returns the kernel, its ParamLayout, a random ``theta`` (only the ``rcor``
    and ``kern`` blocks are load-bearing), the ``Vsub`` / ``rc_draw`` inputs and
    the ``obs`` bundle (avail / chosen).
    """
    rng = np.random.default_rng(seed)
    nc = 3
    nrndcoef = 2
    n_obs = 4
    kernel_dim = nc - 1                       # 2
    nrndtot = nrndcoef + kernel_dim           # 4
    n_rcor = nrndtot * (nrndtot - 1) // 2     # 6
    n_kern = nc - 2                           # 1

    kernel = MvncdKernel(nc, nrndcoef, copula=copula, scal=1.0)
    layout = ParamLayout(
        n_beta=0, n_rcor=n_rcor, n_scal=0, n_lam=0, n_kern=n_kern,
        kern_before_lam=True,
    )

    theta = np.zeros(layout.n_theta, dtype=np.float64)
    sl = layout.slices()
    # unconstrained radial correlation params -> a valid random correlation
    theta[sl["rcor"]] = rng.normal(scale=0.6, size=n_rcor)
    theta[sl["kern"]] = rng.normal(scale=0.5, size=n_kern)

    Vsub = rng.normal(scale=0.8, size=(n_obs, nc))
    rc_draw = rng.normal(scale=1.0, size=(n_obs, nrndcoef))

    # all alternatives available; a random chosen alternative per obs
    avail = np.ones((n_obs, nc), dtype=np.float64)
    chosen = np.zeros((n_obs, nc), dtype=np.float64)
    chosen[np.arange(n_obs), rng.integers(0, nc, size=n_obs)] = 1.0
    obs = SimpleNamespace(avail=avail, chosen=chosen)

    return kernel, layout, theta, Vsub, rc_draw, obs


def _logp(kernel, layout, theta, Vsub, rc_draw, obs):
    """log(p_obs) vector at the given inputs (value-only, no gradients)."""
    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=False)
    return np.log(kev.p_obs)


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("copula", [False, True])
def test_mvncd_kernel_fd_gate(copula):
    seed = 20260716 + int(copula)
    kernel, layout, theta, Vsub, rc_draw, obs = _make_case(seed, copula=copula)
    n_obs, nc = Vsub.shape
    k = kernel.nrndcoef

    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=True)

    # sanity: probabilities in (0, 1]
    assert np.all(kev.p_obs > 0.0) and np.all(kev.p_obs <= 1.0 + 1e-9)

    eps = 1e-6
    worst = {"V": 0.0, "kern": 0.0, "rc": 0.0}

    # --- dlogp_dV : central FD wrt each Vsub[i, c] ------------------------- #
    fd_dV = np.zeros_like(Vsub)
    for i in range(n_obs):
        for c in range(nc):
            Vp = Vsub.copy(); Vp[i, c] += eps
            Vm = Vsub.copy(); Vm[i, c] -= eps
            lp = _logp(kernel, layout, theta, Vp, rc_draw, obs)
            lm = _logp(kernel, layout, theta, Vm, rc_draw, obs)
            fd_dV[i, c] = (lp[i] - lm[i]) / (2.0 * eps)
    worst["V"] = float(np.max(np.abs(fd_dV - kev.dlogp_dV)))
    assert np.allclose(kev.dlogp_dV, fd_dV, atol=1e-4, rtol=0.0), (
        f"dlogp_dV FD mismatch (max |Δ|={worst['V']:.2e})"
    )

    # --- dlogp_dkparams : central FD wrt theta[kern] (affects ALL obs) ----- #
    sl = layout.slices()
    kern_idx = range(sl["kern"].start, sl["kern"].stop)
    fd_dk = np.zeros((n_obs, layout.n_kern))
    for jj, p in enumerate(kern_idx):
        tp = theta.copy(); tp[p] += eps
        tm = theta.copy(); tm[p] -= eps
        lp = _logp(kernel, layout, tp, Vsub, rc_draw, obs)
        lm = _logp(kernel, layout, tm, Vsub, rc_draw, obs)
        fd_dk[:, jj] = (lp - lm) / (2.0 * eps)
    if layout.n_kern > 0:
        worst["kern"] = float(np.max(np.abs(fd_dk - kev.dlogp_dkparams)))
        assert np.allclose(kev.dlogp_dkparams, fd_dk, atol=1e-4, rtol=0.0), (
            f"dlogp_dkparams FD mismatch (max |Δ|={worst['kern']:.2e})"
        )

    # --- dlogp_drc : central FD wrt each rc_draw[i, r] (copula only) -------- #
    fd_drc = np.zeros((n_obs, k))
    for i in range(n_obs):
        for r in range(k):
            rp = rc_draw.copy(); rp[i, r] += eps
            rm = rc_draw.copy(); rm[i, r] -= eps
            lp = _logp(kernel, layout, theta, Vsub, rp, obs)
            lm = _logp(kernel, layout, theta, Vsub, rm, obs)
            fd_drc[i, r] = (lp[i] - lm[i]) / (2.0 * eps)
    worst["rc"] = float(np.max(np.abs(fd_drc - kev.dlogp_drc)))
    if not copula:
        # no copula -> exactly zero, and the probability must not depend on rc
        assert np.allclose(kev.dlogp_drc, 0.0)
        assert worst["rc"] < 1e-10
    else:
        assert np.allclose(kev.dlogp_drc, fd_drc, atol=1e-4, rtol=0.0), (
            f"dlogp_drc FD mismatch (max |Δ|={worst['rc']:.2e})"
        )

    print(
        f"copula={copula}  max|Δ| dlogp_dV={worst['V']:.2e} "
        f"dlogp_dkparams={worst['kern']:.2e} dlogp_drc={worst['rc']:.2e}"
    )
