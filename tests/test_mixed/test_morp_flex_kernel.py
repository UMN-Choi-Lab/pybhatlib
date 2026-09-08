"""Self-contained finite-difference gate for the MORP RectMvncdKernel.

No GAUSS oracle: on a small synthetic MORP observation set (``nord=2``, a random
valid joint correlation over ``nrndtot = nrndcoef + nord`` elements, ordered
thresholds, and observed categories covering the low / middle / high routing
branches) we check that the analytic gradient fields of
:class:`~pybhatlib.models.morp_flex._morp_flex_kernel.RectMvncdKernel` each track
a *central finite difference* of ``log(p_obs)`` of the **same** Python
``pdfrectn`` probability:

* ``dlogp_dV``       vs central-FD wrt the ordinal utilities ``Vsub``;
* ``dlogp_dkparams`` vs central-FD wrt the kernel-owned ``[thresh | kernlam]``
  parameters ``theta[thresh]`` / ``theta[kernlam]``;
* ``dlogp_drc``      vs central-FD wrt the drawn random coefficients ``rc_draw``
  (copula conditional-mean path);
* ``dlogp_domega``   vs central-FD wrt the free off-diagonal correlation
  elements of the joint ``omegastar`` (copula conditional cov + mean path).

The kernel is exercised with ``copula=True`` and both the normal and the
Yeo-Johnson kernel (``yj_kernel`` toggled). The tolerance is loose (1e-4)
because ``pdfrectn``'s analytic MVNCD gradient is an approximation; the gate
enforces that the analytic gradient matches the finite difference of the
identical approximated function, so any bug in the gradient chain (threshold
routing, kernel transform / YJ ``meanyj`` chain, copula conditional mean, or the
joint-correlation ``gcondnewcov`` / ``gcondnewmean`` seam) is caught.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout, thresh_reparam
from pybhatlib.utils._logistic import cdlogit
from pybhatlib.models.morp_flex._morp_flex_kernel import RectMvncdKernel

NORD = 2
NRNDCOEF = 2
N_CATEGORIES = (3, 4)          # dim0 cats {0,1,2}, dim1 cats {0,1,2,3}
NUMTHRESH = tuple(c - 1 for c in N_CATEGORIES)   # (2, 3)
N_THRESH = sum(NUMTHRESH)      # 5
NRNDTOT = NRNDCOEF + NORD      # 4
N_RCOR = NRNDTOT * (NRNDTOT - 1) // 2            # 6

# observed categories per obs, covering low / middle / high in both dims.
Y_ORD = np.array(
    [
        [0, 3],   # dim0 low,    dim1 high
        [1, 1],   # dim0 middle, dim1 middle
        [2, 0],   # dim0 high,   dim1 low
        [1, 2],   # dim0 middle, dim1 middle
    ],
    dtype=np.int64,
)


def _make_case(seed: int, *, yj_kernel: bool):
    """Build a small synthetic MORP kernel scenario (copula on)."""
    rng = np.random.default_rng(seed)
    n_obs = Y_ORD.shape[0]
    n_kernlam = NORD if yj_kernel else 0

    kernel = RectMvncdKernel(
        NORD, NRNDCOEF, N_CATEGORIES,
        copula=True, yj_kernel=yj_kernel, scal=1.0,
    )
    layout = ParamLayout(
        n_beta=0, n_rcor=N_RCOR, n_scal=0, n_lam=0, n_kern=0,
        kern_before_lam=True, n_thresh=N_THRESH, n_kernlam=n_kernlam,
    )

    theta = np.zeros(layout.n_theta, dtype=np.float64)
    sl = layout.slices()
    # ordered-increment thresholds: raw [b0, b1, ...] -> [b0, b0+exp(b1), ...].
    theta[sl["thresh"]] = np.array([-0.5, 0.0, -1.0, 0.0, 0.0]) + rng.normal(
        scale=0.15, size=N_THRESH
    )
    # unconstrained radial correlation params -> a valid random correlation.
    theta[sl["rcor"]] = rng.normal(scale=0.5, size=N_RCOR)
    if yj_kernel:
        theta[sl["kernlam"]] = rng.normal(scale=0.4, size=n_kernlam)

    Vsub = rng.normal(scale=0.6, size=(n_obs, NORD))
    rc_draw = rng.normal(scale=0.8, size=(n_obs, NRNDCOEF))
    obs = SimpleNamespace(y_ord=Y_ORD)

    return kernel, layout, theta, Vsub, rc_draw, obs


def _logp_theta(kernel, layout, theta, Vsub, rc_draw, obs):
    """log(p_obs) vector at the given theta (value-only)."""
    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=False)
    return np.log(kev.p_obs)


def _base_parts(kernel, layout, theta):
    """Recover (omegastar, tau, dtau, xlamker, kernlam) from theta for the
    omega-FD path (perturbs omegastar directly via ``state_from_parts``)."""
    sl = layout.slices()
    kst = kernel.prepare(theta, layout)
    tau, dtau = thresh_reparam(theta[sl["thresh"]], kernel.numthresh, want_grad=True)
    if kernel.yj_kernel:
        kernlam = np.asarray(theta[sl["kernlam"]], dtype=np.float64)
        xlamker = 2.0 * cdlogit(kernlam)
    else:
        kernlam = xlamker = None
    return kst.omegastar.copy(), tau, dtau, xlamker, kernlam


def _offdiag_pairs(k):
    return [(p, q) for p in range(k) for q in range(p + 1, k)]


@pytest.mark.parametrize("yj_kernel", [False, True])
def test_rectmvncd_kernel_fd_gate(yj_kernel):
    seed = 20260716 + int(yj_kernel)
    kernel, layout, theta, Vsub, rc_draw, obs = _make_case(seed, yj_kernel=yj_kernel)
    n_obs = Vsub.shape[0]
    k = kernel.nrndcoef

    kst = kernel.prepare(theta, layout)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=rc_draw, want_grad=True)

    # sanity: probabilities in (0, 1]
    assert np.all(kev.p_obs > 0.0) and np.all(kev.p_obs <= 1.0 + 1e-9)

    eps = 1e-6
    worst = {"V": 0.0, "kern": 0.0, "rc": 0.0, "omega": 0.0}

    # --- dlogp_dV : central FD wrt each Vsub[i, d] ------------------------- #
    fd_dV = np.zeros_like(Vsub)
    for i in range(n_obs):
        for d in range(NORD):
            Vp = Vsub.copy(); Vp[i, d] += eps
            Vm = Vsub.copy(); Vm[i, d] -= eps
            lp = _logp_theta(kernel, layout, theta, Vp, rc_draw, obs)
            lm = _logp_theta(kernel, layout, theta, Vm, rc_draw, obs)
            fd_dV[i, d] = (lp[i] - lm[i]) / (2.0 * eps)
    worst["V"] = float(np.max(np.abs(fd_dV - kev.dlogp_dV)))
    assert np.allclose(kev.dlogp_dV, fd_dV, atol=1e-4, rtol=0.0), (
        f"dlogp_dV FD mismatch (max |Δ|={worst['V']:.2e})"
    )

    # --- dlogp_dkparams : central FD wrt theta[thresh]|theta[kernlam] ------ #
    sl = layout.slices()
    kern_idx = list(range(sl["thresh"].start, sl["thresh"].stop))
    if yj_kernel:
        kern_idx += list(range(sl["kernlam"].start, sl["kernlam"].stop))
    n_kparams = len(kern_idx)
    assert kev.dlogp_dkparams.shape == (n_obs, n_kparams)
    fd_dk = np.zeros((n_obs, n_kparams))
    for jj, p in enumerate(kern_idx):
        tp = theta.copy(); tp[p] += eps
        tm = theta.copy(); tm[p] -= eps
        lp = _logp_theta(kernel, layout, tp, Vsub, rc_draw, obs)
        lm = _logp_theta(kernel, layout, tm, Vsub, rc_draw, obs)
        fd_dk[:, jj] = (lp - lm) / (2.0 * eps)
    worst["kern"] = float(np.max(np.abs(fd_dk - kev.dlogp_dkparams)))
    assert np.allclose(kev.dlogp_dkparams, fd_dk, atol=1e-4, rtol=0.0), (
        f"dlogp_dkparams FD mismatch (max |Δ|={worst['kern']:.2e})"
    )

    # --- dlogp_drc : central FD wrt each rc_draw[i, r] --------------------- #
    fd_drc = np.zeros((n_obs, k))
    for i in range(n_obs):
        for r in range(k):
            rp = rc_draw.copy(); rp[i, r] += eps
            rm = rc_draw.copy(); rm[i, r] -= eps
            lp = _logp_theta(kernel, layout, theta, Vsub, rp, obs)
            lm = _logp_theta(kernel, layout, theta, Vsub, rm, obs)
            fd_drc[i, r] = (lp[i] - lm[i]) / (2.0 * eps)
    worst["rc"] = float(np.max(np.abs(fd_drc - kev.dlogp_drc)))
    assert np.allclose(kev.dlogp_drc, fd_drc, atol=1e-4, rtol=0.0), (
        f"dlogp_drc FD mismatch (max |Δ|={worst['rc']:.2e})"
    )

    # --- dlogp_domega : central FD wrt omegastar free off-diagonals -------- #
    assert kev.dlogp_domega is not None
    omega0, tau, dtau, xlamker, kernlam = _base_parts(kernel, layout, theta)
    pairs = _offdiag_pairs(NRNDTOT)
    assert kev.dlogp_domega.shape == (n_obs, len(pairs))

    def _logp_omega(omega):
        st = kernel.state_from_parts(omega, tau, dtau, xlamker, kernlam)
        ev = kernel.probability(Vsub, obs, st, rc_draw=rc_draw, want_grad=False)
        return np.log(ev.p_obs)

    fd_dom = np.zeros((n_obs, len(pairs)))
    for c, (p, q) in enumerate(pairs):
        op = omega0.copy(); op[p, q] += eps; op[q, p] += eps
        om = omega0.copy(); om[p, q] -= eps; om[q, p] -= eps
        lp = _logp_omega(op)
        lm = _logp_omega(om)
        fd_dom[:, c] = (lp - lm) / (2.0 * eps)
    worst["omega"] = float(np.max(np.abs(fd_dom - kev.dlogp_domega)))
    assert np.allclose(kev.dlogp_domega, fd_dom, atol=1e-4, rtol=0.0), (
        f"dlogp_domega FD mismatch (max |Δ|={worst['omega']:.2e})"
    )

    print(
        f"yj_kernel={yj_kernel}  max|Δ| dlogp_dV={worst['V']:.2e} "
        f"dlogp_dkparams={worst['kern']:.2e} dlogp_drc={worst['rc']:.2e} "
        f"dlogp_domega={worst['omega']:.2e}"
    )
