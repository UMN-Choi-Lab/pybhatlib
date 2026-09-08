"""Self-contained finite-difference gate for the MNP kernel's joint-correlation
gradient (``dlogp_domega``) and the copula draw gradient (``dlogp_drc``).

No GAUSS oracle. On a small synthetic MNP observation set with ``nc = 3`` (so
``A = 2`` -> the OVUS MVNCD is an *exact* bivariate normal CDF and the finite
differences are tight), a random valid ``4x4`` joint correlation ``omegastar``
(``nrndtot = nrndcoef + (nc - 1) = 2 + 2``) and a nonzero kernel scale, we check
that :class:`~pybhatlib.models.mnpkercp._mnpkercp_kernel.MvncdKernel` (copula
path) emits two analytic gradient fields matching central finite differences of
``log(p_obs)`` of the **same** Python ``mvncd`` (OVUS) probability:

* ``dlogp_domega`` vs central-FD wrt the free off-diagonal (correlation)
  elements of the joint ``omegastar``, **holding the drawn ``errbeta3`` fixed**
  (exactly as the kernel consumes it -- the kernel never re-derives ``errbeta3``
  from ``x11chol`` internally). The FD rebuilds the kernel state from the
  perturbed ``omegastar`` via :meth:`MvncdKernel.state_from_omegastar`.
* ``dlogp_drc`` vs central-FD wrt each drawn random coefficient ``errbeta3``.

Free-correlation ordering is row-based strict-upper (``vecndup``) of the full
``nrndtot x nrndtot`` matrix, i.e. ``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pybhatlib.matgradient._radial import newcholparmscaled
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel


# --------------------------------------------------------------------------- #
# synthetic fixture
# --------------------------------------------------------------------------- #
def _offdiag_pairs(K: int):
    """Row-based strict-upper (``vecndup``) index pairs of a ``K x K`` matrix."""
    return [(p, q) for p in range(K) for q in range(p + 1, K)]


def _make_case(seed: int):
    """Build a synthetic ``nc=3`` MNP copula scenario.

    Returns the kernel, a random valid ``4x4`` ``omegastar``, the nonzero
    kernel-scale block ``xscalker``, the ``Vsub`` / ``errbeta3`` inputs and the
    ``obs`` bundle (all alternatives available; random chosen alternative).
    """
    rng = np.random.default_rng(seed)
    nc = 3
    nrndcoef = 2
    n_obs = 4
    kd = nc - 1                                # 2
    nrndtot = nrndcoef + kd                    # 4
    n_kern = nc - 2                            # 1

    kernel = MvncdKernel(nc, nrndcoef, copula=True, scal=1.0)

    # a valid random joint correlation via the radial reparameterization
    xrand = rng.normal(scale=0.6, size=nrndtot * (nrndtot - 1) // 2)
    cholall = np.asarray(newcholparmscaled(xrand, 1.0))
    omegastar = cholall.T @ cholall            # unit-diagonal correlation (4x4)

    xscalker = rng.normal(scale=0.5, size=n_kern)

    Vsub = rng.normal(scale=0.8, size=(n_obs, nc))
    errbeta3 = rng.normal(scale=1.0, size=(n_obs, nrndcoef))

    avail = np.ones((n_obs, nc), dtype=np.float64)
    chosen = np.zeros((n_obs, nc), dtype=np.float64)
    chosen[np.arange(n_obs), rng.integers(0, nc, size=n_obs)] = 1.0
    obs = SimpleNamespace(avail=avail, chosen=chosen)

    return kernel, omegastar, xscalker, Vsub, errbeta3, obs


def _logp(kernel, omegastar, xscalker, Vsub, errbeta3, obs):
    """``log(p_obs)`` from a kernel state built on an explicit ``omegastar``."""
    kst = kernel.state_from_omegastar(omegastar, xscalker)
    kev = kernel.probability(Vsub, obs, kst, rc_draw=errbeta3, want_grad=False)
    return np.log(kev.p_obs)


# --------------------------------------------------------------------------- #
# gate
# --------------------------------------------------------------------------- #
def test_mnpkercp_corr_grad_fd():
    for seed in (20260716, 987654321):
        kernel, omegastar, xscalker, Vsub, errbeta3, obs = _make_case(seed)
        n_obs = Vsub.shape[0]
        k = kernel.nrndcoef
        nrndtot = omegastar.shape[0]
        pairs = _offdiag_pairs(nrndtot)
        n_free = len(pairs)

        kst = kernel.state_from_omegastar(omegastar, xscalker)
        kev = kernel.probability(Vsub, obs, kst, rc_draw=errbeta3, want_grad=True)

        assert kev.dlogp_domega is not None
        assert kev.dlogp_domega.shape == (n_obs, n_free)
        assert np.all(kev.p_obs > 0.0) and np.all(kev.p_obs <= 1.0 + 1e-9)

        eps = 1e-6

        # --- dlogp_domega : FD wrt each free off-diagonal of omegastar ------ #
        fd_dom = np.zeros((n_obs, n_free))
        for j, (p, q) in enumerate(pairs):
            op = omegastar.copy(); op[p, q] += eps; op[q, p] += eps
            om = omegastar.copy(); om[p, q] -= eps; om[q, p] -= eps
            lp = _logp(kernel, op, xscalker, Vsub, errbeta3, obs)
            lm = _logp(kernel, om, xscalker, Vsub, errbeta3, obs)
            fd_dom[:, j] = (lp - lm) / (2.0 * eps)
        worst_dom = float(np.max(np.abs(fd_dom - kev.dlogp_domega)))
        assert np.allclose(kev.dlogp_domega, fd_dom, atol=1e-5, rtol=0.0), (
            f"[seed={seed}] dlogp_domega FD mismatch (max |Δ|={worst_dom:.2e})"
        )

        # --- dlogp_drc : FD wrt each drawn errbeta3[i, r] ------------------- #
        fd_drc = np.zeros((n_obs, k))
        for i in range(n_obs):
            for r in range(k):
                rp = errbeta3.copy(); rp[i, r] += eps
                rm = errbeta3.copy(); rm[i, r] -= eps
                lp = _logp(kernel, omegastar, xscalker, Vsub, rp, obs)
                lm = _logp(kernel, omegastar, xscalker, Vsub, rm, obs)
                fd_drc[i, r] = (lp[i] - lm[i]) / (2.0 * eps)
        worst_drc = float(np.max(np.abs(fd_drc - kev.dlogp_drc)))
        assert np.allclose(kev.dlogp_drc, fd_drc, atol=1e-5, rtol=0.0), (
            f"[seed={seed}] dlogp_drc FD mismatch (max |Δ|={worst_drc:.2e})"
        )

        print(
            f"seed={seed}  max|Δ| dlogp_domega={worst_dom:.2e} "
            f"dlogp_drc={worst_drc:.2e}"
        )
