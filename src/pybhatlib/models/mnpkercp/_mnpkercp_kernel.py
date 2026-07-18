"""MVNCD (OVUS) kernel with rc<->kernel copula for the mixed-panel MNP model.

Implements the :class:`~pybhatlib.mixed._kernel.MixedKernel` protocol for the
GAUSS ``MNPKERCP`` driver (``0714_UTA_request/MNPKERCP.gss`` ``lpr``/``lgd``,
lines ~610-953). Per MSL replication the engine forms the systematic utilities
``Vsub = X @ xmunew`` and hands them (with the drawn random coefficients) to
:meth:`MvncdKernel.probability`, which returns the observed-choice probability
and the three gradient paths the engine chains back through the shared
random-coefficient reparameterization Jacobians.

Kernel math (one obs ``i``, one draw ``r`` -- GAUSS lines ~619-648)
-------------------------------------------------------------------
The chosen alternative is the utility-maximiser, so the observed choice
probability is the MVNCD of the *differenced* (against the chosen alternative)
latent utilities.  With ``Msubq`` the ``(A, nc)`` differencing map (see
:mod:`pybhatlib.models.mnpkercp._mnpkercp_diff`), ``nc-1`` differenced kernel
errors, ``nrndcoef`` random coefficients, and the joint correlation
``omegastar`` over the ``nrndtot = nrndcoef + (nc-1)`` random elements:

**Copula path** (``copula=True``, GAUSS ``_nocorrrcker=0``) -- the kernel error
is *conditioned* on the drawn random coefficients ``errbeta3`` (``rc_draw``)::

    { B1subq, xi2subq } = condition(I, 0, omegastar, errbeta3, indxrand)
    xi3subq  = wdiagker @ xi2subq @ wdiagker
    covker2  = indxnormalize @ xi3subq @ indxnormalize'
    B2subq   = indxnormalize @ B1subq + Vsub
    B3subq   = Msubq @ B2subq
    xisubq   = Msubq @ covker2 @ Msubq'
    P        = cdfmvnanl(B3subq, xisubq, 0)  ==  mvncd(-B3subq, xisubq)

**No-copula path** (``copula=False``, GAUSS ``_nocorrrcker=1``; the panel
default) -- the conditional mean shift vanishes and the kernel covariance is the
*unconditional* differenced kernel block::

    B3subq   = Msubq @ Vsub
    xisubq   = Msubq @ covker2_uncond @ Msubq'   (covker2_uncond built from the
               omegastar kernel block)
    P        = mvncd(-B3subq, xisubq)

The conditional covariance ``xi2subq`` (copula) and the unconditional
kernel-block covariance both depend only on ``omegastar`` -- not on the draw --
so they are built once in :meth:`prepare`; only the conditional *mean*
``B1subq = M_cond @ errbeta3`` (with ``M_cond = X12 X11^{-1}``) is draw-dependent.

Gradients (GAUSS ``lgd`` lines ~815-931)
-----------------------------------------
Let ``dlnP/dB3subq = -grad_a / P`` and ``dP/dxisubq`` come from
:func:`~pybhatlib.gradmvn._mvncd_grad_ovus.mvncd_grad_ovus_analytic` evaluated at
``a = -B3subq`` (so ``dP/dB3subq = -grad_a``).  The kernel fills all four
:class:`~pybhatlib.mixed._kernel.KernelObsResult` fields:

* ``dlogp_dV     = Msubq' @ dlnP/dB3subq``                       (utility path)
* ``dlogp_dkparams`` -- ``dP/dxisubq`` chained through
  ``xisubq -> covker2 -> xi3subq -> wdiagker -> wker -> xscalker`` (the
  sum-of-squares ``logitmod`` kernel-scale reparameterization);
* ``dlogp_drc  = M_cond' @ indxnormalize' @ dlogp_dV``  (copula path; the
  ``errbeta3`` sensitivity of ``B1subq``), **zero** when ``copula=False``.

The self-consistency of ``p_obs`` and its gradients is guaranteed by always
sourcing ``p_obs`` from ``mvncd_grad_ovus_analytic`` (the same OVUS function the
gradient differentiates), so an analytic gradient tracks the finite-difference
of the identical probability function.

No module-level mutable state; the ``xp`` kwarg is accepted for backend parity
(computation is NumPy, matching the analytic-gradient house style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pybhatlib.gradmvn._mvncd_grad_ovus import mvncd_grad_ovus_analytic
from pybhatlib.matgradient._radial import newcholparmscaled
from pybhatlib.mixed._copula import condition, gcondnewcov, gcondnewmean
from pybhatlib.mixed._kernel import KernelObsResult
from pybhatlib.models.mnpkercp._mnpkercp_diff import dm_matrix
from pybhatlib.utils._logistic import gradlogitmod, logitmod


@dataclass(frozen=True)
class MvncdKernelState:
    """Per-evaluation state built by :meth:`MvncdKernel.prepare`.

    All fields are constant across the MSL draws (they depend only on the
    correlation ``omegastar`` and the kernel-scale block ``xscalker``); only the
    conditional mean ``B1subq = M_cond @ errbeta3`` is draw-dependent and is
    formed inside :meth:`MvncdKernel.probability`.

    Attributes
    ----------
    omegastar : NDArray, shape (nrndtot, nrndtot)
        Joint random-coefficient + differenced-kernel correlation matrix.
    covker2 : NDArray, shape (nc, nc)
        Differenced-kernel covariance embedded into the full alternative space
        (``indxnormalize @ (wdiagker @ inner @ wdiagker) @ indxnormalize'``);
        the conditional kernel block (copula) or the unconditional block.
    kernel_inner : NDArray, shape (nc-1, nc-1)
        The matrix ``inner`` sandwiched by ``wdiagker`` -- ``xi2subq`` (copula)
        or the ``omegastar`` kernel block (no-copula). Needed for the
        kernel-scale gradient.
    wdiagker : NDArray, shape (nc-1, nc-1)
        ``diag(wker_full)`` (kernel scales, first entry = reference).
    m_cond : NDArray, shape (nc-1, nrndcoef)
        ``X12 X11^{-1}`` -- sensitivity of the conditional kernel mean to the
        drawn random coefficients. Empty (``(nc-1, 0)``) when ``nrndcoef == 0``.
    indxnormalize : NDArray, shape (nc, nc-1)
        Embedding ``[0; I_{nc-1}]`` (nc-1 alternatives -> nc alternatives).
    dwker_dxscal : NDArray, shape (n_kern, nc-1)
        Jacobian ``d wker_full[k] / d xscalker_free[j]`` (rows = free params).
    indxrand : NDArray, shape (nrndtot,)
        Marginal-set indicator ``[ones(nrndcoef); zeros(nc-1)]`` selecting the
        random-coefficient (conditioning) block of ``omegastar`` (copula only).
    gxcov_domega : NDArray or None, shape (n_free, n_vech)
        Draw-independent gradient of the conditional kernel covariance
        ``xi2subq`` w.r.t. the free correlation elements of the full
        ``omegastar`` (``gcondnewcov``); rows in row-based ``vecndup`` order of
        the ``nrndtot`` matrix, columns in ``vecdup`` order of ``xi2subq``.
        ``None`` when ``copula=False`` (no joint-correlation gradient emitted).
    """

    omegastar: NDArray
    covker2: NDArray
    kernel_inner: NDArray
    wdiagker: NDArray
    m_cond: NDArray
    indxnormalize: NDArray
    dwker_dxscal: NDArray
    indxrand: Optional[NDArray] = None
    gxcov_domega: Optional[NDArray] = None


class MvncdKernel:
    """MVNCD (OVUS) kernel with optional rc<->kernel copula (MNPKERCP).

    Parameters
    ----------
    nc : int
        Number of alternatives (GAUSS ``nc``); the kernel consumes ``nc``
        utility columns and the differenced kernel space has ``nc - 1`` dims.
    nrndcoef : int
        Number of random coefficients; the joint correlation is sized over
        ``nrndtot = nrndcoef + (nc - 1)``.
    copula : bool, default False
        If ``True`` condition the kernel error on the drawn random coefficients
        (GAUSS ``_nocorrrcker=0``); if ``False`` use the unconditional kernel
        covariance and emit ``dlogp_drc = 0`` (GAUSS ``_nocorrrcker=1``, the
        panel default). ``copula=True`` requires ``nrndcoef >= 1``.
    scal : float, default 1.0
        Radial correlation-Cholesky scaling constant (must match the pipeline /
        space value); passed to ``newcholparmscaled``.

    Attributes
    ----------
    n_util : int
        ``nc`` (number of utility columns consumed).
    """

    def __init__(
        self, nc: int, nrndcoef: int, *, copula: bool = False, scal: float = 1.0
    ) -> None:
        if nc < 2:
            raise ValueError(f"nc must be >= 2, got {nc}")
        if nrndcoef < 0:
            raise ValueError(f"nrndcoef must be >= 0, got {nrndcoef}")
        if copula and nrndcoef < 1:
            raise ValueError("copula=True requires nrndcoef >= 1")
        self.n_util: int = int(nc)
        self.nc: int = int(nc)
        self.nrndcoef: int = int(nrndcoef)
        self.kernel_dim: int = int(nc) - 1
        self.n_kern: int = int(nc) - 2
        self.copula: bool = bool(copula)
        self.scal: float = float(scal)

    # ------------------------------------------------------------------
    def kernel_param_names(self) -> list[str]:
        """Return the ``n_kern = nc - 2`` kernel-scale parameter names."""
        return [f"kernscale{i + 1:02d}" for i in range(self.n_kern)]

    # ------------------------------------------------------------------
    def prepare(self, theta: NDArray, layout: Any, *, xp: Any = None) -> MvncdKernelState:
        """Build the per-evaluation kernel state from the parameter vector.

        Reproduces the draw-independent setup of GAUSS ``lpr``/``lgd``:
        ``omegastar`` from the correlation block (``newcholparmscaled``),
        ``wker_full = sqrt(logitmod([0, xscalker]))`` and ``wdiagker`` from the
        kernel-scale block, the conditional (copula) or unconditional kernel
        covariance ``covker2``, and the conditional-mean sensitivity
        ``m_cond = X12 X11^{-1}``.

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Full parameter vector; the ``"rcor"`` and ``"kern"`` slices are read.
        layout : ParamLayout
            Provides the named block slices.
        xp : module, optional
            Array backend (defaults to NumPy).

        Returns
        -------
        MvncdKernelState
        """
        if xp is None:
            xp = np
        theta = np.asarray(theta, dtype=np.float64)
        sl = layout.slices()
        k = self.nrndcoef
        kd = self.kernel_dim

        # --- joint correlation omegastar (GAUSS lines 556-561) --------------
        xrand = theta[sl["rcor"]]
        nrndtot = k + kd
        if nrndtot > 1:
            cholall = newcholparmscaled(xrand, self.scal)
            omegastar = np.asarray(cholall).T @ np.asarray(cholall)
        else:  # single random element -> unit correlation
            omegastar = np.ones((nrndtot, nrndtot), dtype=np.float64)

        return self.state_from_omegastar(omegastar, theta[sl["kern"]], xp=xp)

    # ------------------------------------------------------------------
    def state_from_omegastar(
        self, omegastar: NDArray, xscalker_free: NDArray, *, xp: Any = None
    ) -> MvncdKernelState:
        """Build the kernel state from an explicit joint ``omegastar``.

        Splits the ``omegastar``-dependent setup out of :meth:`prepare` so the
        joint-correlation finite-difference gate can perturb ``omegastar``
        directly (bypassing the radial reparameterization) and rebuild the
        exact state the kernel consumes. :meth:`prepare` calls this after
        forming ``omegastar`` from ``theta``.

        Parameters
        ----------
        omegastar : NDArray, shape (nrndtot, nrndtot)
            Joint random-coefficient + differenced-kernel correlation matrix.
        xscalker_free : NDArray, shape (n_kern,)
            Free kernel-scale parameters (the ``"kern"`` slice of ``theta``).
        xp : module, optional
            Array backend (defaults to NumPy).

        Returns
        -------
        MvncdKernelState
        """
        if xp is None:
            xp = np
        omegastar = np.asarray(omegastar, dtype=np.float64)
        k = self.nrndcoef
        kd = self.kernel_dim

        # --- kernel scale: wker_full = sqrt(logitmod([0, xscalker])) --------
        xscalker_full = np.concatenate(
            [np.zeros(1, dtype=np.float64), np.asarray(xscalker_free, dtype=np.float64)]
        )                                                     # (nc-1,)
        F, ga = gradlogitmod(xscalker_full)                   # F,(nc-1); ga[a,b]=dF_b/dfull_a
        wker_full = np.sqrt(F)                                # (nc-1,)
        wdiagker = np.diag(wker_full)                         # (nc-1, nc-1)
        # d wker_full[k] / d xscalker_free[j] = ga[j+1, k] / (2 wker_full[k]).
        # Rows are the free params (drop the fixed reference param, full idx 0).
        if self.n_kern > 0:
            dwker_dxscal = ga[1:, :] / (2.0 * wker_full[None, :])   # (n_kern, nc-1)
        else:
            dwker_dxscal = np.zeros((0, kd), dtype=np.float64)

        # --- conditional / unconditional kernel covariance ------------------
        indxnormalize = np.vstack(
            [np.zeros((1, kd), dtype=np.float64), np.eye(kd, dtype=np.float64)]
        )                                                     # (nc, nc-1)

        indxrand = None
        gxcov_domega = None
        if self.copula:
            # xi2subq is draw-independent (COVB in condition); get it via the
            # validated conditional-Gaussian routine with a dummy zero draw.
            indxrand = np.concatenate(
                [np.ones(k, dtype=np.float64), np.zeros(kd, dtype=np.float64)]
            )
            nrndtot = k + kd
            _b1_dummy, kernel_inner = condition(
                np.eye(kd), np.zeros(nrndtot), omegastar,
                np.zeros(k), indxrand,
            )                                                 # kernel_inner (nc-1, nc-1)
            # conditional-mean sensitivity  M_cond = X12 X11^{-1}
            X11 = omegastar[:k, :k]
            X12 = omegastar[k:, :k]
            m_cond = X12 @ np.linalg.inv(X11)                 # (nc-1, nrndcoef)
            # draw-independent d(xi2subq)/d(omega free-corr) (gcondnewcov gX).
            _gy_cov, gxcov_domega = gcondnewcov(
                np.eye(kd), omegastar, indxrand,
                cholesky=False, condcov=False,
            )                                                 # (n_free, n_vech)
        else:
            kernel_inner = omegastar[k:, k:].copy()           # (nc-1, nc-1) kernel block
            m_cond = np.zeros((kd, k), dtype=np.float64)

        xi3 = wdiagker @ kernel_inner @ wdiagker              # (nc-1, nc-1)
        covker2 = indxnormalize @ xi3 @ indxnormalize.T       # (nc, nc)

        return MvncdKernelState(
            omegastar=omegastar,
            covker2=covker2,
            kernel_inner=kernel_inner,
            wdiagker=wdiagker,
            m_cond=m_cond,
            indxnormalize=indxnormalize,
            dwker_dxscal=dwker_dxscal,
            indxrand=indxrand,
            gxcov_domega=gxcov_domega,
        )

    # ------------------------------------------------------------------
    def probability(
        self,
        Vsub: NDArray,
        obs: Any,
        kstate: MvncdKernelState,
        *,
        rc_draw: NDArray,
        want_grad: bool,
    ) -> KernelObsResult:
        """Observed-choice MVNCD probability and gradients for one draw.

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nc)
            Systematic utilities per observation and alternative.
        obs : Any
            Per-observation bundle exposing ``avail`` (``(n_obs, nc)``
            availability mask) and ``chosen`` (``(n_obs, nc)`` one-hot chosen
            indicator), mirroring GAUSS ``availmat`` / ``depvar``.
        kstate : MvncdKernelState
            State from :meth:`prepare`.
        rc_draw : NDArray, shape (n_obs, nrndcoef)
            Drawn (correlated) random coefficients ``errbeta3`` entering the
            copula conditional mean; ignored when ``copula=False`` (but always
            supplied so ``dlogp_drc`` has a well-defined width).
        want_grad : bool
            If ``False`` the gradient fields are correctly-shaped zeros.

        Returns
        -------
        KernelObsResult
            ``p_obs`` ``(n_obs,)``, ``dlogp_dV`` ``(n_obs, nc)``,
            ``dlogp_dkparams`` ``(n_obs, n_kern)``, ``dlogp_drc``
            ``(n_obs, nrndcoef)`` (``= d lnP / d errbeta3``), and
            ``dlogp_domega`` ``(n_obs, nrndtot*(nrndtot-1)//2)`` -- the
            joint-correlation gradient over the free off-diagonal elements of
            the full ``omegastar`` in ``vecndup`` order (``None`` when
            ``copula=False``).
        """
        Vsub = np.asarray(Vsub, dtype=np.float64)
        avail = np.asarray(obs.avail, dtype=np.float64)
        chosen = np.asarray(obs.chosen, dtype=np.float64)
        n_obs, nc = Vsub.shape
        k = self.nrndcoef
        kd = self.kernel_dim
        nrndtot = k + kd
        n_free = nrndtot * (nrndtot - 1) // 2
        N = kstate.indxnormalize
        covker2 = kstate.covker2

        rc = np.asarray(rc_draw, dtype=np.float64)
        if rc.ndim == 1:
            rc = rc.reshape(n_obs, -1)

        p_obs = np.zeros(n_obs, dtype=np.float64)
        dlogp_dV = np.zeros((n_obs, nc), dtype=np.float64)
        dlogp_dkparams = np.zeros((n_obs, self.n_kern), dtype=np.float64)
        dlogp_drc = np.zeros((n_obs, k), dtype=np.float64)
        # Joint-correlation gradient: only emitted for an active copula.
        dlogp_domega = (
            np.zeros((n_obs, n_free), dtype=np.float64)
            if (self.copula and want_grad)
            else None
        )

        # row-based upper-triangular (vecdup) pairs of the differenced-kernel
        # covariance ``xi2subq`` (nc-1 dims): columns of ``gcondnewcov``'s gX.
        vech_pairs = [(a, b) for a in range(kd) for b in range(a, kd)]

        for i in range(n_obs):
            chosen_idx = int(np.argmax(chosen[i]))
            Msubq = dm_matrix(chosen_idx, avail[i], nc)       # (A, nc)
            A = Msubq.shape[0]

            if self.copula:
                B1 = kstate.m_cond @ rc[i]                    # (nc-1,)
                B2 = N @ B1 + Vsub[i]                         # (nc,)
            else:
                B2 = Vsub[i]

            B3 = Msubq @ B2                                   # (A,)
            xisubq = Msubq @ covker2 @ Msubq.T               # (A, A)

            a = -B3
            P, grad_a, grad_sigma = mvncd_grad_ovus_analytic(a, xisubq)
            P = max(float(P), 1e-300)
            p_obs[i] = P

            if not want_grad:
                continue

            # dlnP/dB3 = -grad_a / P  (a = -B3 -> dP/dB3 = -grad_a)
            dlnPdB3 = -np.asarray(grad_a, dtype=np.float64) / P     # (A,)

            # --- utility path ---
            dV = Msubq.T @ dlnPdB3                            # (nc,)
            dlogp_dV[i] = dV

            # --- copula path: errbeta3 -> B1 -> B2 -> B3 ---
            if self.copula:
                dlogp_drc[i] = kstate.m_cond.T @ (N.T @ dV)  # (nrndcoef,)

            # Symmetric dP/dxisubq from the row-based vech ``grad_sigma`` --
            # shared by the kernel-scale and joint-correlation paths.
            need_sigma = self.n_kern > 0 or self.copula
            if need_sigma:
                GxiP = np.zeros((A, A), dtype=np.float64)
                idx = 0
                for r in range(A):
                    for c in range(r, A):
                        if r == c:
                            GxiP[r, r] = grad_sigma[idx]
                        else:
                            GxiP[r, c] = GxiP[c, r] = 0.5 * grad_sigma[idx]
                        idx += 1

            # --- kernel-scale path: xisubq -> covker2 -> xi3 -> wker ---
            if self.n_kern > 0:
                # chain to covker2, then to xi3 = W inner W
                Gcov = Msubq.T @ GxiP @ Msubq                # (nc, nc)
                Gxi3 = N.T @ Gcov @ N                        # (nc-1, nc-1)
                YW = kstate.kernel_inner @ kstate.wdiagker
                # dP/dwker_full[m] = 2 (inner W Gxi3)[m,m]  (product rule on W..W)
                dPdwker = 2.0 * np.diag(YW @ Gxi3)           # (nc-1,)
                dlnPdwker = dPdwker / P
                dlogp_dkparams[i] = kstate.dwker_dxscal @ dlnPdwker  # (n_kern,)

            # --- joint-correlation path (copula): d lnP / d omegastar ---
            # Two contributions funnel through ``omegastar`` while ``errbeta3``
            # (``rc[i]``) is held fixed -- the kernel's direct dependence via the
            # conditional covariance ``xi2subq`` and the conditional mean
            # ``B1subq`` (both from ``condition``). Emitted over the free
            # off-diagonal (correlation) elements of the FULL ``nrndtot``
            # matrix, row-based ``vecndup`` order.
            if self.copula:
                # (a) covariance: xisubq = J xi2subq J', J = M N W.  Convert the
                # symmetric d lnP / d xi2subq into a ``vecdup`` cotangent (the
                # off-diagonals count twice) and chain via ``gcondnewcov``.
                W = kstate.wdiagker
                Jmat = Msubq @ N @ W                         # (A, nc-1)
                G2 = Jmat.T @ (GxiP / P) @ Jmat              # (nc-1, nc-1)
                cov_bar = np.array(
                    [G2[aa, bb] if aa == bb else 2.0 * G2[aa, bb]
                     for (aa, bb) in vech_pairs],
                    dtype=np.float64,
                )                                            # (n_vech,)
                dom = kstate.gxcov_domega @ cov_bar          # (n_free,)

                # (b) mean: B1subq = X12 X11^{-1} errbeta3 depends on omegastar
                # (holding ``errbeta3`` fixed) -- ``gcondnewmean``'s gX.
                b1_bar = N.T @ dV                            # (nc-1,) d lnP / d B1
                g_full = np.concatenate([rc[i], np.zeros(kd, dtype=np.float64)])
                _gy, _gmu, gxmean_domega, _gg = gcondnewmean(
                    np.eye(kd), np.zeros(nrndtot), kstate.omegastar,
                    g_full, kstate.indxrand,
                    cholesky=False, condcov=False,
                )                                            # gX (n_free, nc-1)
                dom = dom + gxmean_domega @ b1_bar           # (n_free,)
                dlogp_domega[i] = dom

        return KernelObsResult(
            p_obs=p_obs,
            dlogp_dV=dlogp_dV,
            dlogp_dkparams=dlogp_dkparams,
            dlogp_drc=dlogp_drc,
            dlogp_domega=dlogp_domega,
        )


__all__ = ["MvncdKernel", "MvncdKernelState"]
