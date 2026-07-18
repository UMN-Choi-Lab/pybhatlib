"""Rectangle-MVNCD (``pdfrectn``) kernel with rc<->kernel copula for the
mixed-panel MORP (multivariate ordered response) model.

Implements the :class:`~pybhatlib.mixed._kernel.MixedKernel` protocol for the
GAUSS *Joint Ordered YJ with Cross-Sectional or Panel Random Coefficients*
driver (``0714_UTA_request/Joint Ordered YJ ...gss`` ``lpr`` lines ~700-741).
Per MSL replication the engine forms the systematic ordinal utilities
``Vsub = X @ xmunew`` (one column per ordinal dimension) and hands them, with
the drawn random coefficients, to :meth:`RectMvncdKernel.probability`, which
returns the observed-category probability and the gradient paths the engine
chains back through the shared random-coefficient reparameterization Jacobians.

Kernel math (one obs ``i``, one draw ``r`` -- GAUSS lines ~700-741)
-------------------------------------------------------------------
Each ordinal dimension ``d`` maps its observed category ``y_d`` to a rectangle
region ``[xdown_d, xup_d]`` (middle category), a lower orthant ``(-inf, xgg_d]``
(lowest category) or an upper orthant ``[xgg_d, +inf)`` (highest category),
whose bounds are threshold cut points ``tau_d`` (from the ordered-increment
``thresh`` block). With ``Bsubq = Vsub[i]`` (the ordinal utility mean, all
``nord`` dimensions available) the kernel builds the *centred, kernel-scaled*
thresholds

**NORMAL kernel** (``yj_kernel=False``, GAUSS ``_normker == 1``; ``sig == 1``)::

    tzgg  = sig .* (xgg  - Bsubq)
    tzlow = sig .* (xdown - Bsubq)
    tzup  = sig .* (xup  - Bsubq)

**Yeo-Johnson kernel** (``yj_kernel=True``, GAUSS ``_normker == 0``)::

    zgg  = scsiglamker .* (xgg  - Bsubq) + mulamker;  tzgg  = yjnonp(xlamker, zgg)
    zlow = scsiglamker .* (xdown - Bsubq) + mulamker; tzlow = yjnonp(xlamker, zlow)
    zup  = scsiglamker .* (xup  - Bsubq) + mulamker;  tzup  = yjnonp(xlamker, zup)

with ``xlamker = 2 cdlogit(kernlam)`` and ``(mulamker, siglamker) =
meanyj(xlamker)``, ``scsiglamker = siglamker`` (kernel scale ``wker`` is fixed
to ones for the ordered response, so ``inv(wdiagker) == I``).

**Copula** -- the joint ``(rc, kernel-error)`` vector is ``N(0, omegastar)`` over
``nrndtot = nrndcoef + nord`` elements. Conditioning on the drawn (correlated)
random coefficients ``errbeta3`` (``rc_draw``) gives the kernel-error law
``N(B3subq, xi2subq)`` (GAUSS ``condition``; reused verbatim from the MNP
copula seam)::

    { B3subq, xi2subq } = condition(I_nord, 0, omegastar, errbeta3, indxmarg)

with ``indxmarg = [ones(nrndcoef); zeros(nord)]``.  The observed-category
probability is the routed rectangle-MVNCD of that conditioned law::

    P = pdfrectn(B3subq, xi2subq, tzgg, tzlow, tzup, indxone, indxcomp, indxeq)

``xi2subq`` and ``m_cond = X12 X11^{-1}`` are draw-independent (built once in
:meth:`prepare`); only ``B3subq = m_cond @ errbeta3`` is draw-dependent.  When
``copula=False`` the conditional mean vanishes (``B3subq = 0``) and the kernel
covariance is the unconditional ordinal block ``omegastar[nord, nord]``.

Gradients (GAUSS ``lgd`` ``gradpdfrectn`` chain)
------------------------------------------------
:func:`~pybhatlib.gradmvn._pdfrectn.gradpdfrectn` returns
``(P, gmu, gcov, gxg, gx1, gx2)`` -- ``dP/d`` mean / covariance-``vech`` /
``xgg`` / ``xlow`` / ``xup``.  The kernel fills all five
:class:`~pybhatlib.mixed._kernel.KernelObsResult` fields:

* ``dlogp_dV`` -- the thresholds all depend on ``Bsubq = Vsub`` (via
  ``- Bsubq``), chained through the kernel transform;
* ``dlogp_dkparams`` -- ``[thresh | kernlam]``: ``gxg / gx1 / gx2`` chained
  through the kernel transform back to the threshold cut points (``thresh``
  block, via the ordered-increment Jacobian) and, for the YJ kernel, back to
  ``xlamker`` (``kernlam`` block, both the direct ``yjnonp`` lambda dependence
  and the ``meanyj`` mean/scale dependence);
* ``dlogp_drc`` -- ``m_cond' @ gmu`` (the ``errbeta3`` sensitivity of
  ``B3subq``), zero when ``copula=False``;
* ``dlogp_domega`` -- ``gcov`` routed through ``gcondnewcov`` (conditional
  covariance) plus ``gmu`` routed through ``gcondnewmean`` (conditional mean),
  over the free off-diagonal correlation elements of the full ``omegastar`` in
  row-based ``vecndup`` order -- the **same** MNP copula seam layout. ``None``
  when ``copula=False``.

``p_obs`` is always sourced from ``gradpdfrectn`` when gradients are requested
(and from ``pdfrectn`` otherwise); the two agree to ~1e-12, so an analytic
gradient tracks the finite difference of the identical probability.

No module-level mutable state; ``xp`` is accepted for backend parity
(computation is NumPy, matching the analytic-gradient house style).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from pybhatlib.gradmvn._pdfrectn import gradpdfrectn, pdfrectn
from pybhatlib.matgradient._corr_chol import matndupdiagonefull
from pybhatlib.matgradient._radial import newcholparmscaled
from pybhatlib.mixed._copula import condition, gcondnewcov, gcondnewmean
from pybhatlib.mixed._kernel import KernelObsResult
from pybhatlib.mixed._reparam import thresh_reparam
from pybhatlib.utils._logistic import cdlogit, pdlogit
from pybhatlib.vecup._yj import gradmeanyj, gyjnonp, meanyj, yjnonp

# Yeo-Johnson kernel power ``xlamker`` lives in the open interval ``(0, 2)``.
# ``cdlogit`` saturates to exactly 1.0 for arguments above ~37, which would push
# ``2 * cdlogit`` to exactly 2.0 (or 0.0 in the opposite tail) and make the YJ
# helpers divide by ``(2 - lam) == 0``. Clamp just inside the open interval.
_LAM_EPS = 1e-8


@dataclass(frozen=True)
class RectMvncdKernelState:
    """Per-evaluation state built by :meth:`RectMvncdKernel.prepare`.

    All fields are constant across the MSL draws; only the conditional mean
    ``B3subq = m_cond @ errbeta3`` is draw-dependent and is formed inside
    :meth:`RectMvncdKernel.probability`.

    Attributes
    ----------
    omegastar : NDArray, shape (nrndtot, nrndtot)
        Joint random-coefficient + ordinal-kernel correlation matrix
        (``nrndtot = nrndcoef + nord``).
    tau : NDArray, shape (nthresh,)
        Ordered threshold cut points, concatenated across ordinal dimensions.
    dtau : NDArray, shape (nthresh, nthresh)
        Jacobian ``d tau / d thresh`` (block-diagonal; from ``thresh_reparam``).
    xi2subq : NDArray, shape (nord, nord)
        Conditional (copula) kernel-error covariance -- or the unconditional
        ordinal block ``omegastar[nord, nord]`` when ``copula=False``.
    m_cond : NDArray, shape (nord, nrndcoef)
        Conditional-mean sensitivity ``X12 X11^{-1}``; zeros when
        ``copula=False`` or ``nrndcoef == 0``.
    indxmarg : NDArray, shape (nrndtot,)
        Marginal-set indicator ``[ones(nrndcoef); zeros(nord)]`` selecting the
        random-coefficient (conditioning) block of ``omegastar``.
    gxcov_domega : NDArray or None, shape (n_free, nord*(nord+1)//2)
        Draw-independent gradient of ``vech(xi2subq)`` w.r.t. the free
        off-diagonal correlation elements of ``omegastar`` (``gcondnewcov`` gX),
        rows in ``vecndup`` order. ``None`` when ``copula=False``.
    xlamker : NDArray or None, shape (nord,)
        Yeo-Johnson kernel-error power parameters (``2 cdlogit(kernlam)``);
        ``None`` for the normal kernel.
    kernlam : NDArray or None, shape (nord,)
        Raw ``kernlam`` block (needed for the ``d xlamker / d kernlam`` chain);
        ``None`` for the normal kernel.
    mulamker : NDArray or None, shape (nord,)
        ``meanyj`` mean of the standardized inverse YJ; ``None`` for the normal
        kernel.
    scsiglamker : NDArray or None, shape (nord,)
        ``meanyj`` std (``siglamker``, since ``wker == 1``); ``None`` for the
        normal kernel.
    gmulamker, gsiglamker : NDArray or None, shape (nord,)
        ``d mulamker / d xlamker`` and ``d scsiglamker / d xlamker`` from
        ``gradmeanyj``; ``None`` for the normal kernel.
    """

    omegastar: NDArray
    tau: NDArray
    dtau: NDArray
    xi2subq: NDArray
    m_cond: NDArray
    indxmarg: NDArray
    gxcov_domega: Optional[NDArray] = None
    xlamker: Optional[NDArray] = None
    kernlam: Optional[NDArray] = None
    mulamker: Optional[NDArray] = None
    scsiglamker: Optional[NDArray] = None
    gmulamker: Optional[NDArray] = None
    gsiglamker: Optional[NDArray] = None


class RectMvncdKernel:
    """Rectangle-MVNCD kernel with optional rc<->kernel copula (MORP).

    Parameters
    ----------
    nord : int
        Number of ordinal outcome dimensions (GAUSS ``nord``); the kernel
        consumes ``nord`` systematic-utility columns.
    nrndcoef : int
        Number of random coefficients; the joint correlation is sized over
        ``nrndtot = nrndcoef + nord``.
    n_categories : sequence of int, shape (nord,)
        Observed ordinal category counts per dimension (GAUSS ``ncord``); each
        dimension contributes ``n_categories[d] - 1`` threshold cut points.
    copula : bool, default True
        If ``True`` condition the ordinal kernel errors on the drawn random
        coefficients (GAUSS ``condition``) and emit ``dlogp_drc`` /
        ``dlogp_domega``; if ``False`` use the unconditional ordinal covariance
        block and emit ``dlogp_drc = 0`` / ``dlogp_domega = None``.
        ``copula=True`` requires ``nrndcoef >= 1``.
    yj_kernel : bool, default False
        If ``True`` use the Yeo-Johnson kernel (GAUSS ``_normker == 0``) with an
        ``nord``-parameter ``kernlam`` block; if ``False`` use the standard
        normal kernel (GAUSS ``_normker == 1``; no ``kernlam`` block).
    scal : float, default 1.0
        Radial correlation-Cholesky scaling constant (must match the pipeline /
        space value); passed to ``newcholparmscaled``.
    intordn1 : int, default 20
        Gauss-Hermite node count for ``meanyj`` (YJ kernel only).
    reporting : bool, default False
        Parameter-space convention consumed by :meth:`prepare`.  ``False``
        (default) is the estimation (``lpr``) space: the ``thresh`` block holds
        raw cumulative increments (mapped through ``thresh_reparam``), the
        ``rcor`` block holds radial correlation-Cholesky parameters (mapped
        through ``newcholparmscaled``) and the ``kernlam`` block holds the
        unconstrained pre-image of the Yeo-Johnson power (mapped through
        ``2 * cdlogit``).  ``True`` is the reporting (``lpr1``) space matching
        :class:`~pybhatlib.mixed._reparam.ReportingSpace`: the ``thresh`` block
        *is* the ordered cut points, the ``rcor`` block *is* the off-diagonal
        joint correlation (mapped through ``matndupdiagonefull``) and the
        ``kernlam`` block *is* the Yeo-Johnson power in ``(0, 2)`` directly.  The
        reporting path is gradient-free (used only for prediction / ATE).

    Attributes
    ----------
    n_util : int
        ``nord`` (number of ordinal-utility columns consumed).
    """

    def __init__(
        self,
        nord: int,
        nrndcoef: int,
        n_categories: Any,
        *,
        copula: bool = True,
        yj_kernel: bool = False,
        scal: float = 1.0,
        intordn1: int = 20,
        reporting: bool = False,
        iid: bool = False,
        correst: Optional[NDArray] = None,
    ) -> None:
        if nord < 1:
            raise ValueError(f"nord must be >= 1, got {nord}")
        if nrndcoef < 0:
            raise ValueError(f"nrndcoef must be >= 0, got {nrndcoef}")
        if copula and nrndcoef < 1:
            raise ValueError("copula=True requires nrndcoef >= 1")
        n_categories = tuple(int(c) for c in n_categories)
        if len(n_categories) != nord:
            raise ValueError(
                f"n_categories must have length nord={nord}, got {len(n_categories)}"
            )
        if any(c < 2 for c in n_categories):
            raise ValueError(
                f"each n_categories entry must be >= 2, got {n_categories}"
            )
        self.n_util: int = int(nord)
        self.nord: int = int(nord)
        self.nrndcoef: int = int(nrndcoef)
        self.n_categories: tuple[int, ...] = n_categories
        self.numthresh: tuple[int, ...] = tuple(c - 1 for c in n_categories)
        self.n_thresh: int = int(sum(self.numthresh))
        self.copula: bool = bool(copula)
        self.yj_kernel: bool = bool(yj_kernel)
        self.n_kernlam: int = int(nord) if yj_kernel else 0
        self.scal: float = float(scal)
        self.intordn1: int = int(intordn1)
        self.reporting: bool = bool(reporting)
        self.iid: bool = bool(iid)
        if correst is not None:
            mask = np.asarray(correst, dtype=bool)
            if mask.shape != (nord, nord):
                raise ValueError(
                    f"correst must have shape {(nord, nord)}, got {mask.shape}"
                )
            if not np.array_equal(mask, mask.T) or not np.all(np.diag(mask)):
                raise ValueError("correst must be symmetric with a true diagonal")
            self.correst: Optional[NDArray] = mask
        else:
            self.correst = None
        self.nrndtot: int = int(nrndcoef) + int(nord)
        # per-dimension global offset of the threshold sub-block in ``tau``.
        offs = [0]
        for m in self.numthresh:
            offs.append(offs[-1] + m)
        self._thresh_offset: tuple[int, ...] = tuple(offs[:-1])
        # Per-evaluation cache of the draw-invariant per-obs rectangle regions
        # (``_obs_region`` depends only on ``obs.y_ord`` and ``kst.tau``, both
        # constant across the engine's ``n_rep`` ``probability`` calls). Keyed by
        # the *identity* of ``(obs, tau)``: a new evaluation rebuilds ``kst`` with
        # a fresh ``tau`` array, invalidating the cache. Holding references to
        # ``obs``/``tau`` prevents id reuse, so the identity check is sound; a
        # racing writer at worst forces a redundant recompute (the regions are a
        # pure function of the key), never a wrong result.
        self._region_cache: Optional[tuple[Any, Any, list]] = None

    # ------------------------------------------------------------------
    def kernel_param_names(self) -> list[str]:
        """Return the ``n_thresh + n_kernlam`` kernel-owned parameter names.

        Ordered ``[thresh ... | kernlam ...]``, matching the column order of
        :class:`~pybhatlib.mixed._kernel.KernelObsResult`'s ``dlogp_dkparams``.
        """
        names: list[str] = []
        for d, m in enumerate(self.numthresh):
            names.extend(f"thresh{d + 1:02d}_{j + 1:02d}" for j in range(m))
        if self.yj_kernel:
            names.extend(f"kernlam{d + 1:02d}" for d in range(self.nord))
        return names

    # ------------------------------------------------------------------
    def prepare(
        self, theta: NDArray, layout: Any, *, xp: Any = None
    ) -> RectMvncdKernelState:
        """Build the per-evaluation kernel state from the parameter vector.

        Reproduces the draw-independent setup of GAUSS ``lpr``: ``omegastar``
        from the joint correlation block (``newcholparmscaled``), the ordered
        threshold cut points from the ``thresh`` block, the conditional (copula)
        or unconditional ordinal covariance, the conditional-mean sensitivity
        ``m_cond``, and -- for the YJ kernel -- ``xlamker`` and the ``meanyj``
        mean/scale (with their ``xlamker`` gradients).

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Full parameter vector; the ``"thresh"``, ``"rcor"`` and (for the YJ
            kernel) ``"kernlam"`` slices are read.
        layout : ParamLayout
            Provides the named block slices.
        xp : module, optional
            Array backend (defaults to NumPy).

        Returns
        -------
        RectMvncdKernelState
        """
        if xp is None:
            xp = np
        theta = np.asarray(theta, dtype=np.float64)
        sl = layout.slices()

        bthresh = theta[sl["thresh"]] if "thresh" in sl else np.zeros(0)
        xrand = theta[sl["rcor"]]

        if self.reporting:
            # --- reporting (lpr1) space: natural parameters read directly ----
            # ``thresh`` already holds the ordered cut points, ``rcor`` the
            # off-diagonal joint correlation, ``kernlam`` the YJ power in (0, 2).
            tau = np.asarray(bthresh, dtype=np.float64).ravel()
            dtau = np.eye(tau.shape[0], dtype=np.float64)
            if layout.n_rcor == 0 and self.nrndtot > 1:
                omegastar = np.eye(self.nrndtot, dtype=np.float64)
            elif self.nrndtot > 1:
                omegastar = matndupdiagonefull(xrand)
            else:  # single random element -> unit correlation
                omegastar = np.ones((self.nrndtot, self.nrndtot), dtype=np.float64)
            kernlam = None
            xlamker = None
            if self.yj_kernel:
                xlamker = np.clip(
                    np.asarray(theta[sl["kernlam"]], dtype=np.float64),
                    _LAM_EPS, 2.0 - _LAM_EPS,
                )
                kernlam = xlamker
            return self.state_from_parts(omegastar, tau, dtau, xlamker, kernlam, xp=xp)

        # --- estimation (lpr) space -----------------------------------------
        # --- ordered thresholds (thresh block) ------------------------------
        tau, dtau = thresh_reparam(bthresh, self.numthresh, want_grad=True)

        # --- joint correlation omegastar (rcor block) -----------------------
        if layout.n_rcor == 0 and self.nrndtot > 1:
            omegastar = np.eye(self.nrndtot, dtype=np.float64)
        elif self.nrndtot > 1:
            cholall = newcholparmscaled(xrand, self.scal)
            omegastar = np.asarray(cholall).T @ np.asarray(cholall)
        else:  # single random element -> unit correlation
            omegastar = np.ones((self.nrndtot, self.nrndtot), dtype=np.float64)

        # --- YJ kernel-lam block (kernlam) ----------------------------------
        kernlam = None
        xlamker = None
        if self.yj_kernel:
            kernlam = np.asarray(theta[sl["kernlam"]], dtype=np.float64)
            xlamker = np.clip(2.0 * cdlogit(kernlam), _LAM_EPS, 2.0 - _LAM_EPS)

        return self.state_from_parts(omegastar, tau, dtau, xlamker, kernlam, xp=xp)

    # ------------------------------------------------------------------
    def state_from_parts(
        self,
        omegastar: NDArray,
        tau: NDArray,
        dtau: NDArray,
        xlamker: Optional[NDArray] = None,
        kernlam: Optional[NDArray] = None,
        *,
        xp: Any = None,
    ) -> RectMvncdKernelState:
        """Build the kernel state from explicit ``omegastar`` / thresholds / lam.

        Splits the state construction out of :meth:`prepare` so the
        joint-correlation finite-difference gate can perturb ``omegastar``
        directly (bypassing the radial reparameterization) and rebuild the exact
        state the kernel consumes.

        Parameters
        ----------
        omegastar : NDArray, shape (nrndtot, nrndtot)
            Joint random-coefficient + ordinal-kernel correlation matrix.
        tau : NDArray, shape (nthresh,)
            Ordered threshold cut points (from ``thresh_reparam``).
        dtau : NDArray, shape (nthresh, nthresh)
            Jacobian ``d tau / d thresh``.
        xlamker : NDArray or None, shape (nord,)
            YJ kernel-error powers (``None`` for the normal kernel).
        kernlam : NDArray or None, shape (nord,)
            Raw ``kernlam`` block (``None`` for the normal kernel).
        xp : module, optional
            Array backend (defaults to NumPy).

        Returns
        -------
        RectMvncdKernelState
        """
        if xp is None:
            xp = np
        omegastar = np.asarray(omegastar, dtype=np.float64)
        k = self.nrndcoef
        nord = self.nord
        nrndtot = self.nrndtot
        ordinal = omegastar[k:, k:].copy()
        if self.iid:
            ordinal = np.eye(nord, dtype=np.float64)
        elif self.correst is not None:
            ordinal = np.where(self.correst, ordinal, 0.0)
            np.fill_diagonal(ordinal, 1.0)
        omegastar = omegastar.copy()
        omegastar[k:, k:] = ordinal
        tau = np.asarray(tau, dtype=np.float64).ravel()
        dtau = np.asarray(dtau, dtype=np.float64)

        indxmarg = np.concatenate(
            [np.ones(k, dtype=np.float64), np.zeros(nord, dtype=np.float64)]
        )

        gxcov_domega = None
        if self.copula:
            # xi2subq is draw-independent (COVB in condition); dummy zero draw.
            _b3_dummy, xi2subq = condition(
                np.eye(nord), np.zeros(nrndtot), omegastar,
                np.zeros(k), indxmarg,
            )                                                    # (nord, nord)
            X11 = omegastar[:k, :k]
            X12 = omegastar[k:, :k]
            m_cond = X12 @ np.linalg.inv(X11)                    # (nord, nrndcoef)
            _gy_cov, gxcov_domega = gcondnewcov(
                np.eye(nord), omegastar, indxmarg,
                cholesky=False, condcov=False,
            )                                                    # (n_free, n_vech)
        else:
            xi2subq = omegastar[k:, k:].copy()                   # (nord, nord)
            m_cond = np.zeros((nord, k), dtype=np.float64)

        # --- YJ kernel: meanyj mean/scale and their xlamker gradients --------
        mulamker = scsiglamker = gmulamker = gsiglamker = None
        if self.yj_kernel:
            xlamker = np.asarray(xlamker, dtype=np.float64).ravel()
            mulamker, scsiglamker, gmulamker, gsiglamker = gradmeanyj(
                xlamker, self.intordn1
            )
            # wker == ones(nord) => inv(wdiagker) == I => scsiglamker == siglamker.

        return RectMvncdKernelState(
            omegastar=omegastar,
            tau=tau,
            dtau=dtau,
            xi2subq=xi2subq,
            m_cond=m_cond,
            indxmarg=indxmarg,
            gxcov_domega=gxcov_domega,
            xlamker=xlamker if self.yj_kernel else None,
            kernlam=kernlam if self.yj_kernel else None,
            mulamker=mulamker,
            scsiglamker=scsiglamker,
            gmulamker=gmulamker,
            gsiglamker=gsiglamker,
        )

    # ------------------------------------------------------------------
    def _obs_region(
        self, y_obs: NDArray, tau: NDArray
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray, list[list[tuple[int, float]]]]:
        """Map one obs's observed categories to the routed rectangle region.

        Returns per-dimension raw thresholds and routing indicators plus, for
        each dimension, the list of ``(global_tau_index, dP_source)`` couplings
        used to chain the threshold gradient back to the ``thresh`` block. The
        ``dP_source`` tag is ``0`` for ``xgg``, ``1`` for ``xlow`` and ``2`` for
        ``xup`` (which of ``gxg / gx1 / gx2`` feeds that cut point).
        """
        nord = self.nord
        raw_xg = np.zeros(nord, dtype=np.float64)
        raw_xlow = np.zeros(nord, dtype=np.float64)
        raw_xup = np.zeros(nord, dtype=np.float64)
        indxone = np.zeros(nord, dtype=np.int64)
        indxcomp = np.zeros(nord, dtype=np.int64)
        tau_map: list[list[tuple[int, int]]] = []

        for d in range(nord):
            j = int(y_obs[d])
            ncat = self.n_categories[d]
            off = self._thresh_offset[d]
            ncut = self.numthresh[d]                             # ncat - 1
            couplings: list[tuple[int, int]] = []
            if j <= 0:                                           # lowest category
                indxone[d] = 1
                indxcomp[d] = 0
                raw_xg[d] = tau[off + 0]
                couplings.append((off + 0, 0))                  # xgg
            elif j >= ncat - 1:                                 # highest category
                indxone[d] = 1
                indxcomp[d] = 1
                raw_xg[d] = tau[off + ncut - 1]
                couplings.append((off + ncut - 1, 0))           # xgg
            else:                                               # middle category
                indxone[d] = 0
                indxcomp[d] = 0
                raw_xlow[d] = tau[off + j - 1]
                raw_xup[d] = tau[off + j]
                couplings.append((off + j - 1, 1))              # xlow
                couplings.append((off + j, 2))                  # xup
            tau_map.append(couplings)

        indxeq = np.zeros(nord, dtype=np.int64)                 # pure ordinal
        return raw_xg, raw_xlow, raw_xup, indxone, indxcomp, indxeq, tau_map

    # ------------------------------------------------------------------
    def _obs_regions(self, obs: Any, y_ord: NDArray, tau: NDArray) -> list:
        """Per-obs :meth:`_obs_region` outputs, cached across the draw loop.

        ``_obs_region`` is draw-invariant (a function of ``obs.y_ord`` and
        ``tau`` only), yet the engine calls :meth:`probability` once per MSL
        replication with the same ``obs`` and ``kstate``. Compute the region
        list once per evaluation and reuse it across the ``n_rep`` calls, keyed
        by the identity of ``(obs, tau)`` (see :attr:`_region_cache`).
        """
        cache = self._region_cache
        if cache is not None and cache[0] is obs and cache[1] is tau:
            return cache[2]
        regions = [self._obs_region(y_ord[i], tau) for i in range(len(y_ord))]
        self._region_cache = (obs, tau, regions)
        return regions

    # ------------------------------------------------------------------
    def predict_shares(
        self,
        Vsub: NDArray,
        obs: Any,
        kstate: RectMvncdKernelState,
        *,
        rc_draw: NDArray,
        xp: Any = None,
    ) -> NDArray:
        """Per-observation marginal ordinal category probabilities for one draw.

        Lifts the shipped fixed-coefficient MORP prediction formulation
        (:func:`pybhatlib.models.morp._morp_forecast.morp_predict`) into the
        shared mixed predictor: for each observation the ordinal utilities
        ``Vsub[i]`` centre the (kernel-transformed) threshold cut points, and
        each dimension's marginal category distribution is the univariate normal
        rectangle of the conditional kernel-error law
        ``N(B3subq[d], xi2subq[d, d])`` (the marginal of the joint
        rectangle-MVNCD used by :meth:`probability`).  The shared predictor
        (:func:`pybhatlib.mixed._predict.default_kernel_predict`) calls this once
        per MSL replication and averages over draws, integrating out the mixing
        distribution.

        Because a marginal of a multivariate normal ignores the off-diagonal
        correlation, the per-dimension category probabilities depend only on the
        conditional mean ``B3subq[d]`` and the conditional variance
        ``xi2subq[d, d]`` -- so with ``copula=False`` and no random coefficients
        this reduces *exactly* to the fixed-coefficient MORP marginal (the
        collapse gate).

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nord)
            Systematic ordinal utilities per observation and ordinal dimension
            (``X @ xmunew`` for this replication).
        obs : Any
            Per-observation bundle (unused: prediction returns every category's
            probability, not only the observed one).
        kstate : RectMvncdKernelState
            State from :meth:`prepare` (thresholds ``tau``, conditional
            covariance ``xi2subq``, conditional-mean sensitivity ``m_cond`` and,
            for the Yeo-Johnson kernel, the ``meanyj`` mean/scale and
            ``xlamker``).
        rc_draw : NDArray, shape (n_obs, nrndcoef)
            Drawn (correlated) random coefficients ``errbeta3`` entering the
            copula conditional mean; ignored when ``copula=False``.
        xp : module, optional
            Array backend (defaults to NumPy).

        Returns
        -------
        NDArray, shape (n_obs, sum(n_categories))
            Per-observation marginal category probabilities, concatenated across
            ordinal dimensions in ``dep_vars`` order.  Each dimension's block
            sums to one over its categories.
        """
        if xp is None:
            xp = np
        Vsub = np.asarray(Vsub, dtype=np.float64)
        n_obs = Vsub.shape[0]
        nord = self.nord
        tau = np.asarray(kstate.tau, dtype=np.float64)
        xi2subq = np.asarray(kstate.xi2subq, dtype=np.float64)
        sd = np.sqrt(np.maximum(np.diag(xi2subq), 1e-30))         # (nord,)
        total_cats = int(sum(self.n_categories))

        rc = np.asarray(rc_draw, dtype=np.float64)
        if rc.ndim == 1:
            rc = rc.reshape(n_obs, -1)

        yj = self.yj_kernel
        out = np.zeros((n_obs, total_cats), dtype=np.float64)
        for i in range(n_obs):
            if self.copula:
                B3 = kstate.m_cond @ rc[i]                        # (nord,)
            else:
                B3 = np.zeros(nord, dtype=np.float64)
            col = 0
            for d in range(nord):
                ncat = self.n_categories[d]
                off = self._thresh_offset[d]
                mu_d = Vsub[i, d]
                sd_d = sd[d]
                # Cumulative CDF at each interior cut point -> category probs
                # via telescoping (the upper bound of category j is the lower
                # bound of category j+1); the outer bounds are +/- infinity.
                cdf_prev = 0.0
                for j in range(ncat):
                    if j == ncat - 1:
                        cdf_hi = 1.0
                    else:
                        gap = tau[off + j] - mu_d
                        if yj:
                            z = kstate.scsiglamker[d] * gap + kstate.mulamker[d]
                            tz = float(
                                yjnonp(kstate.xlamker[d:d + 1],
                                       np.array([z], dtype=np.float64))[0]
                            )
                        else:
                            tz = gap
                        cdf_hi = float(ndtr((tz - B3[d]) / sd_d))
                    out[i, col] = max(0.0, cdf_hi - cdf_prev)
                    cdf_prev = cdf_hi
                    col += 1
        if xp is not np:
            out = xp.asarray(out)
        return out

    # ------------------------------------------------------------------
    def probability(
        self,
        Vsub: NDArray,
        obs: Any,
        kstate: RectMvncdKernelState,
        *,
        rc_draw: NDArray,
        want_grad: bool,
    ) -> KernelObsResult:
        """Observed-category rectangle-MVNCD probability and gradients per draw.

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nord)
            Systematic ordinal utilities per observation and ordinal dimension.
        obs : Any
            Per-observation bundle exposing ``y_ord`` (``(n_obs, nord)`` 0-based
            observed ordinal categories; GAUSS ``yiord``).
        kstate : RectMvncdKernelState
            State from :meth:`prepare`.
        rc_draw : NDArray, shape (n_obs, nrndcoef)
            Drawn (correlated) random coefficients ``errbeta3`` entering the
            copula conditional mean; ignored when ``copula=False``.
        want_grad : bool
            If ``False`` the gradient fields are correctly-shaped zeros.

        Returns
        -------
        KernelObsResult
            ``p_obs`` ``(n_obs,)``, ``dlogp_dV`` ``(n_obs, nord)``,
            ``dlogp_dkparams`` ``(n_obs, n_thresh + n_kernlam)``, ``dlogp_drc``
            ``(n_obs, nrndcoef)`` (``= d lnP / d errbeta3``), and
            ``dlogp_domega`` ``(n_obs, nrndtot*(nrndtot-1)//2)`` in ``vecndup``
            order (``None`` when ``copula=False``).
        """
        Vsub = np.asarray(Vsub, dtype=np.float64)
        y_ord = np.asarray(obs.y_ord, dtype=np.int64)
        n_obs = Vsub.shape[0]
        k = self.nrndcoef
        nord = self.nord
        nrndtot = self.nrndtot
        n_free = nrndtot * (nrndtot - 1) // 2
        n_kparams = self.n_thresh + self.n_kernlam

        rc = np.asarray(rc_draw, dtype=np.float64)
        if rc.ndim == 1:
            rc = rc.reshape(n_obs, -1)

        tau = kstate.tau
        dtau = kstate.dtau
        xi2subq = kstate.xi2subq
        yj = self.yj_kernel
        if yj:
            xlamker = kstate.xlamker
            mulamker = kstate.mulamker
            scsiglamker = kstate.scsiglamker
            gmulamker = kstate.gmulamker
            gsiglamker = kstate.gsiglamker
            dxlamker_dkernlam = 2.0 * pdlogit(kstate.kernlam)     # (nord,)

        p_obs = np.zeros(n_obs, dtype=np.float64)
        dlogp_dV = np.zeros((n_obs, nord), dtype=np.float64)
        dlogp_dkparams = np.zeros((n_obs, n_kparams), dtype=np.float64)
        dlogp_drc = np.zeros((n_obs, k), dtype=np.float64)
        dlogp_domega = (
            np.zeros((n_obs, n_free), dtype=np.float64)
            if (self.copula and want_grad)
            else None
        )

        zero_seed = 0.0
        regions = self._obs_regions(obs, y_ord, tau)
        for i in range(n_obs):
            raw_xg, raw_xlow, raw_xup, indxone, indxcomp, indxeq, tau_map = regions[i]
            Bsubq = Vsub[i]                                      # (nord,) all avail

            # --- kernel transform: raw thresholds -> tz --------------------
            gap_gg = raw_xg - Bsubq
            gap_low = raw_xlow - Bsubq
            gap_up = raw_xup - Bsubq
            if not yj:
                # normal kernel: sig == 1
                tzgg = gap_gg
                tzlow = gap_low
                tzup = gap_up
            else:
                zgg = scsiglamker * gap_gg + mulamker
                zlow = scsiglamker * gap_low + mulamker
                zup = scsiglamker * gap_up + mulamker
                tzgg = yjnonp(xlamker, zgg)
                tzlow = yjnonp(xlamker, zlow)
                tzup = yjnonp(xlamker, zup)

            # --- conditional kernel-error law (copula) ---------------------
            if self.copula:
                B3subq = kstate.m_cond @ rc[i]                  # (nord,)
            else:
                B3subq = np.zeros(nord, dtype=np.float64)

            if not want_grad:
                P, _s = pdfrectn(
                    B3subq, xi2subq, tzgg, tzlow, tzup, zero_seed,
                    indxone, indxcomp, indxeq,
                )
                p_obs[i] = max(float(P), 1e-300)
                continue

            P, gmu, gcov, gxg, gx1, gx2, _s = gradpdfrectn(
                B3subq, xi2subq, tzgg, tzlow, tzup, zero_seed,
                indxone, indxcomp, indxeq,
            )
            P = max(float(P), 1e-300)
            p_obs[i] = P
            invP = 1.0 / P

            # d tz / d(kernel arg z) and d tz / d(xlamker), per bound.
            if not yj:
                # tz = raw - Bsubq  => d tz/d gap = 1 (sig == 1).
                dtzgg_dgap = np.ones(nord)
                dtzlow_dgap = np.ones(nord)
                dtzup_dgap = np.ones(nord)
            else:
                _f, glam_gg, gx_gg = gyjnonp(xlamker, zgg)
                _f, glam_low, gx_low = gyjnonp(xlamker, zlow)
                _f, glam_up, gx_up = gyjnonp(xlamker, zup)
                dyj_dz_gg = np.diag(gx_gg)
                dyj_dz_low = np.diag(gx_low)
                dyj_dz_up = np.diag(gx_up)
                dyj_dlam_gg = np.diag(glam_gg)
                dyj_dlam_low = np.diag(glam_low)
                dyj_dlam_up = np.diag(glam_up)
                # d z / d gap = scsiglamker  => d tz / d gap = dyj_dz * scsiglamker
                dtzgg_dgap = dyj_dz_gg * scsiglamker
                dtzlow_dgap = dyj_dz_low * scsiglamker
                dtzup_dgap = dyj_dz_up * scsiglamker

            # --- utility path: d lnP / d Vsub (gap = raw - Bsubq) ----------
            # d gap / d Bsubq = -1 ; only the active bound(s) per dim nonzero.
            dP_dV = -(gxg * dtzgg_dgap + gx1 * dtzlow_dgap + gx2 * dtzup_dgap)
            dlogp_dV[i] = dP_dV * invP

            # --- kernel-owned params: thresholds (thresh block) ------------
            # d tz / d(raw threshold) = + d tz / d gap (gap = raw - Bsubq).
            dP_draw_xg = gxg * dtzgg_dgap
            dP_draw_xlow = gx1 * dtzlow_dgap
            dP_draw_xup = gx2 * dtzup_dgap
            dP_dtau = np.zeros(self.n_thresh, dtype=np.float64)
            for d in range(nord):
                for (g_idx, src) in tau_map[d]:
                    if src == 0:
                        dP_dtau[g_idx] += dP_draw_xg[d]
                    elif src == 1:
                        dP_dtau[g_idx] += dP_draw_xlow[d]
                    else:
                        dP_dtau[g_idx] += dP_draw_xup[d]
            dP_dthresh = dtau.T @ dP_dtau                        # (n_thresh,)
            dlogp_dkparams[i, : self.n_thresh] = dP_dthresh * invP

            # --- kernel-owned params: kernlam block (YJ only) --------------
            if yj:
                # d z / d xlamker = d scsiglamker/d xlamker * gap + d mulamker/d xlamker
                dz_dlam_gg = gsiglamker * gap_gg + gmulamker
                dz_dlam_low = gsiglamker * gap_low + gmulamker
                dz_dlam_up = gsiglamker * gap_up + gmulamker
                dtzgg_dlam = dyj_dlam_gg + dyj_dz_gg * dz_dlam_gg
                dtzlow_dlam = dyj_dlam_low + dyj_dz_low * dz_dlam_low
                dtzup_dlam = dyj_dlam_up + dyj_dz_up * dz_dlam_up
                dP_dxlamker = (
                    gxg * dtzgg_dlam + gx1 * dtzlow_dlam + gx2 * dtzup_dlam
                )                                                # (nord,)
                dP_dkernlam = dP_dxlamker * dxlamker_dkernlam
                dlogp_dkparams[i, self.n_thresh :] = dP_dkernlam * invP

            # --- copula path: errbeta3 -> B3subq ---------------------------
            if self.copula:
                dlogp_drc[i] = (kstate.m_cond.T @ gmu) * invP

                # --- joint-correlation path: d lnP / d omegastar -----------
                # (a) conditional covariance xi2subq: gcov already carries the
                # off-diagonal doubling (vecdup w/ doubling), gcondnewcov's gX is
                # raw vecdup, so dP/domega = gX @ gcov (see MNP seam).
                dom = kstate.gxcov_domega @ (np.asarray(gcov) * invP)  # (n_free,)
                # (b) conditional mean B3subq (errbeta3 held fixed).
                g_full = np.concatenate([rc[i], np.zeros(nord, dtype=np.float64)])
                _gy, _gmu, gxmean_domega, _gg = gcondnewmean(
                    np.eye(nord), np.zeros(nrndtot), kstate.omegastar,
                    g_full, kstate.indxmarg,
                    cholesky=False, condcov=False,
                )                                                # gX (n_free, nord)
                dom = dom + gxmean_domega @ (np.asarray(gmu) * invP)
                dlogp_domega[i] = dom

        return KernelObsResult(
            p_obs=p_obs,
            dlogp_dV=dlogp_dV,
            dlogp_dkparams=dlogp_dkparams,
            dlogp_drc=dlogp_drc,
            dlogp_domega=dlogp_domega,
        )


__all__ = ["RectMvncdKernel", "RectMvncdKernelState"]
