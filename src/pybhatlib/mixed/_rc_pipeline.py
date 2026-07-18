"""Random-coefficient realization pipeline for the mixed MSL engine.

This module implements the byte-identical inner transform of the GAUSS mixed
choice drivers (``MIXMNL.gss`` ``lpr``/``lgd``, lines ~429-602), **batched over
observations**.  Given a person's standard-normal draws already broadcast to the
observation level (``errbeta1``) and the realized reparameterization state
(:class:`~pybhatlib.mixed._reparam.RCState` produced by :meth:`prepare`), the
:meth:`RandomCoefPipeline.realize` step reproduces, for every observation ``q``:

    errbeta3   = x11chol' @ errbeta2                                 (correlate)
    ftemprand  = standyjinvnonpgiven(xlamrnd, mu, sig, errbeta3)     (typed YJ)
    f1rand     = wdiagrand @ ftemprand    (no-log)                   (scale)
               = poslog @ exp(wlog .* poslog' @ ftemprand)
                 + wdiagrand @ posnolog @ posnolog' @ ftemprand      (lognormal)
    xmunew1    = xmu + indxrndvar @ f1rand                           (additive)
    xmunew2    = xmu .* (indxrndvar @ f1rand)                        (multiplicative)
    xmunew     = xmunew1 .* indxvarnonegposlog
                 + xmunew2 .* indxvarnegposlog                       (inject)

together with every analytic Jacobian block that the engine's score assembly
(``MIXED_PANEL_MODELS_PLAN.md`` 2.3) chains back into the shared reparameter
gradient: ``dxmunewdxmu``, ``dxmunewdf1rand``, ``df1randdwdiagrand``,
``df1randdx11chol`` and ``df1randdxlamrnd``.

All Jacobian blocks are computed from the already-validated primitive gradients
(:func:`~pybhatlib.vecup._yj.gradstandyjinvnonpgiven`,
:func:`~pybhatlib.matgradient._corr_chol.ggradchol`) rather than re-derived here.
The two GAUSS matgradient helpers ``gradelproduct`` / ``gradelBproduct`` -- the
Jacobians of a Hadamard product ``a .* b`` w.r.t. ``a`` and ``b`` -- reduce to
``diag(b)`` and ``diag(a)`` respectively and are applied inline.

Batched shape convention (leading axis ``q`` indexes observations, ``K`` is
``nrndcoef``, ``nvarm`` is ``n_beta``, ``ncorr = K*(K-1)//2``):

======================  =====================  =============================
field                   shape                  meaning
======================  =====================  =============================
``errbeta3``            ``(Q, K)``             correlated normal draws
``ftemprand``           ``(Q, K)``             standardized YJ-inverse draws
``f1rand``              ``(Q, K)``             scaled random coefficients
``xmunew1``             ``(Q, nvarm)``         additive-injected coefficients
``xmunew2``             ``(Q, nvarm)``         multiplicative-injected
``xmunew``              ``(Q, nvarm)``         mask-combined coefficients
``jac.dxmunewdxmu``     ``(Q, nvarm, nvarm)``  ``[q,v,w] = d xmunew_w / d xmu_v``
``jac.dxmunewdf1rand``  ``(Q, K, nvarm)``      ``[q,r,v] = d xmunew_v / d f1rand_r``
``jac.df1randdwdiagrand`` ``(Q, K, K)``        ``[q,r,s] = d f1rand_s / d wscalrand_r``
``jac.df1randdx11chol`` ``(Q, ncorr, K)``      ``[q,k,r] = d f1rand_r / d chol_offdiag_k``
``jac.df1randdxlamrnd`` ``(Q, K, K)``          ``[q,r,s] = d f1rand_s / d xlamrnd_r``
======================  =====================  =============================

The Jacobian orientations match the ``einsum`` contractions in the engine's
``_assemble_score`` (plan 2.3) verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.matgradient._corr_chol import ggradchol
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout, RCState
from pybhatlib.vecup._yj import gradstandyjinvnonpgiven, standyjinvnonpgiven

if TYPE_CHECKING:  # avoid a runtime import cycle with the engine / model facades
    from pybhatlib.mixed._spec import MixingSpec


# ---------------------------------------------------------------------------
# Realization cache
# ---------------------------------------------------------------------------

@dataclass
class RCJacobian:
    """Analytic Jacobian blocks of the random-coefficient realization.

    All arrays carry a leading observation axis ``q`` (length ``Q``). The block
    orientations are chosen to match the engine ``_assemble_score`` contractions
    (``MIXED_PANEL_MODELS_PLAN.md`` 2.3):

    * ``dxmunewdxmu``   ``einsum('qvw,qw->qv', dxmunewdxmu, dxmunew)``
    * ``dxmunewdf1rand``  ``einsum('qrv,qv->qr', dxmunewdf1rand, dxmunew)``
    * ``df1randdwdiagrand`` ``einsum('qrs,qs->qr', df1randdwdiagrand, df1)``
    * ``df1randdx11chol`` ``einsum('qkr,qr->qk', df1randdx11chol, df1)``
    * ``df1randdxlamrnd`` ``einsum('qrs,qs->qr', df1randdxlamrnd, df1)``

    Attributes
    ----------
    dxmunewdxmu : ndarray, shape (Q, nvarm, nvarm)
        ``[q, v, w] = d xmunew[q, w] / d xmu[v]`` (diagonal in ``v, w``).
    dxmunewdf1rand : ndarray, shape (Q, nrndcoef, nvarm)
        ``[q, r, v] = d xmunew[q, v] / d f1rand[q, r]`` (obs-independent).
    df1randdwdiagrand : ndarray, shape (Q, nrndcoef, nrndcoef)
        ``[q, r, s] = d f1rand[q, s] / d wscalrand[r]``.
    df1randdx11chol : ndarray, shape (Q, ncorr, nrndcoef)
        ``[q, k, r] = d f1rand[q, r] / d (x11chol off-diagonal element k)``,
        with ``ncorr = nrndcoef*(nrndcoef-1)//2`` in row-based upper-triangular
        order (empty when ``nrndcoef <= 1``).
    df1randdxlamrnd : ndarray, shape (Q, nrndcoef, nrndcoef)
        ``[q, r, s] = d f1rand[q, s] / d xlamrnd[r]``.
    gerrbeta3dx11chol : ndarray, shape (Q, ncorr, nrndcoef)
        ``[q, k, r] = d errbeta3[q, r] / d (x11chol off-diagonal element k)``
        -- the *pre-Yeo-Johnson / pre-scale* correlated-draw sensitivity to the
        random-coefficient Cholesky (``ggradchol(I, x11chol, errbeta1)``). This
        is the copula path's route from the kernel's ``dlogp_drc``
        (``= d lnP / d errbeta3``) to the correlation parameters, distinct from
        ``df1randdx11chol`` which routes the *utility* path through ``f1rand``.
        Empty (``ncorr = 0``) when ``nrndcoef <= 1``.
    """

    dxmunewdxmu: NDArray
    dxmunewdf1rand: NDArray
    df1randdwdiagrand: NDArray
    df1randdx11chol: NDArray
    df1randdxlamrnd: NDArray
    gerrbeta3dx11chol: NDArray


@dataclass
class RealizationCache:
    """Every intermediate of one MSL draw's random-coefficient realization.

    Produced by :meth:`RandomCoefPipeline.realize`. The gradient block bundle
    ``jac`` is populated only when ``want_grad=True`` and is ``None`` otherwise.

    Attributes
    ----------
    errbeta3 : ndarray, shape (Q, nrndcoef)
        Correlated standard-normal draws, ``errbeta1 @ x11chol``.
    ftemprand : ndarray, shape (Q, nrndcoef)
        Standardized inverse-YJ draws (identity for normal coefficients).
    f1rand : ndarray, shape (Q, nrndcoef)
        Scaled random coefficients.
    xmunew1 : ndarray, shape (Q, n_beta)
        Additive-injected coefficient vector per observation.
    xmunew2 : ndarray, shape (Q, n_beta)
        Multiplicative-injected coefficient vector per observation.
    xmunew : ndarray, shape (Q, n_beta)
        Mask-combined coefficient vector consumed by the kernel utility.
    jac : RCJacobian or None
        Analytic Jacobian blocks (``None`` when ``want_grad=False``).
    """

    errbeta3: NDArray
    ftemprand: NDArray
    f1rand: NDArray
    xmunew1: NDArray
    xmunew2: NDArray
    xmunew: NDArray
    jac: Optional[RCJacobian] = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RandomCoefPipeline:
    """Batched random-coefficient realization + analytic Jacobian blocks.

    Wraps an :class:`~pybhatlib.mixed._reparam.EstimationSpace` for the
    ``theta``-to-:class:`~pybhatlib.mixed._reparam.RCState` reparameterization
    (:meth:`prepare`) and implements the per-draw inner transform
    (:meth:`realize`) that maps a broadcast draw matrix ``errbeta1`` to the
    injected coefficient vector ``xmunew`` and its Jacobian blocks.

    Parameters
    ----------
    spec : MixingSpec
        Index masks and random-coefficient counts.
    layout : ParamLayout
        ``[beta | rcor | scal | lam | kern]`` partition of ``theta``.
    spher : bool, default False
        Spherical (True) vs radial (False) correlation Cholesky. Only the radial
        path is implemented (delegated to :class:`EstimationSpace`).
    scal : float, default 1.0
        Scaling constant in ``newcholparmscaled``.
    intordn1 : int, default 20
        Gauss-Hermite node count for ``meanyj``.
    """

    def __init__(
        self,
        spec: "MixingSpec",
        layout: ParamLayout,
        *,
        spher: bool = False,
        scal: float = 1.0,
        intordn1: int = 20,
    ) -> None:
        self.spec = spec
        self.layout = layout
        self.spher = bool(spher)
        self.scal = float(scal)
        self.intordn1 = int(intordn1)
        self._space = EstimationSpace(
            layout, scal=scal, intordn1=intordn1, spher=spher
        )

    @property
    def n_rnd(self) -> int:
        """Number of random coefficients (``spec.nrndcoef``)."""
        return self.spec.nrndcoef

    def prepare(self, theta: NDArray, *, want_grad: bool = False) -> RCState:
        """Realize the ``theta``-independent-of-draw reparameterization state.

        Delegates to :meth:`EstimationSpace.unpack`: builds the correlation
        Cholesky ``x11chol = newcholparmscaled(xrand, scal)``, the scale vector
        ``wscalrand = exp(xscalrand)``, the Yeo-Johnson powers
        ``xlamrnd = 2*cdlogit(xlam)``, the ``meanyj`` mean/std ``(mu, sig)`` and,
        when ``want_grad``, the correlation-Cholesky gradient blocks
        (``gker``/``gscal``/``gtempstar``) and the scale/lambda reparameter
        Jacobians consumed by the engine score.

        Parameters
        ----------
        theta : ndarray, shape (n_theta,)
            Optimizer parameter vector in ``[beta|rcor|scal|lam|kern]`` order.
        want_grad : bool, default False
            Populate the gradient (``d*``) fields of the returned state
            (required before calling :meth:`realize` with ``want_grad=True``).

        Returns
        -------
        RCState
        """
        return self._space.unpack(theta, self.spec, want_grad=want_grad)

    def realize(
        self,
        errbeta1: NDArray,
        rc: RCState,
        *,
        want_grad: bool = True,
        expand: Optional[NDArray] = None,
    ) -> RealizationCache:
        """Batched inner transform of one draw plus its Jacobian blocks.

        Reproduces ``MIXMNL.gss`` ``lpr``/``lgd`` (lines ~433-602) over all
        observations at once.

        The entire realization is *row-wise* in the leading axis: every output
        row (``errbeta3``/``ftemprand``/``f1rand``/``xmunew`` and each Jacobian
        block) depends only on the corresponding input row ``errbeta1[q]`` and
        the draw-invariant reparameterization state ``rc``. For panel data the
        engine broadcasts per-person draws to the observation level
        (``panel.broadcast``), so ``errbeta1`` has *identical* rows within a
        person and the realization is redundant across a person's occasions.
        Passing ``expand=panel.row_to_person`` lets the caller supply
        ``errbeta1`` at *person* granularity (``n_ind`` rows); the realization +
        Jacobians are then computed once per unique person and broadcast to the
        observation level via ``field[expand]``. Because the transform is
        row-wise this is bit-identical to computing at observation granularity
        and cross-sectional models (``n_ind == n_obs``, identity ``expand``) are
        a no-op.

        Parameters
        ----------
        errbeta1 : ndarray, shape (Q, nrndcoef)
            Standard-normal draws. When ``expand is None`` these are already at
            the observation level (``Dmask @ errbeta1temp``, one row per obs).
            When ``expand`` is supplied they are at the *person* level
            (``n_ind`` rows) and the outputs are broadcast to ``len(expand)``
            observation rows.
        rc : RCState
            Reparameterization state from :meth:`prepare`. When
            ``want_grad=True`` its gradient fields (``dmulamrnddxlamrnd``,
            ``dsiglamrnddxlamrnd``) must be populated.
        want_grad : bool, default True
            Compute and attach the :class:`RCJacobian` block bundle.
        expand : ndarray of int, optional
            Row-to-person index (``panel.row_to_person``). When given,
            ``errbeta1`` is at person granularity and each output field's
            leading axis is expanded to the observation level via ``[expand]``
            (exactly reproducing ``panel.broadcast``).

        Returns
        -------
        RealizationCache

        Raises
        ------
        ValueError
            If ``want_grad`` is requested but ``rc`` lacks gradient fields, or if
            ``errbeta1`` does not have ``nrndcoef`` columns.
        """
        xp = array_namespace(errbeta1)  # noqa: F841 - backend parity hook
        spec = self.spec
        K = spec.nrndcoef
        nvarm = spec.n_beta

        errbeta1 = np.asarray(errbeta1, dtype=np.float64)
        if errbeta1.ndim != 2 or errbeta1.shape[1] != K:
            raise ValueError(
                f"errbeta1 must have shape (Q, {K}); got {errbeta1.shape}"
            )
        Q = errbeta1.shape[0]

        wscalrand = np.asarray(rc.wscalrand, dtype=np.float64)  # (K,)
        xlamrnd = np.asarray(rc.xlamrnd, dtype=np.float64)      # (K,)
        mu = np.asarray(rc.mulamrnd, dtype=np.float64)          # (K,)
        sig = np.asarray(rc.siglamrnd, dtype=np.float64)        # (K,)
        xmu = np.asarray(rc.xmu, dtype=np.float64)              # (nvarm,)
        indxrndvar = np.asarray(spec.indxrndvar, dtype=np.float64)  # (nvarm, K)
        nonegmask = np.asarray(spec.indxvarnonegposlog, dtype=np.float64)  # (nvarm,)
        negmask = np.asarray(spec.indxvarnegposlog, dtype=np.float64)      # (nvarm,)

        # --- correlate: errbeta3[q] = x11chol' @ errbeta1[q] ------------------
        if K > 1:
            x11chol = np.asarray(rc.x11chol, dtype=np.float64)  # (K, K)
            errbeta3 = errbeta1 @ x11chol                       # (Q, K)
        else:
            x11chol = np.asarray(rc.x11chol, dtype=np.float64)  # scalar 1.0
            errbeta3 = errbeta1 * float(x11chol)

        # --- typed YJ / normal transform (univariate per coefficient) --------
        if want_grad:
            if rc.dmulamrnddxlamrnd is None or rc.dsiglamrnddxlamrnd is None:
                raise ValueError(
                    "realize(want_grad=True) requires prepare(want_grad=True) "
                    "so the meanyj gradient fields are populated"
                )
            gmuyj = np.asarray(rc.dmulamrnddxlamrnd, dtype=np.float64)
            gsigyj = np.asarray(rc.dsiglamrnddxlamrnd, dtype=np.float64)
            F, glam, gx = gradstandyjinvnonpgiven(
                xlamrnd, mu, sig, gmuyj, gsigyj, errbeta3.T
            )
            ftemprand = F.T          # (Q, K)
            glam = glam.T            # (Q, K)  d ftemprand / d xlamrnd (diag)
            gx = gx.T                # (Q, K)  d ftemprand / d errbeta3 (diag)
        else:
            ftemprand = standyjinvnonpgiven(xlamrnd, mu, sig, errbeta3.T).T
            glam = gx = None

        # --- scale: f1rand (lognormal vs plain scale branch) -----------------
        f1rand, df1randdftemprand_full = self._scale(
            ftemprand, wscalrand, want_grad=want_grad
        )

        # --- inject: xmunew (additive / multiplicative by mask) --------------
        # B[q, v] = (indxrndvar @ f1rand[q]) ; per-obs random contribution.
        B = f1rand @ indxrndvar.T                       # (Q, nvarm)
        xmunew1 = xmu[None, :] + B                       # (Q, nvarm)
        xmunew2 = xmu[None, :] * B                       # (Q, nvarm)
        xmunew = xmunew1 * nonegmask[None, :] + xmunew2 * negmask[None, :]

        cache = RealizationCache(
            errbeta3=errbeta3,
            ftemprand=ftemprand,
            f1rand=f1rand,
            xmunew1=xmunew1,
            xmunew2=xmunew2,
            xmunew=xmunew,
        )
        if not want_grad:
            if expand is not None:
                cache = self._expand_cache(cache, expand)
            return cache

        cache.jac = self._jacobians(
            Q=Q,
            errbeta1=errbeta1,
            x11chol=x11chol,
            ftemprand=ftemprand,
            f1rand=f1rand,
            B=B,
            xmu=xmu,
            wscalrand=wscalrand,
            indxrndvar=indxrndvar,
            nonegmask=nonegmask,
            negmask=negmask,
            glam=glam,
            gx=gx,
            df1randdftemprand_full=df1randdftemprand_full,
        )
        if expand is not None:
            cache = self._expand_cache(cache, expand)
        return cache

    @staticmethod
    def _expand_cache(
        cache: RealizationCache, expand: NDArray
    ) -> RealizationCache:
        """Broadcast a person-granularity cache to the observation level.

        Every field's leading axis is indexed by ``expand``
        (``panel.row_to_person``), reproducing ``panel.broadcast`` exactly.
        Because :meth:`realize` is row-wise this is bit-identical to having
        computed the realization at observation granularity.
        """
        idx = np.asarray(expand, dtype=np.intp)
        jac = cache.jac
        new_jac = None
        if jac is not None:
            new_jac = RCJacobian(
                dxmunewdxmu=jac.dxmunewdxmu[idx],
                dxmunewdf1rand=jac.dxmunewdf1rand[idx],
                df1randdwdiagrand=jac.df1randdwdiagrand[idx],
                df1randdx11chol=jac.df1randdx11chol[idx],
                df1randdxlamrnd=jac.df1randdxlamrnd[idx],
                gerrbeta3dx11chol=jac.gerrbeta3dx11chol[idx],
            )
        return RealizationCache(
            errbeta3=cache.errbeta3[idx],
            ftemprand=cache.ftemprand[idx],
            f1rand=cache.f1rand[idx],
            xmunew1=cache.xmunew1[idx],
            xmunew2=cache.xmunew2[idx],
            xmunew=cache.xmunew[idx],
            jac=new_jac,
        )

    # ------------------------------------------------------------------
    # internal transform / gradient helpers
    # ------------------------------------------------------------------

    def _scale(
        self, ftemprand: NDArray, wscalrand: NDArray, *, want_grad: bool
    ) -> tuple[NDArray, Optional[NDArray]]:
        """Scale branch: ``f1rand`` and (per-obs) ``d f1rand / d ftemprand``.

        Reproduces ``MIXMNL.gss`` lines 437-443 / 560-582. For the plain-scale
        (no log-normal coefficient) branch ``f1rand = wdiagrand @ ftemprand`` and
        ``d f1rand / d ftemprand = diag(wscalrand)`` (obs-independent). The
        log-normal branch mixes an ``exp``-scaled block (log coefficients) with
        the plain-scaled block (remaining coefficients).

        Parameters
        ----------
        ftemprand : ndarray, shape (Q, K)
            Standardized inverse-YJ draws.
        wscalrand : ndarray, shape (K,)
            Scale (std-dev) vector.
        want_grad : bool
            If True also return the per-observation ``d f1rand / d ftemprand``
            Jacobian stack, shape ``(Q, K, K)``.

        Returns
        -------
        f1rand : ndarray, shape (Q, K)
        df1randdftemprand : ndarray or None, shape (Q, K, K)
        """
        spec = self.spec
        Q, K = ftemprand.shape

        if spec.nrndlog == 0:
            f1rand = ftemprand * wscalrand[None, :]
            if not want_grad:
                return f1rand, None
            df = np.zeros((Q, K, K), dtype=np.float64)
            di = np.arange(K)
            df[:, di, di] = wscalrand[None, :]           # diag(wscalrand)
            return f1rand, df

        # --- log-normal branch (MIXMNL 437-440 / 560-576) --------------------
        poslog = np.asarray(spec.poslog, dtype=np.float64)      # (K, nlog)
        posnolog = np.asarray(spec.posnolog, dtype=np.float64)  # (K, nnolog)
        wlog = poslog.T @ wscalrand                             # (nlog,)
        pnp = posnolog @ posnolog.T                             # (K, K)

        hold1 = ftemprand @ poslog                              # (Q, nlog)
        hold2 = np.exp(wlog[None, :] * hold1)                   # (Q, nlog)
        # log block + plain-scaled non-log block
        f1rand = hold2 @ poslog.T + (ftemprand @ pnp) * wscalrand[None, :]
        if not want_grad:
            return f1rand, None

        # d f1rand / d ftemprand, per observation (MIXMNL 562-572), vectorized
        # over ``q`` (identical batched-matmul semantics as the per-q loop):
        #   dhold2[q] = poslog @ diag(hold2[q] * wlog) @ poslog'
        # non-log (plain-scaled) block: wdiagrand @ posnolog @ posnolog'
        base = np.diag(wscalrand) @ pnp                         # (K, K)
        scaled = hold2 * wlog[None, :]                          # (Q, nlog)
        tmp = scaled[:, :, None] * poslog.T[None, :, :]         # (Q, nlog, K)
        df = poslog @ tmp                                       # (Q, K, K)
        df += base[None, :, :]
        return f1rand, df

    def _jacobians(
        self,
        *,
        Q: int,
        errbeta1: NDArray,
        x11chol: NDArray,
        ftemprand: NDArray,
        f1rand: NDArray,
        B: NDArray,
        xmu: NDArray,
        wscalrand: NDArray,
        indxrndvar: NDArray,
        nonegmask: NDArray,
        negmask: NDArray,
        glam: NDArray,
        gx: NDArray,
        df1randdftemprand_full: Optional[NDArray],
    ) -> RCJacobian:
        """Assemble the five analytic Jacobian blocks (batched over obs).

        See the module docstring / :class:`RCJacobian` for exact orientations.
        """
        spec = self.spec
        K = spec.nrndcoef
        nvarm = spec.n_beta
        ncorr = K * (K - 1) // 2
        di = np.arange(K)

        # --- dxmunewdxmu (diagonal in v, w): d xmunew_w / d xmu_v -------------
        # xmunew_w = nonegmask_w*(xmu_w + B_w) + negmask_w*(xmu_w * B_w)
        # d/d xmu_w = nonegmask_w + negmask_w * B_w  (diagonal).
        diag_xmu = nonegmask[None, :] + negmask[None, :] * B     # (Q, nvarm)
        dxmunewdxmu = np.zeros((Q, nvarm, nvarm), dtype=np.float64)
        dv = np.arange(nvarm)
        dxmunewdxmu[:, dv, dv] = diag_xmu

        # --- dxmunewdf1rand: d xmunew_v / d f1rand_r (obs-independent) --------
        # [r, v] = indxrndvar[v, r] * (nonegmask_v + negmask_v * xmu_v)
        coef_v = nonegmask + negmask * xmu                      # (nvarm,)
        dxmunewdf1rand_2d = indxrndvar.T * coef_v[None, :]      # (K, nvarm)
        dxmunewdf1rand = np.broadcast_to(
            dxmunewdf1rand_2d, (Q, K, nvarm)
        ).copy()

        # --- df1randdwdiagrand: d f1rand_s / d wscalrand_r --------------------
        df1randdwdiagrand = self._df1_dwdiag(
            Q=Q, ftemprand=ftemprand, wscalrand=wscalrand
        )

        # --- df1randdftemprand per obs (const in no-log branch) --------------
        if df1randdftemprand_full is None:
            df1_dftemp = np.zeros((K, K), dtype=np.float64)
            df1_dftemp[di, di] = wscalrand                     # diag(wscalrand)

        # --- df1randdxlamrnd: d f1rand_s / d xlamrnd_r ------------------------
        # ftemprand is univariate in xlamrnd -> dftempranddxlamrnd = diag(glam);
        # diag(glam[q]) @ df1_dftemp_q is the row-scaling glam[q][:,None]*M, so
        # the whole block is vectorized over ``q`` with a single elementwise mul.
        if df1randdftemprand_full is not None:
            df1randdxlamrnd = glam[:, :, None] * df1randdftemprand_full
        else:
            df1randdxlamrnd = glam[:, :, None] * df1_dftemp[None, :, :]

        # --- df1randdx11chol: d f1rand_r / d chol_offdiag_k -------------------
        df1randdx11chol = np.zeros((Q, ncorr, K), dtype=np.float64)
        # --- gerrbeta3dx11chol: d errbeta3_r / d chol_offdiag_k (copula path) -
        gerrbeta3dx11chol = np.zeros((Q, ncorr, K), dtype=np.float64)

        if ncorr > 0:
            eyeK = np.eye(K)                                   # hoisted (obs-invariant)
            for q in range(Q):
                df1_dftemp_q = (
                    df1randdftemprand_full[q]
                    if df1randdftemprand_full is not None
                    else df1_dftemp
                )                                              # (K, K)
                # gerrbeta3dx11chol = ggradchol(I, x11chol, errbeta1) (ncorr, K)
                # -- d errbeta3 / d(x11chol off-diagonal); errbeta3 = x11chol' e.
                gerr = ggradchol(
                    eyeK, x11chol, errbeta1[q],
                    cholcov=False, diagL=False,
                )
                gerrbeta3dx11chol[q] = gerr                    # (ncorr, K)
                # dftempranddx11chol = gerr @ diag(gx[q]) ; then @ df1/dftemp.
                dftempranddx11chol_q = gerr * gx[q][None, :]   # (ncorr, K)
                df1randdx11chol[q] = dftempranddx11chol_q @ df1_dftemp_q

        return RCJacobian(
            dxmunewdxmu=dxmunewdxmu,
            dxmunewdf1rand=dxmunewdf1rand,
            df1randdwdiagrand=df1randdwdiagrand,
            df1randdx11chol=df1randdx11chol,
            df1randdxlamrnd=df1randdxlamrnd,
            gerrbeta3dx11chol=gerrbeta3dx11chol,
        )

    def _df1_dwdiag(
        self, *, Q: int, ftemprand: NDArray, wscalrand: NDArray
    ) -> NDArray:
        """``d f1rand_s / d wscalrand_r`` block, shape ``(Q, K, K)``.

        Plain-scale branch (MIXMNL 579): ``diag(ftemprand)`` per observation.
        Log-normal branch (MIXMNL 573-574): the ``exp``-scaled block contributes
        through ``wlog = poslog' @ wscalrand`` in addition to the plain-scaled
        block's ``diag(posnolog @ posnolog' @ ftemprand)``.
        """
        spec = self.spec
        K = spec.nrndcoef
        di = np.arange(K)
        df = np.zeros((Q, K, K), dtype=np.float64)

        if spec.nrndlog == 0:
            df[:, di, di] = ftemprand                          # diag(ftemprand)
            return df

        poslog = np.asarray(spec.poslog, dtype=np.float64)      # (K, nlog)
        posnolog = np.asarray(spec.posnolog, dtype=np.float64)  # (K, nnolog)
        wlog = poslog.T @ wscalrand                             # (nlog,)
        pnp = posnolog @ posnolog.T                             # (K, K)
        hold1 = ftemprand @ poslog                              # (Q, nlog)
        hold2 = np.exp(wlog[None, :] * hold1)                   # (Q, nlog)
        nonlog_diag = (ftemprand @ pnp)                         # (Q, K)
        # Vectorized over ``q`` (same batched-matmul semantics as the per-q
        # loop): d(poslog @ hold2)/d wlog = poslog @ diag(hold2[q]*hold1[q]);
        # lift wlog -> wscalrand via poslog on the input side, then add the
        # plain-scaled non-log block on the diagonal.
        scaled = hold2 * hold1                                  # (Q, nlog)
        tmp = scaled[:, :, None] * poslog.T[None, :, :]         # (Q, nlog, K)
        df = poslog @ tmp                                       # (Q, K, K)
        df[:, di, di] += nonlog_diag
        return df
