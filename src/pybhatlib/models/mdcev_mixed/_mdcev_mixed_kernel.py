"""Logit-Jacobian kernel for the mixed MDCEV MSL engine (Phase 3).

The mixed MDCEV model is the *simplest* of the hard mixed kernels: it mixes over
the **baseline-utility** ``beta`` only, and has **no copula**. Per draw ``r`` the
engine forms the drawn baseline utility ``Vsub = X @ xmunew`` (the same
random-coefficient pipeline as MixMNL); this kernel then feeds ``Vsub`` --
together with the (non-mixed) translation ``gamma`` and the MDCEV kernel scale --
through the **shipped, GAUSS-parity** MDCEV satiation/utility machinery
(:func:`pybhatlib.models.mdcev._mdcev_loglik._compute_satiation_block` /
:func:`~pybhatlib.models.mdcev._mdcev_loglik._compute_utility_terms_from_sat`)
and a batched, masked closed form of the MDCEV partial multivariate-logistic
likelihood to obtain the observed-outcome probability and its gradients.

Vectorization (identity baseline design, batched over obs)
----------------------------------------------------------
The baseline utility ``v[i, :] == Vsub[i, :]`` is the drawn utility itself (the
identity baseline design ``ivm``, ``nvarm == nc``), so ``d v / d beta`` is the
identity and the ``d(.) / d beta`` gradient columns are exactly ``d p / d Vsub``
-- **no division by ``Vsub``** is required. Rather than dispatch the shipped
scalar evaluator once per observation, the kernel evaluates **all** ``n_obs``
observations in a single batched pass (``e1 == n_obs``): the draw-invariant
satiation block and the ``v``/``vdisc``/``vcont`` builder run vectorized once,
and the three consumption branches (some / all / none inside goods consumed) --
which share the *same* standard partial multivariate-logistic closed form -- are
evaluated together over the obs axis by masking consumed vs non-consumed inside
goods (``nonpdfmvlogit`` / ``noncdfmvlogit`` are that form with ``K2 == 0`` /
``K1 == 0``). This replaces the ~15000 per-draw scalar logit dispatches with a
handful of ``(n_obs, nc)`` array ops. The satiation design (``ivg``) is the real
per-observation ``gamma`` design supplied via ``obs``.

Score convention (RAW derivatives)
----------------------------------
Because the engine runs the MDCEV kernel under ``score_convention="divide"``
(GAUSS ``gcomp ./ Pobs``, ``MDCEV.gss`` ``lgd`` line ~812), the four gradient
fields carry **raw probability derivatives** ``d p_obs / d(.)`` (GAUSS
``gcomp``), *not* log-derivatives; the engine divides the assembled ``gcomp`` by
``p_obs`` once to obtain ``d log p_obs / d theta``. The batched closed form
produces the raw derivative ``grad_raw == d p`` directly (the shipped
``lgd``-body ``grad_raw / z`` log-derivative times ``z == p_obs`` cancels), so no
explicit ``* p_obs`` step is needed. This is the one kernel whose ``dlogp_*``
fields are pre-divide (the softmax / MVNCD / rectangle kernels return
log-derivatives and run under ``score_convention="mask"``).

No copula
---------
``dlogp_drc`` is exactly zero (the drawn coefficients enter only through
``Vsub``, never a kernel covariance) and ``dlogp_domega`` is ``None``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pybhatlib.mixed._kernel import KernelObsResult, KernelState
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev._mdcev_loglik import (
    _compute_satiation_block,
    _compute_utility_terms_from_sat,
)


class LogitJacobianKernel:
    """MDCEV logit-Jacobian kernel (kernel-owned ``gamma`` + kernel scale, no copula).

    Implements :class:`~pybhatlib.mixed._kernel.MixedKernel`. The kernel wraps the
    shipped MDCEV log-likelihood / gradient as a per-draw evaluator using an
    identity baseline design (see the module docstring).

    Parameters
    ----------
    n_util : int
        Number of MDCEV alternatives ``nc`` (including the outside good). The
        kernel consumes ``Vsub`` of shape ``(n_obs, nc)``.
    nvargam : int
        Number of translation / satiation (``gamma``) parameters (GAUSS
        ``nvargam``). The kernel-owned parameter block is ``[gamma | kern-scale]``
        of width ``nvargam + 1``.
    control : MDCEVControl, optional
        MDCEV specification control. Only ``utility`` (``"trad"`` / ``"linear"``)
        and ``outside_good_gamma`` are read here; defaults to a traditional
        MDCEV control (``utility="trad"``, ``outside_good_gamma=-1000``).
    eqmatgam : NDArray, optional
        Gamma restriction / equality matrix, shape ``(nvargam, nvargam)``;
        defaults to the identity (GAUSS ``eqmatgam = eye(nvargam)``).

    Attributes
    ----------
    n_util : int
        Number of alternatives ``nc``.
    nvargam : int
        Number of gamma parameters.

    Notes
    -----
    Parameter-block placement (via :class:`~pybhatlib.mixed._reparam.ParamLayout`
    built with ``n_gamma=nvargam``, ``n_kern=1``, ``kern_before_scal=True``):
    the physical ``theta`` order is ``[beta | gamma | rcor | kern | scal | lam]``.
    The kernel reads its own ``gamma`` and ``kern`` (scale) slices in
    :meth:`prepare`; ``dlogp_dkparams`` columns follow that physical order
    ``[gamma (nvargam) | kern-scale (1)]``, which is exactly how the engine's
    ``_assemble_score`` stripes the kernel-owned (non-core) blocks.
    """

    def __init__(
        self,
        n_util: int,
        nvargam: int,
        *,
        control: Optional[MDCEVControl] = None,
        eqmatgam: Optional[NDArray] = None,
    ) -> None:
        self.n_util: int = int(n_util)
        self.nvargam: int = int(nvargam)
        self.control: MDCEVControl = control or MDCEVControl(utility="trad")
        if eqmatgam is None:
            eqmatgam = np.eye(self.nvargam, dtype=np.float64)
        self.eqmatgam: NDArray = np.asarray(eqmatgam, dtype=np.float64)
        if self.eqmatgam.shape != (self.nvargam, self.nvargam):
            raise ValueError(
                f"eqmatgam must be ({self.nvargam}, {self.nvargam}); "
                f"got {self.eqmatgam.shape}"
            )
        self._build_indices()

        # ---- per-evaluation caches (D2/D3) -------------------------------
        # D2: the synthetic per-obs ``dta`` matrix is fully draw-invariant --
        # only ``beta = Vsub[i]`` (which lives in ``x_q``, not ``dta``) varies
        # across MSL draws -- and depends only on ``obs``, so it is built once
        # and reused for every draw and every evaluation. Keyed by ``obs``
        # identity (held by reference to defend against id() reuse).
        self._dta_cache_obs: Any = None
        self._dta_all: Optional[NDArray] = None
        # D3: the draw-invariant satiation/gamma block depends on ``obs`` and the
        # per-eval ``(gamma_raw, log_sigma)`` (carried by the kernel state), so it
        # is rebuilt once per evaluation and reused across the ``n_rep`` draws.
        # It is a single batched (``e1 == n_obs``) block (D4), not a per-obs list.
        self._sat_cache_kstate: Any = None
        self._sat_cache_obs: Any = None
        self._sat_all: Optional[dict] = None

    # ------------------------------------------------------------------
    # constant synthetic-design index layout (identity baseline design)
    # ------------------------------------------------------------------

    def _build_indices(self) -> None:
        """Precompute the constant column layout of the per-obs synthetic ``dta``.

        Column layout (single-observation ``dta`` row):

        ==================  ================================  ==================
        columns             content                           consumed by
        ==================  ================================  ==================
        ``0 : nc``          consumption quantities            ``flagchm``
        ``nc : 2nc``        prices                            ``flagprcm``
        ``2nc``             observation weight (``1.0``)      ``wtind``
        ``2nc+1 : 2nc+1+nc*nc``  identity baseline design     ``ivm``
        ``... : ... + nc*nvargam``  satiation (gamma) design  ``ivg``
        ==================  ================================  ==================

        ``ivm`` is a fixed one-hot block so that ``v[0, k] == beta_k`` and
        ``d v / d beta`` is the identity; ``ivg`` values are filled per
        observation from ``obs.gamma_design``.
        """
        nc = self.n_util
        ng = self.nvargam

        self._flagchm = np.arange(0, nc, dtype=np.intp)
        self._flagprcm = np.arange(nc, 2 * nc, dtype=np.intp)
        self._wtind = int(2 * nc)

        base_ivm = 2 * nc + 1
        # ivm[j*nc + k] points to column (base_ivm + j*nc + k); the identity
        # template puts 1.0 at that column iff j == k.
        self._ivm = base_ivm + np.arange(nc * nc, dtype=np.intp)
        idblock = np.zeros(nc * nc, dtype=np.float64)
        for j in range(nc):
            idblock[j * nc + j] = 1.0
        self._idblock = idblock
        self._base_ivm = int(base_ivm)

        base_ivg = base_ivm + nc * nc
        self._ivg = base_ivg + np.arange(nc * ng, dtype=np.intp)
        self._base_ivg = int(base_ivg)

        self._ncols = int(base_ivg + nc * ng)

        # Factorials 0!..(nc-1)! for the batched partial-CDF closed form: the
        # standard partial multivariate-logistic CDF carries a ``K1!`` factor
        # where ``K1`` is the number of consumed inside goods (0..nc-1).
        facts = np.ones(nc, dtype=np.float64)
        for k in range(2, nc):
            facts[k] = facts[k - 1] * k
        self._facts = facts

    # ------------------------------------------------------------------
    # MixedKernel protocol
    # ------------------------------------------------------------------

    def kernel_param_names(self) -> list[str]:
        """Kernel-owned parameter names in ``dlogp_dkparams`` column order.

        Returns the ``nvargam`` translation names followed by the single MDCEV
        kernel-scale name, i.e. ``["gamma0", ..., "gamma{n-1}", "logsig_ker"]``.
        The length is ``nvargam + 1`` (the ``dlogp_dkparams`` width), which the
        engine stripes across the physical ``gamma`` and ``kern`` layout blocks.
        """
        return [f"gamma{j}" for j in range(self.nvargam)] + ["logsig_ker"]

    def prepare(self, theta: NDArray, layout: Any) -> KernelState:
        """Extract the kernel-owned ``gamma`` and kernel-scale from ``theta``.

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Full parameter vector in ``[beta | gamma | rcor | kern | scal | lam]``
            order.
        layout : ParamLayout
            Provides the ``gamma`` and ``kern`` slices (built with
            ``n_gamma=nvargam``, ``n_kern=1``, ``kern_before_scal=True``).

        Returns
        -------
        KernelState
            A namespace with ``gamma_raw`` (``(nvargam,)``) and ``log_sigma``
            (scalar) -- the MDCEV kernel-error log scale (GAUSS ``xscalker``).
        """
        theta = np.asarray(theta, dtype=np.float64)
        sl = layout.slices()
        gamma_raw = theta[sl["gamma"]].copy()
        kern_block = theta[sl["kern"]]
        log_sigma = float(kern_block[0]) if kern_block.shape[0] else 0.0
        return SimpleNamespace(gamma_raw=gamma_raw, log_sigma=log_sigma)

    def _get_dta_all(
        self,
        obs: Any,
        consumption: NDArray,
        price: NDArray,
        gamma_design: NDArray,
        n_obs: int,
        nc: int,
        ng: int,
    ) -> NDArray:
        """Build (or return cached) the draw-invariant synthetic ``dta`` matrix.

        The full ``(n_obs, ncols)`` synthetic design (consumption | price |
        weight | identity ``ivm`` | per-obs ``ivg``) depends only on ``obs`` and
        is constant across MSL draws, so it is built once and cached per ``obs``
        (D2). Row ``i`` is exactly the single-obs ``dta`` the pre-optimization
        code rebuilt every draw.
        """
        cached = self._dta_all
        if (
            self._dta_cache_obs is obs
            and cached is not None
            and cached.shape == (n_obs, self._ncols)
        ):
            return cached

        base_ivm = self._base_ivm
        base_ivg = self._base_ivg
        dta_all = np.zeros((n_obs, self._ncols), dtype=np.float64)
        dta_all[:, self._flagchm] = consumption
        dta_all[:, self._flagprcm] = price
        dta_all[:, self._wtind] = 1.0
        dta_all[:, base_ivm: base_ivm + nc * nc] = self._idblock
        # ivg[j*nc + k] value == gamma_design[i, k, j] == gamma_design[i].T.ravel();
        # batched over obs: transpose (nc, ng) -> (ng, nc) then flatten row-major.
        dta_all[:, base_ivg: base_ivg + nc * ng] = (
            gamma_design.transpose(0, 2, 1).reshape(n_obs, nc * ng)
        )

        self._dta_cache_obs = obs
        self._dta_all = dta_all
        return dta_all

    def _get_sat_all(
        self,
        kstate: KernelState,
        obs: Any,
        dta_all: NDArray,
        gamma_raw: NDArray,
        log_sigma: float,
        n_obs: int,
        nc: int,
        ng: int,
    ) -> dict:
        """Build (or return cached) the batched draw-invariant satiation block.

        The satiation/gamma block (:func:`_compute_satiation_block`) depends only
        on ``obs`` and the per-eval ``(gamma_raw, log_sigma)`` carried by
        ``kstate`` -- never on the drawn ``beta = Vsub`` -- so it is identical
        across the ``n_rep`` MSL draws of one evaluation (D3) and is computed
        once for **all** observations at ``e1 == n_obs`` (D4), reusing the
        already-vectorized :func:`_compute_satiation_block`. Cached by
        ``(kstate, obs)`` object identity: the engine builds one ``kstate`` per
        evaluation and threads the same object through every draw, so holding the
        reference both keys the cache and defends against id() reuse across
        evaluations.
        """
        if (
            self._sat_cache_kstate is kstate
            and self._sat_cache_obs is obs
            and self._sat_all is not None
            and self._sat_all["e1"] == n_obs
        ):
            return self._sat_all

        # ``beta`` is ignored by the satiation block; use a zero placeholder so
        # x[nvarm:] carries the (draw-invariant) gamma / log_sigma exactly as the
        # per-draw ``x_q`` did.
        x_sat = np.concatenate(
            [np.zeros(nc, dtype=np.float64), gamma_raw,
             np.array([log_sigma], dtype=np.float64)]
        )
        sat_all = _compute_satiation_block(
            x_sat, dta_all, self._ivg,
            self._flagchm, self._flagprcm, self._wtind,
            nc, ng, nc, self.eqmatgam, self.control,
        )

        self._sat_cache_kstate = kstate
        self._sat_cache_obs = obs
        self._sat_all = sat_all
        return sat_all

    def probability(
        self,
        Vsub: NDArray,
        obs: Any,
        kstate: KernelState,
        *,
        rc_draw: NDArray,
        want_grad: bool,
    ) -> KernelObsResult:
        """Observed-outcome probability and raw gradients for one MSL draw.

        Fully vectorized over the ``n_obs`` observations (D4/D5): a single
        batched ``e1 == n_obs`` evaluation of the satiation block and the
        v/vdisc/vcont builder, then a masked closed-form evaluation of the MDCEV
        partial multivariate-logistic likelihood and its analytic gradient over
        the whole obs axis (no per-observation Python loop).

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nc)
            Drawn baseline utilities (``X @ xmunew`` for this replication).
        obs : DesignData / SimpleNamespace
            Must expose ``consumption`` (``(n_obs, nc)`` quantities; column 0 the
            outside good), ``price`` (``(n_obs, nc)``; column 0 the numeraire),
            and ``gamma_design`` (``(n_obs, nc, nvargam)`` satiation covariates;
            ``gamma_design[i, k, j]`` is the covariate for obs ``i``, alternative
            ``k``, gamma parameter ``j``). The outside-good satiation row is
            ignored (forced to ``outside_good_gamma``).
        kstate : KernelState
            The ``(gamma_raw, log_sigma)`` namespace from :meth:`prepare`.
        rc_draw : NDArray, shape (n_obs, n_rnd)
            Drawn random coefficients; used only to size the zero ``dlogp_drc``
            (the MDCEV kernel has no copula).
        want_grad : bool
            If ``False``, ``dlogp_dV`` / ``dlogp_dkparams`` are returned as
            correctly-shaped zeros (``p_obs`` is still computed).

        Returns
        -------
        KernelObsResult
            ``p_obs`` ``(n_obs,)``; ``dlogp_dV`` ``(n_obs, nc)`` and
            ``dlogp_dkparams`` ``(n_obs, nvargam + 1)`` as **raw** probability
            derivatives (see the module docstring); ``dlogp_drc``
            ``(n_obs, n_rnd)`` zeros; ``dlogp_domega`` ``None``.
        """
        Vsub = np.asarray(Vsub, dtype=np.float64)
        n_obs, nc = Vsub.shape
        if nc != self.n_util:
            raise ValueError(
                f"Vsub has {nc} alternatives; kernel expects n_util={self.n_util}"
            )
        ng = self.nvargam

        consumption = np.asarray(obs.consumption, dtype=np.float64)      # (n_obs, nc)
        price = np.asarray(obs.price, dtype=np.float64)                  # (n_obs, nc)
        gamma_design = np.asarray(obs.gamma_design, dtype=np.float64)    # (n_obs, nc, ng)

        gamma_raw = np.asarray(kstate.gamma_raw, dtype=np.float64)       # (ng,)
        log_sigma = float(kstate.log_sigma)

        rc_arr = np.asarray(rc_draw)
        n_rnd = rc_arr.shape[1] if rc_arr.ndim == 2 else 0
        dlogp_drc = np.zeros((n_obs, n_rnd), dtype=np.float64)

        # D2: draw-invariant synthetic ``dta`` (built once per obs, reused across
        # every draw). D3/D4: one batched draw-invariant satiation/gamma block
        # (built once per evaluation at ``e1 == n_obs``, reused across the n_rep
        # draws).
        dta_all = self._get_dta_all(
            obs, consumption, price, gamma_design, n_obs, nc, ng,
        )
        sat = self._get_sat_all(
            kstate, obs, dta_all, gamma_raw, log_sigma, n_obs, nc, ng,
        )

        # D4: the baseline utility is the drawn ``Vsub`` (identity baseline
        # design); reuse the shipped, already-vectorized v/vdisc/vcont builder
        # over ALL observations at once (``e1 == n_obs``) via ``v_override``.
        terms = _compute_utility_terms_from_sat(
            None, dta_all, self._ivm, self._flagchm, self._flagprcm,
            nc, nc, sat, self.control, v_override=Vsub,
        )
        vdisc = terms["vdisc"]        # (n_obs, nc-1)
        vcont = terms["vcont"]        # (n_obs, nc-1)

        # ---- batched MDCEV logit likelihood (D5) -------------------------
        # The three consumption branches (some / all / none inside goods
        # consumed) share ONE closed form: the standard partial multivariate-
        # logistic CDF ``pdfcdfmvlogit(a=vcont_consumed, b=vdisc_nonconsumed)``
        # with ``mu==0``, ``sig==xsigm``. The all-inside ``nonpdfmvlogit`` is
        # that form with ``K2==0`` (no upper-truncated variates); the
        # outside-only ``noncdfmvlogit`` is it with ``K1==0`` (no equality
        # variates). Masking consumed vs non-consumed inside goods evaluates all
        # three uniformly over the obs axis, replacing the per-obs scalar logit
        # dispatch. (The argument is zeroed on masked-out goods before ``exp`` so
        # an irrelevant large-negative utility cannot overflow into an
        # ``inf * 0 == nan``; the kept entries are byte-identical.)
        xsigm = float(sat["xsigm"])
        c = sat["c"]                                            # (n_obs,)
        con = sat["b"]                                          # (n_obs, nc-1) 0/1
        noncon = 1.0 - con

        arg_a = np.where(con > 0.0, -vcont / xsigm, 0.0)
        arg_b = np.where(noncon > 0.0, -vdisc / xsigm, 0.0)
        ea = np.where(con > 0.0, np.exp(arg_a), 0.0)           # exp(-vcont/sig), consumed
        eb = np.where(noncon > 0.0, np.exp(arg_b), 0.0)        # exp(-vdisc/sig), non-consumed

        sum_ea = ea.sum(axis=1)                                # (n_obs,)
        sum_eb = eb.sum(axis=1)                                # (n_obs,)
        D = 1.0 + sum_ea + sum_eb                              # (n_obs,) logistic denom
        n_con = con.sum(axis=1)                                # (n_obs,) #consumed inside
        S_a = (vcont * con).sum(axis=1)                        # (n_obs,) sum consumed vcont
        fact = self._facts[n_con.astype(np.intp)]             # (n_obs,) K1!

        # L == nonpdfcdfmvlogit(vcont_con, vdisc_noncon, 0, xsigm)
        L = (xsigm ** (-n_con)) * fact * np.exp(-S_a / xsigm) / D ** (n_con + 1.0)
        # z == pdisc*pcont: ``c*L`` when any inside good is consumed (pcont==c*L,
        # pdisc==1); ``L`` when none is consumed (pdisc==L, pcont==1, and the
        # Jacobian scalar c degenerates to 1).
        z = np.where(n_con > 0.0, c * L, L)
        z = np.where(z <= 0.0, 1e-4, z)
        p_obs = np.exp(sat["wt"] * np.log(z))                  # wt == 1 -> exp(log z)

        if not want_grad:
            return KernelObsResult(
                p_obs=p_obs,
                dlogp_dV=np.zeros((n_obs, nc), dtype=np.float64),
                dlogp_dkparams=np.zeros((n_obs, ng + 1), dtype=np.float64),
                dlogp_drc=dlogp_drc,
                dlogp_domega=None,
            )

        # ---- batched analytic gradient (RAW d p / d(.), GAUSS gcomp) ------
        # Vectorized re-expression of ``_gradient_from_terms``: ``ggv`` is the
        # baseline-utility gradient (identity design -> d p / d Vsub), ``gg`` the
        # gamma gradient, ``gsiggg`` the log-sigma gradient. Because ``wt == 1``
        # and ``z`` is unfloored here, the RAW derivative is ``grad_raw`` itself
        # (the engine's later ``*p_obs`` on the ``/z`` log-derivative cancels).
        factor = L / xsigm                                     # (n_obs,)
        coefP = ((n_con + 1.0) / D)[:, np.newaxis]             # (n_obs, 1)

        # per-inside-good logit gradients (disjoint masks): gcont on consumed
        # goods (== gradnonpdfcdfmvlogit ga), gdisc on non-consumed (== gb).
        gcont = (factor[:, np.newaxis] * (coefP * ea - 1.0)) * con    # (n_obs, nc-1)
        gdisc = (factor[:, np.newaxis] * (coefP * eb)) * noncon       # (n_obs, nc-1)
        gd_gc = gdisc + gcont                                          # disjoint union
        sum_gd_gc = gd_gc.sum(axis=1, keepdims=True)                  # (n_obs, 1)
        # ggv == c*(sumc(gdisc+gcont) ~ -(gdisc+gcont))
        ggv = c[:, np.newaxis] * np.hstack([sum_gd_gc, -gd_gc])       # (n_obs, nc)

        # gsigdisc (== gradnonpdfcdfmvlogit gsiga / gsigb); gsigcont is always
        # zero in the shipped kernel. Consumed part carries an extra -(L/sig).
        gsig = (
            -(gcont * (vcont / xsigm) + factor[:, np.newaxis] * con)  # consumed
            - gdisc * (vdisc / xsigm)                                 # non-consumed
        )
        gsiggg = c * gsig.sum(axis=1) * xsigm                        # (n_obs,)

        # ggam2: gradient of pcont w.r.t. the gamma utility index (consumed only)
        f = sat["f"]                                                  # (n_obs, nc-1)
        newf = sat["newf"]                                            # (n_obs, nc-1)
        pcont = c * L                                                 # (n_obs,)
        if self.control.utility == "trad":
            c3 = sat["c3"]                                            # (n_obs, nc-1)
            c2 = sat["c2"]                                            # (n_obs,)
            price_inside = sat["price_inside"]                        # (n_obs, nc-1)
            ggam2 = con * (
                (-pcont)[:, np.newaxis] * c3
                + (c2 * (pcont / c))[:, np.newaxis] * price_inside
            ) * f
        else:  # linear
            ggam2 = con * ((-pcont)[:, np.newaxis] / (newf + f)) * f

        gamma_cont_term = (
            c[:, np.newaxis] * (-(newf / f) / (newf + f)) * gcont * f + ggam2
        )                                                            # (n_obs, nc-1)
        ggam = np.hstack([np.zeros((n_obs, 1)), gamma_cont_term])     # (n_obs, nc)

        # map ggam to gamma parameter space via the satiation design: the
        # identity-embedded ``dta[:, ivg_cols]`` equals ``gamma_design[:, k, j]``.
        gg_raw = np.einsum("ik,ikj->ij", ggam, gamma_design)        # (n_obs, ng)
        gg = gg_raw @ self.eqmatgam.T                               # (n_obs, ng)

        dlogp_dkparams = np.hstack([gg, gsiggg[:, np.newaxis]])      # (n_obs, ng+1)

        return KernelObsResult(
            p_obs=p_obs,
            dlogp_dV=ggv,
            dlogp_dkparams=dlogp_dkparams,
            dlogp_drc=dlogp_drc,
            dlogp_domega=None,
        )
