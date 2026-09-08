"""Shared MSL simulation engine for mixed/panel choice models (T0.13).

This is the byte-identical skeleton that all four mixed-model families
(``mixmnl``, ``mnpkercp``, ``morp_flex``, ``mdcev_mixed``) reuse; the only
per-model piece is the inner :class:`~pybhatlib.mixed._kernel.MixedKernel`
probability + score.  The engine (plan MIXED_PANEL_MODELS_PLAN.md 2.2/2.3):

    Halton draws
      -> errbeta1 = Dmask @ draws                  (panel broadcast)
      -> pipeline.realize                           (correlate / YJ / scale /
                                                     inject -> xmunew + Jacobians)
      -> Vsub = X . xmunew
      -> kernel.probability                         (the ONLY differing piece)
      -> floor pcomp BEFORE log
      -> Pprod = exp(Dmask' @ lnP)                  (product across occasions)
      -> average over draws
      -> flooring on z
      -> analytic score chained through BOTH the utility path (``dlogp_dV``) and
         the copula path (``dlogp_drc``; zero for MNL) into the shared
         random-coefficient reparameterization Jacobians.

The engine returns **per-individual** vectors so BHHH standard errors fall out
of the same score matrix (``cov = inv(S'S)``).

Deviations from the verbatim plan 2.2/2.3 pseudocode are documented inline and
in the task report:

* :meth:`RandomCoefPipeline.realize` and :meth:`MixedKernel.probability` are
  called with the engine's ``want_grad`` so a value-only evaluation never needs
  the (absent) reparameterization gradient fields.
* ``_assemble_score`` concatenates the five per-observation blocks along axis 1
  (shape ``(n_obs, n_theta)``); :meth:`ParamLayout.pack` is 1-D only and is not
  used on the batched score.
* the Yeo-Johnson chain multiplies by the *diagonal* of ``rc.dxlamrnddxlam``
  (stored as a ``diag`` matrix by :class:`EstimationSpace`), not the full
  matrix, so the elementwise broadcast against ``(n_obs, n_lam)`` is correct.
* the correlation block is skipped (returned as an empty ``(n_obs, 0)`` slice)
  when ``layout.n_rcor == 0`` (no random coefficients, or a single one), where
  ``gcholeskycor`` / ``gnewcholparmcorscaled`` are degenerate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from pybhatlib.matgradient._corr_chol import gcholeskycor
from pybhatlib.matgradient._radial import gnewcholparmcorscaled
from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._kernel import KernelObsResult, MixedKernel
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline, RealizationCache
from pybhatlib.mixed._reparam import ParamLayout, ParamSpace, RCState
from pybhatlib.vecup._panel import PanelIndex


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MSLConfig:
    """Immutable engine configuration (plan 2.2).

    Parameters
    ----------
    n_rep : int
        Number of MSL replications (GAUSS ``nrep``, default 15).
    spher : bool, default False
        Radial (``False``) vs spherical (``True``) correlation-Cholesky
        parameterization. Carried for parity with the pipeline / space; only the
        radial path is implemented downstream.
    scal : float, default 1.0
        Scaling constant in ``newcholparmscaled`` (must match the pipeline /
        space value).
    intordn1 : int, default 20
        Gauss-Hermite node count for ``meanyj`` (must match the pipeline /
        space value).
    floor_pcomp : float, default 1e-4
        GAUSS ``w1``: per-observation kernel-probability floor applied **before**
        the log.
    floor_z : float, default 1e-4
        GAUSS ``w2``: per-person simulated-probability floor applied after
        averaging over draws.
    score_convention : {"mask", "divide"}, default "mask"
        Per-observation score convention. ``"mask"`` (MNL/MNP/MORP) zeroes the
        score contribution of a floored observation (``gcomp * (Pcomp > w1)``);
        ``"divide"`` (MDCEV, GAUSS line 791) forms ``gcomp / Pobs``.
    """

    n_rep: int
    spher: bool = False
    scal: float = 1.0
    intordn1: int = 20
    floor_pcomp: float = 1e-4
    floor_z: float = 1e-4
    score_convention: str = "mask"


# ---------------------------------------------------------------------------
# Design bundle
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DesignData:
    """Per-observation design consumed by the engine and the kernel.

    Attributes
    ----------
    X : NDArray, shape (n_obs, nc, n_beta)
        Alternative-specific covariates; the engine forms the systematic
        utilities as ``Vsub = einsum('qcv,qv->qc', X, xmunew)`` for each draw.
    obs : Any
        Opaque kernel-specific per-observation bundle threaded to
        :meth:`MixedKernel.probability` unchanged. The softmax kernel reads
        ``obs.avail`` and ``obs.chosen`` (``(n_obs, nc)`` each).
    """

    X: NDArray
    obs: Any


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------

class Tracer:
    """Record named engine intermediates for the GAUSS-parity comparator.

    Flat (once-per-evaluation) intermediates are stored under their name;
    per-replication intermediates are stored keyed by the replication index.
    The engine calls :meth:`record` only when a tracer is attached, so tracing
    is a no-op on the hot path when ``trace is None``.
    """

    def __init__(self) -> None:
        self.flat: dict[str, Any] = {}
        self.per_rep: dict[str, dict[int, Any]] = {}

    def record(self, name: str, value: Any, *, rep: Optional[int] = None) -> None:
        """Record ``value`` under ``name`` (globally, or for replication ``rep``)."""
        if rep is None:
            self.flat[name] = value
        else:
            self.per_rep.setdefault(name, {})[int(rep)] = value

    def get(self, name: str) -> Any:
        """Return the recorded flat value, else the per-replication dict, else ``None``."""
        if name in self.flat:
            return self.flat[name]
        return self.per_rep.get(name)


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------

class MixedMSLEstimator:
    """Byte-identical MSL engine shared by every mixed-model family.

    Parameters
    ----------
    panel : PanelIndex
        Person-index mapping (Dmask broadcast / scatter-sum / panel product).
    draws : DrawSource
        Standard-normal draw strategy returning ``(n_rep, n_ind, n_rnd)``.
    pipeline : RandomCoefPipeline
        Random-coefficient realization (correlate / YJ / scale / inject) plus
        analytic Jacobian blocks.
    kernel : MixedKernel
        Inner probability + score (softmax / MVNCD / rectangle-MVNCD / logit).
    layout : ParamLayout
        ``[beta | rcor | scal | lam | kern]`` partition of ``theta``.
    space : ParamSpace
        Reparameterization strategy (``EstimationSpace`` for optimization). Must
        be compatible with ``pipeline`` (same ``scal`` / ``intordn1`` / ``spher``)
        because its :class:`RCState` feeds :meth:`RandomCoefPipeline.realize`.
    design : DesignData
        Per-observation covariates ``X`` and the kernel ``obs`` bundle.
    weightind : NDArray, shape (n_ind,)
        Per-individual likelihood weights (GAUSS ``weightind``).
    config : MSLConfig
        Engine configuration (replications, floors, score convention, ...).
    trace : Tracer, optional
        When supplied, named intermediates are recorded for the parity
        comparator. ``None`` (default) disables tracing.
    """

    def __init__(
        self,
        panel: PanelIndex,
        draws: DrawSource,
        pipeline: RandomCoefPipeline,
        kernel: MixedKernel,
        layout: ParamLayout,
        space: ParamSpace,
        design: DesignData,
        weightind: NDArray,
        config: MSLConfig,
        trace: Optional[Tracer] = None,
    ) -> None:
        self.panel = panel
        self.draws = draws
        self.pipeline = pipeline
        self.kernel = kernel
        self.layout = layout
        self.space = space
        self.spec = pipeline.spec
        self.design = design
        self.weightind = np.asarray(weightind, dtype=np.float64)
        self.config = config
        self.trace = trace

    # ------------------------------------------------------------------
    # tracing helper
    # ------------------------------------------------------------------

    def _emit(self, name: str, value: Any, *, rep: Optional[int] = None) -> None:
        if self.trace is not None:
            self.trace.record(name, value, rep=rep)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def simulated_loglik(
        self, theta: NDArray, *, want_grad: bool
    ) -> tuple[NDArray, Optional[NDArray]]:
        """Per-individual simulated log-likelihood and (optional) score.

        Body mirrors GAUSS ``lpr`` / ``lgd`` byte-for-byte, obs-batched
        (``einsum``, never a per-observation Python loop).

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Optimizer parameter vector in ``[beta|rcor|scal|lam|kern]`` order.
        want_grad : bool
            If ``True`` also return the per-individual score matrix.

        Returns
        -------
        ll_per_ind : NDArray, shape (n_ind,)
            Weighted per-individual log-likelihood contributions.
        score_per_ind : NDArray or None, shape (n_ind, n_theta)
            Per-individual analytic score when ``want_grad``, else ``None``.
        """
        theta = np.asarray(theta, dtype=np.float64)
        cfg = self.config
        panel = self.panel
        n_ind = panel.n_ind
        n_theta = self.layout.n_theta

        rc = self.space.unpack(theta, self.spec, want_grad=want_grad)
        kst = self.kernel.prepare(theta, self.layout)
        ass = self.draws.draws(n_ind, self.pipeline.n_rnd, cfg.n_rep)

        self._emit("xmu", rc.xmu)
        self._emit("x11chol", rc.x11chol)
        self._emit("omegastar", rc.omegastar)
        self._emit("wscalrand", rc.wscalrand)
        self._emit("xlamrnd", rc.xlamrnd)

        p0 = np.zeros(n_ind, dtype=np.float64)
        g0 = (
            np.zeros((n_ind, n_theta), dtype=np.float64) if want_grad else None
        )

        # Panel dedup: the RC realization is row-wise, so on panel data (multiple
        # obs per person) it is computed once per unique person and broadcast to
        # the observation level inside ``realize`` (``expand=row_to_person``),
        # bit-identically reproducing ``panel.broadcast(ass[r])`` -- avoiding the
        # ~n_obs/n_ind-fold redundant per-obs recompute. For cross-sectional data
        # (``n_ind == n_obs``, e.g. MNL/MNP/MDCEV) there is nothing to dedup, so
        # the pre-existing obs-granularity path runs unchanged (a true no-op).
        dedup_panel = panel.n_ind < panel.n_obs
        for r in range(cfg.n_rep):
            if dedup_panel:
                cache = self.pipeline.realize(
                    ass[r], rc, want_grad=want_grad, expand=panel.row_to_person
                )
            else:
                errbeta1 = panel.broadcast(ass[r])             # (n_obs, n_rnd)
                cache = self.pipeline.realize(
                    errbeta1, rc, want_grad=want_grad
                )
            Vsub = np.einsum("qcv,qv->qc", self.design.X, cache.xmunew)
            # The copula conditions on ``errbeta3`` -- the *correlated,
            # pre-Yeo-Johnson / pre-scale* random-coefficient draw (GAUSS
            # ``condition``'s ``g``), NOT the scaled ``f1rand``. The kernel's
            # ``dlogp_drc`` is therefore ``d lnP / d errbeta3`` and the engine
            # chains it via ``errbeta3 = errbeta1 @ x11chol`` into the
            # random-coefficient sub-block of the joint correlation.
            kev = self.kernel.probability(
                Vsub, self.design.obs, kst,
                rc_draw=cache.errbeta3, want_grad=want_grad,
            )
            lnP = np.log(np.maximum(kev.p_obs, cfg.floor_pcomp))   # floor BEFORE log
            Pprod = panel.logprod(lnP)                             # (n_ind,)
            p0 += Pprod

            if self.trace is not None:
                # obs-level draws recovered for the parity comparator only.
                self._emit("errbeta1", panel.broadcast(ass[r]), rep=r)
            self._emit("errbeta3", cache.errbeta3, rep=r)
            self._emit("ftemprand", cache.ftemprand, rep=r)
            self._emit("f1rand", cache.f1rand, rep=r)
            self._emit("xmunew", cache.xmunew, rep=r)
            self._emit("Vsubq", Vsub, rep=r)
            self._emit("pcomp", kev.p_obs, rep=r)
            self._emit("Pprod", Pprod, rep=r)

            if want_grad:
                gcomp = self._assemble_score(kev, cache, rc, kst)  # (n_obs, n_theta)
                if cfg.score_convention == "mask":
                    gcomp = gcomp * (kev.p_obs[:, None] > cfg.floor_pcomp)
                elif cfg.score_convention == "divide":            # MDCEV (line 791)
                    gcomp = gcomp / kev.p_obs[:, None]
                else:
                    raise ValueError(
                        f"unknown score_convention {cfg.score_convention!r}"
                    )
                gPprodgcomp = Pprod[:, None] * panel.scatter_sum(gcomp)
                g0 += gPprodgcomp
                self._emit("gcomp", gcomp, rep=r)
                self._emit("gPprodgcomp", gPprodgcomp, rep=r)

        z = np.maximum(p0 / cfg.n_rep, cfg.floor_z)
        ll = self.weightind * np.log(z)
        self._emit("p0", p0)
        self._emit("z", z)
        self._emit("final", ll)

        if not want_grad:
            return ll, None

        active = (p0 / cfg.n_rep > cfg.floor_z)
        score = (
            self.weightind[:, None]
            * ((g0 / cfg.n_rep) / z[:, None])
            * active[:, None]
        )
        self._emit("g0", g0)
        self._emit("grad", score)
        return ll, score

    def objective(self, theta: NDArray) -> tuple[float, NDArray]:
        """Negative summed log-likelihood and gradient for ``scipy.minimize``.

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Optimizer parameter vector.

        Returns
        -------
        neg_ll : float
            ``-ll.sum()``.
        neg_grad : NDArray, shape (n_theta,)
            ``-score.sum(0)`` (``jac=True`` convention).
        """
        ll, score = self.simulated_loglik(theta, want_grad=True)
        return -float(ll.sum()), -np.asarray(score).sum(0)

    # ------------------------------------------------------------------
    # score assembly (plan 2.3) -- written ONCE, correct for all kernels
    # ------------------------------------------------------------------

    def _assemble_score(
        self,
        kev: KernelObsResult,
        cache: RealizationCache,
        rc: RCState,
        kst: Any = None,
    ) -> NDArray:
        """Chain both gradient paths through the shared RC Jacobians (plan 2.3).

        Path A (utility -> ``xmunew`` -> ``theta``) via ``kev.dlogp_dV`` feeds
        the scale / correlation / lambda reparameterization chain. The copula
        path enters through ``kev.dlogp_drc`` (``= d lnP / d errbeta3``) and,
        for a kernel that emits the joint-correlation gradient
        (``kev.dlogp_domega is not None``), through ``kev.dlogp_domega``; both
        are routed to the correlation parameters (never to scale / lambda,
        since ``errbeta3`` is pre-scale / pre-Yeo-Johnson). Returns the
        per-observation score, shape ``(n_obs, n_theta)``.

        For kernels without a copula (MNL softmax, MDCEV logit)
        ``kev.dlogp_domega`` is ``None`` and ``kev.dlogp_drc`` is zero, so the
        legacy single-``df1`` path is used unchanged.
        """
        jac = cache.jac
        layout = self.layout
        Xdes = self.design.X
        Q = kev.p_obs.shape[0]
        joint = kev.dlogp_domega is not None

        # --- Path A: utility -> xmunew ------------------------------------
        dxmunew = np.einsum("qc,qcv->qv", kev.dlogp_dV, Xdes)          # (Q, nvarm)
        dbeta = np.asarray(rc.dxmudxmu1, dtype=np.float64) * np.einsum(
            "qvw,qw->qv", jac.dxmunewdxmu, dxmunew
        )                                                             # (Q, nvarm)
        df1_A = np.einsum("qrv,qv->qr", jac.dxmunewdf1rand, dxmunew)   # (Q, K)

        # For a copula kernel emitting ``dlogp_domega`` the ``errbeta3`` (copula)
        # sensitivity is routed to the correlation block directly (via
        # ``errbeta3 = errbeta1 @ x11chol``), NOT through ``f1rand``, so scale /
        # lambda see only the utility path ``df1_A``. For the legacy path
        # (MNL/MDCEV) ``dlogp_drc`` is zero, so adding it is a no-op that keeps
        # the softmax collapse gate byte-identical.
        if joint:
            df1 = df1_A                                               # (Q, K)
        else:
            df1 = df1_A + np.asarray(kev.dlogp_drc, dtype=np.float64)  # (Q, K)

        # --- shared RC reparam chain --------------------------------------
        # scale: d wscalrand / d xscalrand = wscalrand (exp reparam).
        dscal = np.asarray(rc.wscalrand, dtype=np.float64) * np.einsum(
            "qrs,qs->qr", jac.df1randdwdiagrand, df1
        )                                                             # (Q, n_scal)

        # correlation -----------------------------------------------------
        if joint:
            drcor = self._joint_corr_score(kev, cache, rc, kst, df1_A)
        elif layout.n_rcor > 0:
            # cross-sectional / non-joint: theta_rcor -> chol -> f1rand.
            ncorr_rc = jac.df1randdx11chol.shape[1]
            if ncorr_rc == layout.n_rcor:
                dx11 = np.einsum("qkr,qr->qk", jac.df1randdx11chol, df1)   # (Q, ncorr)
                gchol = gcholeskycor(rc.omegastar)                        # (ncorr, ncorr)
                chain = np.asarray(rc.gtempstar, dtype=np.float64) @ np.linalg.inv(gchol)
                drcor = dx11 @ chain.T                                    # (Q, n_rcor)
            else:
                drcor = np.zeros((Q, layout.n_rcor), dtype=np.float64)
        else:
            drcor = np.zeros((Q, 0), dtype=np.float64)

        # Yeo-Johnson power: d xlamrnd / d xlam = diag(2 pdlogit).
        dxlamrnddxlam = np.asarray(rc.dxlamrnddxlam, dtype=np.float64)
        dxlam_diag = (
            np.diagonal(dxlamrnddxlam) if dxlamrnddxlam.ndim == 2 else dxlamrnddxlam
        )
        dlam = dxlam_diag * np.einsum(
            "qrs,qs->qr", jac.df1randdxlamrnd, df1
        )                                                             # (Q, n_lam)

        # kernel-owned params (thresholds / kernel scale / kernel-lam). The
        # kernel returns a single ``dlogp_dkparams`` block whose columns follow
        # ``kernel_param_names()`` order, which matches the *physical* order of
        # the non-core (kernel-owned) layout blocks: ``thresh`` (leads, MORP),
        # ``kern`` (MNP scale, empty for MNL/MORP) and ``kernlam`` (trails, MORP
        # YJ). Striping the columns across whichever of those blocks are present
        # keeps MNL (no kernel params) and MNP (``kern`` only) byte-identical
        # while placing the MORP ``[thresh | kernlam]`` gradient correctly.
        dkparams = np.asarray(kev.dlogp_dkparams, dtype=np.float64)   # (Q, n_kparams)

        # Place each block at its physical slice so the score column order always
        # matches ``theta`` -- honoring ``kern_before_lam`` (MNP interleaves the
        # ``kern`` block between ``scal`` and ``lam``, MNL keeps it last) and the
        # MORP ``thresh``-first / ``kernlam``-last additive blocks.
        core_blocks = {
            "beta": dbeta, "rcor": drcor, "scal": dscal, "lam": dlam,
        }
        out = np.zeros((Q, layout.n_theta), dtype=np.float64)
        col = 0
        for name, s in layout.slices().items():
            if name in core_blocks:
                out[:, s] = core_blocks[name]
            else:  # kernel-owned block(s): thresh / kern / kernlam.
                w = s.stop - s.start
                if w:
                    out[:, s] = dkparams[:, col:col + w]
                col += w
        return out

    # ------------------------------------------------------------------
    # joint-correlation score (MNP copula) -- plan P1.2
    # ------------------------------------------------------------------

    def _joint_corr_score(
        self,
        kev: KernelObsResult,
        cache: RealizationCache,
        rc: RCState,
        kst: Any,
        df1_A: NDArray,
    ) -> NDArray:
        """Chain the total ``d lnP / d omegastar`` to the correlation params.

        Accumulates, over the free off-diagonal elements of the FULL joint
        ``omegastar`` (``nrndtot = nrndcoef + (nc - 1)``, ``vecndup`` order),
        three contributions, then chains once through the radial
        reparameterization ``gnewcholparmcorscaled`` to the ``rcor`` block:

        1. ``kev.dlogp_domega`` -- the kernel's direct dependence via the
           conditional covariance ``xi2subq`` and mean ``B1subq`` (``errbeta3``
           held fixed);
        2. the utility path ``d lnP / d f1rand`` (``df1_A``) routed through
           ``f1rand``'s dependence on the RC-sub-block Cholesky
           (``jac.df1randdx11chol``);
        3. the copula path ``d lnP / d errbeta3`` (``kev.dlogp_drc``) routed
           through ``errbeta3 = errbeta1 @ x11chol``
           (``jac.gerrbeta3dx11chol``).

        Contributions 2 and 3 land on the RC sub-block ``omegastar[:k, :k]``:
        their Cholesky cotangent is mapped to the sub-block correlation
        off-diagonals via ``inv(gcholeskycor(omega_sub))`` and embedded into the
        full ``vecndup`` positions before the single radial chain.

        Returns
        -------
        NDArray, shape (n_obs, n_rcor)
        """
        jac = cache.jac
        layout = self.layout
        Q = kev.p_obs.shape[0]
        k = self.spec.nrndcoef
        scal = self.config.scal

        omega_full = np.asarray(kst.omegastar, dtype=np.float64)      # (nrndtot,nrndtot)
        nrndtot = omega_full.shape[0]
        n_free = nrndtot * (nrndtot - 1) // 2

        if n_free != layout.n_rcor:  # defensive: layout must match the joint size
            return np.zeros((Q, layout.n_rcor), dtype=np.float64)

        # (1) kernel's direct joint-correlation gradient (errbeta3 fixed).
        total = np.array(kev.dlogp_domega, dtype=np.float64)          # (Q, n_free)

        # (2)+(3) x11chol-routed contributions into the RC sub-block.
        ncorr_rc = jac.df1randdx11chol.shape[1]
        if ncorr_rc > 0:
            chol_cot = (
                np.einsum("qkr,qr->qk", jac.df1randdx11chol, df1_A)
                + np.einsum("qkr,qr->qk", jac.gerrbeta3dx11chol, kev.dlogp_drc)
            )                                                        # (Q, ncorr_rc)
            omega_sub = omega_full[:k, :k]
            gchol_sub = gcholeskycor(omega_sub)                      # (ncorr_rc, ncorr_rc)
            # d lnP / d(sub corr off-diag) = inv(gcholeskycor) @ d lnP / d(chol).
            subcorr_cot = chol_cot @ np.linalg.inv(gchol_sub).T      # (Q, ncorr_rc)
            # embed the sub-block off-diagonals into the full vecndup positions.
            sub_to_full = self._sub_to_full_offdiag(k, nrndtot)
            total[:, sub_to_full] += subcorr_cot

        # single radial chain: d lnP / d(rcor) = grad1 @ (d lnP / d corr).
        # grad1[p, j] = d corr_offdiag[j] / d theta_p.
        grad1, _grad2 = gnewcholparmcorscaled(omega_full, scal)      # (n_free, n_free)
        drcor = np.einsum("pj,qj->qp", grad1, total)                 # (Q, n_rcor)
        return drcor

    @staticmethod
    def _sub_to_full_offdiag(k: int, nrndtot: int) -> list[int]:
        """Indices (in the full ``nrndtot`` ``vecndup`` order) of the RC-sub-block
        off-diagonal pairs ``(p, q)`` with ``p < q < k``."""
        full = [(p, q) for p in range(nrndtot) for q in range(p + 1, nrndtot)]
        pos = {pair: i for i, pair in enumerate(full)}
        return [pos[(p, q)] for p in range(k) for q in range(p + 1, k)]


__all__ = ["MSLConfig", "DesignData", "Tracer", "MixedMSLEstimator"]
