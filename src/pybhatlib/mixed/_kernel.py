"""Kernel protocol for the shared mixed/panel MSL engine.

The MSL engine (``mixed/_engine.py``) is byte-identical across the four
mixed-model families; the only per-model piece is the inner **kernel
probability + score**. This module defines that seam:

- :class:`KernelObsResult` — the batched (over obs/draw) result a kernel
  returns for one MSL replication, carrying the observed-outcome probability
  and the three gradient paths the engine chains back through the shared
  random-coefficient reparameterization Jacobians.
- :class:`MixedKernel` — the structural :class:`typing.Protocol` every kernel
  implements (``SoftmaxKernel``, ``MvncdKernel``, ``RectMvncdKernel``,
  ``LogitJacobianKernel``).

The result carries **four** fields (plan §2.3): the fourth, ``dlogp_drc``,
expresses the copula gradient — the derivative of ``log P`` with respect to
the *drawn* random-coefficient vector that enters the kernel covariance
(MNP/MORP). It is exactly zero for kernels without a copula (MNL softmax,
MDCEV logit), but it is a *field*, not an optional return, so the engine's
``_assemble_score`` chains both the utility path (``dlogp_dV``) and the
covariance path (``dlogp_drc``) uniformly for every model.

See ``docs/plans/MIXED_PANEL_MODELS_PLAN.md`` §2.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from numpy.typing import NDArray

# ``prepare`` returns an opaque, per-eval state bundle that the engine threads
# back into ``probability`` unchanged (Cholesky factors, thresholds, kernel
# scale, copula covariance, ...). Each kernel owns its concrete shape; the
# engine never inspects it, so the protocol keeps it deliberately untyped.
KernelState = Any


@dataclass
class KernelObsResult:
    """Batched kernel result for one MSL replication.

    All fields are arrays batched over the ``n_obs`` observations of the
    current draw (the engine calls a kernel once per replication, never per
    observation). ``nc`` is the number of utility differences the kernel
    consumes, ``n_kern`` the number of kernel-only parameters, and ``n_rnd``
    the number of random coefficients.

    Attributes
    ----------
    p_obs : NDArray, shape (n_obs,)
        Probability of the observed outcome for each observation/draw.
    dlogp_dV : NDArray, shape (n_obs, nc)
        Derivative ``d log P / d Vsub`` — the utility path. Chains through
        ``Vsub = X @ xmunew`` into the beta / scale / correlation / lambda
        blocks.
    dlogp_dkparams : NDArray, shape (n_obs, n_kern)
        Derivative ``d log P / d kernel-only params`` (thresholds, kernel
        scale, joint-covariance block). Shape ``(n_obs, 0)`` for kernels with
        no own parameters (MNL softmax).
    dlogp_drc : NDArray, shape (n_obs, n_rnd)
        Derivative ``d log P / d drawn rc`` — the copula path, routed through
        the kernel covariance. All zeros for kernels without a copula (MNL
        softmax, MDCEV logit).

        Semantically this is ``d log P / d errbeta3`` (the *correlated,
        pre-Yeo-Johnson / pre-scale* random-coefficient draw that enters the
        copula conditioning value ``g``), **not** ``d log P / d f1rand``. The
        engine chains it via ``errbeta3 = errbeta1 @ x11chol`` into the
        random-coefficient sub-block of the joint correlation.
    dlogp_domega : NDArray or None, shape (n_obs, nrndtot*(nrndtot-1)//2)
        Derivative ``d log P / d omegastar`` w.r.t. the **free off-diagonal
        (correlation) elements of the FULL joint correlation** ``omegastar``
        (size ``nrndtot = nrndcoef + (nc - 1)``), in row-based strict-upper
        (``vecndup``) order. This accumulates the copula sources that funnel
        through ``omegastar`` while the drawn ``errbeta3`` is held fixed (the
        kernel's direct dependence via the conditional covariance ``xi2subq``
        and the conditional mean ``B1subq``). ``None`` for kernels without an
        active copula (MNL softmax, MDCEV logit, or ``MvncdKernel`` with
        ``copula=False``), so those kernels are unaffected. The engine adds the
        ``errbeta3 -> x11chol`` and utility-path contributions to this total
        before chaining once to the correlation parameters.
    """

    p_obs: NDArray
    dlogp_dV: NDArray
    dlogp_dkparams: NDArray
    dlogp_drc: NDArray
    dlogp_domega: Optional[NDArray] = None


@runtime_checkable
class MixedKernel(Protocol):
    """Structural protocol for a mixed-model inner kernel (plan §2.3).

    A kernel translates per-observation systematic utilities ``Vsub`` (and,
    for copula models, the drawn random coefficients) into the observed-outcome
    probability and its gradient paths. The engine holds one kernel object and
    never inspects which concrete kernel it is; adding a fifth model family is
    one new ``MixedKernel`` implementation plus a thin facade.

    Attributes
    ----------
    n_util : int
        Number of utility (difference) columns the kernel consumes from
        ``Vsub`` — i.e. the ``nc`` in :class:`KernelObsResult`.
    """

    n_util: int

    def kernel_param_names(self) -> list[str]:
        """Return the names of the kernel-only parameters, in ``theta`` order.

        The list length equals ``n_kern`` (the ``dlogp_dkparams`` width) and
        matches the ``kern`` block of the model's :class:`ParamLayout`. Empty
        for kernels with no own parameters (MNL softmax).
        """
        ...

    def prepare(self, theta: NDArray, layout: Any) -> KernelState:
        """Build the per-evaluation kernel state from the full parameter vector.

        Called once per objective evaluation (not per replication). Extracts
        the kernel-only parameter block via ``layout`` and precomputes anything
        constant across draws (Cholesky factors, thresholds, kernel scale,
        copula covariance). Returns an opaque state the engine threads back
        into :meth:`probability` unchanged.

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Full parameter vector (``[beta | rcor | scal | lam | kern]``).
        layout : ParamLayout
            Parameter-block partition; the kernel reads only its ``kern`` slice.

        Returns
        -------
        KernelState
            Opaque per-evaluation state (``None`` for a stateless kernel).
        """
        ...

    def probability(
        self,
        Vsub: NDArray,
        obs: Any,
        kstate: KernelState,
        *,
        rc_draw: NDArray,
        want_grad: bool,
    ) -> KernelObsResult:
        """Compute the observed-outcome probability and gradients for one draw.

        Batched over the ``n_obs`` observations of the current replication.

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nc)
            Systematic utilities per observation and alternative, already
            formed by the engine as ``X @ xmunew`` for this draw.
        obs : DesignData
            Per-observation design bundle. The softmax kernel reads
            ``obs.avail`` (``(n_obs, nc)`` availability mask) and
            ``obs.chosen`` (``(n_obs, nc)`` one-hot chosen indicator), mirroring
            GAUSS ``davunord`` / ``dvunord``.
        kstate : KernelState
            The state returned by :meth:`prepare` for this evaluation.
        rc_draw : NDArray, shape (n_obs, n_rnd)
            The drawn random coefficients for this replication (GAUSS
            ``f1rand``); used only by copula kernels, but always supplied so
            ``dlogp_drc`` has a well-defined width.
        want_grad : bool
            If ``False`` the gradient fields are returned as correctly-shaped
            zeros (the engine ignores them).

        Returns
        -------
        KernelObsResult
            The four-field batched result for this replication.
        """
        ...
