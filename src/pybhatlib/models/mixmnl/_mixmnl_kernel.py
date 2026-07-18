"""Softmax (MNL) kernel for the mixed-model MSL engine — Phase-0 anchor.

The MixMNL family mixes an MNL softmax kernel over random coefficients. Because
the softmax has an exact, MC-noise-free analytic gradient, it is the engine's
Phase-0 anchor: every shared-engine primitive (panel product, reparam Jacobian
chain, flooring/masking) is validated against this kernel before any stochastic
copula kernel is trusted.

:class:`SoftmaxKernel` implements the :class:`~pybhatlib.mixed._kernel.MixedKernel`
protocol. It owns **no** parameters (``n_kern == 0``) and has **no** copula
(``dlogp_drc == 0``); its only job is, per MSL replication, to turn the
engine-formed systematic utilities ``Vsub`` into the chosen-alternative
probability and the softmax score.

GAUSS reference: ``MIXMNL.gss`` ``lpr`` / ``lgd`` — availability-masked softmax
``p2`` and the residual score ``ggv = dvunord - p2`` (line 609). The softmax
math (max-shift + denominator guard) matches
``models/mnl/_mnl_loglik._compute_probabilities`` exactly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.mixed._kernel import KernelObsResult, KernelState


def _softmax_avail(Vsub: NDArray, avail: NDArray) -> NDArray:
    """Availability-masked softmax over alternatives (batched over obs).

    Replicates the softmax block of
    ``models/mnl/_mnl_loglik._compute_probabilities`` (GAUSS: ``p1 = exp(v)``;
    ``p2 = (p1 .* avail) ./ sumc((p1 .* avail)')``) with the same numerical
    guards: subtract the per-row max over *available* alternatives before
    ``exp`` (the shift cancels in the ratio, preventing overflow), and guard a
    zero denominator (all-unavailable row) against division by zero.

    Parameters
    ----------
    Vsub : NDArray, shape (n_obs, nc)
        Systematic utilities per observation and alternative.
    avail : NDArray, shape (n_obs, nc)
        Availability mask (``> 0`` where the alternative is available).

    Returns
    -------
    p2 : NDArray, shape (n_obs, nc)
        Availability-masked, normalised choice probabilities.
    """
    v_max = np.max(np.where(avail > 0, Vsub, -np.inf), axis=1, keepdims=True)
    v_max = np.where(np.isfinite(v_max), v_max, 0.0)      # guard all-unavail row
    # Mask *inside* the exponent (exp(-inf) == 0): an unavailable alternative
    # with utility > v_max + 709 would otherwise overflow exp then hit inf*0=NaN.
    p1 = np.exp(np.where(avail > 0, Vsub - v_max, -np.inf))  # (n_obs, nc) masked
    denom = p1.sum(axis=1, keepdims=True)                  # (n_obs, 1)
    denom = np.where(denom == 0, 1.0, denom)               # guard /0
    return p1 / denom                                      # (n_obs, nc)


class SoftmaxKernel:
    """MNL softmax kernel (no own parameters, no copula).

    Parameters
    ----------
    n_util : int
        Number of alternatives ``nc`` the kernel consumes from ``Vsub``.

    Notes
    -----
    Implements :class:`~pybhatlib.mixed._kernel.MixedKernel`:

    - ``prepare`` is a no-op (the kernel is stateless), returning ``None``.
    - ``kernel_param_names`` is empty (``n_kern == 0``).
    - ``probability`` returns ``p_obs`` = ``p2`` at the chosen alternative,
      ``dlogp_dV`` = ``onehot(chosen) - p2`` (GAUSS ``ggv``),
      ``dlogp_dkparams`` of shape ``(n_obs, 0)``, and ``dlogp_drc`` of zeros.
    """

    def __init__(self, n_util: int) -> None:
        self.n_util: int = int(n_util)

    def kernel_param_names(self) -> list[str]:
        """Return the kernel-only parameter names — empty for the softmax."""
        return []

    def prepare(self, theta: NDArray, layout: object) -> KernelState:
        """No-op: the softmax kernel is stateless (returns ``None``)."""
        return None

    def probability(
        self,
        Vsub: NDArray,
        obs: object,
        kstate: KernelState,
        *,
        rc_draw: NDArray,
        want_grad: bool,
    ) -> KernelObsResult:
        """Chosen-alternative probability and softmax score for one draw.

        Parameters
        ----------
        Vsub : NDArray, shape (n_obs, nc)
            Systematic utilities per observation and alternative.
        obs : DesignData
            Must expose ``avail`` (``(n_obs, nc)`` availability mask) and
            ``chosen`` (``(n_obs, nc)`` one-hot chosen indicator).
        kstate : KernelState
            Ignored (``None`` for this stateless kernel).
        rc_draw : NDArray, shape (n_obs, n_rnd)
            Drawn random coefficients; used only to size the zero
            ``dlogp_drc`` (the softmax kernel has no copula).
        want_grad : bool
            If ``False``, ``dlogp_dV`` is returned as zeros.

        Returns
        -------
        KernelObsResult
            ``p_obs`` ``(n_obs,)``, ``dlogp_dV`` ``(n_obs, nc)``,
            ``dlogp_dkparams`` ``(n_obs, 0)``, ``dlogp_drc`` ``(n_obs, n_rnd)``.
        """
        Vsub = np.asarray(Vsub, dtype=np.float64)
        avail = np.asarray(obs.avail, dtype=np.float64)    # (n_obs, nc)
        chosen = np.asarray(obs.chosen, dtype=np.float64)  # (n_obs, nc) one-hot
        n_obs, nc = Vsub.shape
        n_rnd = np.asarray(rc_draw).shape[1] if np.asarray(rc_draw).ndim == 2 else 0

        p2 = _softmax_avail(Vsub, avail)                   # (n_obs, nc)

        # Probability of the chosen alternative (GAUSS: sumc((p2 .* dvunord)')).
        p_obs = (p2 * chosen).sum(axis=1)                  # (n_obs,)

        # Utility-path score: onehot(chosen) - p2  (GAUSS ggv, MIXMNL lpr 609).
        if want_grad:
            dlogp_dV = chosen - p2                          # (n_obs, nc)
        else:
            dlogp_dV = np.zeros((n_obs, nc), dtype=np.float64)

        # No kernel-only params and no copula for the softmax kernel.
        dlogp_dkparams = np.zeros((n_obs, 0), dtype=np.float64)
        dlogp_drc = np.zeros((n_obs, n_rnd), dtype=np.float64)

        return KernelObsResult(
            p_obs=p_obs,
            dlogp_dV=dlogp_dV,
            dlogp_dkparams=dlogp_dkparams,
            dlogp_drc=dlogp_drc,
        )
