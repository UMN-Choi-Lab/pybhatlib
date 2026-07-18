"""Parameter layout and estimation/reporting reparameterization spaces.

The optimizer parameter vector ``theta`` for a mixed random-coefficient model
is laid out, in GAUSS ``b`` order, as five contiguous blocks::

    theta = [ beta | rcor | scal | lam | kern ]
            <-n_beta-><-n_rcor-><-n_scal-><-n_lam-><-n_kern->

(``MIXMNL.gss`` line 276: ``b = b1 | startker | zeros(nscale) | zeros(numlam)``;
the ``kern`` block holds kernel-only parameters and is empty for the MNL
softmax kernel).

:class:`ParamLayout` owns that block partition. Two :class:`ParamSpace`
strategies map a ``theta`` slice-set plus a :class:`~pybhatlib.mixed._spec.MixingSpec`
into the realized random-coefficient quantities (:class:`RCState`) consumed by
the pipeline and kernel:

* :class:`EstimationSpace` -- the ``lpr``/``lgd`` parameterization used during
  optimization: correlation via ``newcholparmscaled``, scale via ``exp``,
  Yeo-Johnson power via ``2 * cdlogit`` (bounded, unconstrained optimizer).
* :class:`ReportingSpace` -- the ``lpr1`` parameterization used for standard
  errors / reporting: correlation via ``matndupdiagonefull`` (direct
  correlation entries), scale and ``lambda`` entered directly.

The two strategies share the ``meanyj`` mean/std computation and the additive-
vs-multiplicative sign reparameterization of the fixed coefficients, so SE
reporting reuses the estimation chain (see MIXED_PANEL_MODELS_PLAN.md 2.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.matgradient._corr_chol import gcholeskycor, matndupdiagonefull
from pybhatlib.matgradient._radial import (
    gnewcholparmcorscaled,
    newcholparmscaled,
)
from pybhatlib.utils._logistic import cdlogit, gradlogitmod, logitmod, pdlogit
from pybhatlib.utils._safe_reparam import (
    nearest_pd_correlation,
    safe_cholesky,
    safe_exp,
)
from pybhatlib.vecup._yj import gradmeanyj, meanyj

if TYPE_CHECKING:  # avoid a runtime import cycle with _spec
    from pybhatlib.mixed._spec import MixingSpec

# Yeo-Johnson power ``lambda`` lives in the open interval ``(0, 2)``. ``cdlogit``
# saturates to exactly 1.0 for arguments above ~37, which would push
# ``2 * cdlogit`` to exactly 2.0 (or 0.0 in the opposite tail) and make the YJ
# helpers divide by ``(2 - lam) == 0``. Clamp just inside the open interval so a
# saturated optimizer step cannot yield a degenerate ``lam``.
_LAM_EPS = 1e-8


# ---------------------------------------------------------------------------
# Parameter layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamLayout:
    """Contiguous parameter partition over the GAUSS ``b`` blocks.

    The five core blocks are ``beta``, ``rcor``, ``scal``, ``lam``, ``kern``.
    Two orderings of ``kern`` relative to ``lam`` are supported via
    ``kern_before_lam``:

    * default (``False``) -- the MNL softmax ordering
      ``[beta | rcor | scal | lam | kern]`` (``MIXMNL.gss`` line 276), where
      the ``kern`` block is empty;
    * ``True`` -- the MNP kernel ordering
      ``[beta | rcor | scal | kern | lam]`` (``MNPKERCP.gss`` line 554), where
      the kernel-scale block interleaves **between** ``scal`` and ``lam``.

    Two optional MORP blocks (ordered-YJ driver, GAUSS b vector line 477
    ``b = thresh | b1 | startrandker | scalcoef | blamrnd | blamker``) are added
    additively and appear as slice keys **only when their size is non-zero**, so
    the MNL / MNP key lists are unchanged:

    * ``thresh`` -- ``n_thresh`` ordinal threshold parameters that **lead** the
      vector (``sum(n_categories[d] - 1)`` over the ordinal dimensions);
    * ``kernlam`` -- ``n_kernlam`` Yeo-Johnson kernel-lam parameters that
      **trail** the vector (GAUSS ``blamker``, ``nord`` params when
      ``_normker == 0``).

    The MORP physical order is therefore
    ``[thresh | beta | rcor | scal | (kern) | lam | kernlam]`` (``kern`` empty
    since MORP fixes ``wker`` to ones).

    A third optional block ``gamma`` (MDCEV translation / satiation parameters,
    GAUSS ``xgam``) is added additively for the MDCEV logit-Jacobian kernel. It
    follows ``beta`` directly and appears as a slice key **only when its size is
    non-zero**, so the MNL / MNP / MORP key lists are unchanged. Combined with
    ``kern_before_scal`` (the MDCEV kernel-scale ``wscalker`` interleaves
    **between** ``rcor`` and the random-coefficient ``scal`` block, GAUSS
    ``MDCEV.gss`` b vector line ~301), the MDCEV physical order is

    ``[beta | gamma | rcor | kern | scal | lam]``

    matching the GAUSS parameter vector
    ``b = b1 | -1000 | gamma_rest | startker | xscalker | xscalrand | xlam``
    (``kern`` holds the single MDCEV kernel scale; ``gamma`` element 0 is the
    fixed outside-good satiation ``-1000``). A converged GAUSS ``b`` therefore
    plugs into these slices verbatim.

    Block sizes are fixed at construction; :meth:`slices` returns the named
    slices and :meth:`pack` assembles a full ``theta`` from its blocks.

    Parameters
    ----------
    n_beta : int
        Number of fixed-coefficient (``beta``) parameters (GAUSS ``ncoeford``
        for MORP).
    n_rcor : int
        Number of correlation parameters. For the joint kernel space this is
        ``nrndtot * (nrndtot - 1) // 2`` with ``nrndtot = nrndcoef + (nc - 1)``
        (MNP) or ``nrndcoef + nord`` (MORP).
    n_scal : int
        Number of random-coefficient scale parameters.
    n_lam : int
        Number of random-coefficient Yeo-Johnson ``lambda`` parameters.
    n_kern : int
        Number of kernel-scale parameters (``0`` for the MNL softmax and MORP
        kernels; ``nc - 2`` for the MNP kernel).
    kern_before_lam : bool, default False
        If ``True``, place the ``kern`` block between ``scal`` and ``lam``
        (MNP / MORP ordering); otherwise place it last (MNL ordering).
    n_thresh : int, default 0
        Number of leading ordinal threshold parameters (MORP; GAUSS
        ``nthresh``). ``0`` omits the ``thresh`` slice key entirely.
    n_kernlam : int, default 0
        Number of trailing YJ kernel-lam parameters (MORP; GAUSS ``blamker``).
        ``0`` omits the ``kernlam`` slice key entirely.
    n_gamma : int, default 0
        Number of MDCEV translation / satiation (``gamma``) parameters (GAUSS
        ``nvargam``), placed immediately after ``beta``. ``0`` omits the
        ``gamma`` slice key entirely (MNL / MNP / MORP unchanged).
    kern_before_scal : bool, default False
        If ``True`` (MDCEV), place the ``kern`` block **between** ``rcor`` and
        ``scal`` (the MDCEV kernel scale precedes the random-coefficient scales,
        GAUSS ``xscalker`` then ``xscalrand``). Mutually exclusive with
        ``kern_before_lam``.
    """

    n_beta: int
    n_rcor: int
    n_scal: int
    n_lam: int
    n_kern: int
    kern_before_lam: bool = False
    n_thresh: int = 0
    n_kernlam: int = 0
    n_gamma: int = 0
    kern_before_scal: bool = False

    def __post_init__(self) -> None:
        if self.kern_before_lam and self.kern_before_scal:
            raise ValueError(
                "kern_before_lam (MNP) and kern_before_scal (MDCEV) are "
                "mutually exclusive kernel-scale placements"
            )

    @property
    def n_theta(self) -> int:
        """Total parameter count over every (possibly empty) block."""
        return (
            self.n_thresh
            + self.n_beta
            + self.n_gamma
            + self.n_rcor
            + self.n_scal
            + self.n_lam
            + self.n_kern
            + self.n_kernlam
        )

    def _order(self) -> list[str]:
        """Block names in physical ``theta`` order.

        ``thresh`` leads and ``kernlam`` trails, each included only when its
        size is positive (so MNL / MNP key lists are unchanged). ``kern`` is
        placed by ``kern_before_lam`` (MNP) or ``kern_before_scal`` (MDCEV); the
        optional MDCEV ``gamma`` block follows ``beta`` when non-empty.
        """
        if self.kern_before_lam:
            core = ["beta", "rcor", "scal", "kern", "lam"]
        elif self.kern_before_scal:
            core = ["beta", "rcor", "kern", "scal", "lam"]
        else:
            core = ["beta", "rcor", "scal", "lam", "kern"]
        order: list[str] = []
        if self.n_thresh:
            order.append("thresh")
        for name in core:
            order.append(name)
            if name == "beta" and self.n_gamma:
                order.append("gamma")
        if self.n_kernlam:
            order.append("kernlam")
        return order

    def _sizes(self) -> dict[str, int]:
        return {
            "thresh": self.n_thresh,
            "beta": self.n_beta,
            "gamma": self.n_gamma,
            "rcor": self.n_rcor,
            "scal": self.n_scal,
            "lam": self.n_lam,
            "kern": self.n_kern,
            "kernlam": self.n_kernlam,
        }

    def slices(self) -> dict[str, slice]:
        """Return the named block slices into a length-``n_theta`` vector.

        Returns
        -------
        dict[str, slice]
            Contiguous, non-overlapping, gap-free slices whose union covers
            ``range(n_theta)``. Always includes ``"beta"``, ``"rcor"``,
            ``"scal"``, ``"lam"``, ``"kern"``; additionally ``"thresh"`` (first)
            and ``"kernlam"`` (last) when those blocks are non-empty, and
            ``"gamma"`` (after ``"beta"``) when ``n_gamma > 0`` (MDCEV).
            Insertion order follows the physical block order.
        """
        sizes = self._sizes()
        out: dict[str, slice] = {}
        start = 0
        for name in self._order():
            stop = start + sizes[name]
            out[name] = slice(start, stop)
            start = stop
        return out

    def pack(
        self,
        dbeta: NDArray,
        drcor: NDArray,
        dscal: NDArray,
        dlam: NDArray,
        dkern: NDArray,
        *,
        dthresh: Optional[NDArray] = None,
        dkernlam: Optional[NDArray] = None,
        dgamma: Optional[NDArray] = None,
    ) -> NDArray:
        """Assemble a full ``theta`` vector from its blocks.

        Inverse of slicing with :meth:`slices`. Each block is flattened to 1-D
        and length-checked against the layout before concatenation, in the
        physical block order. The five core blocks stay positional for
        backward compatibility with the MNL / MNP callers; the two MORP blocks
        are keyword-only and default to empty.

        Parameters
        ----------
        dbeta, drcor, dscal, dlam, dkern : array_like
            Core block contents (scalars or arrays; empty blocks accept any
            0-length input). Always passed in the fixed logical order
            ``(beta, rcor, scal, lam, kern)`` regardless of physical ordering.
        dthresh : array_like, optional
            Leading MORP threshold block; defaults to empty (``None`` -> length
            0), which is required when ``n_thresh == 0``.
        dkernlam : array_like, optional
            Trailing MORP kernel-lam block; defaults to empty.
        dgamma : array_like, optional
            MDCEV translation (``gamma``) block that follows ``beta``; defaults
            to empty (required when ``n_gamma == 0``).

        Returns
        -------
        ndarray, shape (n_theta,)
            Concatenated parameter vector (float64).

        Raises
        ------
        ValueError
            If any block length differs from the corresponding layout size.
        """
        vals = {
            "thresh": dthresh,
            "beta": dbeta,
            "gamma": dgamma,
            "rcor": drcor,
            "scal": dscal,
            "lam": dlam,
            "kern": dkern,
            "kernlam": dkernlam,
        }
        sizes = self._sizes()
        parts = []
        for name in self._order():
            raw = vals[name]
            if raw is None:
                arr = np.zeros(0, dtype=np.float64)
            else:
                arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).reshape(-1)
            if arr.shape[0] != sizes[name]:
                raise ValueError(
                    f"block {name!r} has length {arr.shape[0]}, "
                    f"expected {sizes[name]}"
                )
            parts.append(arr)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


# ---------------------------------------------------------------------------
# MNP kernel-scale (sum-of-squares) reparameterization
# ---------------------------------------------------------------------------

def wker_reparam(
    xscalker: NDArray, *, want_grad: bool = False, xp=None
) -> tuple[NDArray, Optional[NDArray]]:
    r"""Sum-of-squares (softmax) reparameterization of the MNP kernel scale.

    Ports the ``wker`` construction of ``MNPKERCP.gss`` (line ~479 / ~554)::

        bscalker = 0 | xscalker            # prepend a fixed 0 -> length (nc-1)
        bscalker = sqrt(logitmod(bscalker))
        wker     = bscalker[2:]            # drop the first entry -> length (nc-2)

    Prepending the fixed ``0`` and applying ``logitmod`` (softmax) makes the
    squared entries a probability vector summing to one, i.e. the kernel-scale
    vector satisfies ``sum(wker_full ** 2) == 1`` -- the sum-of-squares
    normalization GAUSS reports. The first (reference) entry is dropped so the
    ``n_kern = nc - 2`` free parameters ``xscalker`` map to an ``n_kern``-vector
    ``wker``.

    Parameters
    ----------
    xscalker : array_like, shape (n_kern,)
        Free kernel-scale parameters (GAUSS ``x[nvar+nrndtcor+nrndcoef+1 : ...]``),
        with ``n_kern == nc - 2``.
    want_grad : bool, default False
        If True, also return the ``(n_kern, n_kern)`` Jacobian
        ``d wker[i] / d xscalker[j]``.
    xp : backend, optional
        Array backend. Inferred from ``xscalker`` if not given.

    Returns
    -------
    wker : ndarray, shape (n_kern,)
        Kernel-scale vector (the dropped-reference sum-of-squares scales).
    dwker : ndarray or None, shape (n_kern, n_kern)
        Jacobian ``d wker[i] / d xscalker[j]`` when ``want_grad``, else ``None``.

    Notes
    -----
    With ``full = [0, xscalker]``, ``F = logitmod(full)`` and
    ``wker[i] = sqrt(F[i + 1])``, the Jacobian is

    .. math::

        \frac{\partial w_i}{\partial x_j}
          = \frac{1}{2 \sqrt{F_{i+1}}}\, \frac{\partial F_{i+1}}{\partial x_j}
          = \frac{1}{2\, w_i}\, g_{a}[j + 1,\, i + 1],

    where ``g_a`` is the softmax Jacobian ``d F_b / d full_a`` from
    :func:`~pybhatlib.utils._logistic.gradlogitmod` (only the ``xscalker``
    columns ``a = j + 1`` are varied; the reference column ``a = 0`` is fixed).
    """
    if xp is None:
        xp = array_namespace(xscalker)

    xscalker = xp.array(xscalker, dtype=xp.float64).reshape(-1)
    n_kern = xscalker.shape[0]
    full = xp.concatenate([xp.zeros(1, dtype=xp.float64), xscalker])

    if want_grad:
        F, ga = gradlogitmod(full, xp=xp)
    else:
        F = logitmod(full, xp=xp)

    wfull = xp.sqrt(F)
    wker = wfull[1:]

    if not want_grad:
        return wker, None

    if n_kern == 0:
        return wker, xp.zeros((0, 0), dtype=xp.float64)

    # dwker[i, j] = (1 / (2 * wker[i])) * ga[j + 1, i + 1]
    #            = 0.5 * (ga[1:, 1:]).T[i, j] / wker[i]
    sub = ga[1:, 1:]  # (n_kern, n_kern); ga[a, b] = dF_b/dfull_a
    dwker = 0.5 * sub.T / wker[:, None]
    return wker, dwker


# ---------------------------------------------------------------------------
# MORP ordinal-threshold (cumulative-increment) reparameterization
# ---------------------------------------------------------------------------

def thresh_reparam(
    bthresh: NDArray,
    numthresh: Sequence[int],
    *,
    want_grad: bool = False,
    xp=None,
) -> tuple[NDArray, Optional[NDArray]]:
    r"""Cumulative-increment reparam of ordinal thresholds -> ordered cut points.

    Ports the MORP threshold parameterization (ordered-YJ driver): the leading
    ``nthresh``-block of ``theta`` is partitioned into per-ordinal sub-blocks of
    sizes ``numthresh``. Within each sub-block the raw increments
    ``(b_0, b_1, \dots, b_{m-1})`` map to strictly-ordered cut points

    .. math::

        \tau_0 = b_0, \qquad \tau_k = \tau_{k-1} + e^{b_k}\ \ (k \ge 1),

    i.e. ``tau = cumsum([b_0, exp(b_1), ..., exp(b_{m-1})])``. The ``exp`` of the
    spacings guarantees ``tau_0 < tau_1 < \dots`` for any real ``bthresh`` (the
    identification-preserving monotone parameterization used by the fixed MORP
    facade, ``models/morp/_morp_loglik._unpack_morp_params``).

    Parameters
    ----------
    bthresh : array_like, shape (nthresh,)
        Raw threshold parameters (GAUSS ``x[1:nthresh]``), concatenated across
        the ordinal dimensions in ``numthresh`` order.
    numthresh : sequence of int, shape (nord,)
        Per-ordinal threshold counts (``n_categories[d] - 1``). Must sum to
        ``len(bthresh)``.
    want_grad : bool, default False
        If True, also return the ``(nthresh, nthresh)`` Jacobian
        ``d tau[i] / d bthresh[j]`` (block-diagonal; lower-triangular within
        each ordinal block).
    xp : backend, optional
        Array backend. Inferred from ``bthresh`` if not given.

    Returns
    -------
    tau : ndarray, shape (nthresh,)
        Ordered cut points, concatenated across ordinal dimensions.
    dtau : ndarray or None, shape (nthresh, nthresh)
        Jacobian ``d tau / d bthresh`` when ``want_grad``, else ``None``.

    Notes
    -----
    Within a block, ``d tau_k / d b_j = 1`` for ``j = 0`` (all ``k``), and for
    ``j \ge 1`` equals ``e^{b_j}`` when ``k \ge j`` else ``0``. Writing
    ``g_j = 1`` if ``j = 0`` else ``e^{b_j}``, the block Jacobian is
    ``J[k, j] = \mathbf{1}[k \ge j]\, g_j`` -- a lower-triangular matrix scaled
    column-wise by ``g``.

    Raises
    ------
    ValueError
        If ``sum(numthresh) != len(bthresh)`` or any count is negative.
    """
    if xp is None:
        xp = array_namespace(bthresh)

    b = xp.array(bthresh, dtype=xp.float64).reshape(-1)
    numthresh = [int(m) for m in numthresh]
    if any(m < 0 for m in numthresh):
        raise ValueError(f"numthresh entries must be non-negative, got {numthresh}")
    total = int(sum(numthresh))
    if total != int(b.shape[0]):
        raise ValueError(
            f"sum(numthresh)={total} does not match len(bthresh)={int(b.shape[0])}"
        )

    tau_blocks: list[NDArray] = []
    row_blocks: list[NDArray] = []
    offset = 0
    for m in numthresh:
        if m == 0:
            continue
        blk = b[offset : offset + m]
        # incr = [b_0, exp(b_1), ..., exp(b_{m-1})]; tau = cumsum(incr).
        incr = xp.concatenate([blk[:1], xp.exp(blk[1:])])
        tau_blocks.append(xp.cumsum(incr))

        if want_grad:
            # g_j = d incr_j / d b_j = 1 (j == 0) else exp(b_j) == incr_j.
            g = xp.concatenate([xp.ones(1, dtype=xp.float64), xp.exp(blk[1:])])
            # J[k, j] = 1[k >= j] * g_j (lower-triangular, column-scaled).
            tri = xp.zeros((m, m), dtype=xp.float64)
            ii, jj = xp.tril_indices(m)
            tri[ii, jj] = 1.0
            jblk = tri * g[None, :]
            # Pad to full width and stack for a block-diagonal Jacobian.
            left = xp.zeros((m, offset), dtype=xp.float64)
            right = xp.zeros((m, total - offset - m), dtype=xp.float64)
            row_blocks.append(xp.concatenate([left, jblk, right], axis=1))
        offset += m

    if total == 0:
        tau = xp.zeros(0, dtype=xp.float64)
    else:
        tau = xp.concatenate(tau_blocks)

    if not want_grad:
        return tau, None

    if total == 0:
        return tau, xp.zeros((0, 0), dtype=xp.float64)
    dtau = xp.concatenate(row_blocks, axis=0)
    return tau, dtau


# ---------------------------------------------------------------------------
# Realized random-coefficient state
# ---------------------------------------------------------------------------

@dataclass
class RCState:
    """Realized random-coefficient reparameterization for one ``theta``.

    Produced by :meth:`ParamSpace.unpack`. Non-gradient fields are always
    populated; the ``d*`` gradient fields are populated only when
    ``want_grad=True`` and are ``None`` otherwise.

    Attributes
    ----------
    xmu : ndarray, shape (n_beta,)
        Fixed coefficients after the sign reparameterization.
    x11chol : ndarray, shape (nrndcoef, nrndcoef)
        Upper-triangular correlation Cholesky factor (``x11chol' x11chol =
        omegastar``); the scalar ``1.0`` when ``nrndcoef <= 1``.
    omegastar : ndarray, shape (nrndcoef, nrndcoef)
        Random-coefficient correlation matrix.
    wscalrand : ndarray, shape (nrndcoef,)
        Random-coefficient scale (std-dev) vector.
    wdiagrand : ndarray, shape (nrndcoef, nrndcoef)
        ``diag(wscalrand)``.
    xlamrnd : ndarray, shape (nrndcoef,)
        Yeo-Johnson power parameters in ``(0, 2)``.
    mulamrnd, siglamrnd : ndarray, shape (nrndcoef,)
        Mean and std of the YJ-inverse of a standard normal, from ``meanyj``.
    dxmudxmu1 : ndarray or None, shape (n_beta,)
        Diagonal of ``d xmu / d beta`` (estimation space only).
    gker, gscal : ndarray or None
        Correlation-Cholesky gradient blocks from ``gnewcholparmcorscaled``.
    gtempstar : ndarray or None
        Alias of ``gker`` used by the score chain.
    dwscalranddxscalrand : ndarray or None, shape (nrndcoef, nrndcoef)
        Diagonal Jacobian ``d wscalrand / d xscalrand`` (``= diag(wscalrand)``).
    dxlamrnddxlam : ndarray or None, shape (nrndcoef, nrndcoef)
        Diagonal Jacobian ``d xlamrnd / d xlam`` (``= diag(2 pdlogit)``).
    dmulamrnddxlamrnd, dsiglamrnddxlamrnd : ndarray or None, shape (nrndcoef,)
        Gradients of the ``meanyj`` outputs w.r.t. ``xlamrnd``.
    """

    xmu: NDArray
    x11chol: NDArray
    omegastar: NDArray
    wscalrand: NDArray
    wdiagrand: NDArray
    xlamrnd: NDArray
    mulamrnd: NDArray
    siglamrnd: NDArray
    dxmudxmu1: Optional[NDArray] = None
    gker: Optional[NDArray] = None
    gscal: Optional[NDArray] = None
    gtempstar: Optional[NDArray] = None
    dwscalranddxscalrand: Optional[NDArray] = None
    dxlamrnddxlam: Optional[NDArray] = None
    dmulamrnddxlamrnd: Optional[NDArray] = None
    dsiglamrnddxlamrnd: Optional[NDArray] = None


def _reparam_xmu(
    bmu: NDArray, spec: "MixingSpec", *, want_grad: bool
) -> tuple[NDArray, Optional[NDArray]]:
    """Sign reparameterization of the fixed coefficients (GAUSS ``xmu`` block).

    Reproduces ``MIXMNL.gss`` ``lpr``/``lgd`` (lines 396-406 / 491-507): with no
    strict-sign variables ``xmu = bmu`` and ``d xmu / d bmu = 1``; otherwise the
    ``varneg`` positions carry ``-exp(bmu)`` and the ``varpos`` positions carry
    ``+exp(bmu)`` (the remaining positions stay linear).

    Parameters
    ----------
    bmu : ndarray, shape (n_beta,)
        Raw ``beta`` block of ``theta``.
    spec : MixingSpec
        Provides the sign masks.
    want_grad : bool
        If True also return the diagonal ``d xmu / d bmu``.

    Returns
    -------
    xmu : ndarray, shape (n_beta,)
    dxmudxmu1 : ndarray or None, shape (n_beta,)
        Diagonal Jacobian when ``want_grad``, else ``None``.
    """
    bmu = np.asarray(bmu, dtype=np.float64)
    if spec.nvarneg + spec.nvarpos == 0:
        xmu = bmu.copy()
        dxmudxmu1 = np.ones(spec.n_beta, dtype=np.float64) if want_grad else None
        fixed = spec.fix_location_zero_mask > 0
        xmu[fixed] = 0.0
        if want_grad:
            dxmudxmu1[fixed] = 0.0
        return xmu, dxmudxmu1

    # Indexed assignment (GAUSS MIXMNL.gss:397-401): only the sign-constrained
    # rows are exponentiated, so an unconstrained coefficient > ~709 cannot
    # produce ``exp(bmu) = inf`` and poison the linear rows via ``0 * inf = NaN``.
    neg = spec.indxvarneg > 0
    pos = spec.indxvarpos > 0
    xmu = bmu.copy()
    xmu[neg] = -safe_exp(bmu[neg])
    xmu[pos] = safe_exp(bmu[pos])
    dxmudxmu1 = None
    if want_grad:
        dxmudxmu1 = spec.indxvarnonegpos.astype(float).copy()
        dxmudxmu1[neg] = -safe_exp(bmu[neg])
        dxmudxmu1[pos] = safe_exp(bmu[pos])
    fixed = spec.fix_location_zero_mask > 0
    xmu[fixed] = 0.0
    if want_grad:
        dxmudxmu1[fixed] = 0.0
    return xmu, dxmudxmu1


class ParamSpace:
    """Strategy interface mapping ``theta`` -> :class:`RCState`.

    Concrete strategies (:class:`EstimationSpace`, :class:`ReportingSpace`)
    realize the two GAUSS parameterizations over a common
    :class:`~pybhatlib.mixed._spec.MixingSpec`.

    Parameters
    ----------
    layout : ParamLayout
        Block partition of ``theta``.
    scal : float, default 1.0
        Scaling constant in the correlation-Cholesky parameterization.
    intordn1 : int, default 20
        Gauss-Hermite node count for ``meanyj``.
    spher : bool, default False
        Spherical (True) vs radial (False) correlation Cholesky. Only the
        radial path is implemented.
    """

    def __init__(
        self,
        layout: ParamLayout,
        *,
        scal: float = 1.0,
        intordn1: int = 20,
        spher: bool = False,
    ) -> None:
        if spher:
            raise NotImplementedError(
                "spherical correlation parameterization (spher=True) is not "
                "yet supported in the mixed engine"
            )
        self.layout = layout
        self.scal = float(scal)
        self.intordn1 = int(intordn1)
        self.spher = bool(spher)

    def unpack(
        self, theta: NDArray, spec: "MixingSpec", *, want_grad: bool = False
    ) -> RCState:  # pragma: no cover - abstract
        """Realize the random-coefficient quantities for ``theta``.

        Parameters
        ----------
        theta : ndarray, shape (n_theta,)
            Optimizer parameter vector.
        spec : MixingSpec
            Index masks and counts.
        want_grad : bool, default False
            Populate the gradient (``d*``) fields of the returned state.

        Returns
        -------
        RCState
        """
        raise NotImplementedError


class EstimationSpace(ParamSpace):
    """The ``lpr``/``lgd`` estimation-space parameterization.

    Correlation via ``newcholparmscaled`` (unconstrained -> Cholesky factor),
    scale via ``exp``, YJ power via ``2 * cdlogit`` (maps the real line into the
    open ``(0, 2)`` interval). Supplies the full gradient chain when
    ``want_grad=True``.
    """

    def unpack(
        self, theta: NDArray, spec: "MixingSpec", *, want_grad: bool = False
    ) -> RCState:
        theta = np.asarray(theta, dtype=np.float64)
        sl = self.layout.slices()
        k = spec.nrndcoef

        xmu, dxmudxmu1 = _reparam_xmu(theta[sl["beta"]], spec, want_grad=want_grad)

        xrand = theta[sl["rcor"]]
        xscalrand = theta[sl["scal"]]
        xlam = theta[sl["lam"]]

        # --- correlation Cholesky (MIXMNL 411-419, 521-533) -----------------
        # Joint (MNP) case: the ``rcor`` block parameterizes the full
        # ``nrndtot = nrndcoef + kernel_dim`` correlation, but the pipeline
        # correlates only the ``nrndcoef`` random-coefficient draws through the
        # RC sub-block Cholesky (GAUSS ``x11 = posrand' omegastar posrand``,
        # ``x11chol = chol(x11)``; MNPKERCP.gss lines 612-617). The kernel reads
        # its own joint ``omegastar`` from ``theta`` directly, so ``rc.omegastar``
        # here is the RC sub-block (kept ``k x k`` for the pipeline score chain).
        gker = gscal = gtempstar = None
        joint = spec.nrndtot > spec.nrndcoef
        if self.layout.n_rcor == 0 and k > 1:
            x11chol = np.eye(k, dtype=np.float64)
            omegastar = np.eye(k, dtype=np.float64)
            if want_grad:
                gker = np.zeros((0, 0), dtype=np.float64)
                gtempstar = gker
        elif joint:
            cholall = newcholparmscaled(xrand, self.scal)
            omegastar_joint = np.asarray(cholall).T @ np.asarray(cholall)
            omegastar = omegastar_joint[:k, :k].copy()
            if k > 1:
                x11chol = safe_cholesky(omegastar)[0].T
                if want_grad:
                    gker, gscal = gnewcholparmcorscaled(omegastar, self.scal)
                    gtempstar = gker
            else:
                x11chol = np.array(1.0)
                if want_grad:
                    gker = np.array(0.0)
                    gtempstar = gker
        elif k > 1:
            x11chol = newcholparmscaled(xrand, self.scal)
            omegastar = x11chol.T @ x11chol
            if want_grad:
                gker, gscal = gnewcholparmcorscaled(omegastar, self.scal)
                gtempstar = gker
        else:
            x11chol = np.array(1.0)
            omegastar = np.ones((k, k), dtype=np.float64)
            if want_grad:
                gker = np.array(0.0)
                gtempstar = gker

        # --- scale (MIXMNL 421-425, 535-541) --------------------------------
        wscalrand = safe_exp(xscalrand)
        wdiagrand = np.diag(wscalrand)
        dwscalranddxscalrand = np.diag(wscalrand) if want_grad else None

        # --- Yeo-Johnson power (MIXMNL 426-427, 542-544) --------------------
        actlam = np.asarray(spec.actlam, dtype=np.float64)
        mapped_lam = np.clip(2.0 * cdlogit(xlam), _LAM_EPS, 2.0 - _LAM_EPS)
        xlamrnd = actlam * mapped_lam + (1.0 - actlam)
        dxlamrnddxlam = (
            np.diag(actlam * 2.0 * pdlogit(xlam)) if want_grad else None
        )
        if want_grad:
            mulamrnd, siglamrnd, dmulamrnddxlamrnd, dsiglamrnddxlamrnd = gradmeanyj(
                xlamrnd, self.intordn1
            )
        else:
            mulamrnd, siglamrnd = meanyj(xlamrnd, self.intordn1)
            dmulamrnddxlamrnd = dsiglamrnddxlamrnd = None

        return RCState(
            xmu=xmu,
            x11chol=x11chol,
            omegastar=omegastar,
            wscalrand=wscalrand,
            wdiagrand=wdiagrand,
            xlamrnd=xlamrnd,
            mulamrnd=mulamrnd,
            siglamrnd=siglamrnd,
            dxmudxmu1=dxmudxmu1,
            gker=gker,
            gscal=gscal,
            gtempstar=gtempstar,
            dwscalranddxscalrand=dwscalranddxscalrand,
            dxlamrnddxlam=dxlamrnddxlam,
            dmulamrnddxlamrnd=dmulamrnddxlamrnd,
            dsiglamrnddxlamrnd=dsiglamrnddxlamrnd,
        )


class ReportingSpace(ParamSpace):
    """The ``lpr1`` reporting-space parameterization (standard errors).

    Correlation entries are read directly via ``matndupdiagonefull`` (the
    ``rcor`` block *is* the off-diagonal correlation), and both the scale and
    ``lambda`` blocks are entered directly (no ``exp`` / no ``cdlogit``). This
    space carries no gradient blocks: ``want_grad`` is accepted but the ``d*``
    fields remain ``None`` (GAUSS ``lpr1`` has no gradient companion).
    """

    def unpack(
        self, theta: NDArray, spec: "MixingSpec", *, want_grad: bool = False
    ) -> RCState:
        del want_grad  # reporting space never emits reparam gradients
        theta = np.asarray(theta, dtype=np.float64)
        sl = self.layout.slices()
        k = spec.nrndcoef

        # xmu is the reported coefficient directly (no exp reparam)
        xmu = theta[sl["beta"]].copy()
        xmu[spec.fix_location_zero_mask > 0] = 0.0

        xrand = theta[sl["rcor"]]
        # --- correlation from direct entries (MIXMNL 657-663) ---------------
        if self.layout.n_rcor == 0 and k > 1:
            omegastar = np.eye(k, dtype=np.float64)
            x11chol = np.eye(k, dtype=np.float64)
        elif k > 1:
            omegastar = nearest_pd_correlation(matndupdiagonefull(xrand))
            # GAUSS chol() returns upper-triangular R with R' R = omega
            x11chol = safe_cholesky(omegastar)[0].T
        else:
            omegastar = np.ones((k, k), dtype=np.float64)
            x11chol = np.array(1.0)

        # --- scale and lambda entered directly (MIXMNL 665-668) -------------
        wscalrand = theta[sl["scal"]].copy()
        wdiagrand = np.diag(wscalrand)
        actlam = np.asarray(spec.actlam, dtype=np.float64)
        xlamrnd = actlam * theta[sl["lam"]] + (1.0 - actlam)
        mulamrnd, siglamrnd = meanyj(xlamrnd, self.intordn1)

        return RCState(
            xmu=xmu,
            x11chol=x11chol,
            omegastar=omegastar,
            wscalrand=wscalrand,
            wdiagrand=wdiagrand,
            xlamrnd=xlamrnd,
            mulamrnd=mulamrnd,
            siglamrnd=siglamrnd,
        )


# gcholeskycor is imported for downstream score assembly (see _assemble_score
# in the engine); re-exported here so the reparam module is the single owner of
# the estimation/reporting correlation transforms.
__all__ = [
    "ParamLayout",
    "RCState",
    "ParamSpace",
    "EstimationSpace",
    "ReportingSpace",
    "wker_reparam",
    "thresh_reparam",
    "gcholeskycor",
]
