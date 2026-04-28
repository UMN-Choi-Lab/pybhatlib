"""MORP model control structure.

Configures the estimation procedure for the Multivariate Ordered Response
Probit (MORP) model, analogous to BHATLIB's morpControl struct.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = ["MORPControl", "morp_control_replace", "morp_control_asdict"]

# Canonical field names for MORPControl (used by shims below).
#
# IMPORTANT — when adding/removing a field on ``MORPControl.__init__``,
# update this tuple in lock-step.  A missed update is *silent*:
# ``morp_control_replace`` will reject the field as "unknown" while
# ``morp_control_asdict`` will simply omit it, which may then surface as
# a NotImplementedError or a missing attribute deep inside ``morp_loglik``.
_MORP_CONTROL_FIELDS = (
    "iid",
    "correst",
    "heteronly",
    "method",
    "spherical",
    "maxiter",
    "tol",
    "optimizer",
    "verbose",
    "seed",
    "startb",
)


class MORPControl:
    """Control structure for MORP model estimation.

    Attributes
    ----------
    iid : bool
        If True, assume independent errors across dimensions.
    correst : NDArray or None
        Correlation restriction matrix. None means no restrictions.
    heteronly : bool
        If True, estimate only heteroscedastic errors (no correlations).
    method : str
        MVNCD method: "me", "ovus", "tvbs", "bme", "ovbs", "ssj", "scipy".
    spherical : bool
        If True, use spherical parameterization for correlation matrix.
    maxiter : int
        Maximum optimizer iterations.
    tol : float
        Convergence tolerance (gradient norm).
    optimizer : str
        Optimization method: "bfgs" or "lbfgsb".
    verbose : int
        Verbosity: 0=silent, 1=summary, 2=per-iteration.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values.
    analytic_grad : bool
        If True (default), use the analytic log-likelihood gradient when
        ``method`` is one of ``{"me", "ovus"}``. Otherwise, or when
        ``method`` selects an unsupported MVNCD approximation, fall back
        silently to central finite differences.

    Notes
    -----
    The ``indep`` parameter and attribute are deprecated. Use ``iid`` instead.
    Passing ``indep=True`` at construction emits a :class:`DeprecationWarning`
    and sets ``iid`` accordingly.

    Examples
    --------
    >>> MORPControl(iid=True)         # canonical
    >>> MORPControl(indep=True)       # deprecated, emits DeprecationWarning
    """

    def __init__(
        self,
        iid: bool = False,
        correst: NDArray | None = None,
        heteronly: bool = False,
        method: str = "ovus",
        spherical: bool = True,
        maxiter: int = 200,
        tol: float = 1e-5,
        optimizer: Literal["bfgs", "lbfgsb"] = "bfgs",
        verbose: int = 1,
        seed: int | None = None,
        startb: NDArray | None = None,
        analytic_grad: bool = True,
        *,
        # Deprecated alias — must be keyword-only to avoid position conflicts
        indep: bool | None = None,
    ) -> None:
        if indep is not None:
            warnings.warn(
                "MORPControl(indep=...) is deprecated; use MORPControl(iid=...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            iid = bool(indep)

        self.iid = iid
        self.correst = correst
        self.heteronly = heteronly
        self.method = method
        self.spherical = spherical
        self.maxiter = maxiter
        self.tol = tol
        self.optimizer = optimizer
        self.verbose = verbose
        self.seed = seed
        self.startb = startb
        self.analytic_grad = analytic_grad

    # ------------------------------------------------------------------
    # Backward-compat property for ctrl.indep / ctrl.indep = v
    # ------------------------------------------------------------------

    @property
    def indep(self) -> bool:
        """Deprecated alias for :attr:`iid`.

        .. deprecated::
            Use :attr:`iid` instead.
        """
        warnings.warn(
            "MORPControl.indep is deprecated; use MORPControl.iid instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.iid

    @indep.setter
    def indep(self, value: bool) -> None:
        warnings.warn(
            "MORPControl.indep is deprecated; use MORPControl.iid instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.iid = bool(value)

    def __repr__(self) -> str:
        return (
            f"MORPControl(iid={self.iid!r}, heteronly={self.heteronly!r}, "
            f"method={self.method!r}, spherical={self.spherical!r}, "
            f"maxiter={self.maxiter!r}, tol={self.tol!r}, "
            f"optimizer={self.optimizer!r}, verbose={self.verbose!r}, "
            f"seed={self.seed!r}, analytic_grad={self.analytic_grad!r})"
        )


# ---------------------------------------------------------------------------
# dataclasses-compatibility shims
# ---------------------------------------------------------------------------
# MORPControl is a plain class (not a @dataclass) so that it can support the
# ``indep=`` keyword-only deprecation alias.  Code that previously relied on
# ``dataclasses.replace`` / ``dataclasses.asdict`` should use these helpers.


def morp_control_replace(ctrl: "MORPControl", **changes: Any) -> "MORPControl":
    """Return a new :class:`MORPControl` with selected fields replaced.

    Drop-in replacement for ``dataclasses.replace(ctrl, ...)`` for callers
    that were using the old ``@dataclass``-based ``MORPControl``.

    Parameters
    ----------
    ctrl : MORPControl
        The control instance to copy.
    **changes
        Fields to override.  Only canonical field names are accepted;
        the deprecated ``indep`` alias is intentionally *not* supported here.

    Returns
    -------
    MORPControl
        A fresh instance with ``changes`` applied on top of ``ctrl``'s values.

    Raises
    ------
    TypeError
        If any key in ``changes`` is not a recognised MORPControl field.

    Examples
    --------
    >>> ctrl = MORPControl(maxiter=100)
    >>> ctrl2 = morp_control_replace(ctrl, maxiter=500, verbose=0)
    >>> ctrl2.maxiter
    500
    """
    unknown = set(changes) - set(_MORP_CONTROL_FIELDS)
    if unknown:
        raise TypeError(
            f"morp_control_replace() got unknown field(s): {sorted(unknown)}"
        )
    current: dict[str, Any] = {f: getattr(ctrl, f) for f in _MORP_CONTROL_FIELDS}
    current.update(changes)
    return MORPControl(**current)


def morp_control_asdict(ctrl: "MORPControl") -> dict[str, Any]:
    """Return a shallow-copy dict of all canonical :class:`MORPControl` fields.

    Drop-in replacement for ``dataclasses.asdict(ctrl)`` for callers that
    were using the old ``@dataclass``-based ``MORPControl``.

    Parameters
    ----------
    ctrl : MORPControl
        The control instance to convert.

    Returns
    -------
    dict
        Mapping of field name → current value for all canonical fields.

    Examples
    --------
    >>> ctrl = MORPControl(iid=True, verbose=0)
    >>> d = morp_control_asdict(ctrl)
    >>> d["iid"]
    True
    """
    return {f: getattr(ctrl, f) for f in _MORP_CONTROL_FIELDS}
