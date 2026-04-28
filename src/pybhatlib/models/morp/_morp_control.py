"""MORP model control structure.

Configures the estimation procedure for the Multivariate Ordered Response
Probit (MORP) model, analogous to BHATLIB's morpControl struct.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from numpy.typing import NDArray


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
