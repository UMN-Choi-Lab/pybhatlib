"""MDCEV model control structure.

Configures the estimation procedure for the Multiple Discrete-Continuous
Extreme Value (MDCEV) model.  Both the traditional outside-good utility
and the linear outside-good utility specifications are supported via the
``utility`` attribute.

References
----------
Bhat, C.R. (2008), "The Multiple Discrete-Continuous Extreme Value (MDCEV)
Model: Role of Utility Function Parameters, Identification Considerations,
and Model Extensions," Transportation Research Part B, Vol. 42, No. 3,
pp. 274-303.

Bhat, C.R. (2018), "A New Flexible Multiple Discrete-Continuous Extreme
Value (MDCEV) Choice Model," Transportation Research Part B.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MDCEVControl:
    """Control structure for MDCEV model estimation.

    Attributes
    ----------
    utility : str
        Outside-good utility specification:
        ``"trad"``   – traditional MDCEV (Bhat 2008); the Jacobian term
                       includes all alternatives and vdisc/vcont are
                       differenced from ln(qty_outside).
        ``"linear"`` – linear outside-good MDCEV (Bhat 2018); the
                       Jacobian uses only inside goods and vdisc/vcont
                       are not adjusted by ln(qty_outside).
    maxiter : int
        Maximum number of optimizer iterations (_max_maxIters in GAUSS).
    optimizer : str
        Optimization method. ``"bfgs"`` mirrors the GAUSS default
        (``_max_Options = { bfgs stepbt }``); ``"lbfgsb"`` is also
        supported.
    tol : float
        Convergence tolerance (gradient norm).
    want_covariance : bool
        If True, compute the parameter covariance matrix at convergence
        (_max_CovPar in GAUSS).
    analytic_grad : bool
        If True, use the analytic gradient (``lgd`` procedure in GAUSS).
        If False, use numerical gradient.
    se_method : str
        Standard-error method: "bhhh" (default, matches GAUSS _max_CovPar=2),
        "hessian" (inverse observed information), or "sandwich" (robust).
    se_diagnostic : bool
        If True, compute *all three* SE estimators (BHHH, observed Hessian,
        sandwich) at convergence so they can be compared side by side; the
        reported ``se`` still follows ``se_method``. When False (default) only
        the building blocks needed for ``se_method`` are computed — this avoids
        the expensive observed-Hessian finite-difference pass when only BHHH is
        wanted (parity with MORP/MNP ``se_diagnostic``).
    verbose : int
        Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values. If None, defaults to zeros for
        beta/gamma parameters and 0 for log(sigma), matching
        ``b = zeros(nvarm,1)|-1000|zeros(nvargam-1,1)|0`` in GAUSS.
    outside_good_gamma : float
        Fixed gamma utility value for the outside good.  Set to -1000
        in GAUSS (``u[:,0] = -1000``) so the outside good has no
        satiation parameter.
    weight_var : str or None
        Column name for observation weights.  If None, all weights = 1
        (equivalent to ``{ weight,wtind } = indices(dataset,"uno")``
        in GAUSS).
    """

    utility: Literal["trad", "linear"] = "trad"
    maxiter: int = 2000
    optimizer: Literal["bfgs", "lbfgsb"] = "bfgs"
    tol: float = 1e-5
    want_covariance: bool = True
    analytic_grad: bool = True
    se_method: Literal["bhhh", "hessian", "sandwich"] = "bhhh"
    se_diagnostic: bool = False
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
    outside_good_gamma: float = -1000.0
    weight_var: str | None = None
