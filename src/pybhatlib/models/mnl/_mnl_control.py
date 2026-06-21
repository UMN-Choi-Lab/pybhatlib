"""MNL model control structure.

Equivalent to BHATLIB's MNL configuration globals, this dataclass
configures the estimation procedure for the Multinomial Logit (MNL) model.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MNLControl:
    """Control structure for MNL model estimation.

    Attributes
    ----------
    maxiter : int
        Maximum number of optimizer iterations (_max_maxIters in GAUSS).
    optimizer : str
        Optimization method. "newton" mirrors the GAUSS default
        (``_max_Options = { newton stepbt }``); "bfgs" and "lbfgsb" are
        also supported.
    tol : float
        Optimizer tolerance passed to scipy (``xtol`` for Newton-CG,
        ``gtol`` for BFGS/CG).
    tol_check : float
        Gradient-norm threshold used to flag convergence after the fit.
    want_covariance : bool
        If True, compute the parameter covariance matrix at convergence
        (_max_CovPar in GAUSS).
    analytic_grad : bool
        If True, use the analytic gradient (``lgd`` procedure in GAUSS).
        If False, use numerical gradient.
    analytic_hess : bool
        If True, use the analytic Hessian (``lhs`` procedure in GAUSS).
        Only used when optimizer = "newton".
    se_method : str
        Standard-error method: "bhhh" (default), "hessian" (inverse observed
        information), or "sandwich" (robust).
    verbose : int
        Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values. If None, defaults to zeros
        (matching ``b = zeros(ncoefunord, 1)`` in GAUSS).
    """

    maxiter: int = 2000
    optimizer: Literal["newton", "bfgs", "lbfgsb"] = "newton"
    tol: float = 1e-8
    tol_check: float = 1e-5
    want_covariance: bool = True
    analytic_grad: bool = True
    analytic_hess: bool = True
    se_method: Literal["bhhh", "hessian", "sandwich"] = "bhhh"
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
