"""MNP model control structure.

Equivalent to BHATLIB's mnpControl struct, this dataclass configures
the estimation procedure for the Multinomial Probit model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MNPControl:
    """Control structure for MNP model estimation.

    Attributes
    ----------
    iid : bool
        If True, assume IID error terms (homoscedastic, uncorrelated).
        If False, estimate flexible covariance structure.
    mix : bool
        If True, include random coefficients (mixed MNP).
    indep : bool
        If True, assume independence across ordinal outcomes.
    correst : NDArray or None
        Correlation restriction matrix. Upper-diagonal matrix with 1s on
        diagonal, 1s in off-diagonal positions indicate active correlations.
        None means no restrictions (full correlation).
    heteronly : bool
        If True, estimate only heteroscedastic errors (no correlations).
    randdiag : bool
        If True, random coefficients covariance is diagonal (uncorrelated).
    nseg : int
        Number of mixture-of-normals segments. 1 = single normal,
        >1 = discrete mixture of normals.
    method : str
        MVNCD approximation method: "me", "ovus", "tvbs", "bme", "ovbs",
        "ssj", "scipy".
    spherical : bool
        If True, use spherical parameterization for correlation matrix.
    scal : float
        Scaling factor for starting values.
    IID_first : bool
        If True, run IID model first to obtain starting values.
    want_covariance : bool
        If True, compute parameter covariance matrix at convergence.
    seed10 : int
        Secondary seed for QMC sequences.
    perms : int
        Number of permutations for MVNCD variable reordering.
    maxiter : int
        Maximum number of optimizer iterations.
    tol : float
        Convergence tolerance (gradient norm).
    optimizer : str
        Optimization method: "bfgs", "lbfgsb", "torch_adam", "torch_lbfgs".
    verbose : int
        Verbosity: 0=silent, 1=summary, 2=per-iteration.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values for parameters.
    """

    iid: bool = False
    mix: bool = False
    indep: bool = False
    correst: NDArray | None = None
    heteronly: bool = False
    randdiag: bool = False
    nseg: int = 1
    method: str = "ovus"
    spherical: bool = True
    scal: float = 1.0
    IID_first: bool = True
    want_covariance: bool = True
    seed10: int = 1234
    perms: int = 0
    maxiter: int = 200
    tol: float = 1e-5
    optimizer: Literal["bfgs", "lbfgsb", "torch_adam", "torch_lbfgs"] = "bfgs"
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
