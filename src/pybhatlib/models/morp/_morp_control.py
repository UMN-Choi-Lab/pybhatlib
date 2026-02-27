"""MORP model control structure.

Configures the estimation procedure for the Multivariate Ordered Response
Probit (MORP) model, analogous to BHATLIB's morpControl struct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MORPControl:
    """Control structure for MORP model estimation.

    Attributes
    ----------
    indep : bool
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
        Optimization method.
    verbose : int
        Verbosity: 0=silent, 1=summary, 2=per-iteration.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values.
    """

    indep: bool = False
    correst: NDArray | None = None
    heteronly: bool = False
    method: str = "ovus"
    spherical: bool = True
    maxiter: int = 200
    tol: float = 1e-5
    optimizer: Literal["bfgs", "lbfgsb"] = "bfgs"
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
