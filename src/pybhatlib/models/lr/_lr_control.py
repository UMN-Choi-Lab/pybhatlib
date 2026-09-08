"""Controls for analytical, single-outcome linear regression."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LRControl:
    """Covariance options; estimation always uses analytical least squares.

    ``hessian`` is classical OLS covariance, ``sandwich`` is White robust
    covariance, and ``bhhh`` is inverse Gaussian score cross-products.
    ``df_correction`` multiplies covariance by N / (N - K); with the default
    hessian method this gives the usual unbiased residual-variance estimator.
    Classical inference uses Student's t; robust/BHHH inference uses normal
    asymptotics. No optimizer or simulation controls are needed.

    ``verbose`` follows the package convention: 0 = silent, 1 = summary.
    """

    se_method: Literal["hessian", "sandwich", "bhhh"] = "hessian"
    want_covariance: bool = True
    df_correction: bool = True
    verbose: int = 1

    def __post_init__(self):
        if self.se_method not in ("hessian", "sandwich", "bhhh"):
            raise ValueError("se_method must be 'hessian', 'sandwich', or 'bhhh'")
