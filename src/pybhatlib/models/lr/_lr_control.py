"""Controls for analytical, single-outcome linear regression.

Unlike the optimizer-based models there are no iteration, tolerance, or
simulation settings: the least-squares solution is computed in closed form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class LRControl:
    """Covariance options for :class:`LRModel`; estimation is always analytical.

    Attributes
    ----------
    se_method : {"hessian", "sandwich", "bhhh"}
        ``"hessian"`` is the classical OLS covariance ``sigma2 * (X'X)^-1``;
        ``"sandwich"`` is the White heteroscedasticity-robust covariance;
        ``"bhhh"`` is the inverse of the Gaussian score cross-product.
    want_covariance : bool
        If False, skip inference: ``se`` / ``t_stat`` / ``p_value`` /
        ``cov_matrix`` are filled with NaN.
    df_correction : bool
        Multiply the covariance by ``N / (N - K)``.  With the default
        ``"hessian"`` method this yields the usual unbiased residual-variance
        estimator ``SSE / (N - K)``; with ``"sandwich"`` it is the HC1 form.
    verbose : int
        Package convention: 0 = silent, 1 = summary.

    Notes
    -----
    Classical (``"hessian"``) inference uses Student's t with ``N - K``
    degrees of freedom; robust and BHHH inference use normal asymptotics.
    """

    se_method: Literal["hessian", "sandwich", "bhhh"] = "hessian"
    want_covariance: bool = True
    df_correction: bool = True
    verbose: int = 1

    def __post_init__(self) -> None:
        if self.se_method not in ("hessian", "sandwich", "bhhh"):
            raise ValueError("se_method must be 'hessian', 'sandwich', or 'bhhh'")
