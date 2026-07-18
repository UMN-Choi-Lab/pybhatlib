"""Utility functions."""

from pybhatlib.utils._seeds import set_seed
from pybhatlib.utils._validation import check_symmetric, check_positive_definite
from pybhatlib.utils._quadrature import SUPPORTED_ORDERS, gauss_hermite
from pybhatlib.utils._logistic import (
    cdlogit,
    d_lam_d_lamnew,
    gradlogitmod,
    gradpdlogit,
    lam_from_lamnew,
    logitmod,
    pdlogit,
)
from pybhatlib.utils._safe_reparam import (
    corr_from_angle,
    nearest_pd_correlation,
    safe_cholesky,
    safe_exp,
    softmax,
)

__all__ = [
    "set_seed",
    "check_symmetric",
    "check_positive_definite",
    "gauss_hermite",
    "SUPPORTED_ORDERS",
    "cdlogit",
    "pdlogit",
    "gradpdlogit",
    "lam_from_lamnew",
    "d_lam_d_lamnew",
    "logitmod",
    "gradlogitmod",
    "safe_exp",
    "corr_from_angle",
    "safe_cholesky",
    "nearest_pd_correlation",
    "softmax",
]
