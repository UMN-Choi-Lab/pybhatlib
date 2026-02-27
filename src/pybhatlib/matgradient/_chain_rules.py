"""Matrix chain rule helpers for BHATLIB gradient computations.

BHATLIB uses row-based arrangement for all gradient matrices. The chain rule
follows the form:

    dA/d(omega) = dA/dOmega @ dOmega/d(omega)

Note: due to row-based arrangement, the chain rule is applied in this exact
order (not reversed), as described on p. 4 of the BHATLIB paper (Eq. 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace


def chain_grad(
    dA_dOmega: NDArray,
    dOmega_dparam: NDArray,
    *,
    xp=None,
) -> NDArray:
    """Apply matrix chain rule: dA/dparam = dOmega/dparam @ dA/dOmega.

    In BHATLIB's row-based convention (Eq. 3 of the paper):
        dA/d(omega) = dOmega/d(omega) @ dA/dOmega

    where dimensions are:
        dA/dOmega: (n_omega, n_a)
        dOmega/dparam: (n_param, n_omega)
        result: (n_param, n_a)

    Parameters
    ----------
    dA_dOmega : ndarray, shape (n_omega, n_a)
        Gradient of output w.r.t. intermediate matrix.
    dOmega_dparam : ndarray, shape (n_param, n_omega)
        Gradient of intermediate w.r.t. parameters.
    xp : backend, optional
        Array backend.

    Returns
    -------
    result : ndarray, shape (n_param, n_a)
        Gradient of output w.r.t. parameters.
    """
    if xp is None:
        xp = array_namespace(dA_dOmega, dOmega_dparam)

    return xp.matmul(dOmega_dparam, dA_dOmega)


def chain_grad_omega_theta(
    dA_dOmega: NDArray,
    dOmega_domegastar: NDArray,
    domegastar_dtheta: NDArray,
    *,
    xp=None,
) -> NDArray:
    """Full chain rule: dA/dTheta = dOmega*/dTheta @ dOmega/dOmega* @ dA/dOmega.

    This computes the gradient of A w.r.t. the unconstrained spherical
    parameters theta, passing through correlation and covariance matrices.

    Parameters
    ----------
    dA_dOmega : ndarray, shape (n_omega, n_a)
        Gradient of A w.r.t. covariance elements.
    dOmega_domegastar : ndarray, shape (n_corr, n_omega)
        Gradient of covariance w.r.t. correlation elements (from gradcovcor).
    domegastar_dtheta : ndarray, shape (n_theta, n_corr_full)
        Gradient of correlation w.r.t. theta (from grad_corr_theta).
    xp : backend, optional

    Returns
    -------
    result : ndarray, shape (n_theta, n_a)
    """
    if xp is None:
        xp = array_namespace(dA_dOmega, dOmega_domegastar, domegastar_dtheta)

    # dA/dOmega* = dOmega/dOmega* @ dA/dOmega
    dA_domegastar = xp.matmul(dOmega_domegastar, dA_dOmega)

    # dA/dTheta = dOmega*/dTheta @ dA/dOmega*
    # Need to handle dimension matching: domegastar_dtheta maps theta to
    # full upper-tri of Omega*, but dA_domegastar only uses off-diagonal corr elements
    return xp.matmul(domegastar_dtheta, dA_domegastar)
