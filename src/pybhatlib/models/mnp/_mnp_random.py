"""Random-coefficient covariance in correlation/log-SD coordinates."""

import numpy as np

from pybhatlib.matgradient._radial import newcholparmscaled, gnewcholparmcorscaled
from pybhatlib.utils._safe_reparam import safe_exp


def random_covariance(parameters, n_rand, control):
    """Decode [radial correlations | log SDs], as in mixed MNP.

    randdiag omits the correlation block; it never changes scale coordinates.
    Return the lower Cholesky factor and the structural covariance.
    """
    n_corr = 0 if control.randdiag else n_rand * (n_rand - 1) // 2
    scales = safe_exp(parameters[n_corr:n_corr + n_rand])
    upper = (newcholparmscaled(parameters[:n_corr], control.scal)
             if n_corr else np.eye(n_rand))
    lower = scales[:, None] * upper.T
    return lower, lower @ lower.T


def random_covariance_adjoint(adjoint, parameters, n_rand, control):
    """Chain covariance derivatives to radial correlations and log SDs."""
    n_corr = 0 if control.randdiag else n_rand * (n_rand - 1) // 2
    scales = safe_exp(parameters[n_corr:n_corr + n_rand])
    upper = (newcholparmscaled(parameters[:n_corr], control.scal)
             if n_corr else np.eye(n_rand))
    corr = upper.T @ upper
    scale_gradient = 2.0 * scales * ((adjoint * corr) @ scales)
    if not n_corr:
        return scale_gradient
    jacobian, _ = gnewcholparmcorscaled(corr, control.scal)
    corr_gradient = np.array([
        2.0 * adjoint[i, j] * scales[i] * scales[j]
        for i in range(n_rand) for j in range(i + 1, n_rand)
    ])
    return np.r_[jacobian @ corr_gradient, scale_gradient]
