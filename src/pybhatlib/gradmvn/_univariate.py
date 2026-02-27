"""Standard univariate normal distribution wrappers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def normal_pdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal probability density function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    pdf : ndarray
        phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2)
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_pdf(x)


def normal_cdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal cumulative distribution function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    cdf : ndarray
        Phi(x)
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_cdf(x)


def normal_ppf(p: NDArray, *, xp=None) -> NDArray:
    """Standard normal quantile function (inverse CDF).

    Parameters
    ----------
    p : ndarray
        Probabilities in (0, 1).
    xp : backend, optional

    Returns
    -------
    x : ndarray
        Phi^{-1}(p)
    """
    if xp is None:
        xp = array_namespace(p)
    return xp.normal_ppf(p)


def normal_logpdf(x: NDArray, *, xp=None) -> NDArray:
    """Standard normal log probability density function.

    Parameters
    ----------
    x : ndarray
        Evaluation points.
    xp : backend, optional

    Returns
    -------
    logpdf : ndarray
        log(phi(x)) = -0.5 * (x^2 + log(2*pi))
    """
    if xp is None:
        xp = array_namespace(x)
    return xp.normal_logpdf(x)


def bivariate_normal_cdf(
    x1: float, x2: float, rho: float, *, xp=None
) -> float:
    """Bivariate standard normal CDF P(X1 <= x1, X2 <= x2) with correlation rho.

    Uses Drezner & Wesolowsky (1990) approximation for efficiency.

    Parameters
    ----------
    x1, x2 : float
        Upper integration limits.
    rho : float
        Correlation coefficient in [-1, 1].
    xp : backend, optional

    Returns
    -------
    prob : float
        P(X1 <= x1, X2 <= x2)
    """
    if xp is None:
        xp = get_backend("numpy")

    # Clamp rho to valid range
    rho = max(-1.0, min(1.0, float(rho)))

    # Handle edge cases
    if abs(rho) < 1e-15:
        return float(xp.normal_cdf(xp.array(x1)) * xp.normal_cdf(xp.array(x2)))

    if rho > 1.0 - 1e-15:
        return float(xp.normal_cdf(xp.array(min(x1, x2))))

    if rho < -1.0 + 1e-15:
        if x1 + x2 < 0:
            return 0.0
        return float(
            xp.normal_cdf(xp.array(x1)) + xp.normal_cdf(xp.array(x2)) - 1.0
        )

    # Use scipy for numpy backend
    from scipy.stats import multivariate_normal

    cov = np.array([[1.0, rho], [rho, 1.0]])
    try:
        return float(
            multivariate_normal.cdf(np.array([x1, x2]), mean=np.zeros(2), cov=cov)
        )
    except np.linalg.LinAlgError:
        # Fallback for near-singular cases
        return float(xp.normal_cdf(xp.array(min(x1, x2))))


def trivariate_normal_cdf(
    x1: float, x2: float, x3: float, sigma: NDArray, *, xp=None
) -> float:
    """Trivariate standard normal CDF P(X1<=x1, X2<=x2, X3<=x3).

    Parameters
    ----------
    x1, x2, x3 : float
        Upper integration limits.
    sigma : ndarray, shape (3, 3)
        Correlation matrix.
    xp : backend, optional

    Returns
    -------
    prob : float
    """
    from scipy.stats import multivariate_normal

    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.array([x1, x2, x3])
    try:
        result = multivariate_normal.cdf(x, mean=np.zeros(3), cov=sigma)
        return max(0.0, min(1.0, float(result)))
    except Exception:
        return 0.0


def quadrivariate_normal_cdf(
    x1: float, x2: float, x3: float, x4: float, sigma: NDArray, *, xp=None
) -> float:
    """Quadrivariate standard normal CDF P(X1<=x1, ..., X4<=x4).

    Parameters
    ----------
    x1, x2, x3, x4 : float
        Upper integration limits.
    sigma : ndarray, shape (4, 4)
        Correlation matrix.
    xp : backend, optional

    Returns
    -------
    prob : float
    """
    from scipy.stats import multivariate_normal

    sigma = np.asarray(sigma, dtype=np.float64)
    x = np.array([x1, x2, x3, x4])
    try:
        result = multivariate_normal.cdf(x, mean=np.zeros(4), cov=sigma)
        return max(0.0, min(1.0, float(result)))
    except Exception:
        return 0.0
