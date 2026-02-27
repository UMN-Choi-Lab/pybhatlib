"""Additional distribution functions: logistic, skew-normal, skew-t, Gumbel.

These are used in various econometric models supported by BHATLIB.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import array_namespace, get_backend


# --- Logistic distribution ---


def logistic_pdf(x: float, mu: float = 0.0, s: float = 1.0) -> float:
    """Logistic distribution PDF."""
    z = (x - mu) / s
    ez = np.exp(-z)
    return ez / (s * (1 + ez) ** 2)


def logistic_cdf(x: float, mu: float = 0.0, s: float = 1.0) -> float:
    """Logistic distribution CDF."""
    z = (x - mu) / s
    return 1.0 / (1.0 + np.exp(-z))


def mv_logistic_cdf(x: NDArray, sigma: NDArray, *, xp=None) -> float:
    """Multivariate logistic CDF approximation.

    Uses the relationship between multivariate logistic and multivariate normal
    distributions for approximation.
    """
    if xp is None:
        xp = get_backend("numpy")
    # Scale factor: logistic variance = pi^2/3 * s^2
    # For standard logistic, variance = pi^2/3
    # Approximate via MVN with scaled covariance
    scale = np.sqrt(3.0) / np.pi
    x_scaled = np.asarray(x) * scale
    sigma_np = np.asarray(sigma)
    sigma_scaled = sigma_np * scale**2

    from pybhatlib.gradmvn._mvncd import mvncd

    return mvncd(xp.array(x_scaled), xp.array(sigma_scaled), xp=xp)


# --- Skew-normal distribution ---


def skew_normal_pdf(
    x: float, alpha: float = 0.0, mu: float = 0.0, sigma: float = 1.0
) -> float:
    """Skew-normal distribution PDF.

    f(x) = (2/sigma) * phi((x-mu)/sigma) * Phi(alpha*(x-mu)/sigma)

    Parameters
    ----------
    x : float
        Evaluation point.
    alpha : float
        Skewness parameter.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    """
    z = (x - mu) / sigma
    return 2.0 / sigma * norm.pdf(z) * norm.cdf(alpha * z)


def skew_normal_cdf(
    x: float, alpha: float = 0.0, mu: float = 0.0, sigma: float = 1.0
) -> float:
    """Skew-normal distribution CDF (Owen's T function approximation)."""
    z = (x - mu) / sigma
    # For alpha = 0, reduces to standard normal CDF
    if abs(alpha) < 1e-15:
        return float(norm.cdf(z))

    # Use the relationship: F(x; alpha) = Phi(z) - 2*T(z, alpha)
    # where T is Owen's T function
    from scipy.special import owens_t

    return float(norm.cdf(z) - 2.0 * owens_t(z, alpha))


# --- Skew-t distribution ---


def skew_t_pdf(
    x: float,
    alpha: float = 0.0,
    nu: float = 5.0,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> float:
    """Skew-t distribution PDF.

    Parameters
    ----------
    x : float
        Evaluation point.
    alpha : float
        Skewness parameter.
    nu : float
        Degrees of freedom (> 2 for finite variance).
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    """
    from scipy.special import gamma as gamma_fn

    z = (x - mu) / sigma
    t_pdf = (
        gamma_fn((nu + 1) / 2)
        / (gamma_fn(nu / 2) * np.sqrt(nu * np.pi))
        * (1 + z**2 / nu) ** (-(nu + 1) / 2)
    )
    # Skew-t: 2 * t_pdf * T_nu+1(alpha * z * sqrt((nu+1)/(z^2+nu)))
    from scipy.stats import t as t_dist

    arg = alpha * z * np.sqrt((nu + 1) / (z**2 + nu))
    return float(2.0 / sigma * t_pdf * t_dist.cdf(arg, df=nu + 1))


# --- Gumbel (Type I extreme value) distribution ---


def gumbel_pdf(x: float, mu: float = 0.0, beta: float = 1.0) -> float:
    """Gumbel (maximum) distribution PDF.

    f(x) = (1/beta) * exp(-(z + exp(-z))) where z = (x - mu) / beta
    """
    z = (x - mu) / beta
    return (1.0 / beta) * np.exp(-(z + np.exp(-z)))


def gumbel_cdf(x: float, mu: float = 0.0, beta: float = 1.0) -> float:
    """Gumbel (maximum) distribution CDF.

    F(x) = exp(-exp(-(x-mu)/beta))
    """
    z = (x - mu) / beta
    return np.exp(-np.exp(-z))


def reverse_gumbel_pdf(x: float, mu: float = 0.0, beta: float = 1.0) -> float:
    """Reverse Gumbel (minimum) distribution PDF.

    f(x) = (1/beta) * exp(z - exp(z)) where z = (x - mu) / beta
    """
    z = (x - mu) / beta
    return (1.0 / beta) * np.exp(z - np.exp(z))


def reverse_gumbel_cdf(x: float, mu: float = 0.0, beta: float = 1.0) -> float:
    """Reverse Gumbel (minimum) distribution CDF.

    F(x) = 1 - exp(-exp((x-mu)/beta))
    """
    z = (x - mu) / beta
    return 1.0 - np.exp(-np.exp(z))
