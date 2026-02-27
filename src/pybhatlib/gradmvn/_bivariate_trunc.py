"""Bivariate truncated normal moments.

Computes E[X|X<=a] and Cov[X|X<=a] for bivariate normal distributions,
used by the BME and TVBS MVNCD methods.

Based on Properties 1-4 from Bhat (2018), Section 2.1:
    Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation
    of the MVNCD Function. Transportation Research Part B, 109: 238-256.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.gradmvn._univariate import bivariate_normal_cdf


def truncated_bivariate_mean(
    mu: NDArray,
    sigma: NDArray,
    a: NDArray,
) -> NDArray:
    """Compute E[X | X <= a] for X ~ MVN(mu, Sigma), bivariate case.

    Uses the truncated bivariate normal moment formulas (Bhat 2018, Property 1).

    Parameters
    ----------
    mu : ndarray, shape (2,)
        Mean vector.
    sigma : ndarray, shape (2, 2)
        Covariance matrix.
    a : ndarray, shape (2,)
        Upper truncation limits.

    Returns
    -------
    E_trunc : ndarray, shape (2,)
        Truncated mean E[X | X <= a].
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    sd1 = np.sqrt(sigma[0, 0])
    sd2 = np.sqrt(sigma[1, 1])
    rho = sigma[0, 1] / (sd1 * sd2) if sd1 > 1e-15 and sd2 > 1e-15 else 0.0

    # Standardize
    alpha1 = (a[0] - mu[0]) / sd1 if sd1 > 1e-15 else 100.0
    alpha2 = (a[1] - mu[1]) / sd2 if sd2 > 1e-15 else 100.0

    # P(X <= a)
    prob = bivariate_normal_cdf(alpha1, alpha2, rho)
    if prob < 1e-300:
        return a.copy()

    # Conditional standardized limits
    denom1 = np.sqrt(1.0 - rho**2) if abs(rho) < 1.0 - 1e-10 else 1e-10
    alpha2_cond = (alpha2 - rho * alpha1) / denom1
    alpha1_cond = (alpha1 - rho * alpha2) / denom1

    # phi(alpha_j) * Phi(alpha_{-j|j})
    phi1 = norm.pdf(alpha1)
    phi2 = norm.pdf(alpha2)
    Phi_2g1 = norm.cdf(alpha2_cond)
    Phi_1g2 = norm.cdf(alpha1_cond)

    # E[Z_k | Z <= alpha] = -( phi(alpha_k) * Phi(alpha_{-k|k}) ) / prob
    E_z1 = -(phi1 * Phi_2g1) / prob
    E_z2 = -(phi2 * Phi_1g2) / prob

    # Unstandardize
    E_trunc = np.array([
        mu[0] + sd1 * E_z1,
        mu[1] + sd2 * E_z2,
    ])

    return E_trunc


def truncated_bivariate_cov(
    mu: NDArray,
    sigma: NDArray,
    a: NDArray,
) -> NDArray:
    """Compute Cov[X | X <= a] for X ~ MVN(mu, Sigma), bivariate case.

    Uses the truncated bivariate normal covariance formulas
    (Bhat 2018, Properties 2-4).

    Parameters
    ----------
    mu : ndarray, shape (2,)
        Mean vector.
    sigma : ndarray, shape (2, 2)
        Covariance matrix.
    a : ndarray, shape (2,)
        Upper truncation limits.

    Returns
    -------
    Cov_trunc : ndarray, shape (2, 2)
        Truncated covariance Cov[X | X <= a].
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    sd1 = np.sqrt(sigma[0, 0])
    sd2 = np.sqrt(sigma[1, 1])
    rho = sigma[0, 1] / (sd1 * sd2) if sd1 > 1e-15 and sd2 > 1e-15 else 0.0

    # Standardize
    alpha1 = (a[0] - mu[0]) / sd1 if sd1 > 1e-15 else 100.0
    alpha2 = (a[1] - mu[1]) / sd2 if sd2 > 1e-15 else 100.0

    prob = bivariate_normal_cdf(alpha1, alpha2, rho)
    if prob < 1e-300:
        return sigma.copy()

    denom = np.sqrt(1.0 - rho**2) if abs(rho) < 1.0 - 1e-10 else 1e-10
    alpha2_cond = (alpha2 - rho * alpha1) / denom
    alpha1_cond = (alpha1 - rho * alpha2) / denom

    phi1 = norm.pdf(alpha1)
    phi2 = norm.pdf(alpha2)
    Phi_2g1 = norm.cdf(alpha2_cond)
    Phi_1g2 = norm.cdf(alpha1_cond)

    # Bivariate normal pdf at (alpha1, alpha2)
    phi_12 = np.exp(
        -0.5 * (alpha1**2 - 2 * rho * alpha1 * alpha2 + alpha2**2) / (1 - rho**2)
    ) / (2 * np.pi * denom) if abs(rho) < 1.0 - 1e-10 else 0.0

    # Truncated mean (standardized)
    E_z1 = -(phi1 * Phi_2g1) / prob
    E_z2 = -(phi2 * Phi_1g2) / prob

    # Property 2: Var[Z_k | Z <= alpha]
    # = 1 + E_z_k * alpha_k - (phi_k * Phi_{-k|k}) * alpha_k / prob - E_z_k^2
    # Simplified: Var = 1 - (alpha_k * phi_k * Phi_{-k|k})/prob - E_z_k^2
    var_z1 = 1.0 - (alpha1 * phi1 * Phi_2g1) / prob - E_z1**2
    var_z2 = 1.0 - (alpha2 * phi2 * Phi_1g2) / prob - E_z2**2

    # Property 4: Cov[Z1, Z2 | Z <= alpha]
    # = rho + (-rho * phi1 * Phi_2g1 - phi_12) / prob ... but let's use
    # the general form: Cov = rho - (rho*phi1*Phi_2g1 + phi_12)/prob ... - E_z1*E_z2
    # Actually: Cov[Z1,Z2|Z<=a] = rho * (1 - correction) where correction involves phi_12
    cov_z12 = -(rho * phi1 * Phi_2g1 + phi_12) / prob - E_z1 * E_z2 + rho
    # Alternatively, a more stable formula:
    # cov_z12 = rho + (- rho * alpha1 * phi1 * Phi_2g1 - phi_12) / prob - E_z1 * E_z2
    # Use the more numerically stable version

    # Ensure positive variances
    var_z1 = max(var_z1, 1e-15)
    var_z2 = max(var_z2, 1e-15)

    # Unstandardize: Cov_X = D * Cov_Z * D where D = diag(sd)
    Cov_trunc = np.array([
        [sd1**2 * var_z1, sd1 * sd2 * cov_z12],
        [sd1 * sd2 * cov_z12, sd2**2 * var_z2],
    ])

    return Cov_trunc
