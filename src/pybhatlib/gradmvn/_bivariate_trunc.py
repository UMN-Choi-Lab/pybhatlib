"""Bivariate truncated normal moments.

Computes E[X|X<=a] and Cov[X|X<=a] for bivariate normal distributions,
used by the BME and TVBS MVNCD methods.

Based on Properties 1-4 from Bhat (2018), Section 2.1:
    Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation
    of the MVNCD Function. Transportation Research Part B, 109: 238-256.

Property 1 (Eq. 3-5):
    lambda_i = -(delta_i + rho * delta_j) / Phi_2
    where delta_i = phi(a_i) * Phi((a_j - rho*a_i) / sqrt(1 - rho^2))

Property 2 (Eq. 5 + Appendix A):
    Var[Z_i | Z<=a] = 1 - (w_i*delta_i + rho^2*w_j*delta_j
                            - (1-rho^2)*rho*phi_12) / Phi_2 - lambda_i^2
    Cov[Z_1,Z_2|Z<=a] = rho - (rho*w_1*delta_1 + rho*w_2*delta_2
                                - (1-rho^2)*phi_12) / Phi_2 - lambda_1*lambda_2

where w_i = a_i, phi_12 = bivariate_normal_pdf(a_1, a_2, rho).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.gradmvn._univariate import bivariate_normal_cdf


def _bvn_pdf(a1: float, a2: float, rho: float) -> float:
    """Standard bivariate normal PDF at (a1, a2) with correlation rho."""
    if abs(rho) >= 1.0 - 1e-10:
        return 0.0
    onemrho2 = 1.0 - rho**2
    exponent = -(a1**2 - 2.0 * rho * a1 * a2 + a2**2) / (2.0 * onemrho2)
    return np.exp(exponent) / (2.0 * np.pi * np.sqrt(onemrho2))


def truncated_bivariate_mean(
    mu: NDArray,
    sigma: NDArray,
    a: NDArray,
) -> NDArray:
    """Compute E[X | X <= a] for X ~ MVN(mu, Sigma), bivariate case.

    Uses Bhat (2018) Property 1, Eq. (3-4):
        lambda_i = -(delta_i + rho * delta_j) / Phi_2

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
    rho = max(-0.9999, min(0.9999, rho))

    # Standardize
    alpha1 = (a[0] - mu[0]) / sd1 if sd1 > 1e-15 else 100.0
    alpha2 = (a[1] - mu[1]) / sd2 if sd2 > 1e-15 else 100.0

    # P(Z <= alpha)
    prob = bivariate_normal_cdf(alpha1, alpha2, rho)
    if prob < 1e-300:
        return a.copy()

    # Conditional standardized limits
    denom = np.sqrt(1.0 - rho**2) if abs(rho) < 1.0 - 1e-10 else 1e-10
    alpha2_cond = (alpha2 - rho * alpha1) / denom
    alpha1_cond = (alpha1 - rho * alpha2) / denom

    # delta_i = phi(alpha_i) * Phi((alpha_j - rho*alpha_i) / sqrt(1-rho^2))
    phi1 = norm.pdf(alpha1)
    phi2 = norm.pdf(alpha2)
    Phi_2g1 = norm.cdf(alpha2_cond)
    Phi_1g2 = norm.cdf(alpha1_cond)

    delta1 = phi1 * Phi_2g1
    delta2 = phi2 * Phi_1g2

    # Bhat (2018) Eq. (4): lambda_i = -(delta_i + rho * delta_j) / Phi_2
    lambda1 = -(delta1 + rho * delta2) / prob
    lambda2 = -(delta2 + rho * delta1) / prob

    # Unstandardize
    E_trunc = np.array([
        mu[0] + sd1 * lambda1,
        mu[1] + sd2 * lambda2,
    ])

    return E_trunc


def truncated_bivariate_cov(
    mu: NDArray,
    sigma: NDArray,
    a: NDArray,
) -> NDArray:
    """Compute Cov[X | X <= a] for X ~ MVN(mu, Sigma), bivariate case.

    Uses Bhat (2018) Property 2, Eq. (5) + Appendix A:
        Var[Z_i] = 1 - (w_i*delta_i + rho^2*w_j*delta_j
                        - (1-rho^2)*rho*phi_12) / Phi_2 - lambda_i^2
        Cov[Z_1,Z_2] = rho - (rho*w_1*delta_1 + rho*w_2*delta_2
                                - (1-rho^2)*phi_12) / Phi_2 - lambda_1*lambda_2

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
    rho = max(-0.9999, min(0.9999, rho))

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

    delta1 = phi1 * Phi_2g1
    delta2 = phi2 * Phi_1g2

    # Truncated mean (standardized) — Eq. (4)
    lambda1 = -(delta1 + rho * delta2) / prob
    lambda2 = -(delta2 + rho * delta1) / prob

    # Bivariate normal PDF at (alpha1, alpha2)
    phi_12 = _bvn_pdf(alpha1, alpha2, rho)

    # Bhat (2018) Eq. (5) + Appendix A
    # w_i = alpha_i (the standardized truncation limit)
    w1 = alpha1
    w2 = alpha2
    onemrho2 = 1.0 - rho**2

    # Var[Z_1 | Z <= alpha]
    var_z1 = (1.0
              - (w1 * delta1 + rho**2 * w2 * delta2
                 - onemrho2 * rho * phi_12) / prob
              - lambda1**2)

    # Var[Z_2 | Z <= alpha] (by symmetry, swap indices)
    var_z2 = (1.0
              - (w2 * delta2 + rho**2 * w1 * delta1
                 - onemrho2 * rho * phi_12) / prob
              - lambda2**2)

    # Cov[Z_1, Z_2 | Z <= alpha]
    cov_z12 = (rho
               - (rho * w1 * delta1 + rho * w2 * delta2
                  - onemrho2 * phi_12) / prob
               - lambda1 * lambda2)

    # Ensure positive variances
    var_z1 = max(var_z1, 1e-15)
    var_z2 = max(var_z2, 1e-15)

    # Unstandardize: Cov_X = D * Cov_Z * D where D = diag(sd)
    Cov_trunc = np.array([
        [sd1**2 * var_z1, sd1 * sd2 * cov_z12],
        [sd1 * sd2 * cov_z12, sd2**2 * var_z2],
    ])

    return Cov_trunc
