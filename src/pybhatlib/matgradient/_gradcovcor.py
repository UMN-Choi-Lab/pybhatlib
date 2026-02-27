"""Gradient of covariance w.r.t. std devs and correlations.

Implements the gradcovcor procedure from BHATLIB's Matgradient.src.

For a covariance matrix Omega = omega * Omega* * omega where:
- omega is a diagonal matrix of standard deviations
- Omega* is the correlation matrix

This computes dOmega/d(omega) and dOmega/d(Omega*).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


@dataclass
class GradCovCorResult:
    """Result of gradcovcor computation.

    Attributes
    ----------
    glitomega : ndarray, shape (K, K*(K+1)//2)
        Gradient of upper-triangular covariance elements w.r.t. K std deviations.
        Each column provides derivatives of a covariance element w.r.t. the K
        std dev (litomega) elements.
    gomegastar : ndarray, shape (K*(K-1)//2, K*(K+1)//2)
        Gradient of upper-triangular covariance elements w.r.t. correlation elements.
        Each column provides derivatives of a covariance element w.r.t. the
        K*(K-1)/2 correlation elements.
    """

    glitomega: NDArray
    gomegastar: NDArray


def gradcovcor(capomega: NDArray, *, xp=None) -> GradCovCorResult:
    """Compute gradients of covariance matrix w.r.t. std devs and correlations.

    For Omega = omega * Omega* * omega, computes dOmega/d(omega) and dOmega/d(Omega*).
    All matrices follow BHATLIB's row-based arrangement convention.

    Parameters
    ----------
    capomega : ndarray, shape (K, K)
        Covariance matrix (must be symmetric, positive semi-definite).
    xp : backend, optional
        Array backend. Inferred from input if not provided.

    Returns
    -------
    result : GradCovCorResult
        Contains glitomega and gomegastar gradient matrices.
    """
    if xp is None:
        xp = array_namespace(capomega)

    capomega = xp.array(capomega, dtype=xp.float64)
    K = capomega.shape[0]
    n_cov = K * (K + 1) // 2  # number of upper-triangular elements
    n_corr = K * (K - 1) // 2  # number of off-diagonal correlation elements

    # Extract standard deviations (sqrt of diagonal)
    omega_diag = xp.sqrt(xp.diagonal(capomega))

    # Compute correlation matrix: Omega* = inv(omega) @ Omega @ inv(omega)
    omega_inv = xp.zeros((K, K), dtype=xp.float64)
    for i in range(K):
        if omega_diag[i] > 0:
            omega_inv[i, i] = 1.0 / omega_diag[i]
    omegastar = omega_inv @ capomega @ omega_inv

    # --- Compute glitomega: dOmega/d(omega) ---
    # Omega_{ij} = omega_i * omega_j * Omega*_{ij}
    # dOmega_{ij}/d(omega_k) = delta_{ik} * omega_j * Omega*_{ij}
    #                        + omega_i * delta_{jk} * Omega*_{ij}
    glitomega = xp.zeros((K, n_cov), dtype=xp.float64)

    col = 0
    for i in range(K):
        for j in range(i, K):
            # Covariance element (i, j) = omega_i * omega_j * Omega*_{ij}
            for k in range(K):
                deriv = 0.0
                if k == i:
                    deriv += omega_diag[j] * omegastar[i, j]
                if k == j:
                    deriv += omega_diag[i] * omegastar[i, j]
                glitomega[k, col] = deriv
            col += 1

    # Check if capomega is already a correlation matrix (all diagonal = 1)
    is_corr = all(abs(capomega[i, i] - 1.0) < 1e-10 for i in range(K))
    if is_corr:
        # If input is a correlation matrix, glitomega is ad hoc value of 1
        glitomega = xp.ones((K, n_cov), dtype=xp.float64)

    # --- Compute gomegastar: dOmega/d(Omega*) ---
    # Omega_{ij} = omega_i * omega_j * Omega*_{ij}
    # dOmega_{ij}/d(Omega*_{kl}) = omega_i * omega_j * delta_{ik} * delta_{jl}
    #   (only for upper-triangular off-diagonal elements of Omega*)
    gomegastar = xp.zeros((n_corr, n_cov), dtype=xp.float64)

    col = 0
    for i in range(K):
        for j in range(i, K):
            # Row index in gomegastar for correlation element (k, l) with k < l
            row = 0
            for k in range(K):
                for l in range(k + 1, K):
                    if i == k and j == l:
                        gomegastar[row, col] = omega_diag[i] * omega_diag[j]
                    elif i == l and j == k:
                        gomegastar[row, col] = omega_diag[i] * omega_diag[j]
                    row += 1
            col += 1

    return GradCovCorResult(glitomega=glitomega, gomegastar=gomegastar)
