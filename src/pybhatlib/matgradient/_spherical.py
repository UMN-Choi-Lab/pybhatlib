"""Spherical parameterization for positive-definite correlation matrices.

Implements the theta-to-correlation transformation used in BHATLIB for
unconstrained optimization of correlation matrices.

The idea: parameterize each column of the Cholesky factor L of the correlation
matrix Omega* using spherical coordinates (angles theta). Since L @ L.T is
automatically PD, any unconstrained theta yields a valid correlation matrix.

For a K x K correlation matrix:
- Number of free parameters: K*(K-1)//2 angles
- Each angle theta_{ij} (for j < i) is in (-pi, pi) but unconstrained in practice
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def theta_to_corr(theta: NDArray, K: int, *, xp=None) -> NDArray:
    """Convert unconstrained angles to a positive-definite correlation matrix.

    Uses spherical parameterization: the (i,j) element of the Cholesky factor L
    is defined as a product of sines and cosines of the angles.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained angle parameters.
    K : int
        Dimension of the correlation matrix.
    xp : backend, optional
        Array backend.

    Returns
    -------
    corr : ndarray, shape (K, K)
        Positive-definite correlation matrix with unit diagonal.
    """
    if xp is None:
        xp = array_namespace(theta)

    theta = xp.array(theta, dtype=xp.float64)
    n_params = K * (K - 1) // 2

    # Build the lower-triangular Cholesky-like factor via spherical coords
    # L[i,j] for j <= i, with L having unit-norm rows to ensure unit diagonal in L@L.T
    L = xp.zeros((K, K), dtype=xp.float64)

    idx = 0
    for i in range(K):
        for j in range(i + 1):
            if j == 0 and i == 0:
                L[i, j] = 1.0
            elif j == 0:
                # Product of sines of all previous angles for this row
                prod_sin = 1.0
                for m in range(i):
                    prod_sin *= np.sin(theta[idx + m]) if hasattr(theta, '__len__') else np.sin(float(theta))
                # Actually, use angles indexed for row i
                prod_sin = 1.0
                for m in range(j, i):
                    t_idx = _angle_index(i, m, K)
                    prod_sin *= np.sin(float(theta[t_idx]))
                L[i, j] = prod_sin
            elif j < i:
                prod_sin = 1.0
                for m in range(j):
                    t_idx = _angle_index(i, m, K)
                    prod_sin *= np.sin(float(theta[t_idx]))
                t_idx = _angle_index(i, j, K)
                L[i, j] = prod_sin * np.cos(float(theta[t_idx]))
            else:  # j == i
                prod_sin = 1.0
                for m in range(i):
                    t_idx = _angle_index(i, m, K)
                    prod_sin *= np.sin(float(theta[t_idx]))
                L[i, j] = prod_sin

    # Fix: use a cleaner implementation
    L = _build_cholesky_from_angles(theta, K, xp)
    corr = L @ xp.transpose(L)

    # Ensure exact unit diagonal
    for i in range(K):
        corr[i, i] = 1.0

    return corr


def _angle_index(i: int, j: int, K: int) -> int:
    """Map (i, j) with j < i to flat index in theta vector."""
    # Angles are stored row by row: row 1 has 1 angle, row 2 has 2, etc.
    return i * (i - 1) // 2 + j


def _build_cholesky_from_angles(theta: NDArray, K: int, xp) -> NDArray:
    """Build lower-triangular matrix L from spherical angles.

    L is constructed so that each row has unit norm, ensuring L@L.T
    has unit diagonal (i.e., is a correlation matrix).

    For row i (0-indexed):
      L[i, 0] = cos(theta[i,0])                                    if i > 0
      L[i, j] = cos(theta[i,j]) * prod_{m=0}^{j-1} sin(theta[i,m])  for 0 < j < i
      L[i, i] = prod_{m=0}^{i-1} sin(theta[i,m])
      L[0, 0] = 1
    """
    L = xp.zeros((K, K), dtype=xp.float64)
    L[0, 0] = 1.0

    for i in range(1, K):
        for j in range(i + 1):
            prod_sin = 1.0
            for m in range(j):
                t_idx = _angle_index(i, m, K)
                prod_sin *= np.sin(float(theta[t_idx]))

            if j < i:
                t_idx = _angle_index(i, j, K)
                L[i, j] = prod_sin * np.cos(float(theta[t_idx]))
            else:  # j == i
                L[i, j] = prod_sin

    return L


def grad_corr_theta(theta: NDArray, K: int, *, xp=None) -> NDArray:
    """Compute Jacobian dOmega*/dTheta of the spherical parameterization.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained angle parameters.
    K : int
        Dimension of the correlation matrix.
    xp : backend, optional
        Array backend.

    Returns
    -------
    jac : ndarray, shape (K*(K-1)//2, K*(K+1)//2)
        Jacobian mapping changes in theta to changes in upper-triangular
        elements of the correlation matrix (row-based order).
    """
    if xp is None:
        xp = array_namespace(theta)

    theta = xp.array(theta, dtype=xp.float64)
    n_theta = K * (K - 1) // 2
    n_corr_upper = K * (K + 1) // 2

    # Use numerical differentiation as a robust baseline
    # (Analytic version can be added for performance later)
    eps = 1e-7
    jac = xp.zeros((n_theta, n_corr_upper), dtype=xp.float64)

    corr_base = theta_to_corr(theta, K, xp=xp)
    base_vec = _corr_to_upper_vec(corr_base, K, xp)

    for p in range(n_theta):
        theta_plus = xp.copy(theta)
        theta_plus[p] += eps
        corr_plus = theta_to_corr(theta_plus, K, xp=xp)
        plus_vec = _corr_to_upper_vec(corr_plus, K, xp)

        theta_minus = xp.copy(theta)
        theta_minus[p] -= eps
        corr_minus = theta_to_corr(theta_minus, K, xp=xp)
        minus_vec = _corr_to_upper_vec(corr_minus, K, xp)

        for q in range(n_corr_upper):
            jac[p, q] = (plus_vec[q] - minus_vec[q]) / (2.0 * eps)

    return jac


def _corr_to_upper_vec(corr: NDArray, K: int, xp) -> NDArray:
    """Extract upper-triangular elements row-by-row from correlation matrix."""
    n = K * (K + 1) // 2
    vec = xp.zeros((n,), dtype=xp.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            vec[idx] = corr[i, j]
            idx += 1
    return vec
