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


def theta_to_corr(
    theta: NDArray, K: int, *, constrained: bool = False, xp=None
) -> NDArray:
    """Convert unconstrained angles to a positive-definite correlation matrix.

    Uses spherical parameterization: the (i,j) element of the Cholesky factor L
    is defined as a product of sines and cosines of the angles.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained angle parameters (mapped to (0, pi) via logistic),
        or constrained angles in (0, pi) if constrained=True.
    K : int
        Dimension of the correlation matrix.
    constrained : bool, default False
        If True, theta is already in (0, pi) and no logistic mapping is applied.
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

    # Map unconstrained parameters to (0, pi) via pi * logistic(theta)
    # GAUSS ref: cholspherparmunconst line 3726: sstar = pi*cdlogit(sdoubstar)
    if not constrained:
        theta = np.pi / (1.0 + np.exp(-theta))

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

    # GAUSS ref (lines 4202-4205): threshold tiny trig values to exact zero
    COS_THRESH = 6.12324e-17
    SIN_THRESH = 1.22465e-16

    for i in range(1, K):
        for j in range(i + 1):
            prod_sin = 1.0
            for m in range(j):
                t_idx = _angle_index(i, m, K)
                s = np.sin(float(theta[t_idx]))
                if abs(s) <= SIN_THRESH:
                    s = 0.0
                prod_sin *= s

            if j < i:
                t_idx = _angle_index(i, j, K)
                c = np.cos(float(theta[t_idx]))
                if abs(c) <= COS_THRESH:
                    c = 0.0
                L[i, j] = prod_sin * c
            else:  # j == i
                L[i, j] = prod_sin

    return L


def _corr_upper_index(a: int, b: int, K: int) -> int:
    """Map (a, b) with a <= b to flat index in upper-tri correlation vector."""
    return a * K - a * (a - 1) // 2 + (b - a)


def grad_corr_theta(
    theta: NDArray, K: int, *, constrained: bool = False, xp=None
) -> NDArray:
    """Compute analytic Jacobian dOmega*/dTheta of spherical parameterization.

    Each unconstrained parameter theta_p is mapped to an angle in (0, pi) via
    the logistic function, then used as a spherical coordinate in the Cholesky
    factor L. Since parameter theta_{p,q} only affects row p of L, the
    derivative dL/dtheta is sparse and the chain through corr = L @ L^T only
    touches row/column p of the correlation matrix.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained angle parameters (or constrained if constrained=True).
    K : int
        Dimension of the correlation matrix.
    constrained : bool, default False
        If True, theta is already in (0, pi). Must match theta_to_corr usage.
    xp : backend, optional
        Array backend (unused; computation uses NumPy internally).

    Returns
    -------
    jac : ndarray, shape (K*(K-1)//2, K*(K+1)//2)
        Jacobian mapping changes in theta to changes in upper-triangular
        elements of the correlation matrix (row-based order).
    """
    theta = np.asarray(theta, dtype=np.float64)
    n_theta = K * (K - 1) // 2
    n_corr_upper = K * (K + 1) // 2

    # Save unconstrained theta for logistic chain rule
    theta_u = theta.copy()

    # Map to constrained angles in (0, pi) if needed
    if not constrained:
        theta = np.pi / (1.0 + np.exp(-theta))

    # Precompute trig values with GAUSS thresholds (matching _build_cholesky)
    COS_THRESH = 6.12324e-17
    SIN_THRESH = 1.22465e-16
    cos_v = np.cos(theta)
    sin_v = np.sin(theta)
    cos_v[np.abs(cos_v) <= COS_THRESH] = 0.0
    sin_v[np.abs(sin_v) <= SIN_THRESH] = 0.0

    # Build lower-triangular Cholesky factor L from precomputed trig values
    L = np.zeros((K, K), dtype=np.float64)
    L[0, 0] = 1.0
    for i in range(1, K):
        for j in range(i + 1):
            prod_sin = 1.0
            for m in range(j):
                prod_sin *= sin_v[_angle_index(i, m, K)]
            if j < i:
                L[i, j] = prod_sin * cos_v[_angle_index(i, j, K)]
            else:
                L[i, j] = prod_sin

    jac = np.zeros((n_theta, n_corr_upper), dtype=np.float64)

    for p in range(1, K):
        for q in range(p):
            theta_idx = _angle_index(p, q, K)

            # --- Compute dL_row: d(L[p,:])/d(angle_{p,q}) ---
            dL_row = np.zeros(K, dtype=np.float64)

            # Prefix product: prod_{m=0}^{q-1} sin(angle_{p,m})
            prefix = 1.0
            for m in range(q):
                prefix *= sin_v[_angle_index(p, m, K)]

            # Direct term at j=q: d/dangle of [cos(angle) * prefix]
            dL_row[q] = -sin_v[_angle_index(p, q, K)] * prefix

            # Sine chain: for j > q, sin(angle_{p,q}) appears in the
            # product. Replace it with cos(angle_{p,q}).
            cos_q = cos_v[_angle_index(p, q, K)]
            suffix = 1.0  # accumulates prod_{m=q+1}^{j-1} sin(angle_{p,m})
            for j in range(q + 1, p + 1):
                if j < p:
                    dL_row[j] = prefix * cos_q * suffix * cos_v[
                        _angle_index(p, j, K)
                    ]
                    suffix *= sin_v[_angle_index(p, j, K)]
                else:  # j == p (diagonal entry)
                    dL_row[j] = prefix * cos_q * suffix

            # --- Chain through corr = L @ L^T ---
            # Only row p / column p of corr are affected.

            # corr[a, p] for a < p
            for a in range(p):
                val = 0.0
                for k in range(a + 1):  # L[a,k] non-zero for k <= a
                    val += L[a, k] * dL_row[k]
                jac[theta_idx, _corr_upper_index(a, p, K)] = val

            # corr[p, p] (diagonal — always 1, gradient should be ~0)
            jac[theta_idx, _corr_upper_index(p, p, K)] = (
                2.0 * np.dot(L[p, :p + 1], dL_row[:p + 1])
            )

            # corr[p, b] for b > p
            for b in range(p + 1, K):
                val = 0.0
                for k in range(p + 1):  # dL_row[k] non-zero for k <= p
                    val += dL_row[k] * L[b, k]
                jac[theta_idx, _corr_upper_index(p, b, K)] = val

    # Chain rule for unconstrained → constrained mapping:
    # d(angle)/d(theta_u) = pi * sigmoid(theta_u) * (1 - sigmoid(theta_u))
    if not constrained:
        sig = 1.0 / (1.0 + np.exp(-theta_u))
        chain = np.pi * sig * (1.0 - sig)
        for p_idx in range(n_theta):
            jac[p_idx, :] *= chain[p_idx]

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
