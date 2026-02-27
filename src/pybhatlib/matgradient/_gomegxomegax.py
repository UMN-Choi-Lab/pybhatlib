"""Gradient of A = X Omega X' w.r.t. symmetric matrix Omega.

Implements the gomegxomegax procedure from BHATLIB's Matgradient.src.

For A = X @ Omega @ X.T where Omega is a symmetric (K x K) matrix and
X is an (N x K) matrix, this computes dA/dOmega following the row-based
arrangement convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace, get_backend


def gomegxomegax(X: NDArray, omega: NDArray, *, xp=None) -> NDArray:
    """Compute dA/dOmega for A = X @ Omega @ X.T.

    Uses BHATLIB's row-based arrangement: rows of the gradient correspond to
    upper-triangular elements of Omega (row by row), and columns correspond
    to upper-triangular elements of A.

    Parameters
    ----------
    X : ndarray, shape (N, K)
        Design matrix.
    omega : ndarray, shape (K, K)
        Symmetric matrix (covariance or correlation).
    xp : backend, optional
        Array backend.

    Returns
    -------
    grad : ndarray, shape (K*(K+1)//2, N*(N+1)//2)
        Gradient dA/dOmega in row-based arrangement.

    Notes
    -----
    For the paper's example (p. 3):
    X (2x3), Omega (3x3) → A (2x2)
    grad has shape (6, 3) — 6 upper-tri elements of Omega, 3 of A.

    The gradient entry (r, c) gives da_{ij}/domega_{kl} where
    (i,j) is the c-th upper-triangular element of A and
    (k,l) is the r-th upper-triangular element of Omega.
    """
    if xp is None:
        xp = array_namespace(X, omega)

    X = xp.array(X, dtype=xp.float64)
    omega = xp.array(omega, dtype=xp.float64)

    N, K = X.shape[0], X.shape[1]
    n_omega = K * (K + 1) // 2  # upper-tri elements of Omega
    n_a = N * (N + 1) // 2      # upper-tri elements of A

    grad = xp.zeros((n_omega, n_a), dtype=xp.float64)

    # A_{ij} = sum_{k,l} X_{ik} * Omega_{kl} * X_{jl}
    # dA_{ij}/dOmega_{kl} = X_{ik} * X_{jl} + X_{il} * X_{jk} (if k != l)
    #                     = X_{ik} * X_{jk}                     (if k == l)

    # Enumerate upper-triangular indices
    omega_indices = []
    for k in range(K):
        for l in range(k, K):
            omega_indices.append((k, l))

    a_indices = []
    for i in range(N):
        for j in range(i, N):
            a_indices.append((i, j))

    for r, (k, l) in enumerate(omega_indices):
        for c, (i, j) in enumerate(a_indices):
            if k == l:
                grad[r, c] = X[i, k] * X[j, k]
            else:
                grad[r, c] = X[i, k] * X[j, l] + X[i, l] * X[j, k]

    return grad
