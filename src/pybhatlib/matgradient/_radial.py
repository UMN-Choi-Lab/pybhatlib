"""Radial parameterization for positive-definite correlation matrices.

Implements the Van Oest (2019) radial parameterization used in BHATLIB.
Maps unconstrained parameters to (-1,1) via tanh, then builds the Cholesky
factor through accumulative sine products.  Each row of the upper-triangular
Cholesky factor L has unit norm, so L^T @ L is automatically a valid
correlation matrix.

The Cholesky structure is identical to the spherical parameterization
(Pinheiro & Bates), but the mapping to (-1,1) via tanh is numerically
simpler than cos/sin of angles in [0, pi].

References
----------
Van Oest, R. (2019). A new coefficient of interrater agreement: The
    challenge of highly unequal category proportions. Psychological
    Methods, 24(4), 439-451.

GAUSS reference: ``newcholparm`` (line 3752), ``gnewcholparm`` (line 4048).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def radial_to_corr(theta: NDArray, K: int) -> NDArray:
    """Convert unconstrained parameters to PD correlation matrix.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained parameters (mapped to (-1,1) via tanh).
    K : int
        Dimension of the correlation matrix.

    Returns
    -------
    corr : ndarray, shape (K, K)
        Positive-definite correlation matrix with unit diagonal.
    """
    theta = np.asarray(theta, dtype=np.float64)
    c = np.tanh(theta)
    s = np.sqrt(1.0 - c**2)  # = sech(theta)

    L = _build_radial_cholesky(c, s, K)
    corr = L.T @ L

    # Ensure exact unit diagonal
    np.fill_diagonal(corr, 1.0)
    return corr


def _build_radial_cholesky(c: NDArray, s: NDArray, K: int) -> NDArray:
    """Build upper-triangular Cholesky factor from radial parameters.

    Each row of L has unit norm, guaranteeing L^T @ L has unit diagonal.

    Parameters
    ----------
    c : ndarray, shape (K*(K-1)//2,)
        tanh(theta) values in (-1, 1).
    s : ndarray, shape (K*(K-1)//2,)
        sqrt(1-c^2) values in [0, 1].
    K : int
        Matrix dimension.

    Returns
    -------
    L : ndarray, shape (K, K)
        Upper-triangular Cholesky factor.
    """
    # Expand to K×K upper-triangular matrices with ones on diagonal
    Scos = np.eye(K, dtype=np.float64)
    Ssin = np.eye(K, dtype=np.float64)

    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            Scos[i, j] = c[idx]
            Ssin[i, j] = s[idx]
            idx += 1

    # Accumulative sine products (GAUSS newcholparm loop)
    for r in range(1, K):
        Scos[r:K, r:K] *= Ssin[r - 1, r:K]

    return Scos


def _param_index(p: int, q: int, K: int) -> int:
    """Map (p, q) with p < q to flat index in theta vector."""
    return p * K - p * (p + 1) // 2 + (q - p - 1)


def _corr_upper_index(a: int, b: int, K: int) -> int:
    """Map (a, b) with a <= b to flat index in upper-tri correlation vector."""
    return a * K - a * (a - 1) // 2 + (b - a)


def grad_radial_theta(theta: NDArray, K: int) -> NDArray:
    """Analytic Jacobian d(corr_upper)/d(theta) for radial parameterization.

    Derives the gradient by computing dL/dtheta for each parameter (which
    affects only one column of L), then chains through corr = L^T @ L.

    Parameters
    ----------
    theta : ndarray, shape (K*(K-1)//2,)
        Unconstrained parameters.
    K : int
        Dimension of the correlation matrix.

    Returns
    -------
    jac : ndarray, shape (K*(K-1)//2, K*(K+1)//2)
        Jacobian: jac[p_idx, c_idx] = d(corr_upper[c_idx]) / d(theta[p_idx]).
    """
    n_theta = K * (K - 1) // 2
    n_corr_upper = K * (K + 1) // 2
    theta = np.asarray(theta, dtype=np.float64)

    c = np.tanh(theta)
    s = np.sqrt(1.0 - c**2)

    # Build Cholesky factor L (upper-triangular)
    L = _build_radial_cholesky(c, s, K)

    # Store c, s in matrix form for lookup
    c_mat = np.zeros((K, K), dtype=np.float64)
    s_mat = np.ones((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i + 1, K):
            c_mat[i, j] = c[idx]
            s_mat[i, j] = s[idx]
            idx += 1

    jac = np.zeros((n_theta, n_corr_upper), dtype=np.float64)

    for p in range(K):
        for q in range(p + 1, K):
            theta_idx = _param_index(p, q, K)

            # --- Compute dL[:,q] for parameter theta_{pq} ---
            dL_q = np.zeros(K, dtype=np.float64)

            # Direct term: dL[p,q]/dtheta = s_{pq}^2 * prod_{m<p} s_{mq}
            prod_s = 1.0
            for m in range(p):
                prod_s *= s_mat[m, q]
            dL_q[p] = s_mat[p, q] ** 2 * prod_s

            # Sine product terms: dL[i,q]/dtheta = -c_{pq} * L[i,q]
            for i in range(p + 1, q + 1):
                dL_q[i] = -c_mat[p, q] * L[i, q]

            # --- Chain through corr = L^T @ L ---
            # Only entries in row q or column q of corr are affected.

            # corr[a,q] for a < q:  dcorr = sum_k L[k,a] * dL_q[k]
            for a in range(q):
                val = 0.0
                for k in range(a + 1):  # L[k,a] is non-zero only for k <= a
                    val += L[k, a] * dL_q[k]
                jac[theta_idx, _corr_upper_index(a, q, K)] = val

            # corr[q,q] (diagonal): dcorr = 2 * sum_k L[k,q] * dL_q[k]
            jac[theta_idx, _corr_upper_index(q, q, K)] = (
                2.0 * np.dot(L[:q + 1, q], dL_q[:q + 1])
            )

            # corr[q,b] for b > q:  dcorr = sum_k dL_q[k] * L[k,b]
            for b in range(q + 1, K):
                val = 0.0
                for k in range(q + 1):  # dL_q[k] is non-zero for k=p..q
                    val += dL_q[k] * L[k, b]
                jac[theta_idx, _corr_upper_index(q, b, K)] = val

    return jac
