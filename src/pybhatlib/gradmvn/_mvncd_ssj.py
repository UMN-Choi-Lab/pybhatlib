"""Switzer-Solow-Joe (SSJ) simulation-based MVNCD using Quasi-Monte Carlo.

Implements the QMC-based separation-of-variables approach for computing
the multivariate normal CDF, as described in:

    Genz, A. (1992). Numerical Computation of Multivariate Normal Probabilities.
    Journal of Computational and Graphical Statistics, 1(2): 141-149.

    Joe, H. (1995). Approximations to Multivariate Normal Rectangle
    Probabilities Based on Conditional Expectations.
    Journal of the American Statistical Association, 90: 957-964.

The method uses Cholesky decomposition + separation-of-variables to convert
the K-dimensional integral into nested 1D integrals evaluated by QMC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.utils._qmc import halton_sequence


def _mvncd_ssj(
    a: np.ndarray,
    sigma: np.ndarray,
    *,
    n_draws: int = 1000,
    seed: int | None = None,
) -> float:
    """Compute P(X <= a) for X ~ MVN(0, Sigma) using QMC simulation.

    Uses the Genz (1992) separation-of-variables trick with Halton
    quasi-random sequences for efficient computation.

    Parameters
    ----------
    a : ndarray, shape (K,)
        Upper integration limits.
    sigma : ndarray, shape (K, K)
        Covariance matrix (symmetric positive-definite).
    n_draws : int
        Number of QMC draws. More draws = higher accuracy.
    seed : int or None
        Random seed for QMC scrambling.

    Returns
    -------
    prob : float
        Estimated P(X <= a).
    """
    K = len(a)

    if K == 0:
        return 1.0

    if K == 1:
        sd = np.sqrt(sigma[0, 0])
        return float(norm.cdf(a[0] / sd))

    # Cholesky decomposition: sigma = C @ C.T
    try:
        C = np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        # Not PD, regularize
        eigvals = np.linalg.eigvalsh(sigma)
        reg = max(1e-10, -eigvals.min() + 1e-10)
        C = np.linalg.cholesky(sigma + reg * np.eye(K))

    # Generate Halton sequence in [0,1]^(K-1)
    # We need K-1 dimensions since the last variable is deterministic
    # given the others (separation of variables).
    U = halton_sequence(n_draws, K - 1, seed=seed)

    # Genz separation-of-variables algorithm
    # P(X <= a) = integral over [0,1]^{K-1} of e(u) du
    # where e(u) is computed by the variable-separation scheme
    prob_sum = 0.0

    for i in range(n_draws):
        u = U[i]  # (K-1,) in [0,1]
        e_prod = 1.0

        # y stores conditional random draws
        y = np.zeros(K, dtype=np.float64)

        for k in range(K):
            # Conditional upper limit
            s = 0.0
            for j in range(k):
                s += C[k, j] * y[j]

            cond_upper = (a[k] - s) / C[k, k]
            Phi_upper = norm.cdf(cond_upper)

            e_prod *= Phi_upper

            if e_prod < 1e-300:
                break

            # Set y[k] for subsequent conditioning (except last variable)
            if k < K - 1:
                # Inverse CDF: y[k] = Phi^{-1}(u[k] * Phi(cond_upper))
                p_k = u[k] * Phi_upper
                if p_k < 1e-15:
                    y[k] = cond_upper - 5.0
                elif p_k > 1.0 - 1e-15:
                    y[k] = cond_upper
                else:
                    y[k] = norm.ppf(p_k)

        prob_sum += e_prod

    prob = prob_sum / n_draws
    return max(0.0, min(1.0, prob))
