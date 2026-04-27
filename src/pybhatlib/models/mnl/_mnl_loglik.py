"""MNL log-likelihood, analytic gradient, and analytic Hessian.


Model
-----
For observation q choosing alternative i from available set A_q:

    V_{qk} = X_{qk} @ beta                         (systematic utility)
    P_{qi} = exp(V_{qi}) / sum_{k in A_q} exp(V_{qk})   (MNL probability)
    LL     = sum_q  ln P_{q, chosen(q)}

Parameter vector layout
-----------------------
    x = beta,  shape (numunord,)

where ``numunord`` = number of columns in the specification matrix
``ivunord`` (mirrors GAUSS ``numunord = cols(ivunord)``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _compute_probabilities(
    x: NDArray,
    dta: NDArray,
    indxivunord: NDArray,
    davunord: NDArray,
    nc: int,
    numunord: int,
) -> tuple[NDArray, NDArray]:
    """Compute MNL probabilities for all observations.

    Implements the utility and softmax calculation shared by ``lpr``,
    ``lgd``, and ``lhs`` (GAUSS: v2, v, p1, p2 blocks).

    Parameters
    ----------
    x : NDArray, shape (numunord,)
        Coefficient vector (beta).
    dta : NDArray, shape (e1, n_cols)
        Data batch.
    indxivunord : NDArray, shape (nc * numunord,)
        Column indices into ``dta`` for the specification matrix, laid out
        row-major over alternatives (alt 0 first, then alt 1, ...).
    davunord : NDArray, shape (nc,)
        Column indices of availability indicators in ``dta``.
    nc : int
        Number of alternatives.
    numunord : int
        Number of variables per alternative.

    Returns
    -------
    p2 : NDArray, shape (e1, nc)
        Choice probabilities (availability-masked and normalised).
    v : NDArray, shape (e1, nc)
        Systematic utilities.
    """
    e1 = dta.shape[0]

    # Systematic utility for each observation and alternative
    # GAUSS: v2 = (ones(nc,1) .*. x) *~ (dta[.,indxivunord])'
    #        then accumulate into v via sumc over blocks
    v = np.zeros((e1, nc), dtype=np.float64)
    for k in range(nc):
        cols_k = indxivunord[k * numunord: (k + 1) * numunord]
        v[:, k] = dta[:, cols_k] @ x                           # (e1,)

    # Availability-masked softmax
    # GAUSS: p1 = exp(v); p2 = (p1 .* avail) ./ sumc((p1 .* avail)')
    avail = dta[:, davunord]                                    # (e1, nc)
    p1    = np.exp(v) * avail                                   # (e1, nc)  masked
    denom = p1.sum(axis=1, keepdims=True)                       # (e1, 1)
    denom = np.where(denom == 0, 1.0, denom)                   # guard /0
    p2    = p1 / denom                                          # (e1, nc)

    return p2, v


# ---------------------------------------------------------------------------
# Log-likelihood  (GAUSS: proc lpr)
# ---------------------------------------------------------------------------

def mnl_loglik(
    x: NDArray,
    dta: NDArray,
    indxivunord: NDArray,
    davunord: NDArray,
    dvunord: NDArray,
    nc: int,
    numunord: int,
) -> NDArray:
    """Per-observation log-likelihood for the MNL model.

    Implements the ``lpr`` procedure from MNLcasenew.gss.

    Parameters
    ----------
    x : NDArray, shape (numunord,)
        Coefficient vector.
    dta : NDArray, shape (e1, n_cols)
        Data batch.
    indxivunord : NDArray, shape (nc * numunord,)
        Column indices for IV specification (row-major over alts).
    davunord : NDArray, shape (nc,)
        Column indices of availability indicators.
    dvunord : NDArray, shape (nc,)
        Column indices of choice indicators (0/1 dummies).
    nc : int
        Number of alternatives.
    numunord : int
        Number of variables per alternative.

    Returns
    -------
    ll_obs : NDArray, shape (e1,)
        Log-likelihood contribution for each observation.
    """
    p2, _ = _compute_probabilities(x, dta, indxivunord, davunord, nc, numunord)

    # Probability of chosen alternative
    # GAUSS: z = sumc((p2 .* dta[.,dvunord])')
    chosen = dta[:, dvunord]                                    # (e1, nc)  0/1
    z = (p2 * chosen).sum(axis=1)                              # (e1,)

    # Floor to avoid log(0), matching GAUSS missrv fallback
    z = np.maximum(z, 1e-4)
    return np.log(z)


# ---------------------------------------------------------------------------
# Analytic gradient  (GAUSS: proc lgd)
# ---------------------------------------------------------------------------

def mnl_gradient(
    x: NDArray,
    dta: NDArray,
    indxivunord: NDArray,
    davunord: NDArray,
    dvunord: NDArray,
    nc: int,
    numunord: int,
) -> NDArray:
    """Per-observation analytic gradient of the MNL log-likelihood.

    Implements the ``lgd`` procedure from MNLcasenew.gss.

    For the MNL model the score for observation q is:

        g_q = sum_{k} (y_{qk} - P_{qk}) * X_{qk}

    where y_{qk} = 1 if q chose k, 0 otherwise.

    Parameters
    ----------
    x : NDArray, shape (numunord,)
    dta : NDArray, shape (e1, n_cols)
    indxivunord : NDArray, shape (nc * numunord,)
    davunord : NDArray, shape (nc,)
    dvunord : NDArray, shape (nc,)
    nc : int
    numunord : int

    Returns
    -------
    grad_obs : NDArray, shape (e1, numunord)
        Per-observation gradient contributions.
    """
    e1 = dta.shape[0]
    p2, _ = _compute_probabilities(x, dta, indxivunord, davunord, nc, numunord)

    # Residual: choice indicator minus predicted probability
    # GAUSS: g1 = dta[.,dvunord] - p2
    chosen = dta[:, dvunord]                                    # (e1, nc)
    g1 = chosen - p2                                            # (e1, nc)

    # Map residuals back to coefficient space via design matrix
    # GAUSS: g4 = reshape(sumc(g4 .* reshape(ylarge', nc, e1*numunord))', numunord, e1)'
    # Equivalent: for each coefficient j, sum over alts of g1[:,k] * X_{qkj}
    grad = np.zeros((e1, numunord), dtype=np.float64)
    for k in range(nc):
        cols_k = indxivunord[k * numunord: (k + 1) * numunord]
        grad  += g1[:, k: k + 1] * dta[:, cols_k]             # (e1, numunord)

    return grad


# ---------------------------------------------------------------------------
# Analytic Hessian  (GAUSS: proc lhs)
# ---------------------------------------------------------------------------

def mnl_hessian(
    x: NDArray,
    dta: NDArray,
    indxivunord: NDArray,
    davunord: NDArray,
    dvunord: NDArray,
    ddind: int,
    nc: int,
    numunord: int,
) -> NDArray:
    """Analytic Hessian of the MNL log-likelihood.

    Implements the ``lhs`` procedure from MNLcasenew.gss.

    The Hessian is:

        H = -sum_q sum_k P_{qk} * (X_{qk} - Xbar_q)(X_{qk} - Xbar_q).T

    where Xbar_q = sum_k P_{qk} * X_{qk}  is the probability-weighted
    mean covariate for observation q.

    Parameters
    ----------
    x : NDArray, shape (numunord,)
    dta : NDArray, shape (e1, n_cols)
    indxivunord : NDArray, shape (nc * numunord,)
    davunord : NDArray, shape (nc,)
    dvunord : NDArray, shape (nc,)
    ddind : int
        Column index of the UNO (ones) variable used as weight in GAUSS
        (``_dd`` / ``dd``).
    nc : int
    numunord : int

    Returns
    -------
    hess : NDArray, shape (numunord, numunord)
        Negative Hessian (positive semi-definite).
    """
    e1 = dta.shape[0]
    p2, _ = _compute_probabilities(x, dta, indxivunord, davunord, nc, numunord)

    # Probability-weighted mean covariate for each observation
    # GAUSS: g4 built with p2 (not residual) then h3 = ones(nc,1) .*. g4
    #        ymxbar = h2' - h3   (X_{qk} - Xbar_q)
    xbar = np.zeros((e1, numunord), dtype=np.float64)           # (e1, numunord)
    for k in range(nc):
        cols_k = indxivunord[k * numunord: (k + 1) * numunord]
        xbar  += p2[:, k: k + 1] * dta[:, cols_k]

    # Build (X_{qk} - Xbar_q) for all q and k, weighted by P_{qk}
    # GAUSS: hesse = ymxbar' * (uno .* (vecr(p2') .* ymxbar))
    hess = np.zeros((numunord, numunord), dtype=np.float64)
    uno  = dta[:, ddind]                                        # (e1,)  weights (ones)
    for k in range(nc):
        cols_k  = indxivunord[k * numunord: (k + 1) * numunord]
        xk      = dta[:, cols_k]                               # (e1, numunord)
        dev     = xk - xbar                                    # (e1, numunord)
        w       = (uno * p2[:, k])[:, np.newaxis]              # (e1, 1)
        hess   += dev.T @ (w * dev)

    return -hess                                                # negative Hessian
