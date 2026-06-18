"""MDCEV model prediction and forecasting.

Provides predicted consumption shares for new observations or
counterfactual scenarios, using Monte Carlo simulation over error draws
via ``simtradmdcev`` from mvlogit.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.models.mdcev._mdcev_results import MDCEVResults


def mdcev_predict(
    results: MDCEVResults,
    X_new: NDArray,
    X_gam_new: NDArray,
    price_new: NDArray,
    n_draws: int = 1000,
    seed: int = 1234,
) -> NDArray:
    """Predict mean consumption shares for new observations.

    Uses Monte Carlo simulation over error draws (via ``simtradmdcev``)
    to compute expected consumption indicator probabilities for each
    alternative.

    Parameters
    ----------
    results : MDCEVResults
        Fitted MDCEV model results.
    X_new : NDArray, shape (N, nc, nvarm)
        Baseline utility design matrix for N new observations.
    X_gam_new : NDArray, shape (N, nc, nvargam)
        Satiation utility design matrix.
    price_new : NDArray, shape (N, nc)
        Price matrix (first column = outside-good price, typically ones).
    n_draws : int
        Number of Monte Carlo error draws per observation.
    seed : int
        Random seed.

    Returns
    -------
    shares : NDArray, shape (N, nc)
        Mean predicted consumption share for each alternative.
    """
    from pybhatlib.gradmvn import simtradmdcev

    ctrl    = results.control
    b       = results.b_reported
    nc      = X_new.shape[1]
    nvarm   = X_new.shape[2]
    nvargam = X_gam_new.shape[2]

    beta   = b[:nvarm]
    xgam   = b[nvarm: nvarm + nvargam]
    sigma  = results.sigma

    N      = X_new.shape[0]
    shares = np.zeros((N, nc), dtype=np.float64)
    rng    = np.random.default_rng(seed)

    for q in range(N):
        v_q = X_new[q] @ beta                                  # (nc,)

        u_q = X_gam_new[q] @ xgam                             # (nc,)
        u_q[0] = ctrl.outside_good_gamma if ctrl else -1000.0
        gamma_q  = np.exp(u_q[1:])                            # (nc-1,)
        price_q  = price_new[q]                               # (nc,)

        # vtilde_{k,1} = (v_1 - ln p_1) - (v_k - ln p_k)  for k = 1..nc-1
        v1       = v_q[0] - np.log(price_q[0])
        v_inside = v_q[1:] - np.log(price_q[1:])
        a_q      = (v1 - v_inside).reshape(-1, 1)             # (nc-1, 1)

        m_est = max(1, int((a_q > 0).sum()))

        _, draws = simtradmdcev(
            a=a_q,
            m=m_est,
            n=n_draws,
            sig=sigma,
            psi=v_q.reshape(-1, 1),
            gamma=gamma_q.reshape(-1, 1),
            price=price_q[1:].reshape(-1, 1),
            seed=int(rng.integers(1, 2**31)),
        )

        eps     = draws[:, :nc - 1]                           # (n_draws, nc-1)
        v_sim   = v_inside[np.newaxis, :] + eps               # (n_draws, nc-1)
        consumed = (v_sim > v1).astype(np.float64)

        shares[q, 0]  = 1.0                                   # outside good always consumed
        shares[q, 1:] = consumed.mean(axis=0)

    row_sums = shares.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return shares / row_sums


def mdcev_predict_choice(
    results: MDCEVResults,
    X_new: NDArray,
    X_gam_new: NDArray,
    price_new: NDArray,
    n_draws: int = 1000,
    seed: int = 1234,
) -> NDArray:
    """Predict the most likely consumed alternative for each observation.

    Parameters
    ----------
    results : MDCEVResults
    X_new : NDArray, shape (N, nc, nvarm)
    X_gam_new : NDArray, shape (N, nc, nvargam)
    price_new : NDArray, shape (N, nc)
    n_draws : int
    seed : int

    Returns
    -------
    choices : NDArray, shape (N,)
        Index (0-based) of the most likely consumed alternative for each
        observation (0 = outside good if only it is consumed).
    """
    shares = mdcev_predict(results, X_new, X_gam_new, price_new,
                           n_draws=n_draws, seed=seed)
    return np.argmax(shares, axis=1)
