"""MDCEV model prediction and forecasting.

Provides predicted consumption shares for new observations or
counterfactual scenarios, using Monte Carlo simulation over error draws
via ``simtradmdcev`` from mvlogit.py.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.models.mdcev._mdcev_model import _ensure_special_cols
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults


def _normalize_mdcev_forecast_inputs(
    results,
    X_new,
    X_gam_new,
    price_new,
    budget,
):
    """Allow either a results object or raw arrays as the first positional argument."""
    if isinstance(results, MDCEVResults) or results is None:
        return results, X_new, X_gam_new, price_new, budget

    if not isinstance(results, np.ndarray):
        raise TypeError(
            "The first argument must be either an MDCEVResults object or the "
            "X_new array when omitting results."
        )

    X_new_actual = results
    X_gam_new_actual = X_new
    price_new_actual = X_gam_new
    budget_actual = price_new

    if (
        X_gam_new_actual is None
        or price_new_actual is None
        or budget_actual is None
    ):
        raise ValueError(
            "If omitting the results object, call as "
            "mdcev_forecast(X_new, X_gam_new, price_new, budget, ...)"
        )

    return None, X_new_actual, X_gam_new_actual, price_new_actual, budget_actual


def _extract_mdcev_forecast_params(
    results: MDCEVResults | None,
    b_reported: NDArray | None,
    sigma: float | None,
    outside_good_gamma: float | None,
) -> tuple[NDArray, float, float]:
    """Extract forecast parameters from a results object or raw parameter inputs."""
    if results is not None:
        b = results.b_reported
        sigma_val = results.sigma
        if outside_good_gamma is None:
            ctrl = results.control
            outside_good_gamma = (
                ctrl.outside_good_gamma if ctrl is not None else -1000.0
            )
    else:
        if b_reported is None:
            raise ValueError(
                "Either a results object or both b_reported and sigma must be provided"
            )
        b = np.asarray(b_reported, dtype=np.float64)
        sigma_val = float(sigma) if sigma is not None else np.nan
        if outside_good_gamma is None:
            outside_good_gamma = -1000.0

    return b, sigma_val, outside_good_gamma


def mdcev_predict(
    results: MDCEVResults | np.ndarray | None = None,
    X_new: NDArray | None = None,
    X_gam_new: NDArray | None = None,
    price_new: NDArray | None = None,
    n_draws: int = 1000,
    seed: int = 1234,
    b_reported: NDArray | None = None,
    sigma: float | None = None,
    outside_good_gamma: float | None = None,
    **kwargs,
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

    Notes
    -----
    Either ``results`` or both ``b_reported`` and ``sigma`` must be provided.
    """
    from pybhatlib.gradmvn._logit_dist import simtradmdcev

    results, X_new, X_gam_new, price_new, _ = _normalize_mdcev_forecast_inputs(
        results, X_new, X_gam_new, price_new, None
    )
    b, sigma, outside_good_gamma = _extract_mdcev_forecast_params(
        results, b_reported, sigma, outside_good_gamma
    )
    nc = X_new.shape[1]
    nvarm = X_new.shape[2]
    nvargam = X_gam_new.shape[2]

    expected_len = nvarm + nvargam
    if b.size == expected_len + 1:
        if np.isnan(sigma):
            sigma = float(b[-1])
        else:
            if not np.allclose(sigma, b[-1], rtol=1e-8, atol=1e-12):
                raise ValueError(
                    "Conflicting sigma values: provided sigma does not match trailing sigma "
                    "in b_reported"
                )
        b = b[:expected_len]
    elif b.size == expected_len:
        if np.isnan(sigma):
            raise ValueError(
                "Parameter vector provides no trailing sigma; sigma must be supplied separately"
            )
    else:
        raise ValueError(
            f"Parameter vector length mismatch: expected {expected_len} or {expected_len+1} "
            f"(with trailing sigma), got {b.size}"
        )

    beta = b[:nvarm]
    xgam = b[nvarm: nvarm + nvargam]

    N      = X_new.shape[0]
    shares = np.zeros((N, nc), dtype=np.float64)
    rng    = np.random.default_rng(seed)

    for q in range(N):
        v_q = X_new[q] @ beta                                  # (nc,)

        u_q = X_gam_new[q] @ xgam                             # (nc,)
        u_q[0] = outside_good_gamma
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


def prepare_mdcev_forecast_data(
    model,
    df: pd.DataFrame,
    changevar: Sequence[str] | None = None,
    changeval: Sequence[float] | None = None,
    budget_col: str = "tot",
) -> tuple[NDArray, NDArray, NDArray, NDArray, pd.DataFrame]:
    """Prepare MDCEV forecast matrices from validation or counterfactual data.

    This helper validates optional change variables, ensures special dataset
    columns (``uno`` and ``sero``), and builds the design matrices used by
    ``mdcev_forecast``.
    """
    changevar = [] if changevar is None else list(changevar)
    changeval = [] if changeval is None else list(changeval)

    if len(changevar) != len(changeval):
        raise ValueError("changevar and changeval must have the same length")

    df_mod = df.copy()
    for name, val in zip(changevar, changeval):
        if name not in df_mod.columns:
            raise KeyError(f"change variable '{name}' not found in forecast data")
        df_mod[name] = val

    _ensure_special_cols(df_mod)

    nc = model.n_alts
    utility_spec = model.utility_spec
    gamma_spec = model.gamma_spec
    availability = model.availability

    n_obs = len(df_mod)
    nvarm = utility_spec.shape[1]
    nvargam = gamma_spec.shape[1]

    X_new = np.zeros((n_obs, nc, nvarm), dtype=np.float64)
    X_gam_new = np.zeros((n_obs, nc, nvargam), dtype=np.float64)
    price_new = np.ones((n_obs, nc), dtype=np.float64)

    for k in range(nc):
        for j in range(nvarm):
            col = utility_spec[k, j]
            if col in df_mod.columns:
                X_new[:, k, j] = df_mod[col].to_numpy(dtype=np.float64)
        for j in range(nvargam):
            colg = gamma_spec[k, j]
            if colg in df_mod.columns:
                X_gam_new[:, k, j] = df_mod[colg].to_numpy(dtype=np.float64)
        avail_var = availability[k]
        if avail_var in df_mod.columns:
            price_new[:, k] = df_mod[avail_var].to_numpy(dtype=np.float64)

    if budget_col in df_mod.columns:
        budget = df_mod[budget_col].to_numpy(dtype=np.float64)
    else:
        budget = np.ones(n_obs, dtype=np.float64)

    return X_new, X_gam_new, price_new, budget, df_mod


def _mdcev_forecast_allocation(
    v_vals: NDArray,
    prices: NDArray,
    f1_vals: NDArray,
    budget: float,
    num_outside: int = 1,
) -> NDArray:
    """Compute a single MDCEV allocation forecast for one observation."""
    nc = v_vals.size
    fc = np.zeros(nc, dtype=np.float64)
    outside_idx = np.arange(num_outside)
    inside_idx = np.arange(num_outside, nc)
    sorted_inside = inside_idx[np.argsort(v_vals[inside_idx])[::-1]]

    N = np.sum(prices[outside_idx] * v_vals[outside_idx])
    D = float(budget) if budget != 0 else 1.0
    lambda_val = N / D
    consumed: list[int] = []

    for idx in sorted_inside:
        if v_vals[idx] < lambda_val:
            break
        consumed.append(int(idx))
        N += f1_vals[idx] * prices[idx] * v_vals[idx]
        D += f1_vals[idx] * prices[idx]
        lambda_val = N / D

    fc[outside_idx] = v_vals[outside_idx] / lambda_val
    for idx in consumed:
        fc[idx] = (v_vals[idx] / lambda_val - 1.0) * f1_vals[idx]

    return fc


def mdcev_forecast(
    results: MDCEVResults | NDArray | None = None,
    X_new: NDArray | None = None,
    X_gam_new: NDArray | None = None,
    price_new: NDArray | None = None,
    budget: NDArray | None = None,
    n_replications: int = 200,
    seed: int = 1234,
    num_outside: int = 1,
    b_reported: NDArray | None = None,
    sigma: float | None = None,
    outside_good_gamma: float | None = None,
) -> NDArray:
    """Simulate MDCEV allocation forecasts for new observations.

    Returns a stacked forecast matrix with shape
    ``(n_replications * N, nc)``.

    Notes
    -----
    Either ``results`` or both ``b_reported`` and ``sigma`` must be provided.
    """
    from numpy.random import default_rng

    results, X_new, X_gam_new, price_new, budget = _normalize_mdcev_forecast_inputs(
        results, X_new, X_gam_new, price_new, budget
    )
    b, sigma, outside_good_gamma = _extract_mdcev_forecast_params(
        results, b_reported, sigma, outside_good_gamma
    )
    nc = X_new.shape[1]
    nvarm = X_new.shape[2]
    nvargam = X_gam_new.shape[2]

    expected_len = nvarm + nvargam
    if b.size == expected_len + 1:
        if np.isnan(sigma):
            sigma = float(b[-1])
        else:
            if not np.allclose(sigma, b[-1], rtol=1e-8, atol=1e-12):
                raise ValueError(
                    "Conflicting sigma values: provided sigma does not match trailing sigma "
                    "in b_reported"
                )
        b = b[:expected_len]
    elif b.size == expected_len:
        if np.isnan(sigma):
            raise ValueError(
                "Parameter vector provides no trailing sigma; sigma must be supplied separately"
            )
    else:
        raise ValueError(
            f"Parameter vector length mismatch: expected {expected_len} or {expected_len+1} "
            f"(with trailing sigma), got {b.size}"
        )

    beta = b[:nvarm]
    xgam = b[nvarm: nvarm + nvargam]

    N = X_new.shape[0]
    v = X_new @ beta
    u = X_gam_new @ xgam

    u[:, 0] = outside_good_gamma

    f1 = np.exp(u)
    rng = default_rng(seed)

    budget_arr = np.asarray(budget, dtype=np.float64).reshape(-1)
    if budget_arr.shape[0] != N:
        raise ValueError("budget must have the same number of rows as X_new")

    forecasts = []
    for _ in range(n_replications):
        as_draw = rng.gumbel(loc=0.0, scale=sigma, size=(N, nc))
        v_draw = np.exp(v + as_draw) / price_new

        rep_fc = np.zeros((N, nc), dtype=np.float64)
        for i in range(N):
            rep_fc[i, :] = _mdcev_forecast_allocation(
                v_draw[i, :],
                price_new[i, :],
                f1[i, :],
                budget_arr[i],
                num_outside=num_outside,
            )
        forecasts.append(rep_fc)

    return np.vstack(forecasts)


def mdcev_predict_choice(
    results: MDCEVResults | np.ndarray | None = None,
    X_new: NDArray | None = None,
    X_gam_new: NDArray | None = None,
    price_new: NDArray | None = None,
    n_draws: int = 1000,
    seed: int = 1234,
    b_reported: NDArray | None = None,
    sigma: float | None = None,
    outside_good_gamma: float | None = None,
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
    shares = mdcev_predict(
        results=results,
        X_new=X_new,
        X_gam_new=X_gam_new,
        price_new=price_new,
        n_draws=n_draws,
        seed=seed,
        b_reported=b_reported,
        sigma=sigma,
        outside_good_gamma=outside_good_gamma,
    )
    return np.argmax(shares, axis=1)
