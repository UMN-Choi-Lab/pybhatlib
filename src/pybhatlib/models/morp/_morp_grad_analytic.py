"""Analytic gradient of the MORP log-likelihood.

The MORP model has, for each observation n and outcome dimension d,
the latent equation::

    Y_d^* = X_{n,d} @ beta + eps_d,    eps ~ MVN(0, Sigma)

with thresholds ``tau_d`` partitioning the real line into ordered
categories. For an observed outcome vector ``y_n``, the rectangle
probability is::

    P_n = P(L_n <= eps <= U_n)
        = sum_{s in {0,1}^K} (-1)^|s| * MVNCD(c_s; Sigma)         (1)

where ``L_n[d] = tau_d[y_d-1] - mu_d``, ``U_n[d] = tau_d[y_d] - mu_d``,
``mu_d = X_{n,d} @ beta``, and ``c_s[d] = U_n[d]`` if ``s_d = 0`` else
``L_n[d]``. This module implements the gradient of ``log P_n`` w.r.t.
all model parameters by chain rule through (1):

    * ``beta`` --> via ``mu`` --> via ``L`` and ``U``
    * thresholds ``delta_{d,j}`` --> via ``L`` and ``U``
      (``tau_{d,0}`` is free; ``tau_{d,j} = tau_{d,j-1} + exp(delta_{d,j})``)
    * scales (heteroscedastic) and correlation parameters --> via ``Sigma``

The MVNCD building blocks (``mvncd_grad_me_analytic``,
``mvncd_grad_ovus_analytic``) and the spherical Jacobian
``grad_corr_theta`` are reused as-is from the MNP analytic gradient
infrastructure.

References
----------
Bhat, C. R. (2018). New matrix-based methods for the analytic evaluation
of the multivariate cumulative normal distribution function.
Transportation Research Part B, 109, 238-256.

GAUSS reference (UTAcode_0402): ``gradcdrectmvnanl`` in
``gradients mvn.src``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pybhatlib.utils._safe_reparam import safe_exp

from pybhatlib.gradmvn._mvncd_grad_analytic import mvncd_grad_me_analytic
from pybhatlib.gradmvn._mvncd_grad_ovus import mvncd_grad_ovus_analytic
from pybhatlib.matgradient._spherical import grad_corr_theta, theta_to_corr
from pybhatlib.models.morp._morp_control import MORPControl


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def morp_analytic_gradient(
    theta: NDArray,
    X: NDArray,
    y: NDArray,
    n_dims: int,
    n_categories: list[int],
    n_beta: int,
    control: MORPControl,
    *,
    return_per_obs: bool = False,
) -> tuple[float, NDArray] | tuple[float, NDArray, NDArray]:
    """Compute MORP negative mean log-likelihood and its analytic gradient.

    Uses ME or OVUS for the per-vertex MVNCD and chains through the
    rectangle inclusion-exclusion identity. Falls back to FD only when
    invoked through ``morp_loglik`` with an unsupported MVNCD method.

    When ``return_per_obs=True`` the per-observation score matrix
    ``S[q, k] = d log P_q / d theta_k`` is also returned (same quantity the
    BHHH/sandwich estimators otherwise obtain by finite-differencing the
    per-observation log-likelihood, but in a single analytic pass).

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector in the same layout assumed by
        ``_unpack_morp_params`` and ``count_morp_params``.
    X : ndarray, shape (N, n_dims, n_beta)
        Design matrix.
    y : ndarray, shape (N, n_dims)
        Observed ordinal outcomes (0-based category indices).
    n_dims : int
        Number of ordinal outcome dimensions (``D = K``).
    n_categories : list of int
        Categories per dimension.
    n_beta : int
        Number of regression coefficients.
    control : MORPControl
        Model control structure (must have ``method in {"me", "ovus"}``).

    Returns
    -------
    nll : float
        Negative mean log-likelihood.
    grad : ndarray, shape (n_params,)
        Gradient of ``nll`` w.r.t. ``theta``.
    scores : ndarray, shape (N, n_params)
        Per-observation scores ``d log P_q / d theta`` (only when
        ``return_per_obs=True``).
    """
    theta = np.asarray(theta, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    N = X.shape[0]
    K = n_dims
    n_params = len(theta)
    scores = np.zeros((N, n_params), dtype=np.float64) if return_per_obs else None

    # ---- Unpack & precompute ----
    layout = _layout_indices(n_beta, n_dims, n_categories, control)
    beta, thresholds, delta_log, sigma, scales, corr, corr_jac = (
        _unpack_with_components(
            theta, n_beta, n_dims, n_categories, control, layout,
        )
    )

    use_ovus = control.method == "ovus"

    total_ll = 0.0
    grad = np.zeros(n_params, dtype=np.float64)

    # Per-observation loop. The N is typically small for MORP fits,
    # and the dominant cost per obs is MVNCD evaluation, not Python overhead.
    for q in range(N):
        # Per-dimension utilities
        mu_q = np.einsum('dv,v->d', X[q], beta)  # (K,)

        # Lower / upper limits and which dims have finite bounds
        lower = np.empty(K, dtype=np.float64)
        upper = np.empty(K, dtype=np.float64)
        has_lower = np.empty(K, dtype=bool)
        has_upper = np.empty(K, dtype=bool)
        for d in range(K):
            j = int(y[q, d])
            tau_d = thresholds[d]
            if j == 0:
                lower[d] = -np.inf
                has_lower[d] = False
            else:
                lower[d] = tau_d[j - 1] - mu_q[d]
                has_lower[d] = True
            if j == n_categories[d] - 1:
                upper[d] = np.inf
                has_upper[d] = False
            else:
                upper[d] = tau_d[j] - mu_q[d]
                has_upper[d] = True

        # ---- Rectangle probability + gradient via inclusion-exclusion ----
        prob_q, dP_dlower, dP_dupper, dP_dsigma_vech = _rect_prob_and_grad(
            lower, upper, has_lower, has_upper, sigma, K, use_ovus,
        )
        prob_q = max(prob_q, 1e-300)
        total_ll += np.log(prob_q)
        inv_p = 1.0 / prob_q

        # ---- Beta gradient ----
        # dL[k]/dbeta_v = -X[q,k,v]; dU[k]/dbeta_v = -X[q,k,v]
        # dP/dbeta_v = -sum_k (dP/dL[k] + dP/dU[k]) * X[q,k,v]
        sum_dP_dz = dP_dlower + dP_dupper  # (K,)
        # NB: -inf bounds carry zero contribution by construction (set by
        # _rect_prob_and_grad).
        grad_beta_q = -X[q].T @ sum_dP_dz  # (n_beta,)
        grad[layout["beta"]] += inv_p * grad_beta_q
        if scores is not None:
            scores[q, layout["beta"]] = inv_p * grad_beta_q

        # ---- Threshold gradient ----
        # tau_d[j] = tau_d[0] + sum_{k=1..j} exp(delta_d_k)
        # For obs q with y_d = j:
        #   lower[d] uses tau_d[j-1] (touches delta entries 0..j-1)
        #   upper[d] uses tau_d[j]   (touches delta entries 0..j)
        # dP/d(tau_d[k]) influence flows from (dP/dlower[d], dP/dupper[d]).
        # Then dtau_d[k]/d(theta_thr_d_m) is:
        #   m == 0: 1                    (raw level)
        #   m >= 1: exp(delta_d_m) if m <= k else 0
        for d in range(K):
            slot = layout["thr"][d]
            if len(slot) == 0:
                continue
            j = int(y[q, d])
            # dP/dtau_d[j-1] (from lower) and dP/dtau_d[j] (from upper)
            # in real-line space.
            dP_dtau = np.zeros(n_categories[d] - 1, dtype=np.float64)
            if has_lower[d]:
                dP_dtau[j - 1] += dP_dlower[d]
            if has_upper[d]:
                dP_dtau[j] += dP_dupper[d]
            # Chain through cumulative log-spacing parameterization.
            grad_thr_d = _tau_to_param_grad(dP_dtau, delta_log[d])
            grad[slot] += inv_p * grad_thr_d
            if scores is not None:
                scores[q, slot] = inv_p * grad_thr_d

        # ---- Sigma chain (scales + correlations) ----
        if not control.iid and dP_dsigma_vech is not None:
            adj_Sigma = _vech_to_symmetric(dP_dsigma_vech, K)
            adj_lp = _adj_sigma_to_params(
                adj_Sigma, scales, corr, corr_jac, K, control,
            )
            slot = layout["cov"]
            if slot is not None and adj_lp is not None:
                grad[slot] += inv_p * adj_lp
                if scores is not None:
                    scores[q, slot] = inv_p * adj_lp

    nll = -total_ll / N
    grad = -grad / N
    if return_per_obs:
        return nll, grad, scores
    return nll, grad


# ---------------------------------------------------------------------------
# Layout / unpacking
# ---------------------------------------------------------------------------


def _layout_indices(
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
) -> dict:
    """Compute slot indices into ``theta`` for each parameter block.

    Returns
    -------
    layout : dict
        ``{"beta": slice, "thr": [arr_d], "cov": slice or None}``
    """
    idx = 0
    beta_slice = slice(idx, idx + n_beta)
    idx += n_beta

    thr_slots = []
    for d in range(n_dims):
        n_thresh = max(0, n_categories[d] - 1)
        if n_thresh == 0:
            thr_slots.append(np.array([], dtype=np.int64))
        else:
            thr_slots.append(np.arange(idx, idx + n_thresh, dtype=np.int64))
        idx += n_thresh

    cov_slice = None
    if not control.iid:
        n_corr = n_dims * (n_dims - 1) // 2
        if control.heteronly:
            n_cov = n_dims - 1
        elif getattr(control, "fix_scales", False):
            # Scales locked at 1 (GAUSS BHATLIB unit-variance convention):
            # the cov block is just the correlation params.
            n_cov = n_corr
        else:
            n_cov = (n_dims - 1) + n_corr
        if n_cov > 0:
            cov_slice = slice(idx, idx + n_cov)
            idx += n_cov

    return {"beta": beta_slice, "thr": thr_slots, "cov": cov_slice}


def _unpack_with_components(
    theta: NDArray,
    n_beta: int,
    n_dims: int,
    n_categories: list[int],
    control: MORPControl,
    layout: dict,
) -> tuple:
    """Unpack ``theta`` and return building blocks for the gradient chain.

    Returns
    -------
    beta : (n_beta,) array
    thresholds : list of (J_d - 1,) arrays
    delta_log : list of (J_d - 1,) arrays
        Raw log-spacing params: ``delta_log[d][0]`` is the first threshold
        directly; ``delta_log[d][m]`` for m>=1 satisfies
        ``tau[m] = tau[m-1] + exp(delta_log[d][m])``.
    sigma : (K, K) array
    scales : (K,) array (with first entry = 1) when heteroscedastic/full,
        else None.
    corr : (K, K) correlation matrix (None for iid/heteronly).
    corr_jac : Jacobian d(corr_vech)/d(theta_corr) shape (n_corr, K(K+1)/2),
        or None.
    """
    beta = theta[layout["beta"]]

    thresholds: list[NDArray] = []
    delta_log: list[NDArray] = []
    for d in range(n_dims):
        slot = layout["thr"][d]
        if slot.size == 0:
            thresholds.append(np.array([]))
            delta_log.append(np.array([]))
            continue
        raw = theta[slot]
        # raw[0] is tau_d[0]; raw[m>=1] is delta_d_m where
        # tau_d[m] = tau_d[m-1] + exp(raw[m])
        tau_d = np.empty_like(raw)
        tau_d[0] = raw[0]
        for m in range(1, len(raw)):
            tau_d[m] = tau_d[m - 1] + safe_exp(raw[m])
        thresholds.append(tau_d)
        delta_log.append(raw.copy())

    sigma, scales, corr, corr_jac = _build_sigma_components(
        theta, n_dims, control, layout,
    )

    return beta, thresholds, delta_log, sigma, scales, corr, corr_jac


def _build_sigma_components(
    theta: NDArray,
    n_dims: int,
    control: MORPControl,
    layout: dict,
) -> tuple[NDArray, NDArray | None, NDArray | None, NDArray | None]:
    """Build Sigma and its components (scales, corr, corr_jac) from theta."""
    K = n_dims
    if control.iid or layout["cov"] is None:
        return np.eye(K, dtype=np.float64), None, None, None

    cov_params = theta[layout["cov"]]
    if control.heteronly:
        scales = np.ones(K, dtype=np.float64)
        scales[1:] = safe_exp(cov_params[: K - 1])
        sigma = np.diag(scales ** 2)
        return sigma, scales, None, None

    # Full covariance: scales then correlation params (scales absent when
    # ``fix_scales=True`` — all latent-utility scales locked at 1, matching
    # GAUSS BHATLIB MORP's unit-variance ordered-probit identification).
    fix_scales = getattr(control, "fix_scales", False)
    if fix_scales:
        n_scale = 0
        scales = np.ones(K, dtype=np.float64)
    else:
        n_scale = K - 1
        scales = np.ones(K, dtype=np.float64)
        scales[1:] = safe_exp(cov_params[:n_scale])

    n_corr = K * (K - 1) // 2
    corr_theta = cov_params[n_scale: n_scale + n_corr]
    if control.spherical:
        corr = theta_to_corr(corr_theta, K)
        corr_jac = grad_corr_theta(corr_theta, K)
    else:
        # tanh parameterization
        corr = np.eye(K, dtype=np.float64)
        c_idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                corr[i, j] = np.tanh(corr_theta[c_idx])
                corr[j, i] = corr[i, j]
                c_idx += 1
        corr_jac = None  # signaled by None: chain manually below

    D = np.diag(scales)
    sigma = D @ corr @ D
    return sigma, scales, corr, corr_jac


# ---------------------------------------------------------------------------
# Rectangle probability + gradient via inclusion-exclusion
# ---------------------------------------------------------------------------


def _rect_prob_and_grad(
    lower: NDArray,
    upper: NDArray,
    has_lower: NDArray,
    has_upper: NDArray,
    sigma: NDArray,
    K: int,
    use_ovus: bool,
) -> tuple[float, NDArray, NDArray, NDArray | None]:
    """Compute ``P(lower <= eps <= upper)`` and its gradient.

    Uses inclusion-exclusion::

        P = sum_{s in {0,1}^K} (-1)^{|s|} * F(c_s; Sigma)

    where ``F`` is the MVNCD with upper limit ``c_s``. For each vertex we
    obtain ``dF/dc_s`` and ``dF/dSigma_vech`` from
    ``mvncd_grad_{me,ovus}_analytic``, and aggregate by sign and bit
    pattern.

    Vertices that select a ``-inf`` lower bound contribute 0 to ``P`` and
    have zero gradient w.r.t. all parameters except possibly Sigma; we
    skip them entirely (their gradient w.r.t. Sigma is also 0 because
    the vertex CDF itself is 0).

    Parameters
    ----------
    lower, upper : (K,) arrays
        Per-dimension bounds. ``lower[d] = -inf`` if ``has_lower[d]`` is
        False; analogous for upper.
    has_lower, has_upper : (K,) bool arrays
    sigma : (K, K) covariance matrix
    K : int
        Dimension.
    use_ovus : bool
        If True use the OVUS gradient kernel; else ME.

    Returns
    -------
    prob : float
    dP_dlower : (K,) array
        Aggregate gradient w.r.t. ``lower[k]``. Zero where bound is -inf.
    dP_dupper : (K,) array
        Aggregate gradient w.r.t. ``upper[k]``. Zero where bound is +inf.
    dP_dsigma_vech : (K(K+1)/2,) array or None
        Aggregate gradient w.r.t. row-based upper-triangular ``Sigma``.
    """
    grad_kernel = (
        mvncd_grad_ovus_analytic if use_ovus else mvncd_grad_me_analytic
    )

    n_vech = K * (K + 1) // 2
    prob = 0.0
    dP_dlower = np.zeros(K, dtype=np.float64)
    dP_dupper = np.zeros(K, dtype=np.float64)
    dP_dsigma_vech = np.zeros(n_vech, dtype=np.float64)

    for s in range(1 << K):
        c = np.empty(K, dtype=np.float64)
        sign = 1
        skip = False
        # Tracks for each k whether this vertex used `lower` (vs `upper`)
        lower_mask = np.zeros(K, dtype=bool)

        for d in range(K):
            if s & (1 << d):
                if not has_lower[d]:
                    skip = True
                    break
                c[d] = lower[d]
                lower_mask[d] = True
                sign *= -1
            else:
                if not has_upper[d]:
                    # Upper bound is +inf: this vertex's CDF is the marginal
                    # over the remaining (finite) bounds. We can collapse
                    # the +inf component out, but mvncd_grad_*_analytic does
                    # not accept +inf in `a`. Drop the dim and recurse on
                    # the surviving sub-problem.
                    skip = True
                    break
                c[d] = upper[d]

        if skip:
            # Handle +inf upper bounds: collapse those dimensions out.
            collapsed = _collapsed_vertex_grad(
                s, lower, upper, has_lower, has_upper,
                sigma, K, use_ovus,
            )
            if collapsed is None:
                continue
            sign_eff, prob_v, ga_v, gv_v, alive_idx, lower_mask_alive = collapsed
            prob += sign_eff * prob_v
            # Map ga_v back to full-K dP_dlower / dP_dupper for the alive dims
            for kk, full_d in enumerate(alive_idx):
                if lower_mask_alive[kk]:
                    dP_dlower[full_d] += sign_eff * ga_v[kk]
                else:
                    dP_dupper[full_d] += sign_eff * ga_v[kk]
            # Sigma vech: pad alive sub-block back to the full vech.
            _accumulate_sigma_vech_subblock(
                dP_dsigma_vech, gv_v, alive_idx, K, sign_eff,
            )
            continue

        # Full-K vertex: all dims have finite limits in this combination.
        prob_v, ga_v, gv_v = grad_kernel(c, sigma)
        # Short-circuit denormal / underflowed vertices: ga_v from ME/OVUS
        # at deep tails can be subnormal noise that contaminates the
        # accumulated dP_dlower/dP_dupper without contributing to prob.
        # Mirrors the early-return in _grad_me_adjoint at line 380
        # (PR #8 review P1).
        if prob_v < 1e-300:
            continue
        prob += sign * prob_v
        for d in range(K):
            if lower_mask[d]:
                dP_dlower[d] += sign * ga_v[d]
            else:
                dP_dupper[d] += sign * ga_v[d]
        dP_dsigma_vech += sign * gv_v

    return prob, dP_dlower, dP_dupper, dP_dsigma_vech


def _collapsed_vertex_grad(
    s: int,
    lower: NDArray,
    upper: NDArray,
    has_lower: NDArray,
    has_upper: NDArray,
    sigma: NDArray,
    K: int,
    use_ovus: bool,
):
    """Handle a vertex of the inclusion-exclusion sum where some
    coordinates are ``+inf`` (upper bound) or ``-inf`` (lower bound).

    A coordinate that selected ``+inf`` integrates out; we keep only the
    "alive" dims with finite vertex coordinates, evaluate MVNCD on the
    sub-block of Sigma, and multiply the sign by ``(-1)^|s_alive_lower|``.

    A coordinate that selected ``-inf`` makes the entire vertex CDF zero;
    we return ``None``.
    """
    alive_idx = []
    lower_mask_alive = []
    sign = 1
    for d in range(K):
        if s & (1 << d):
            if not has_lower[d]:
                # P(... <= -inf, ...) = 0
                return None
            alive_idx.append(d)
            lower_mask_alive.append(True)
            sign *= -1
        else:
            if has_upper[d]:
                alive_idx.append(d)
                lower_mask_alive.append(False)
            # else: collapse this coord (upper = +inf integrates to 1)

    if len(alive_idx) == 0:
        # All upper bounds were +inf and no lower bound was selected;
        # this only happens for s=0 with all has_upper False, in which
        # case the rectangle is the full space and P=1. The caller's
        # full-vertex loop should have produced this; here we just return
        # a constant contribution.
        return sign, 1.0, np.array([]), np.array([]), [], []

    alive_idx_arr = np.array(alive_idx, dtype=np.int64)
    K_a = len(alive_idx)
    c_alive = np.empty(K_a, dtype=np.float64)
    for kk, d in enumerate(alive_idx):
        c_alive[kk] = lower[d] if lower_mask_alive[kk] else upper[d]
    sigma_alive = sigma[np.ix_(alive_idx_arr, alive_idx_arr)]

    grad_kernel = (
        mvncd_grad_ovus_analytic if use_ovus else mvncd_grad_me_analytic
    )
    prob_v, ga_v, gv_v = grad_kernel(c_alive, sigma_alive)
    return sign, prob_v, ga_v, gv_v, alive_idx, lower_mask_alive


def _accumulate_sigma_vech_subblock(
    full_vech: NDArray,
    sub_vech: NDArray,
    alive_idx: list[int],
    K: int,
    sign: int,
) -> None:
    """Add ``sign * sub_vech`` (defined on the alive sub-block) into
    ``full_vech`` using the row-based upper-triangular index mapping."""
    K_a = len(alive_idx)
    if K_a == 0:
        return
    # Build sub-block matrix and scatter back into full vech.
    sub_idx = 0
    for ai in range(K_a):
        for aj in range(ai, K_a):
            i_full = alive_idx[ai]
            j_full = alive_idx[aj]
            full_pos = _vech_index(i_full, j_full, K)
            full_vech[full_pos] += sign * sub_vech[sub_idx]
            sub_idx += 1


def _vech_index(i: int, j: int, K: int) -> int:
    """Row-based upper-triangular flat index for (i, j) with i <= j."""
    if i > j:
        i, j = j, i
    return i * K - i * (i - 1) // 2 + (j - i)


def _vech_to_symmetric(vech: NDArray, K: int) -> NDArray:
    """Convert row-based upper-tri vech to a symmetric full matrix.

    Off-diagonal values represent both (i,j) and (j,i); each side gets
    half of the vech entry to match the convention used by
    ``mvncd_grad_*_analytic``.
    """
    S = np.zeros((K, K), dtype=np.float64)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            if i == j:
                S[i, i] = vech[idx]
            else:
                S[i, j] = vech[idx] / 2.0
                S[j, i] = vech[idx] / 2.0
            idx += 1
    return S


# ---------------------------------------------------------------------------
# Threshold chain rule
# ---------------------------------------------------------------------------


def _tau_to_param_grad(
    dP_dtau: NDArray,
    delta_log: NDArray,
) -> NDArray:
    """Chain ``dP/d(tau_d[k])`` to ``dP/d(theta_thr_d_m)``.

    Parameterization: ``tau_d[0] = theta_d[0]`` and
    ``tau_d[k] = tau_d[k-1] + exp(theta_d[k])`` for ``k >= 1``.

    The Jacobian is therefore lower-triangular: changing ``theta_d[m]``
    shifts ``tau_d[k]`` for all ``k >= m`` by ``exp(theta_d[m])`` (or by
    ``1`` if ``m = 0``).

    Parameters
    ----------
    dP_dtau : (J - 1,) array
        Gradients w.r.t. each threshold position.
    delta_log : (J - 1,) array
        Raw threshold parameters: ``delta_log[0]`` is ``tau_d[0]`` itself,
        ``delta_log[k]`` for ``k >= 1`` is the log-spacing.

    Returns
    -------
    grad_theta : (J - 1,) array
        Gradient w.r.t. the raw threshold parameters.
    """
    n = len(dP_dtau)
    grad = np.zeros(n, dtype=np.float64)
    if n == 0:
        return grad
    # For m = 0: dtau_d[k]/d(theta_d[0]) = 1 for all k >= 0.
    grad[0] = float(np.sum(dP_dtau))
    # For m >= 1: dtau_d[k]/d(theta_d[m]) = exp(theta_d[m]) if k >= m else 0.
    for m in range(1, n):
        grad[m] = float(safe_exp(delta_log[m]) * np.sum(dP_dtau[m:]))
    return grad


# ---------------------------------------------------------------------------
# Sigma chain rule (scales + correlation)
# ---------------------------------------------------------------------------


def _adj_sigma_to_params(
    adj_Sigma: NDArray,
    scales: NDArray | None,
    corr: NDArray | None,
    corr_jac: NDArray | None,
    K: int,
    control: MORPControl,
) -> NDArray | None:
    """Chain ``adj_Sigma`` (full matrix adjoint) to gradient w.r.t. the
    raw covariance parameters in ``theta``.

    Layout of returned vector: ``[d log scale_2, ..., d log scale_K]``
    followed (for full covariance) by the correlation theta block.

    For ``Sigma = D corr D`` with ``D = diag(scales)``,
    ``scales[0] = 1`` (reference)::

        d Sigma_{ij} / d log scale_d = (Sigma_{id} delta_{dj} + Sigma_{dj} delta_{di})

    so::

        adj_log_scale_d = 2 * scales[d] * sum_j adj_Sigma[d, j] * corr[d, j] * scales[j]

    For correlation::

        d corr_{ij} / d theta_p = corr_jac[p, vech_idx(i,j)]

    and the contribution from correlation to ``Sigma`` carries a factor of
    ``scales[i] * scales[j]``. Off-diagonal vech entries are walked as
    summing ``adj_Sigma[i,j] + adj_Sigma[j,i]``.
    """
    if control.iid:
        return None

    if control.heteronly:
        # scales_diag = exp(theta), so d Sigma_dd / d theta_d
        # = 2 scales_d * d scales_d/d theta_d = 2 scales_d^2.
        # Only scales[1:] are estimated (scales[0] = 1).
        adj = np.zeros(K - 1, dtype=np.float64)
        for d in range(1, K):
            adj[d - 1] = adj_Sigma[d, d] * 2.0 * scales[d] ** 2
        return adj

    # Full: scales (K-1 free, or 0 when fix_scales=True) then correlations.
    fix_scales = getattr(control, "fix_scales", False)
    n_scale = 0 if fix_scales else K - 1
    n_corr = K * (K - 1) // 2
    out = np.zeros(n_scale + n_corr, dtype=np.float64)

    # ---- Scale block ----
    # For dim d (1-indexed in theta), differentiating Sigma = D corr D:
    # d Sigma_{ij}/d log scale_d  =  Sigma_{ij} * (delta_{di} + delta_{dj})
    #                              =  scales_i scales_j corr_ij * (delta_{di} + delta_{dj})
    # adj_log_scale_d = 2 * scales_d * sum_j adj_Sigma[d, j] * corr[d, j] * scales[j]
    # but scales[0] = 1 is fixed, so we only emit slots 1..K-1.
    # When fix_scales=True, all scales are locked at 1 — no slots to emit.
    if not fix_scales:
        for d in range(1, K):
            s = 0.0
            for j in range(K):
                s += adj_Sigma[d, j] * corr[d, j] * scales[j]
            out[d - 1] = 2.0 * s * scales[d]

    # ---- Correlation block ----
    if n_corr > 0:
        # Build adj_corr_vech in row-based upper-tri order over (i,j) with
        # i <= j. Off-diagonal entries collect both (i,j) and (j,i).
        n_corr_upper = K * (K + 1) // 2
        adj_corr_vech = np.zeros(n_corr_upper, dtype=np.float64)
        vidx = 0
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    adj_corr_vech[vidx] = 0.0  # corr diag fixed at 1
                else:
                    adj_corr_vech[vidx] = (
                        2.0 * adj_Sigma[i, j] * scales[i] * scales[j]
                    )
                vidx += 1

        if corr_jac is not None:
            # Spherical: chain through corr_jac : (n_corr, n_corr_upper)
            out[n_scale: n_scale + n_corr] = corr_jac @ adj_corr_vech
        else:
            # Tanh parameterization. corr[i,j] = tanh(corr_theta[c]) for
            # i<j (row-based). d/d corr_theta = 1 - tanh^2 = 1 - corr^2.
            # Walk the upper-tri (i<j) only.
            c_idx = 0
            v_idx = 0
            for i in range(K):
                for j in range(i, K):
                    if i < j:
                        out[n_scale + c_idx] = (
                            adj_corr_vech[v_idx] * (1.0 - corr[i, j] ** 2)
                        )
                        c_idx += 1
                    v_idx += 1

    return out
