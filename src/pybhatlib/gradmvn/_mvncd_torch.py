"""PyTorch GPU-accelerated MVNCD for K=2 (bivariate normal CDF + gradient).

Implements the Genz BVND algorithm in fully vectorized PyTorch tensor
operations, enabling GPU acceleration for large-N datasets.

All functions operate on batches of N observations in parallel.
For N < ~5000, the NumPy+Numba path in _mvncd_grad_analytic.py is faster.
For N >= 5000, the GPU path provides 10-100x speedup.

References
----------
Genz, A. (2004). Numerical computation of rectangular bivariate and
trivariate normal and t probabilities. Statistics and Computing, 14, 251-260.
"""

from __future__ import annotations

import math

import torch

# Gauss-Legendre 20-point quadrature (10 symmetric pairs)
# Raw values; device-specific tensors are cached in _gl_cache below.
_GL20_W_VALUES = [
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091,
    0.0832767415767048, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183820, 0.1491729864726037,
    0.1527533871307258,
]
_GL20_X_VALUES = [
    0.9931285991850949, 0.9639719272779138, 0.9122344282513259,
    0.8391169718222188, 0.7463319064601508, 0.6360536807265150,
    0.5108670019508271, 0.3737060887154195, 0.2277858511416451,
    0.0765265211334973,
]

_gl_cache: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_gl_tables(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GL20 weights and abscissae on the given device, cached."""
    key = (device, dtype)
    if key not in _gl_cache:
        _gl_cache[key] = (
            torch.tensor(_GL20_W_VALUES, dtype=dtype, device=device),
            torch.tensor(_GL20_X_VALUES, dtype=dtype, device=device),
        )
    return _gl_cache[key]


# Compiled function cache (lazy, populated on first use)
_compiled_cache: dict[str, object] = {}


def get_compiled_grad_perobs():
    """Return torch.compiled version of mvncd_grad_batch_k2_perobs_torch."""
    key = "grad_perobs"
    if key not in _compiled_cache:
        _compiled_cache[key] = torch.compile(
            mvncd_grad_batch_k2_perobs_torch, mode="default",
        )
    return _compiled_cache[key]


def get_compiled_grad_shared():
    """Return torch.compiled version of mvncd_grad_batch_k2_torch."""
    key = "grad_shared"
    if key not in _compiled_cache:
        _compiled_cache[key] = torch.compile(
            mvncd_grad_batch_k2_torch, mode="default",
        )
    return _compiled_cache[key]

_TWOPI = 2.0 * math.pi
_INV_SQRT_2PI = 1.0 / math.sqrt(_TWOPI)
_SQRT_TWOPI = math.sqrt(_TWOPI)


def _ndtr_torch(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF."""
    return 0.5 * torch.erfc(-x * (1.0 / math.sqrt(2.0)))


def _npdf_torch(x: torch.Tensor) -> torch.Tensor:
    """Standard normal PDF."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _bvn_low_branch(
    dh: torch.Tensor,
    dk: torch.Tensor,
    r: torch.Tensor,
    gl20_w: torch.Tensor,
    gl20_x: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """Low-to-moderate correlation branch (|rho| < 0.925), branchless.

    Computed for ALL N observations; caller uses torch.where to select.
    """
    hk = dh * dk
    hs = (dh * dh + dk * dk) / 2.0
    asr = torch.asin(r)  # (N,)

    # (N, 10, 2) quadrature grid
    quad_arg = asr[:, None, None] * (signs[None, None, :] * gl20_x[None, :, None] + 1.0) / 2.0
    sn = torch.sin(quad_arg)
    denom = (1.0 - sn * sn).clamp(min=1e-30)
    exponent = (sn * hk[:, None, None] - hs[:, None, None]) / denom

    contrib = gl20_w[None, :, None] * torch.exp(exponent)
    bvn = contrib.sum(dim=(1, 2))
    bvn = bvn * asr / (2.0 * _TWOPI)
    bvn = bvn + _ndtr_torch(-dh) * _ndtr_torch(-dk)
    return bvn


def _bvn_high_branch(
    dh: torch.Tensor,
    dk: torch.Tensor,
    r: torch.Tensor,
    gl20_w: torch.Tensor,
    gl20_x: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    """High correlation branch (|rho| >= 0.925), branchless.

    Computed for ALL N observations; caller uses torch.where to select.
    """
    # Flip sign of dk for negative rho
    neg_r = r < 0
    k = torch.where(neg_r, -dk, dk)
    hk = dh * k

    ass1 = (1.0 - r) * (1.0 + r)
    a_val = torch.sqrt(ass1.clamp(min=1e-30))
    bs = (dh - k) ** 2
    c = (4.0 - hk) / 8.0
    d = (12.0 - hk) / 16.0

    # First term
    asr_h = torch.where(
        ass1 > 1e-30,
        -(bs / ass1.clamp(min=1e-30) + hk) / 2.0,
        torch.full_like(ass1, -200.0),
    )
    term1 = torch.where(
        asr_h > -100,
        a_val * torch.exp(asr_h.clamp(min=-200.0)) * (
            1.0 - c * (bs - ass1) * (1.0 - d * bs / 5.0) / 3.0
            + c * d * ass1 * ass1 / 5.0
        ),
        torch.zeros_like(a_val),
    )

    # Second term
    b = torch.sqrt(bs.clamp(min=0.0))
    second_valid = (-hk < 100) & (a_val > 1e-30)
    term2 = torch.where(
        second_valid,
        torch.exp((-hk / 2.0).clamp(min=-200.0)) * _SQRT_TWOPI
        * _ndtr_torch(-b / a_val.clamp(min=1e-30))
        * b * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0),
        torch.zeros_like(a_val),
    )

    bvn = term1 - term2

    # GL20 quadrature correction
    a_half = a_val / 2.0

    xs = (a_half[:, None, None] * (signs[None, None, :] * gl20_x[None, :, None] + 1.0)) ** 2
    rs = torch.sqrt((1.0 - xs).clamp(min=0.0))

    valid = (xs > 1e-30) & (rs > 1e-30)
    asr2 = -(bs[:, None, None] / xs.clamp(min=1e-30) + hk[:, None, None]) / 2.0
    exp_valid = valid & (asr2 > -100)

    exp_term = torch.where(
        exp_valid,
        torch.exp((-hk[:, None, None] * (1.0 - rs) / (2.0 * (1.0 + rs).clamp(min=1e-30))).clamp(min=-200.0))
        / rs.clamp(min=1e-30)
        - (1.0 + c[:, None, None] * xs * (1.0 + d[:, None, None] * xs)),
        torch.zeros_like(xs),
    )
    gl_contrib = torch.where(
        exp_valid,
        a_half[:, None, None] * gl20_w[None, :, None] * torch.exp(asr2.clamp(min=-200.0)) * exp_term,
        torch.zeros_like(xs),
    )
    bvn = bvn + gl_contrib.sum(dim=(1, 2))
    bvn = -bvn / _TWOPI

    # Final adjustment based on sign of rho
    bvn = torch.where(
        neg_r,
        -bvn + torch.where(k > dh, _ndtr_torch(k) - _ndtr_torch(dh), torch.zeros_like(bvn)),
        bvn + _ndtr_torch(-torch.maximum(dh, k)),
    )
    return bvn


def bvn_cdf_torch(
    x1: torch.Tensor,
    x2: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Vectorized bivariate standard normal CDF via Genz BVND algorithm.

    Computes P(Z1 <= x1, Z2 <= x2) where (Z1, Z2) ~ N(0, [[1, rho], [rho, 1]])
    for N observations in parallel.

    Fully branchless — compatible with torch.compile / CUDA graphs.
    Both low and high correlation branches are computed for all observations,
    then torch.where selects the correct result per element.

    Parameters
    ----------
    x1, x2 : Tensor, shape (N,)
        Upper integration limits.
    rho : Tensor, shape (N,) or scalar
        Correlation coefficient(s).

    Returns
    -------
    prob : Tensor, shape (N,)
        Bivariate normal CDF values.
    """
    device = x1.device
    dtype = x1.dtype
    N = x1.shape[0]

    gl20_w, gl20_x = _get_gl_tables(device, dtype)
    signs = torch.tensor([1.0, -1.0], device=device, dtype=dtype)

    # Broadcast rho if scalar
    if rho.dim() == 0:
        rho = rho.expand(N)

    dh = -x1
    dk = -x2
    abs_r = rho.abs()

    # --- Edge case results (computed for all N, selected via torch.where) ---
    result_zero = _ndtr_torch(x1) * _ndtr_torch(x2)
    result_pos = _ndtr_torch(torch.minimum(x1, x2))
    val_neg = (_ndtr_torch(x1) + _ndtr_torch(x2) - 1.0)
    result_neg = torch.where((x1 + x2) >= 0, val_neg.clamp(min=0.0), torch.zeros_like(val_neg))

    # --- Both branches computed for all observations ---
    # Clamp rho away from ±1 for numerical safety in the regular branches
    rho_safe = rho.clamp(-0.99999, 0.99999)
    bvn_low = _bvn_low_branch(dh, dk, rho_safe, gl20_w, gl20_x, signs)
    bvn_high = _bvn_high_branch(dh, dk, rho_safe, gl20_w, gl20_x, signs)

    # --- Select: low vs high correlation ---
    bvn = torch.where(abs_r < 0.925, bvn_low, bvn_high)

    # --- Select: edge cases vs regular ---
    zero_rho = abs_r < 1e-15
    pos_one = rho > 1.0 - 1e-15
    neg_one = rho < -1.0 + 1e-15

    result = torch.where(zero_rho, result_zero,
             torch.where(pos_one, result_pos,
             torch.where(neg_one, result_neg, bvn)))

    return result.clamp(0.0, 1.0)


def mvncd_grad_batch_k2_torch(
    a_all: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized BVN gradient on GPU with shared covariance.

    Parameters
    ----------
    a_all : Tensor, shape (N, 2)
    sigma : Tensor, shape (2, 2) — shared covariance

    Returns
    -------
    prob, grad_a, grad_sigma_vech : Tensors
    """
    s11 = sigma[0, 0]
    s12 = sigma[0, 1]
    s22 = sigma[1, 1]
    sd1 = torch.sqrt(s11.clamp(min=1e-30))
    sd2 = torch.sqrt(s22.clamp(min=1e-30))
    rho = (s12 / (sd1 * sd2)).clamp(-0.9999, 0.9999)
    rhotilde = torch.sqrt(1.0 - rho * rho)

    w1 = a_all[:, 0] / sd1
    w2 = a_all[:, 1] / sd2

    prob = bvn_cdf_torch(w1, w2, rho)
    prob = prob.clamp(min=1e-300)

    tr1 = (w2 - rho * w1) / rhotilde
    tr2 = (w1 - rho * w2) / rhotilde

    phi_w1 = _npdf_torch(w1)
    phi_w2 = _npdf_torch(w2)
    Phi_tr1 = _ndtr_torch(tr1)
    Phi_tr2 = _ndtr_torch(tr2)

    gw1 = phi_w1 * Phi_tr1
    gw2 = phi_w2 * Phi_tr2
    phi_tr1 = _npdf_torch(tr1)
    grho = (1.0 / rhotilde) * phi_w1 * phi_tr1

    grad_a = torch.stack([gw1 / sd1, gw2 / sd2], dim=1)

    grad_s11 = -(gw1 * w1 + grho * rho) / (2.0 * s11)
    grad_s12 = grho / (sd1 * sd2)
    grad_s22 = -(gw2 * w2 + grho * rho) / (2.0 * s22)
    grad_sigma_vech = torch.stack([grad_s11, grad_s12, grad_s22], dim=1)

    return prob, grad_a, grad_sigma_vech


def mvncd_grad_batch_k2_perobs_torch(
    a_all: torch.Tensor,
    sigma_all: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized BVN gradient on GPU with per-observation covariance.

    Parameters
    ----------
    a_all : Tensor, shape (N, 2)
    sigma_all : Tensor, shape (N, 2, 2) — per-obs covariance

    Returns
    -------
    prob, grad_a, grad_sigma_vech : Tensors
    """
    s11 = sigma_all[:, 0, 0]
    s12 = sigma_all[:, 0, 1]
    s22 = sigma_all[:, 1, 1]
    sd1 = torch.sqrt(s11.clamp(min=1e-30))
    sd2 = torch.sqrt(s22.clamp(min=1e-30))
    rho = (s12 / (sd1 * sd2)).clamp(-0.9999, 0.9999)
    rhotilde = torch.sqrt(1.0 - rho * rho)

    w1 = a_all[:, 0] / sd1
    w2 = a_all[:, 1] / sd2

    prob = bvn_cdf_torch(w1, w2, rho)
    prob = prob.clamp(min=1e-300)

    tr1 = (w2 - rho * w1) / rhotilde
    tr2 = (w1 - rho * w2) / rhotilde

    phi_w1 = _npdf_torch(w1)
    phi_w2 = _npdf_torch(w2)
    Phi_tr1 = _ndtr_torch(tr1)
    Phi_tr2 = _ndtr_torch(tr2)

    gw1 = phi_w1 * Phi_tr1
    gw2 = phi_w2 * Phi_tr2
    phi_tr1 = _npdf_torch(tr1)
    grho = (1.0 / rhotilde) * phi_w1 * phi_tr1

    grad_a = torch.stack([gw1 / sd1, gw2 / sd2], dim=1)

    grad_s11 = -(gw1 * w1 + grho * rho) / (2.0 * s11)
    grad_s12 = grho / (sd1 * sd2)
    grad_s22 = -(gw2 * w2 + grho * rho) / (2.0 * s22)
    grad_sigma_vech = torch.stack([grad_s11, grad_s12, grad_s22], dim=1)

    return prob, grad_a, grad_sigma_vech
