"""Atomic gradient helpers for MVNCD analytic gradients.

Translates the following GAUSS procedures from BHATLIB:
- gradnoncdfn (line 2489): ∂Φ((x-μ)/σ)/∂(μ,σ²,x)
- gradcdfbvn (line 2512): ∂BVN(w₁,w₂,ρ)/∂(w₁,w₂,ρ)
- gradcdfbvnbycdfn (line 2548): ∂[BVN/Φ]/∂(w₁,w₂,ρ)
- gradunivariatenormaltrunc (line 2120): ∂(μ̃,σ̃²)/∂(μ,σ²,w)
- gradbivariatenormaltrunc (line 2138): ∂(μ̃,Σ̃)/∂(μ,Σ,w) for bivariate
- gradnoncdfbvn (line 2525): ∂BVN/∂(μ,Σ,x) non-standard
- gradnoncdfbvnbycdfn (line 2561): ∂[BVN/Φ]/∂(μ,Σ,x) non-standard

All functions are verified against finite differences in tests/test_gradmvn/test_trunc_grads.py.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.gradmvn._univariate import bivariate_normal_cdf


# ---------------------------------------------------------------------------
# 1. grad_noncdfn: ∂Φ((x-μ)/σ)/∂(μ, σ², x)
# GAUSS: gradnoncdfn (line 2489)
# ---------------------------------------------------------------------------

def grad_noncdfn(
    mu: float, sig2: float, x: float,
) -> tuple[float, float, float]:
    """Gradient of non-standard univariate normal CDF.

    Computes ∂Φ((x-μ)/σ)/∂(μ, σ², x) where σ = √σ².

    Parameters
    ----------
    mu : float
        Mean.
    sig2 : float
        Variance (must be > 0).
    x : float
        Upper integration limit.

    Returns
    -------
    g_mu : float
        ∂Φ/∂μ = -φ(w)/σ
    g_sig2 : float
        ∂Φ/∂σ² = -w·φ(w)/(2σ²)
    g_x : float
        ∂Φ/∂x = φ(w)/σ
    """
    invsig2 = 1.0 / sig2
    invsig = np.sqrt(invsig2)
    w = (x - mu) * invsig
    phi_w = norm.pdf(w)

    g_mu = -invsig * phi_w
    g_sig2 = phi_w * (-0.5 * invsig2 * w)
    g_x = invsig * phi_w

    return g_mu, g_sig2, g_x


# ---------------------------------------------------------------------------
# 2. grad_cdf_bvn: ∂BVN(w₁,w₂,ρ)/∂(w₁, w₂, ρ)
# GAUSS: gradcdfbvn (line 2512)
# ---------------------------------------------------------------------------

def grad_cdf_bvn(
    w1: float, w2: float, rho: float,
) -> tuple[float, float, float]:
    """Gradient of standard bivariate normal CDF.

    Computes ∂BVN(w₁,w₂,ρ)/∂(w₁, w₂, ρ) where BVN = P(Z₁≤w₁, Z₂≤w₂).

    Parameters
    ----------
    w1, w2 : float
        Upper integration limits.
    rho : float
        Correlation coefficient in (-1, 1).

    Returns
    -------
    gw1, gw2, grho : float
        Partial derivatives w.r.t. w₁, w₂, ρ.
    """
    rho = max(-0.9999, min(0.9999, rho))
    rhotilde = np.sqrt(1.0 - rho**2)

    tr1 = (w2 - rho * w1) / rhotilde
    tr2 = (w1 - rho * w2) / rhotilde

    gw1 = norm.pdf(w1) * norm.cdf(tr1)
    gw2 = norm.pdf(w2) * norm.cdf(tr2)
    grho = (1.0 / rhotilde) * norm.pdf(w1) * norm.pdf(tr1)

    return gw1, gw2, grho


# ---------------------------------------------------------------------------
# 3. grad_cdf_bvn_by_cdfn: ∂[BVN(w₁,w₂,ρ)/Φ(w₁)]/∂(w₁, w₂, ρ)
# GAUSS: gradcdfbvnbycdfn (line 2548)
# ---------------------------------------------------------------------------

def grad_cdf_bvn_by_cdfn(
    w1: float, w2: float, rho: float,
) -> tuple[float, float, float]:
    """Gradient of BVN/Φ ratio (used by OVUS screening).

    Computes ∂[BVN(w₁,w₂,ρ)/Φ(w₁)]/∂(w₁, w₂, ρ).

    Parameters
    ----------
    w1, w2 : float
        Upper integration limits.
    rho : float
        Correlation coefficient in (-1, 1).

    Returns
    -------
    gw1, gw2, grho : float
        Partial derivatives w.r.t. w₁, w₂, ρ.
    """
    bivarcdf = bivariate_normal_cdf(w1, w2, rho)
    univarcdf = norm.cdf(w1)

    gw1_bvn, gw2_bvn, grho_bvn = grad_cdf_bvn(w1, w2, rho)

    # Quotient rule: d(f/g) = (g·df - f·dg) / g²
    gw1 = (univarcdf * gw1_bvn - bivarcdf * norm.pdf(w1)) / (univarcdf**2)
    gw2 = gw2_bvn / univarcdf
    grho = grho_bvn / univarcdf

    return gw1, gw2, grho


# ---------------------------------------------------------------------------
# 4. grad_univariate_normal_trunc: ∂(μ̃, σ̃²)/∂(μ, σ², w)
# GAUSS: gradunivariatenormaltrunc (line 2120)
# ---------------------------------------------------------------------------

def grad_univariate_normal_trunc(
    mu: float, sig2: float, trpoint: float,
) -> tuple[NDArray, NDArray]:
    """Gradient of univariate truncated normal moments.

    For Z ~ N(μ, σ²), computes ∂μ̃/∂(μ,σ²,w) and ∂σ̃²/∂(μ,σ²,w)
    where μ̃ = E[Z|Z≤w], σ̃² = Var[Z|Z≤w], w = trpoint.

    Parameters
    ----------
    mu : float
        Mean of the untruncated distribution.
    sig2 : float
        Variance of the untruncated distribution (> 0).
    trpoint : float
        Truncation point (upper limit).

    Returns
    -------
    dmutrunc : ndarray, shape (3,)
        (∂μ̃/∂μ, ∂μ̃/∂σ², ∂μ̃/∂w)
    dsigtrunc : ndarray, shape (3,)
        (∂σ̃²/∂μ, ∂σ̃²/∂σ², ∂σ̃²/∂w)
    """
    sig = np.sqrt(sig2)
    w = (trpoint - mu) / sig

    phi_w = norm.pdf(w)
    Phi_w = norm.cdf(w)
    if Phi_w < 1e-300:
        Phi_w = 1e-300

    lam = -phi_w / Phi_w

    # dλ/dw = λ(λ - w)  (standard result for inverse Mills ratio derivative)
    dlamdw = lam * (lam - w)

    # ∂μ̃/∂(μ, σ², w)
    # μ̃ = μ + σ·λ
    # GAUSS: dmutrunc = (1-dlamdw) | ((lam-dlamdw*w)/(2*sig)) | dlamdw
    dmu_dmu = 1.0 - dlamdw
    dmu_dsig2 = (lam - dlamdw * w) / (2.0 * sig)
    dmu_dw = dlamdw
    dmutrunc = np.array([dmu_dmu, dmu_dsig2, dmu_dw])

    # ∂σ̃²/∂(μ, σ², w)
    # σ̃² = σ²·(1 + λ(w - λ))
    dsigtemp = dlamdw * (w - 2.0 * lam)
    dsigtruncdw = sig2 * (lam + dsigtemp)  # = sig^2 * (lam + dlamdw*(w-2*lam))
    dsigtemp1 = (1.0 / sig) * dsigtruncdw
    dsigtemp2 = 1.0 + lam * (w - lam) - (w / 2.0) * (lam + dsigtemp)

    dsig_dmu = -dsigtemp1
    dsig_dsig2 = dsigtemp2
    dsig_dw = dsigtemp1
    dsigtrunc = np.array([dsig_dmu, dsig_dsig2, dsig_dw])

    return dmutrunc, dsigtrunc


# ---------------------------------------------------------------------------
# 5. grad_bivariate_normal_trunc: ∂(μ̃, Σ̃)/∂(a, Σ, w) for bivariate
# GAUSS: gradbivariatenormaltrunc (line 2138)
# ---------------------------------------------------------------------------

def grad_bivariate_normal_trunc(
    mu: NDArray, cov: NDArray, trpoint: NDArray,
) -> tuple[NDArray, NDArray]:
    """Gradient of bivariate truncated normal moments.

    For X ~ N₂(μ, Σ), computes gradients of:
      μ̃ = E[X|X≤a] (2-vector)
      Σ̃ = Cov[X|X≤a] (upper-tri: Σ̃₁₁, Σ̃₁₂, Σ̃₂₂)
    w.r.t. 7 inputs: (a₁, a₂, σ₁², σ₁₂, σ₂², w₁, w₂).

    Parameters
    ----------
    mu : ndarray, shape (2,)
        Mean vector.
    cov : ndarray, shape (2, 2)
        Covariance matrix.
    trpoint : ndarray, shape (2,)
        Truncation points (upper limits).

    Returns
    -------
    dmuderiv : ndarray, shape (7, 2)
        Jacobian of truncated mean w.r.t. (a₁, a₂, σ₁², σ₁₂, σ₂², w₁, w₂).
    domgderiv : ndarray, shape (7, 3)
        Jacobian of truncated covariance upper-tri (Σ̃₁₁, Σ̃₁₂, Σ̃₂₂)
        w.r.t. (a₁, a₂, σ₁², σ₁₂, σ₂², w₁, w₂).
    """
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    trpoint = np.asarray(trpoint, dtype=np.float64)

    # Standard deviations and their inverses
    sigorig = np.sqrt(np.diag(cov))
    sigorigsq = sigorig**2
    sigoriginv = 1.0 / np.maximum(sigorig, 1e-15)
    sigoriginvrev = sigoriginv[::-1]

    # Standardize
    newtrpoint = (trpoint - mu) * sigoriginv
    rho = cov[0, 1] / (sigorig[0] * sigorig[1]) if sigorig[0] > 1e-15 and sigorig[1] > 1e-15 else 0.0
    rho = max(-0.9999, min(0.9999, rho))

    P = bivariate_normal_cdf(newtrpoint[0], newtrpoint[1], rho)
    if P < 1e-300:
        P = 1e-300

    newtrpointrev = newtrpoint[::-1]
    rhotilde = np.sqrt(1.0 - rho**2)
    rhotilde2 = rhotilde**2

    # Conditional standardized limits
    tr1 = (newtrpoint - rho * newtrpointrev) / rhotilde
    tr2 = tr1[::-1]

    trcomp = rho * newtrpoint - newtrpointrev
    trcomprev = trcomp[::-1]

    pd1 = norm.pdf(newtrpoint)
    pd2 = pd1[::-1]
    cd1 = norm.cdf(tr1)
    cd2 = cd1[::-1]

    del1 = pd1 * cd2
    del2 = pd2 * cd1
    pdf2 = (1.0 / rhotilde) * pd1[0] * norm.pdf(tr1[1])

    # Truncated mean (standardized): Bhat (2018) Eq. (4)
    mu_trunc = (-rho * del2 - del1) / P

    # --- Mean gradient (dmuderiv) ---
    # Each row corresponds to an input, each column to a mean component
    dmutilde1dw1 = (1.0 / P) * sigorig * (newtrpoint * del1 + del1 * (-mu_trunc))
    dmutilde1dw2 = (1.0 / P) * (-sigorig) * (
        rhotilde2 * pdf2 + (mu_trunc - rho * newtrpointrev) * del2
    )
    dmutildedrho = (1.0 / P) * (-sigorig) * (del2 + pdf2 * (mu_trunc - newtrpoint))

    dmutilde1da1 = 1.0 - sigoriginv * dmutilde1dw1
    dmutilde1da2 = -sigoriginvrev * dmutilde1dw2

    dmutilde1dsig1 = -(
        sigoriginv * newtrpoint * dmutilde1dw1
        - mu_trunc
        + rho * sigoriginv * dmutildedrho
    )
    dmutilde1dsig2 = -(
        sigoriginvrev * newtrpointrev * dmutilde1dw2
        + rho * sigoriginvrev * dmutildedrho
    )
    dmutilde1dsig1sq = 0.5 * sigoriginv * dmutilde1dsig1
    dmutilde1dsig2sq = 0.5 * sigoriginvrev * dmutilde1dsig2
    dmutildedsig12 = np.prod(sigoriginv) * dmutildedrho

    dmutilde1dtr1 = sigoriginv * dmutilde1dw1
    dmutilde1dtr2 = sigoriginvrev * dmutilde1dw2

    # Assemble: 7 inputs × 2 mean components
    # Row order: (a₁, a₂, σ₁², σ₁₂, σ₂², w₁, w₂)
    # GAUSS: dmuderiv1 uses component [0] for mu_1, with inputs in order:
    #   (da1[0], da2[0], dsig1sq[0], dsig12[0], dsig2sq[0], dtr1[0], dtr2[0])
    # dmuderiv2 uses component [1] for mu_2, with SWAPPED indices:
    #   (da2[1], da1[1], dsig2sq[1], dsig12[1], dsig1sq[1], dtr2[1], dtr1[1])
    dmuderiv = np.zeros((7, 2))
    # Column 0: ∂μ̃₁/∂(inputs)
    dmuderiv[0, 0] = dmutilde1da1[0]
    dmuderiv[1, 0] = dmutilde1da2[0]
    dmuderiv[2, 0] = dmutilde1dsig1sq[0]
    dmuderiv[3, 0] = dmutildedsig12[0]
    dmuderiv[4, 0] = dmutilde1dsig2sq[0]
    dmuderiv[5, 0] = dmutilde1dtr1[0]
    dmuderiv[6, 0] = dmutilde1dtr2[0]
    # Column 1: ∂μ̃₂/∂(inputs) — note the GAUSS swap pattern
    dmuderiv[0, 1] = dmutilde1da2[1]
    dmuderiv[1, 1] = dmutilde1da1[1]
    dmuderiv[2, 1] = dmutilde1dsig2sq[1]
    dmuderiv[3, 1] = dmutildedsig12[1]
    dmuderiv[4, 1] = dmutilde1dsig1sq[1]
    dmuderiv[5, 1] = dmutilde1dtr2[1]
    dmuderiv[6, 1] = dmutilde1dtr1[1]

    # --- Covariance gradient (domgderiv) ---
    # Truncated variance (standardized): Bhat (2018) Eq. (5)
    sig_trunc = (
        (P - newtrpoint * pd1 * cd2 - rho**2 * newtrpointrev * pd2 * cd1
         + rhotilde * rho * pd2 * norm.pdf(tr1)) / P
        - mu_trunc**2
    )

    # Truncated cross-covariance (standardized)
    sig12_trunc = (
        (rho * P
         - rho * newtrpoint[0] * pd1[0] * cd2[0]
         + rhotilde * pd1[0] * norm.pdf(tr2[0])
         - rho * newtrpointrev[0] * pd2[0] * cd1[0]) / P
        - np.prod(mu_trunc)
    )

    # Derivatives of truncated variance w.r.t. standardized limits
    dsigtilde1dw1 = (
        -(del1 / P) * (-newtrpoint**2 + sig_trunc + mu_trunc**2)
        - 2.0 * mu_trunc * (dmutilde1dw1 / sigorig)
    )
    dsigtilde1dw2 = (
        -(1.0 / P) * (
            rhotilde2 * (rho * pdf2 * newtrpointrev + newtrpoint * pdf2 - del2)
            - rho**2 * newtrpointrev**2 * del2
            + (sig_trunc + mu_trunc**2) * del2
        )
        - 2.0 * mu_trunc * (dmutilde1dw2 / sigorig)
    )
    dsigtildedrho = (
        -(1.0 / P) * (
            pdf2 * (-newtrpoint**2)
            + 2.0 * rho * newtrpointrev * del2
            - 2.0 * pdf2 * rhotilde2
            + (sig_trunc + mu_trunc**2) * pdf2
        )
        - 2.0 * mu_trunc * (dmutildedrho / sigorig)
    )

    # Derivatives of truncated cross-covariance
    dsigtilde = (1.0 / P) * (
        (rho * newtrpoint) * (rho * pdf2 + newtrpoint * del1)
        - pdf2 * newtrpoint
        - del1 * (sig12_trunc + np.prod(mu_trunc))
    )
    dsigtilde12dw1 = dsigtilde[0] - np.sum(
        mu_trunc * np.array([dmutilde1dw2[1], dmutilde1dw1[0]]) / sigorig[::-1]
    )
    dsigtilde12dw2 = dsigtilde[1] - np.sum(
        mu_trunc * np.array([dmutilde1dw1[1], dmutilde1dw2[0]]) / sigorig[::-1]
    )
    dsigtilde12drho = (
        1.0
        - (1.0 / P) * (
            (-np.prod(newtrpoint)) * pdf2
            + np.sum(newtrpoint * del1)
            + (np.prod(mu_trunc) + sig12_trunc) * pdf2
        )
        - np.sum(mu_trunc * dmutildedrho[::-1] * sigoriginvrev)
    )

    # Transform from standardized to original scale
    domg1dtr1 = sigorig * dsigtilde1dw1
    domg12dtr = sigorig[::-1] * np.array([dsigtilde12dw1, dsigtilde12dw2])
    domg1dtr2 = sigorigsq * sigoriginvrev * dsigtilde1dw2
    domg1da1 = -domg1dtr1
    domg12da = -domg12dtr
    domg1da2 = -domg1dtr2

    domg1s1 = sig_trunc + dsigtilde1dw1 * newtrpoint * (-0.5) + dsigtildedrho * rho * (-0.5)
    domg1s12 = (sigorig / sigorig[::-1]) * dsigtildedrho
    domg1s2 = (-0.5) * (sigorigsq / sigorigsq[::-1]) * (
        dsigtilde1dw2 * newtrpointrev + dsigtildedrho * rho
    )
    domg12s1 = 0.5 * (sigorig[::-1] / sigorig) * (
        sig12_trunc
        - newtrpoint * np.array([dsigtilde12dw1, dsigtilde12dw2])
        - rho * dsigtilde12drho
    )
    domg12s12 = dsigtilde12drho

    # Assemble: 7 inputs × 3 covariance components (Σ̃₁₁, Σ̃₁₂, Σ̃₂₂)
    domgderiv = np.zeros((7, 3))

    # Column 0: ∂Σ̃₁₁/∂(inputs) — uses component [0] throughout
    domgderiv[0, 0] = domg1da1[0]
    domgderiv[1, 0] = domg1da2[0]
    domgderiv[2, 0] = domg1s1[0]
    domgderiv[3, 0] = domg1s12[0]
    domgderiv[4, 0] = domg1s2[0]
    domgderiv[5, 0] = domg1dtr1[0]
    domgderiv[6, 0] = domg1dtr2[0]

    # Column 1: ∂Σ̃₁₂/∂(inputs) — mixed components
    domgderiv[0, 1] = domg12da[0]
    domgderiv[1, 1] = domg12da[1]
    domgderiv[2, 1] = domg12s1[0]
    domgderiv[3, 1] = domg12s12
    domgderiv[4, 1] = domg12s1[1]
    domgderiv[5, 1] = domg12dtr[0]
    domgderiv[6, 1] = domg12dtr[1]

    # Column 2: ∂Σ̃₂₂/∂(inputs) — uses component [1] with SWAPPED indices
    domgderiv[0, 2] = domg1da2[1]
    domgderiv[1, 2] = domg1da1[1]
    domgderiv[2, 2] = domg1s2[1]
    domgderiv[3, 2] = domg1s12[1]
    domgderiv[4, 2] = domg1s1[1]
    domgderiv[5, 2] = domg1dtr2[1]
    domgderiv[6, 2] = domg1dtr1[1]

    return dmuderiv, domgderiv


# ---------------------------------------------------------------------------
# 6. _standardize_bvn: helper to standardize bivariate inputs
# ---------------------------------------------------------------------------

def _standardize_bvn(
    mu: NDArray, cov: NDArray, x: NDArray,
) -> tuple[NDArray, NDArray, float]:
    """Standardize bivariate inputs to (w, sqrtom, rho)."""
    mu = np.asarray(mu, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)

    om = np.diag(cov)
    sqrtom = np.sqrt(np.maximum(om, 1e-30))
    w = (x - mu) / sqrtom
    rho = cov[0, 1] / (sqrtom[0] * sqrtom[1])
    rho = max(-0.9999, min(0.9999, rho))

    return w, sqrtom, rho


def _destandardize_bvn_grad(
    gw: NDArray, grho: float, w: NDArray, sqrtom: NDArray, rho: float,
) -> tuple[NDArray, NDArray, NDArray]:
    """Chain-rule: transform (gw, grho) in standardized space to (gmu, gcov, gx).

    Implements the GAUSS gradcorcov (line 3409) logic for the K=2 case.

    gcov is a 3-vector: (∂/∂Σ₁₁, ∂/∂Σ₁₂, ∂/∂Σ₂₂) upper-triangular order.
    """
    sig2 = sqrtom**2

    gmu = -gw / sqrtom
    gx = -gmu

    # ∂(·)/∂Σ via chain rule through standardization
    # ∂w_i/∂Σ_ii = -w_i/(2·Σ_ii), ∂ρ/∂Σ_ii = -ρ/(2·Σ_ii), ∂ρ/∂Σ_12 = 1/(σ₁σ₂)
    gcov = np.array([
        -(gw[0] * w[0] + grho * rho) / (2.0 * sig2[0]),
        grho / (sqrtom[0] * sqrtom[1]),
        -(gw[1] * w[1] + grho * rho) / (2.0 * sig2[1]),
    ])

    return gmu, gcov, gx


# ---------------------------------------------------------------------------
# 7. grad_noncdfbvn: ∂BVN/∂(μ, Σ, x) non-standard
# GAUSS: gradnoncdfbvn (line 2525)
# ---------------------------------------------------------------------------

def grad_noncdfbvn(
    mu: NDArray, cov: NDArray, x: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Gradient of non-standard bivariate normal CDF.

    Computes ∂P(X≤x)/∂(μ, Σ, x) where X ~ N₂(μ, Σ).

    Parameters
    ----------
    mu : ndarray, shape (2,)
        Mean vector.
    cov : ndarray, shape (2, 2)
        Covariance matrix.
    x : ndarray, shape (2,)
        Upper integration limits.

    Returns
    -------
    gmu : ndarray, shape (2,)
        ∂P/∂μ
    gcov : ndarray, shape (3,)
        ∂P/∂(Σ₁₁, Σ₁₂, Σ₂₂) upper-triangular order.
    gx : ndarray, shape (2,)
        ∂P/∂x
    """
    w, sqrtom, rho = _standardize_bvn(mu, cov, x)
    gw1, gw2, grho = grad_cdf_bvn(w[0], w[1], rho)
    gw = np.array([gw1, gw2])

    return _destandardize_bvn_grad(gw, grho, w, sqrtom, rho)


# ---------------------------------------------------------------------------
# 8. grad_noncdfbvn_by_cdfn: ∂[BVN/Φ]/∂(μ, Σ, x) non-standard
# GAUSS: gradnoncdfbvnbycdfn (line 2561)
# ---------------------------------------------------------------------------

def grad_noncdfbvn_by_cdfn(
    mu: NDArray, cov: NDArray, x: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """Gradient of non-standard BVN/Φ ratio (OVUS screening).

    Computes ∂[P(X≤x)/Φ((x₁-μ₁)/σ₁)]/∂(μ, Σ, x) where X ~ N₂(μ, Σ).

    Parameters
    ----------
    mu : ndarray, shape (2,)
        Mean vector.
    cov : ndarray, shape (2, 2)
        Covariance matrix.
    x : ndarray, shape (2,)
        Upper integration limits.

    Returns
    -------
    gmu : ndarray, shape (2,)
        ∂(BVN/Φ)/∂μ
    gcov : ndarray, shape (3,)
        ∂(BVN/Φ)/∂(Σ₁₁, Σ₁₂, Σ₂₂) upper-triangular order.
    gx : ndarray, shape (2,)
        ∂(BVN/Φ)/∂x
    """
    w, sqrtom, rho = _standardize_bvn(mu, cov, x)
    gw1, gw2, grho = grad_cdf_bvn_by_cdfn(w[0], w[1], rho)
    gw = np.array([gw1, gw2])

    return _destandardize_bvn_grad(gw, grho, w, sqrtom, rho)
