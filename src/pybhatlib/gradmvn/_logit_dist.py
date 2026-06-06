"""Multivariate logistic distribution functions, partial cumulative distributions,
and simulation routines.

Implements the PDF, CDF, gradient, and simulation functions used in BHATLIB for
multivariate logistic error structures.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from itertools import combinations


# ---------------------------------------------------------------------------
# Helper: combinatorial index matrix 
# ---------------------------------------------------------------------------

def combinate(n: int, k: int) -> np.ndarray:
    """
    Return an (M x k) integer array whose rows are all size-k subsets
    of {1, 2, ..., n} (1-based indices, matching GAUSS convention).

    Parameters
    ----------
    n : int
        Total number of elements.
    k : int
        Subset size.

    Returns
    -------
    np.ndarray, shape (M, k)
        Each row is one combination (1-based).
    """
    combos = list(combinations(range(1, n + 1), k))
    return np.array(combos, dtype=int)  # shape (M, k)


# ---------------------------------------------------------------------------
# Standard multivariate logistic PDF
# ---------------------------------------------------------------------------

def pdfmvlogit(a: np.ndarray) -> np.ndarray:
    """
    Standard multivariate logistic density function.

    Parameters
    ----------
    a : np.ndarray, shape (K, Q)
        Abscissae matrix; K = number of variates, Q = number of observations.

    Returns
    -------
    np.ndarray, shape (Q,)
        Density evaluated at each of the Q observations.

    Notes
    -----
    The K-variate standard multivariate logistic density is:

        f(a) = K! * prod(exp(-a_k)) / (1 + sum(exp(-a_k)))^(K+1)
             = K! * exp(-sum(a)) / (1 + sum(exp(-a)))^(K+1)
    """
    a = np.atleast_2d(a)
    m = a.shape[0]                     # number of variates K
    f = np.exp(-a)                     # (K, Q)
    denom = 1.0 + f.sum(axis=0)        # (Q,)
    numer = np.exp(-a.sum(axis=0))     # (Q,)
    return float(np.prod(np.arange(1, m + 1))) * (denom ** (-(m + 1))) * numer


# ---------------------------------------------------------------------------
# Gradient of standard multivariate logistic PDF
# ---------------------------------------------------------------------------

def gradpdfmvlogit(a: np.ndarray) -> np.ndarray:
    """
    Gradient of the standard multivariate logistic density w.r.t. a.

    Parameters
    ----------
    a : np.ndarray, shape (K, Q)
        Abscissae; K = number of variates, Q = number of observations.

    Returns
    -------
    np.ndarray, shape (K, Q)
        Gradient matrix: element (k, q) = d f(a[:,q]) / d a[k,q].
    """
    a = np.atleast_2d(a)
    m = a.shape[0]
    f = np.exp(-a)                              # (K, Q)
    d = 1.0 + f.sum(axis=0)                     # (Q,)
    pf = pdfmvlogit(a)                          # (Q,)
    # h[k, q] = (m+1) * exp(-a[k,q]) / d[q]  -  1
    h = (m + 1) * (f / d[np.newaxis, :]) - 1.0  # (K, Q)
    return pf[np.newaxis, :] * h                 # (K, Q)


# ---------------------------------------------------------------------------
# Non-standard multivariate logistic PDF
# ---------------------------------------------------------------------------

def nonpdfmvlogit(
    a: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> np.ndarray:
    """
    Non-standard multivariate logistic density function.

    Parameters
    ----------
    a   : np.ndarray, shape (K, Q)  – abscissae.
    mu  : np.ndarray, shape (K, Q) or (K, 1)  – location parameters.
    sig : np.ndarray, shape (K, Q) or (K, 1)  – scale parameters.

    Returns
    -------
    np.ndarray, shape (Q,)
        Density values.

    Notes
    -----
    f_nonstandard(a; mu, sig) = (1 / prod(sig)) * f_standard((a - mu) / sig)
    """
    a = np.atleast_2d(a)
    mu = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    aa = (a - mu) / sig
    return (1.0 / sig.prod(axis=0)) * pdfmvlogit(aa)


# ---------------------------------------------------------------------------
# Gradient of non-standard multivariate logistic PDF
# ---------------------------------------------------------------------------

def gradnonpdfmvlogit(
    a: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the non-standard multivariate logistic density.

    Parameters
    ----------
    a   : np.ndarray, shape (K, Q)
    mu  : np.ndarray, shape (K, Q) or (K, 1)
    sig : np.ndarray, shape (K, Q) or (K, 1)

    Returns
    -------
    ga   : np.ndarray, shape (K, Q)  – gradient w.r.t. a.
    gmu  : np.ndarray, shape (K, Q) or (K, 1)  – gradient w.r.t. mu.
    gsig : np.ndarray, shape (K, Q) or (K, 1)  – gradient w.r.t. sig.
    """
    a = np.atleast_2d(a)
    mu = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)

    
    single_mu = (mu.shape[1] == 1)
    single_sig = (sig.shape[1] == 1)

    aa = (a - mu) / sig
    aa1 = gradpdfmvlogit(aa)                          # (K, Q)
    prsig = (1.0 / sig.prod(axis=0))                  # (Q,)  or scalar

    ga = prsig[np.newaxis, :] * (aa1 / sig)           # (K, Q)
    gmu = -ga
    gsig = prsig[np.newaxis, :] * ((aa1 * (-aa)) / sig) \
         + (-prsig[np.newaxis, :] / sig) * pdfmvlogit(aa)[np.newaxis, :]

    if single_mu:
        gmu = gmu.sum(axis=1, keepdims=True)           # (K, 1)
    if single_sig:
        gsig = gsig.sum(axis=1, keepdims=True)         # (K, 1)

    return ga, gmu, gsig


# ===========================================================================
# Standard partial cumulative multivariate logistic CDF and its gradient
# ===========================================================================

def pdfcdfmvlogit(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Standard partial cumulative multivariate logistic distribution.

    Computes Pr(eta_1 = a, eta_2 < b) where eta = (eta_1 | eta_2) is
    standard multivariate logistic distributed.

    Parameters
    ----------
    a : np.ndarray, shape (K1, Q)
        Abscissae for equality conditions (K1 "equality variates").
    b : np.ndarray, shape (K2, Q)
        Upper truncation points (K2 "inequality variates").

    Returns
    -------
    np.ndarray, shape (Q,)
        Probability values.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    m = a.shape[0]                              # K1
    d = np.vstack([a, b])                       # (K1+K2, Q)
    numer = np.exp(-a).prod(axis=0)             # (Q,)
    denom = (1.0 + np.exp(-d).sum(axis=0))      # (Q,)
    c = numer / (denom ** (m + 1))
    return float(np.prod(np.arange(1, m + 1))) * c


# ---------------------------------------------------------------------------
# Gradient of standard partial cumulative multivariate logistic CDF
# ---------------------------------------------------------------------------

def gradpdfcdfmvlogit(
    a: np.ndarray,
    b: np.ndarray,
):
    """
    Gradient of the standard partial cumulative multivariate logistic
    distribution w.r.t. a and b.

    Parameters
    ----------
    a : np.ndarray, shape (K1, Q)
    b : np.ndarray, shape (K2, Q)

    Returns
    -------
    ga : np.ndarray, shape (K1, Q)  – gradient w.r.t. a.
    gb : np.ndarray, shape (K2, Q)  – gradient w.r.t. b.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    m = a.shape[0]
    f  = np.exp(-a)                                           # (K1, Q)
    f1 = np.exp(-b)                                           # (K2, Q)
    d  = 1.0 + f.sum(axis=0) + f1.sum(axis=0)                # (Q,)
    pf = pdfcdfmvlogit(a, b)                                  # (Q,)

    # Gradient w.r.t. a: same structure as gradpdfmvlogit but uses combined d
    h  = (m + 1) * (f  / d[np.newaxis, :]) - 1.0             # (K1, Q)
    ga = pf[np.newaxis, :] * h                                # (K1, Q)

    # Gradient w.r.t. b: only through denominator
    gb = (pf[np.newaxis, :] * (m + 1)) * (f1 / d[np.newaxis, :])  # (K2, Q)

    return ga, gb


# ---------------------------------------------------------------------------
# Non-standard partial cumulative multivariate logistic CDF
# ---------------------------------------------------------------------------

def nonpdfcdfmvlogit(
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> np.ndarray:
    """
    Non-standard partial cumulative multivariate logistic distribution.

    Computes Pr(eta_1 = a, eta_2 < b) where (eta_1 | eta_2) is
    non-standard multivariate logistic with location mu and scale sig.

    Parameters
    ----------
    a   : np.ndarray, shape (K1, Q)
    b   : np.ndarray, shape (K2, Q)
    mu  : np.ndarray, shape (K1+K2, Q) or (K1+K2, 1)
          First K1 rows correspond to a, next K2 to b.
    sig : np.ndarray, shape (K1+K2, Q) or (K1+K2, 1)
          First K1 rows correspond to a, next K2 to b.

    Returns
    -------
    np.ndarray, shape (Q,)
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    mu = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    k1 = a.shape[0]
    k2 = b.shape[0]

    aa = (a - mu[:k1, :])  / sig[:k1, :]
    bb = (b - mu[k1:k1+k2, :]) / sig[k1:k1+k2, :]
    scale = (1.0 / sig[:k1, :].prod(axis=0))
    return scale * pdfcdfmvlogit(aa, bb)


# ---------------------------------------------------------------------------
# Gradient of non-standard partial cumulative multivariate logistic CDF
# ---------------------------------------------------------------------------

def gradnonpdfcdfmvlogit(
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the non-standard partial cumulative multivariate logistic
    distribution.

    Parameters
    ----------
    a   : np.ndarray, shape (K1, Q)
    b   : np.ndarray, shape (K2, Q)
    mu  : np.ndarray, shape (K1+K2, Q) or (K1+K2, 1)
    sig : np.ndarray, shape (K1+K2, Q) or (K1+K2, 1)

    Returns
    -------
    ga   : np.ndarray, shape (K1, Q)
    gb   : np.ndarray, shape (K2, Q)
    gmua : np.ndarray, shape (K1, Q) or (K1, 1)  – grad w.r.t. mu_a.
    gmub : np.ndarray, shape (K2, Q) or (K2, 1)  – grad w.r.t. mu_b.
    gsiga: np.ndarray, shape (K1, Q) or (K1, 1)  – grad w.r.t. sig_a.
    gsigb: np.ndarray, shape (K2, Q) or (K2, 1)  – grad w.r.t. sig_b.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    mu = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    k1 = a.shape[0]
    k2 = b.shape[0]
    single_mu  = (mu.shape[1]  == 1)
    single_sig = (sig.shape[1] == 1)

    siga = sig[:k1, :]
    sigb = sig[k1:k1+k2, :]
    mua  = mu[:k1, :]
    mub  = mu[k1:k1+k2, :]

    aa = (a - mua) / siga
    bb = (b - mub) / sigb

    gaa, gbb = gradpdfcdfmvlogit(aa, bb)                  # (K1,Q), (K2,Q)
    prsig = (1.0 / siga.prod(axis=0))                     # (Q,)

    ga   = prsig[np.newaxis, :] * (gaa / siga)            # (K1, Q)
    gb   = prsig[np.newaxis, :] * (gbb / sigb)            # (K2, Q)
    gmua = -ga
    gmub = -gb

    gsiga = (prsig[np.newaxis, :] * ((gaa * (-aa)) / siga)
             + (-prsig[np.newaxis, :] / siga)
               * pdfcdfmvlogit(aa, bb)[np.newaxis, :])
    gsigb = prsig[np.newaxis, :] * ((gbb * (-bb)) / sigb)

    if single_mu:
        gmua = gmua.sum(axis=1, keepdims=True)
        gmub = gmub.sum(axis=1, keepdims=True)
    if single_sig:
        gsiga = gsiga.sum(axis=1, keepdims=True)
        gsigb = gsigb.sum(axis=1, keepdims=True)

    return ga, gb, gmua, gmub, gsiga, gsigb


# ===========================================================================
# pdfcdfmvlogitc  –  combination of CDF/complement/PDF
# ===========================================================================

def pdfcdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> float:
    """
    Standard partial cumulative multivariate logistic: combines upper CDF,
    lower CDF complement, and density.

    Computes Pr(Z = a, X < b, Y > c) where (Z | X | Y) is standard
    multivariate logistic.

    Parameters
    ----------
    a : np.ndarray, shape (M, 1)
        Density evaluation points (equality) for variates Z.
    b : np.ndarray, shape (K, 1)
        Upper truncation points for variates X  (-inf to b).
    c : np.ndarray, shape (M, 1)
        Lower truncation points for variates Y  (c to +inf).

        Tip: to compute Pr(Y > b) only, pass a = np.full((M,1), 1000.0).

    Returns
    -------
    float
        Pr(Z = a, X < b, Y > c).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    c = np.atleast_2d(c)
    a1 = a.shape[0]   # M

    a2   = pdfcdfmvlogit(a, b)          # base term
    com1 = 0.0

    for i in range(1, a1 + 1):
        com = 0.0
        yi  = combinate(a1, i)          # (nC_i, i)
        for row in yi:
            idx  = row - 1              # 0-based
            c2   = c[idx, :]            # (i, Q) – selected rows
            c4   = ((-1) ** i) * pdfcdfmvlogit(a, np.vstack([b, c2]))
            com  = com + c4
        com1 = com1 + com

    return float(np.atleast_1d(a2 + com1)[0])


# ---------------------------------------------------------------------------
# Non-standard version of pdfcdfmvlogitc
# ---------------------------------------------------------------------------

def nonpdfcdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> float:
    """
    Non-standard partial cumulative multivariate logistic combining upper
    CDF, lower CDF complement, and density.

    Computes Pr(Z = a, X < b, Y > c) where (Z | X | Y) is non-standard
    multivariate logistic with location mu and scale sig.

    Parameters
    ----------
    a   : np.ndarray, shape (M, 1)
    b   : np.ndarray, shape (K, 1)
    c   : np.ndarray, shape (M, 1)
    mu  : np.ndarray, shape (K + 2*M, 1)
          Location parameters for (Z | X | Y); Z and Y must share dimension M.
    sig : np.ndarray, shape (K + 2*M, 1)
          Scale parameters for (Z | X | Y).

    Returns
    -------
    float
        Pr(Z = a, X < b, Y > c).
    """
    a   = np.atleast_2d(a)
    b   = np.atleast_2d(b)
    c   = np.atleast_2d(c)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)

    a1 = a.shape[0]
    b1 = b.shape[0]

    a2   = nonpdfcdfmvlogit(a, b, mu[:a1+b1], sig[:a1+b1])
    com1 = 0.0

    for i in range(1, a1 + 1):
        com = 0.0
        yi  = combinate(a1, i)
        for row in yi:
            idx = row - 1                           # 0-based
            c2  = c[idx, :]
            mu_ext  = np.vstack([mu[:a1+b1],  mu[a1+b1+idx, :]])
            sig_ext = np.vstack([sig[:a1+b1], sig[a1+b1+idx, :]])
            c4  = ((-1) ** i) * nonpdfcdfmvlogit(
                a, np.vstack([b, c2]), mu_ext, sig_ext
            )
            com = com + c4
        com1 = com1 + com

    return float(np.atleast_1d(a2 + com1)[0])


# ---------------------------------------------------------------------------
# Gradient of non-standard pdfcdfmvlogitc
# ---------------------------------------------------------------------------

def gradnonpdfcdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the non-standard partial cumulative multivariate logistic
    distribution (combining upper CDF, lower CDF complement, and density).

    Parameters
    ----------
    a   : np.ndarray, shape (M, 1)
    b   : np.ndarray, shape (K, 1)
    c   : np.ndarray, shape (M, 1)
    mu  : np.ndarray, shape (K + 2*M, 1)
    sig : np.ndarray, shape (K + 2*M, 1)

    Returns
    -------
    ga   : np.ndarray, shape (M, 1)
    gb   : np.ndarray, shape (K, 1)
    gc   : np.ndarray, shape (M, 1)
    gmu  : np.ndarray, shape (K + 2*M, 1)
    gsig : np.ndarray, shape (K + 2*M, 1)
    """
    a   = np.atleast_2d(a)
    b   = np.atleast_2d(b)
    c   = np.atleast_2d(c)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)

    a1 = a.shape[0]
    b1 = b.shape[0]

    # Base-term gradients
    ga1, gb1, gmua1, gmub1, gsiga1, gsigb1 = gradnonpdfcdfmvlogit(
        a, b, mu[:a1+b1], sig[:a1+b1]
    )

    # Accumulation arrays (all (M,1) or (K,1))
    gcomba   = np.zeros((a1, 1))
    gcombbb  = np.zeros((b1, 1))
    gcombcc  = np.zeros((a1, 1))
    gcombmua = np.zeros((a1, 1))
    gcombmub = np.zeros((b1, 1))
    gcombmuc = np.zeros((a1, 1))
    gcombsiga= np.zeros((a1, 1))
    gcombsigb= np.zeros((b1, 1))
    gcombsigc= np.zeros((a1, 1))

    for i in range(1, a1 + 1):
        gcoma    = np.zeros((a1, 1))
        gcomb_   = np.zeros((b1, 1))
        gcomc    = np.zeros((a1, 1))
        gcommua  = np.zeros((a1, 1))
        gcommub  = np.zeros((b1, 1))
        gcommuc  = np.zeros((a1, 1))
        gcomsiga = np.zeros((a1, 1))
        gcomsigb = np.zeros((b1, 1))
        gcomsigc = np.zeros((a1, 1))

        yi = combinate(a1, i)
        for row in yi:
            idx  = row - 1                   # 0-based
            c2   = c[idx, :]                 # (i, 1)

            mu_ext  = np.vstack([mu[:a1+b1],  mu[a1+b1+idx, :]])
            sig_ext = np.vstack([sig[:a1+b1], sig[a1+b1+idx, :]])

            gca1, gcb1, gcmua1, gcmub1, gcsiga1, gcsigb1 = gradnonpdfcdfmvlogit(
                a, np.vstack([b, c2]), mu_ext, sig_ext
            )
            sign_i = (-1) ** i

            gcoma   += sign_i * gca1
            gcomb_  += sign_i * gcb1[:b1]

            gcomc1 = np.zeros((a1, 1))
            gcomc1[idx, :] = sign_i * gcb1[b1:]
            gcomc += gcomc1

            gcommua += sign_i * gcmua1

            gcommub_tmp = np.zeros((b1, 1))
            gcommub_tmp += sign_i * gcmub1[:b1]
            gcommub += gcommub_tmp

            gcommuc1 = np.zeros((a1, 1))
            gcommuc1[idx, :] = sign_i * gcmub1[b1:]
            gcommuc += gcommuc1

            gcomsiga += sign_i * gcsiga1

            gcomsigb_tmp = np.zeros((b1, 1))
            gcomsigb_tmp += sign_i * gcsigb1[:b1]
            gcomsigb += gcomsigb_tmp

            gcomsigc1 = np.zeros((a1, 1))
            gcomsigc1[idx, :] = sign_i * gcsigb1[b1:]
            gcomsigc += gcomsigc1

        gcomba   += gcoma
        gcombbb  += gcomb_
        gcombcc  += gcomc
        gcombmua += gcommua
        gcombmub += gcommub
        gcombmuc += gcommuc
        gcombsiga+= gcomsiga
        gcombsigb+= gcomsigb
        gcombsigc+= gcomsigc

    ga   = gcomba   + ga1
    gb   = gcombbb  + gb1
    gc   = gcombcc
    gmu  = np.vstack([(gcombmua + gmua1), (gcombmub + gmub1), gcombmuc])
    gsig = np.vstack([(gcombsiga + gsiga1), (gcombsigb + gsigb1), gcombsigc])

    return ga, gb, gc, gmu, gsig


# ===========================================================================
# Simulation routines
# ===========================================================================

def simmdcev(
    a: np.ndarray,
    m: int,
    n: int,
    sig: float,
    seed: int,
):
    """
    Draw error terms for the true MDCEV (Multiple Discrete-Continuous
    Extreme Value) model.

    Parameters
    ----------
    a    : np.ndarray, shape (K-1, 1)
           v-tilde_{k,1} values for each inside good (consumed goods first).
    m    : int
           Number of consumed inside goods (1 <= m <= K-1).
    n    : int
           Number of successful draws required.
    sig  : float
           Scale parameter for extreme value draws.
    seed : int
           Random seed (for reproducibility; sets numpy RNG state).

    Returns
    -------
    seed_out : int
        Updated seed (simulated as the original seed + draws used).
    rfinal   : np.ndarray, shape (n, K-1 + 1)
        Error term realisations.  Columns 0..K-2 are realisations for the
        discrete components of all inside goods (first m columns correspond
        to consumed goods); the last column is the outside-good realisation.

    Notes
    -----
    The acceptance-rejection sampler draws batches of 4*n candidates and
    keeps only rows that satisfy the MDCEV ordering constraint.
    """
    a = np.atleast_2d(a)
    if a.shape[1] != 1:
        a = a.T                         # ensure (K-1, 1)
    arows = a.shape[0]
    a_flat = a.ravel()                  # (K-1,)

    rng      = np.random.default_rng(seed)
    rfinal   = np.empty((0, arows + 1))
    draws_used = 0

    while rfinal.shape[0] < n:
        batch = 4 * n
        draws_used += batch
        err1 = rng.uniform(0.0, 1.0, size=(batch, arows + 1))  # (batch, K)
        r    = (-np.log(-np.log(err1))) * sig                   # (batch, K)
        r1   = r[:, :arows] - a_flat[np.newaxis, :]            # (batch, K-1)
        r1min = r1[:, :m].min(axis=1)                           # (batch,)

        if m < arows:
            r1max = r1[:, m:arows].max(axis=1)                  # (batch,)
            keep  = r1min > r1max                               # (batch,) bool
            if keep.sum() >= 1:
                r1min_k  = r1min[keep]
                r1max_k  = r1max[keep]
                rinside  = r[keep, :arows]
                rout1    = err1[keep, arows]
                routside = (
                    (1.0 - rout1) * np.exp(-np.exp(-r1max_k / sig))
                    + rout1       * np.exp(-np.exp(-r1min_k / sig))
                )
                routside = (-np.log(-np.log(routside))) * sig
                riter    = np.column_stack([rinside, routside])
                rfinal   = np.vstack([rfinal, riter])
        else:
            rinside  = r[:, :arows]
            rout1    = err1[:, arows]
            routside = rout1 * np.exp(-np.exp(-r1min / sig))
            routside = (-np.log(-np.log(routside))) * sig
            riter    = np.column_stack([rinside, routside])
            rfinal   = np.vstack([rfinal, riter])

    seed_out = seed + draws_used
    return seed_out, rfinal[:n, :]


def simtradmdcev(
    a: np.ndarray,
    m: int,
    n: int,
    sig: float,
    psi: np.ndarray,
    gamma: np.ndarray,
    price: np.ndarray,
    seed: int,
):
    """
    Draw error terms for the traditional MDCEV model.

    Parameters
    ----------
    a     : np.ndarray, shape (K-1, 1)
            v-tilde_{k,1} values for each inside good (consumed goods first).
    m     : int
            Number of consumed inside goods (1 <= m <= K-1).
    n     : int
            Number of successful draws required.
    sig   : float
            Scale parameter for extreme value draws.
    psi   : np.ndarray, shape (K, 1)
            Deterministic part of psi baseline utility.
    gamma : np.ndarray, shape (K-1, 1)
            gamma_k values for inside goods.
    price : np.ndarray, shape (K-1, 1)
            p_k values (prices) for inside goods.
    seed  : int
            Random seed.

    Returns
    -------
    seed_out : int
        Updated seed.
    rfinal   : np.ndarray, shape (n, K)
        Error term realisations.  Columns 0..K-2 correspond to the discrete
        components of all inside goods (first m columns = consumed goods);
        the last column is the outside-good realisation.

    Notes
    -----
    In the traditional MDCEV model there is no separate continuous-component
    error term; the output therefore has K columns rather than K-1+m.
    The acceptance-rejection mechanism is identical to simmdcev.
    """
    a = np.atleast_2d(a)
    if a.shape[1] != 1:
        a = a.T
    arows = a.shape[0]
    a_flat = a.ravel()

    rng      = np.random.default_rng(seed)
    rfinal   = np.empty((0, arows + 1))
    draws_used = 0

    while rfinal.shape[0] < n:
        batch = 4 * n
        draws_used += batch
        err1 = rng.uniform(0.0, 1.0, size=(batch, arows + 1))
        r    = (-np.log(-np.log(err1))) * sig
        r1   = r[:, :arows] - a_flat[np.newaxis, :]
        r1min = r1[:, :m].min(axis=1)

        if m < arows:
            r1max = r1[:, m:arows].max(axis=1)
            keep  = r1min > r1max
            if keep.sum() >= 1:
                r1min_k  = r1min[keep]
                r1max_k  = r1max[keep]
                rinside  = r[keep, :arows]
                rout1    = err1[keep, arows]
                routside = (
                    (1.0 - rout1) * np.exp(-np.exp(-r1max_k / sig))
                    + rout1       * np.exp(-np.exp(-r1min_k / sig))
                )
                routside = (-np.log(-np.log(routside))) * sig
                riter    = np.column_stack([rinside, routside])
                rfinal   = np.vstack([rfinal, riter])
        else:
            rinside  = r[:, :arows]
            rout1    = err1[:, arows]
            routside = rout1 * np.exp(-np.exp(-r1min / sig))
            routside = (-np.log(-np.log(routside))) * sig
            riter    = np.column_stack([rinside, routside])
            rfinal   = np.vstack([rfinal, riter])

    seed_out = seed + draws_used
    return seed_out, rfinal[:n, :]

































































# ---------------------------------------------------------------------------
# Standard multivariate logistic CDF
# ---------------------------------------------------------------------------

def cdfmvlogit(a: np.ndarray) -> np.ndarray:
    """
    Standard multivariate logistic cumulative distribution function.

    Parameters
    ----------
    a : np.ndarray, shape (K, Q)
        Abscissae matrix; K = number of variates, Q = number of observations.

    Returns
    -------
    np.ndarray, shape (Q,)
        Pr(X < a) for each of the Q observations.
    """
    a = np.atleast_2d(a)
    f = np.exp(-a)                        # (K, Q)
    return (1.0 + f.sum(axis=0)) ** (-1)  # (Q,)


# ---------------------------------------------------------------------------
# Gradient of standard multivariate logistic CDF
# ---------------------------------------------------------------------------

def gradcdfmvlogit(a: np.ndarray) -> np.ndarray:
    """
    Gradient of the standard multivariate logistic CDF w.r.t. a.

    Parameters
    ----------
    a : np.ndarray, shape (K, Q)
        Abscissae matrix; K = number of variates, Q = number of observations.

    Returns
    -------
    np.ndarray, shape (K, Q)
        Gradient matrix: element (k, q) = d Pr(X < a[:,q]) / d a[k,q].
    """
    a = np.atleast_2d(a)
    f = np.exp(-a)                        # (K, Q)
    h = (1.0 + f.sum(axis=0)) ** (-1)    # (Q,)
    return h[np.newaxis, :] * (f * h[np.newaxis, :])  # (K, Q)


# ---------------------------------------------------------------------------
# Non-standard multivariate logistic CDF
# ---------------------------------------------------------------------------

def noncdfmvlogit(
    a: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> np.ndarray:
    """
    Non-standard multivariate logistic CDF.

    Parameters
    ----------
    a   : np.ndarray, shape (K, Q)
          Abscissae matrix; K = number of variates, Q = number of observations.
    mu  : np.ndarray, shape (K, Q) or (K, 1)
          Location parameters.
    sig : np.ndarray, shape (K, Q) or (K, 1)
          Scale parameters.

    Returns
    -------
    np.ndarray, shape (Q,)
        Pr(X < a) evaluated at each of the Q observations.
    """
    a   = np.atleast_2d(a)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    aa  = (a - mu) / sig
    f   = np.exp(-aa)
    return (1.0 + f.sum(axis=0)) ** (-1)  # (Q,)


# ---------------------------------------------------------------------------
# Gradient of non-standard multivariate logistic CDF
# ---------------------------------------------------------------------------

def gradnoncdfmvlogit(
    a: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the non-standard multivariate logistic CDF.

    Parameters
    ----------
    a   : np.ndarray, shape (K, Q)
          Abscissae matrix; K = number of variates, Q = number of observations.
    mu  : np.ndarray, shape (K, Q) or (K, 1)
          Location parameters.
    sig : np.ndarray, shape (K, Q) or (K, 1)
          Scale parameters.

    Returns
    -------
    ga   : np.ndarray, shape (K, Q)
           Gradient w.r.t. a.
    gmu  : np.ndarray, shape (K, Q) or (K, 1)
           Gradient w.r.t. mu.
    gsig : np.ndarray, shape (K, Q) or (K, 1)
           Gradient w.r.t. sig.
    """
    a   = np.atleast_2d(a)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)

    single_mu  = (mu.shape[1]  == 1)
    single_sig = (sig.shape[1] == 1)

    aa   = (a - mu) / sig
    aa1  = gradcdfmvlogit(aa)              # (K, Q)
    aa2  = aa1 / sig                       # (K, Q)  – ga
    aamu = aa1 * (-1.0 / sig)             # (K, Q)
    aasig = (aa1 * (-aa)) / sig           # (K, Q)

    if single_mu:
        aamu  = aamu.sum(axis=1, keepdims=True)   # (K, 1)
    if single_sig:
        aasig = aasig.sum(axis=1, keepdims=True)  # (K, 1)

    return aa2, aamu, aasig


# ===========================================================================
# Complement of the standard multivariate logistic CDF
# ===========================================================================

def cdfmvlogitcomp(b: np.ndarray) -> float:
    """
    Complement of the standard multivariate logistic CDF.

    Computes Pr(Y > b) where Y is multivariate logistic distributed, using
    the inclusion-exclusion identity.

    Parameters
    ----------
    b : np.ndarray, shape (K, 1)
        Truncation points from below; b is a K x 1 column vector.

    Returns
    -------
    float
        Pr(Y > b).

    Notes
    -----
    Digitisation issues can occasionally push (1 + com1) slightly below zero;
    in that case 0.0 is returned.
    """
    b  = np.atleast_2d(b)
    b1 = b.shape[0]
    com1 = 0.0
    for i in range(1, b1 + 1):
        yi = combinate(b1, i)                          # (M, i)
        for row in yi:
            idx = row - 1                              # 0-based
            b2  = b[idx, :]                            # (i, 1)
            com1 += ((-1) ** i) * cdfmvlogit(b2).item()
    result = 1.0 + com1
    return 0.0 if result < 0.0 else result


# ---------------------------------------------------------------------------
# Gradient of the complement of the standard multivariate logistic CDF
# ---------------------------------------------------------------------------

def gradcdfmvlogitcomp(b: np.ndarray) -> np.ndarray:
    """
    Gradient of the complement of the standard multivariate logistic CDF.

    Parameters
    ----------
    b : np.ndarray, shape (K, 1)
        Truncation points from below (K x 1 column vector).

    Returns
    -------
    np.ndarray, shape (K, 1)
        Gradient vector gb: element k = d Pr(Y > b) / d b[k].
    """
    b  = np.atleast_2d(b)
    b1 = b.shape[0]
    gcombnew = np.zeros((b1, 1))
    for i in range(1, b1 + 1):
        gcomb = np.zeros((b1, 1))
        yi = combinate(b1, i)                          # (M, i)
        for row in yi:
            idx    = row - 1                           # 0-based
            b2     = b[idx, :]                         # (i, 1)
            gcom   = ((-1) ** i) * gradcdfmvlogit(b2)  # (i, 1)
            gcomb1 = np.zeros((b1, 1))
            gcomb1[idx, :] = gcom
            gcomb += gcomb1
        gcombnew += gcomb
    return gcombnew


# ---------------------------------------------------------------------------
# Non-standard complement CDF
# ---------------------------------------------------------------------------

def noncdfmvlogitcomp(
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> float:
    """
    Complement of the non-standard multivariate logistic CDF.

    Computes Pr(Y > b) where Y ~ mvlogit(mu, sig).

    Parameters
    ----------
    b   : np.ndarray, shape (K, 1)
          Truncation points from below.
    mu  : np.ndarray, shape (K, 1)
          Location parameters.
    sig : np.ndarray, shape (K, 1)
          Scale parameters.

    Returns
    -------
    float
        Pr(Y > b).
    """
    b   = np.atleast_2d(b)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    bb  = (b - mu) / sig
    return cdfmvlogitcomp(bb)


# ---------------------------------------------------------------------------
# Gradient of the non-standard complement CDF
# ---------------------------------------------------------------------------

def gradnoncdfmvlogitcomp(
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the complement of the non-standard multivariate logistic CDF.

    Parameters
    ----------
    b   : np.ndarray, shape (K, 1)
          Truncation points from below.
    mu  : np.ndarray, shape (K, 1)
          Location parameters.
    sig : np.ndarray, shape (K, 1)
          Scale parameters.

    Returns
    -------
    gb   : np.ndarray, shape (K, 1)  – gradient w.r.t. b.
    gmu  : np.ndarray, shape (K, 1)  – gradient w.r.t. mu.
    gsig : np.ndarray, shape (K, 1)  – gradient w.r.t. sig.
    """
    b   = np.atleast_2d(b)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    bb   = (b - mu) / sig
    gbb  = gradcdfmvlogitcomp(bb)     # (K, 1)
    gbb1 = gbb / sig                  # (K, 1)  – gb
    gsig = (-gbb * bb) / sig          # (K, 1)
    return gbb1, -gbb1, gsig


# ===========================================================================
# Combined CDF: Pr(X < a, Y > b)
# ===========================================================================

def cdfmvlogitc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Combined standard multivariate logistic CDF and its complement.

    Computes Pr(X < a, Y > b) using inclusion-exclusion.

    Parameters
    ----------
    a : np.ndarray, shape (K, 1)
        Upper truncation points (−∞ to a).
    b : np.ndarray, shape (M, 1)
        Lower truncation points (b to +∞).

    Returns
    -------
    float
        Pr(X < a, Y > b).

    Notes
    -----
    To obtain Pr(Y > b) alone, set a = 1000 * ones(K, 1).
    Digitisation issues can occasionally push (a2 + com1) slightly below zero;
    in that case 0.0 is returned.
    """
    a  = np.atleast_2d(a)
    b  = np.atleast_2d(b)
    b1 = b.shape[0]
    a2   = cdfmvlogit(a).item()
    com1 = 0.0
    for i in range(1, b1 + 1):
        yi = combinate(b1, i)                              # (M, i)
        for row in yi:
            idx = row - 1                                  # 0-based
            b2  = b[idx, :]                                # (i, 1)
            com1 += ((-1) ** i) * cdfmvlogit(np.vstack([a, b2])).item()
    result = a2 + com1
    return 0.0 if result < 0.0 else result


# ---------------------------------------------------------------------------
# Gradient of the combined CDF
# ---------------------------------------------------------------------------

def gradcdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
):
    """
    Gradient of the combined standard multivariate logistic CDF.

    Parameters
    ----------
    a : np.ndarray, shape (K, 1)
        Upper truncation points.
    b : np.ndarray, shape (M, 1)
        Lower truncation points.

    Returns
    -------
    ga : np.ndarray, shape (K, 1)  – gradient w.r.t. a.
    gb : np.ndarray, shape (M, 1)  – gradient w.r.t. b.

    Notes
    -----
    To obtain gradients for Pr(Y > b) alone, set a = 1000 * ones(K, 1).
    """
    a  = np.atleast_2d(a)
    b  = np.atleast_2d(b)
    a1 = a.shape[0]
    b1 = b.shape[0]
    ga1      = gradcdfmvlogit(a)           # (K, 1)
    gcombnew = np.zeros((b1, 1))
    gcomanew = np.zeros((a1, 1))
    for i in range(1, b1 + 1):
        gcoma  = np.zeros((a1, 1))
        gcomb  = np.zeros((b1, 1))
        yi = combinate(b1, i)              # (M, i)
        for row in yi:
            idx    = row - 1              # 0-based
            b2     = b[idx, :]            # (i, 1)
            gcom   = ((-1) ** i) * gradcdfmvlogit(np.vstack([a, b2]))  # (K+i, 1)
            gcoma += gcom[:a1, :]
            gcomb1 = np.zeros((b1, 1))
            gcomb1[idx, :] = gcom[a1:, :]
            gcomb += gcomb1
        gcomanew += gcoma
        gcombnew += gcomb
    return gcomanew + ga1, gcombnew


# ---------------------------------------------------------------------------
# Non-standard combined CDF
# ---------------------------------------------------------------------------

def noncdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
) -> float:
    """
    Combined non-standard multivariate logistic CDF and its complement.

    Computes Pr(X < a, Y > b) where (X, Y) ~ mvlogit(mu, sig).

    Parameters
    ----------
    a   : np.ndarray, shape (K, 1)
          Upper truncation points.
    b   : np.ndarray, shape (M, 1)
          Lower truncation points.
    mu  : np.ndarray, shape (K+M, 1)
          Location parameters for (X | Y).
    sig : np.ndarray, shape (K+M, 1)
          Scale parameters for (X | Y).

    Returns
    -------
    float
        Pr(X < a, Y > b).

    Notes
    -----
    To obtain Pr(Y > b) alone, set a = 1000 * ones(K, 1).
    """
    a   = np.atleast_2d(a)
    b   = np.atleast_2d(b)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    m   = a.shape[0]
    h   = b.shape[0]
    aa  = (a - mu[:m])  / sig[:m]
    bb  = (b - mu[m:m + h]) / sig[m:m + h]
    return cdfmvlogitc(aa, bb)


# ---------------------------------------------------------------------------
# Gradient of the non-standard combined CDF
# ---------------------------------------------------------------------------

def gradnoncdfmvlogitc(
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    sig: np.ndarray,
):
    """
    Gradient of the combined non-standard multivariate logistic CDF.

    Parameters
    ----------
    a   : np.ndarray, shape (K, 1)
          Upper truncation points (−∞ to a).
    b   : np.ndarray, shape (M, 1)
          Lower truncation points (b to +∞).
    mu  : np.ndarray, shape (K+M, 1)
          Location parameters for (X | Y).
    sig : np.ndarray, shape (K+M, 1)
          Scale parameters for (X | Y).

    Returns
    -------
    ga   : np.ndarray, shape (K, 1)    – gradient w.r.t. a.
    gb   : np.ndarray, shape (M, 1)    – gradient w.r.t. b.
    gmu  : np.ndarray, shape (K+M, 1)  – gradient w.r.t. mu.
    gsig : np.ndarray, shape (K+M, 1)  – gradient w.r.t. sig.
    """
    a   = np.atleast_2d(a)
    b   = np.atleast_2d(b)
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    m    = a.shape[0]
    h    = b.shape[0]
    mua  = mu[:m];    mub  = mu[m:m + h]
    siga = sig[:m];   sigb = sig[m:m + h]
    aa   = (a - mua) / siga
    bb   = (b - mub) / sigb
    gaa, gbb = gradcdfmvlogitc(aa, bb)    # (K, 1), (M, 1)
    gaa1  =  gaa / siga                   # (K, 1)  – ga
    gbb1  =  gbb / sigb                   # (M, 1)  – gb
    gsiga = (-gaa * aa) / siga            # (K, 1)
    gsigb = (-gbb * bb) / sigb            # (M, 1)
    gmu   = np.vstack([-gaa1, -gbb1])     # (K+M, 1)
    gsig  = np.vstack([gsiga, gsigb])     # (K+M, 1)
    return gaa1, gbb1, gmu, gsig


# ===========================================================================
# Rectangular integration helpers
# ===========================================================================

def _rectcombs(
    x1: np.ndarray,
    x2: np.ndarray,
):
    """
    Build the sign vector and abscissae matrix for rectangular integration
    via inclusion-exclusion.

    Each row of ``comb`` holds one vertex of the hyper-rectangle [x1, x2].
    The matching entry in ``sign`` is +1 or −1 according to how many lower
    bounds are active (inclusion-exclusion parity).

    Parameters
    ----------
    x1 : np.ndarray, shape (K, 1)   – lower truncation points.
    x2 : np.ndarray, shape (K, 1)   – upper truncation points.

    Returns
    -------
    sign : np.ndarray, shape (2**K,)       – ±1 for each vertex.
    comb : np.ndarray, shape (2**K, K)     – abscissae at each vertex.
    """
    x1 = np.atleast_2d(x1).ravel()
    x2 = np.atleast_2d(x2).ravel()
    K  = x1.shape[0]
    n_vertices = 2 ** K
    sign_out = np.empty(n_vertices, dtype=float)
    comb_out = np.empty((n_vertices, K), dtype=float)
    for v in range(n_vertices):
        bits       = [(v >> k) & 1 for k in range(K)]   # 0 = x2, 1 = x1
        n_lower    = sum(bits)
        sign_out[v] = (-1) ** n_lower
        comb_out[v] = np.where(bits, x1, x2)
    return sign_out, comb_out


def _rectcombsforgrad(
    x1: np.ndarray,
    x2: np.ndarray,
):
    """
    Like _rectcombs but also returns index masks that map each vertex back to
    x1 and x2 for gradient accumulation.

    Returns
    -------
    sign  : np.ndarray, shape (2**K,)
    comb  : np.ndarray, shape (2**K, K)
    indxa : np.ndarray, shape (2**K, K)  – 1 where vertex coordinate = x1.
    indxb : np.ndarray, shape (2**K, K)  – 1 where vertex coordinate = x2.
    """
    x1 = np.atleast_2d(x1).ravel()
    x2 = np.atleast_2d(x2).ravel()
    K  = x1.shape[0]
    n_vertices = 2 ** K
    sign_out  = np.empty(n_vertices, dtype=float)
    comb_out  = np.empty((n_vertices, K), dtype=float)
    indxa_out = np.zeros((n_vertices, K), dtype=float)
    indxb_out = np.zeros((n_vertices, K), dtype=float)
    for v in range(n_vertices):
        bits = [(v >> k) & 1 for k in range(K)]
        n_lower = sum(bits)
        sign_out[v] = (-1) ** n_lower
        for k in range(K):
            if bits[k]:
                comb_out[v, k]  = x1[k]
                indxa_out[v, k] = 1.0
            else:
                comb_out[v, k]  = x2[k]
                indxb_out[v, k] = 1.0
    return sign_out, comb_out, indxa_out, indxb_out


# ===========================================================================
# Rectangular integration of the non-standard multivariate logistic CDF
# ===========================================================================

def cdrectmvlogit(
    mu: np.ndarray,
    sig: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
) -> float:
    """
    Rectangular integration of the non-standard multivariate logistic CDF.

    Computes Pr(x1 < X < x2) using inclusion-exclusion over all 2**K vertices
    of the hyper-rectangle.

    Parameters
    ----------
    mu  : np.ndarray, shape (K, 1)  – location parameters.
    sig : np.ndarray, shape (K, 1)  – scale parameters.
    x1  : np.ndarray, shape (K, 1)  – lower truncation points.
    x2  : np.ndarray, shape (K, 1)  – upper truncation points.

    Returns
    -------
    float
        Pr(x1 < X < x2).

    Notes
    -----
    To obtain Pr(X > x1), set x2 = 1000 * ones(K, 1).
    To obtain Pr(X < x2), set x1 = -1000 * ones(K, 1).
    """
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    x1  = np.atleast_2d(x1)
    x2  = np.atleast_2d(x2)
    sign, comb = _rectcombs(x1, x2)
    P = 0.0
    for j in range(len(sign)):
        vertex = comb[j, :].reshape(-1, 1)   # (K, 1)
        P += sign[j] * noncdfmvlogit(vertex, mu, sig).item()
    return 0.0 if P < 0.0 else P


# ---------------------------------------------------------------------------
# Gradient of the rectangular integration
# ---------------------------------------------------------------------------

def gradcdrectmvlogit(
    mu: np.ndarray,
    sig: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
):
    """
    Gradient of the rectangular multivariate logistic probability.

    Parameters
    ----------
    mu  : np.ndarray, shape (K, 1)  – location parameters.
    sig : np.ndarray, shape (K, 1)  – scale parameters.
    x1  : np.ndarray, shape (K, 1)  – lower truncation points.
    x2  : np.ndarray, shape (K, 1)  – upper truncation points.

    Returns
    -------
    P    : float                      – rectangular probability value.
    gmu  : np.ndarray, shape (K, 1)  – gradient w.r.t. mu.
    gsig : np.ndarray, shape (K, 1)  – gradient w.r.t. sig.
    gx1  : np.ndarray, shape (K, 1)  – gradient w.r.t. x1.
    gx2  : np.ndarray, shape (K, 1)  – gradient w.r.t. x2.
    """
    mu  = np.atleast_2d(mu)
    sig = np.atleast_2d(sig)
    x1  = np.atleast_2d(x1)
    x2  = np.atleast_2d(x2)
    K   = x1.shape[0]
    sign, comb, indxa, indxb = _rectcombsforgrad(x1, x2)
    gmu1  = np.zeros((K, 1))
    gsig1 = np.zeros((K, 1))
    gx1a  = np.zeros((K, 1))
    gx2b  = np.zeros((K, 1))
    P = 0.0
    for j in range(len(sign)):
        vertex = comb[j, :].reshape(-1, 1)         # (K, 1)
        P     += sign[j] * noncdfmvlogit(vertex, mu, sig).item()
        gx, gmu, gsig = gradnoncdfmvlogit(vertex, mu, sig)
        gmu1  += sign[j] * gmu
        gsig1 += sign[j] * gsig
        gxx    = sign[j] * gx                      # (K, 1)
        gx1a  += gxx * indxa[j, :].reshape(-1, 1)
        gx2b  += gxx * indxb[j, :].reshape(-1, 1)
    return P, gmu1, gsig1, gx1a, gx2b