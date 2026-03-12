"""Compound gradient helpers for conditional covariance and truncated moments.

Translates the following GAUSS procedures from BHATLIB's Gradmvn.src:
- gcondcov: gradient of scaled Schur complement Y·(X22-X12·X11⁻¹·X12')·Y'
- gcondmeantrunc: gradient of conditional mean through truncation (TODO)
- gcondcovtrunc: gradient of conditional covariance through truncation (TODO)

All functions follow BHATLIB's row-based arrangement convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.gradmvn._bivariate_trunc import (
    truncated_bivariate_cov,
    truncated_bivariate_mean,
)
from pybhatlib.gradmvn._trunc_grads import (
    grad_bivariate_normal_trunc,
    grad_univariate_normal_trunc,
)
from pybhatlib.matgradient._mat_grad_helpers import (
    gaomegab,
    gbothxomegax,
    ginverse_sym,
)


# ---------------------------------------------------------------------------
# Helper: vech index and block-to-vech permutation
# ---------------------------------------------------------------------------

def _vech_index(i: int, j: int, K: int) -> int:
    """Position of element (i, j) with i <= j in row-major vech of K×K matrix.

    vech ordering (row-by-row upper triangle):
        (0,0), (0,1), ..., (0,K-1), (1,1), (1,2), ..., (K-1,K-1)
    """
    return i * K - i * (i - 1) // 2 + (j - i)


def _block_to_vech_perm(dimdiff: int, dim1: int) -> NDArray:
    """Permutation from stacked block derivatives to vech(X) ordering.

    When X is partitioned as::

        X = [[X11, X12'],
             [X12, X22 ]]

    with X11 (dimdiff×dimdiff), X12 (dim1×dimdiff), X22 (dim1×dim1),
    the gradient rows are naturally stacked as:

        [vech(X11); vec_colmajor(X12); vech(X22)]

    where vec_colmajor(X12) means X12 elements ordered column-by-column
    (matching GAUSS's column-major reshape convention).

    This function returns a permutation array ``perm`` such that
    ``stacked[perm, :]`` reorders the rows into vech(X) ordering.

    Parameters
    ----------
    dimdiff : int
        Size of the top-left block X11 (number of conditioning variables).
    dim1 : int
        Size of the bottom-right block X22 (remaining variables).

    Returns
    -------
    perm : ndarray of int, shape (dim2*(dim2+1)/2,)
        Permutation indices into the stacked block array.
    """
    dim2 = dimdiff + dim1
    ddcov = dimdiff * (dimdiff + 1) // 2   # vech size of X11
    n_vech = dim2 * (dim2 + 1) // 2

    perm = np.empty(n_vech, dtype=np.intp)
    idx = 0
    for i in range(dim2):
        for j in range(i, dim2):
            if i < dimdiff and j < dimdiff:
                # X11 block
                perm[idx] = _vech_index(i, j, dimdiff)
            elif i < dimdiff and j >= dimdiff:
                # Cross-block: X[i, j] = X12[j - dimdiff, i]  (symmetric)
                # In stacked array, X12 is stored column-by-column:
                #   col 0: X12[0,0], X12[1,0], ..., X12[dim1-1,0]
                #   col 1: X12[0,1], X12[1,1], ..., X12[dim1-1,1]
                # Index: ddcov + (col * dim1 + row)
                row = j - dimdiff
                col = i
                perm[idx] = ddcov + col * dim1 + row
            else:
                # X22 block
                perm[idx] = ddcov + dim1 * dimdiff + _vech_index(
                    i - dimdiff, j - dimdiff, dim1
                )
            idx += 1

    return perm


# ---------------------------------------------------------------------------
# gcondcov: gradient of scaled Schur complement
# ---------------------------------------------------------------------------

def gcondcov(Y: NDArray, X: NDArray) -> tuple[NDArray, NDArray]:
    r"""Gradient of A = Y · Σ_cond · Y' w.r.t. Y and vech(X).

    Here Σ_cond is the Schur complement from partitioning X:

    .. math::

        \Sigma_{\text{cond}} = X_{22} - X_{12} X_{11}^{-1} X_{12}^T

    with X partitioned as::

        X = [[X11, X12'],    (dimdiff × dimdiff,  dimdiff × dim1)
             [X12, X22 ]]    (dim1 × dimdiff,     dim1 × dim1)

    and A = Y · Σ_cond · Y' where Y is diagonal (dim1 × dim1).

    When dim1 == dim2 (dimdiff == 0), reduces to A = Y·X·Y'.

    Parameters
    ----------
    Y : ndarray, shape (dim1, dim1)
        Diagonal scaling matrix (only diagonal elements are free).
    X : ndarray, shape (dim2, dim2)
        Full symmetric matrix to be partitioned.

    Returns
    -------
    ggY : ndarray, shape (dim1, dd1cov)
        Gradient d vech(A) / d diag(Y).
    ggX : ndarray, shape (dim2*(dim2+1)/2, dd1cov)
        Gradient d vech(A) / d vech(X).

    Notes
    -----
    GAUSS ref: gcondcov (line 2323), with ``_condcov=1``.
    dd1cov = dim1*(dim1+1)/2 is the vech size of A.
    """
    dim1 = Y.shape[0]
    dim2 = X.shape[0]
    dimdiff = dim2 - dim1

    if dimdiff == 0:
        # No conditioning: A = Y·X·Y'
        ggY, ggX = gbothxomegax(
            Y, X, x1_symmetric=True, x1_diagonal=True,
        )
        return ggY, ggX

    # --- Partition X ---
    X11 = X[:dimdiff, :dimdiff]
    X12 = X[dimdiff:, :dimdiff]   # dim1 × dimdiff
    X22 = X[dimdiff:, dimdiff:]

    invX11 = np.linalg.inv(X11)
    X22condcov = X22 - X12 @ invX11 @ X12.T

    # --- Outer layer: A = Y · X22condcov · Y' ---
    # Y diagonal, X22condcov symmetric
    ggY, gg2 = gbothxomegax(
        Y, X22condcov, x1_symmetric=True, x1_diagonal=True,
    )
    # ggY: (dim1, dd1cov)  — gradient w.r.t. diag(Y)
    # gg2: (dd1cov, dd1cov) — gradient w.r.t. vech(X22condcov)

    # --- Inner layer: X22condcov = X22 - X12·invX11·X12' ---
    # Gradient of X12·invX11·X12' w.r.t. X12 and invX11
    gg22, gg23 = gbothxomegax(X12, invX11)
    # gg22: (dim1*dimdiff, dd1cov) — d vech(X12·invX11·X12')/d vec(X12)
    # gg23: (ddcov, dd1cov) — d vech(X12·invX11·X12')/d vech(invX11)

    # Chain rule through negative sign and gg2
    gg22 = -gg22 @ gg2   # (dim1*dimdiff, dd1cov)

    # Chain rule for invX11: d/d vech(X11) = ginverse_sym(X11) @ d/d vech(invX11)
    gg23 = -ginverse_sym(X11) @ gg23 @ gg2   # (ddcov, dd1cov)

    # --- Reassemble into vech(X) ordering ---
    # Stack: [vech(X11) derivatives; vec(X12) derivatives; vech(X22) derivatives]
    # But gg22 rows are in row-major vec(X12) order, while the permutation
    # expects column-major order (matching GAUSS convention).
    # Reorder gg22 from row-major to column-major:
    #   row-major index (r*dimdiff + c) → column-major index (c*dim1 + r)
    gg22_colmaj = np.empty_like(gg22)
    for r in range(dim1):
        for c in range(dimdiff):
            gg22_colmaj[c * dim1 + r, :] = gg22[r * dimdiff + c, :]

    stacked = np.vstack([gg23, gg22_colmaj, gg2])

    perm = _block_to_vech_perm(dimdiff, dim1)
    ggX = stacked[perm, :]

    return ggY, ggX


# ---------------------------------------------------------------------------
# gcondmeantrunc: gradient of conditional mean through truncation
# ---------------------------------------------------------------------------

def gcondmeantrunc(
    Y: NDArray, mu: NDArray, X: NDArray, C: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""Gradient of conditional mean through truncation.

    Computes gradients of:

    .. math::

        \text{condmean} = \mu_2 + Y \, X_{12} \, X_{11}^{-1}
                          \bigl(E[Z_1 | Z_1 \le C] - \mu_1\bigr)

    where :math:`Z_1 \sim N(\mu_1, X_{11})`, :math:`\mu_1 = \mu_{0:\text{dimdiff}}`,
    :math:`\mu_2 = \mu_{\text{dimdiff}:}`, and Y is diagonal (dim1 × dim1).

    Only supports dimdiff ∈ {1, 2} (ME and OVUS conditioning).

    Parameters
    ----------
    Y : ndarray, shape (dim1, dim1)
        Diagonal scaling matrix.
    mu : ndarray, shape (dim2,)
        Mean vector of the full distribution.
    X : ndarray, shape (dim2, dim2)
        Full symmetric covariance matrix.
    C : ndarray, shape (dimdiff,)
        Upper truncation points for the first dimdiff variables.

    Returns
    -------
    ggY : ndarray, shape (dim1, dim1)
        Gradient d condmean / d diag(Y), diagonal matrix.
    gmu : ndarray, shape (dim2, dim1)
        Gradient d condmean / d mu.
    ggX : ndarray, shape (dim2*(dim2+1)/2, dim1)
        Gradient d condmean / d vech(X).
    gC : ndarray, shape (dimdiff, dim1)
        Gradient d condmean / d C.

    Notes
    -----
    GAUSS ref: gcondmeantrunc (line 2224), with ``_condcovmeantrunc=1``.
    """
    dim1 = Y.shape[0]
    dim2 = X.shape[0]
    dimdiff = dim2 - dim1
    ddcov = dimdiff * (dimdiff + 1) // 2
    dd1cov = dim1 * (dim1 + 1) // 2

    # --- Partition X ---
    X11 = X[:dimdiff, :dimdiff]
    X12 = X[dimdiff:, :dimdiff]   # dim1 × dimdiff
    invX11 = np.linalg.inv(X11)

    # gmutilde: chain rule multiplier from mu through the Schur conditional mean
    # condmean depends on mu[:dimdiff] through -Y @ X12 @ invX11 @ mu[:dimdiff]
    # and on mu[dimdiff:] directly as identity.
    gmutilde = (Y @ X12 @ invX11).T   # (dimdiff, dim1)
    gmu = np.vstack([-gmutilde, np.eye(dim1)])  # (dim2, dim1)

    # --- Forward truncated moments and their gradients ---
    if dimdiff == 1:
        sig2 = float(X11[0, 0])
        sig1 = np.sqrt(sig2)
        w = (C[0] - mu[0]) / sig1
        phi_w = norm.pdf(w)
        Phi_w = max(norm.cdf(w), 1e-300)
        lam = -phi_w / Phi_w
        mutrunc = np.array([mu[0] + sig1 * lam])

        dmutrunc, _ = grad_univariate_normal_trunc(float(mu[0]), sig2, float(C[0]))
        gmutrunc = dmutrunc.reshape(-1, 1)   # (3, 1)
    elif dimdiff == 2:
        mutrunc = truncated_bivariate_mean(mu[:dimdiff], X11, C)
        gmutrunc, _ = grad_bivariate_normal_trunc(mu[:dimdiff], X11, C)
        # gmutrunc shape: (7, 2)
    else:
        raise ValueError(f"dimdiff must be 1 or 2, got {dimdiff}")

    # --- ggY: gradient w.r.t. diag(Y) ---
    gY1 = X12 @ invX11 @ (mutrunc - mu[:dimdiff])   # (dim1,)
    ggY = np.diag(gY1)   # (dim1, dim1)

    # --- Chain rule through truncated moments ---
    gmutildenew = gmutrunc @ gmutilde   # (3, dim1) or (7, dim1)
    gmu[:dimdiff, :] += gmutildenew[:dimdiff, :]

    # --- gX12: gradient w.r.t. vec(X12) ---
    v = invX11 @ (mutrunc - mu[:dimdiff])   # (dimdiff,)
    gX12 = np.kron(Y, v.reshape(-1, 1))     # (dim1*dimdiff, dim1)

    # Reorder from row-major vec(X12) to column-major vec(X12)
    gX12_colmaj = np.empty_like(gX12)
    for r in range(dim1):
        for c in range(dimdiff):
            gX12_colmaj[c * dim1 + r, :] = gX12[r * dimdiff + c, :]

    # --- gX11: gradient w.r.t. vech(X11) ---
    gX11 = ginverse_sym(X11) @ gaomegab(
        (Y @ X12).T, mutrunc - mu[:dimdiff],
    )
    gX11 += gmutildenew[dimdiff:dimdiff + ddcov, :]

    # --- Assemble into vech(X) ordering ---
    # X22 does not affect condmean, so its gradient block is zeros
    stacked = np.vstack([gX11, gX12_colmaj, np.zeros((dd1cov, dim1))])
    perm = _block_to_vech_perm(dimdiff, dim1)
    ggX = stacked[perm, :]

    # --- Truncation point gradient ---
    gC = gmutildenew[dimdiff + ddcov:, :]

    return ggY, gmu, ggX, gC


# ---------------------------------------------------------------------------
# gcondcovtrunc: gradient of conditional covariance through truncation
# ---------------------------------------------------------------------------

def gcondcovtrunc(
    Y: NDArray, mu: NDArray, X: NDArray, C: NDArray,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""Gradient of conditional covariance through truncation.

    Computes gradients of:

    .. math::

        \text{condcov} = Y \cdot \text{Var}(Z_2 | Z_1 \le C) \cdot Y'

    where by the law of total variance:

    .. math::

        \text{Var}(Z_2 | Z_1 \le C) = \Sigma_{\text{cond}}
            + X_{12} X_{11}^{-1} \tilde\Sigma X_{11}^{-1} X_{12}'

    with :math:`\Sigma_{\text{cond}} = X_{22} - X_{12} X_{11}^{-1} X_{12}'`
    (Schur complement) and :math:`\tilde\Sigma = \text{Cov}[Z_1 | Z_1 \le C]`
    (truncated covariance of :math:`Z_1 \sim N(\mu_1, X_{11})`).

    Only supports dimdiff in {1, 2} (ME and OVUS conditioning).

    Parameters
    ----------
    Y : ndarray, shape (dim1, dim1)
        Diagonal scaling matrix.
    mu : ndarray, shape (dim2,)
        Mean vector of the full distribution.
    X : ndarray, shape (dim2, dim2)
        Full symmetric covariance matrix.
    C : ndarray, shape (dimdiff,)
        Upper truncation points for the first dimdiff variables.

    Returns
    -------
    ggY : ndarray, shape (dim1, dd1cov)
        Gradient d vech(condcov) / d diag(Y).
    gmu : ndarray, shape (dim2, dd1cov)
        Gradient d vech(condcov) / d mu.
    ggX : ndarray, shape (dim2*(dim2+1)/2, dd1cov)
        Gradient d vech(condcov) / d vech(X).
    gC : ndarray, shape (dimdiff, dd1cov)
        Gradient d vech(condcov) / d C.

    Notes
    -----
    GAUSS ref: gcondcovtrunc (line 2389), with ``_condcovmeantrunc=2``.
    dd1cov = dim1*(dim1+1)/2 is the vech size of condcov.
    """
    dim1 = Y.shape[0]
    dim2 = X.shape[0]
    dimdiff = dim2 - dim1
    ddcov = dimdiff * (dimdiff + 1) // 2
    dd1cov = dim1 * (dim1 + 1) // 2

    # --- Partition X ---
    X11 = X[:dimdiff, :dimdiff]
    X12 = X[dimdiff:, :dimdiff]   # dim1 × dimdiff
    X22 = X[dimdiff:, dimdiff:]
    invX11 = np.linalg.inv(X11)

    # --- Compute truncated covariance Sigma_tilde ---
    if dimdiff == 1:
        sig2 = float(X11[0, 0])
        sig1 = np.sqrt(sig2)
        w = (C[0] - mu[0]) / sig1
        phi_w = norm.pdf(w)
        Phi_w = max(norm.cdf(w), 1e-300)
        lam = -phi_w / Phi_w
        sig_trunc_val = sig2 * (1.0 + lam * (w - lam))
        sig_trunc = np.array([[max(sig_trunc_val, 1e-15)]])
    elif dimdiff == 2:
        sig_trunc = truncated_bivariate_cov(mu[:dimdiff], X11, C)
    else:
        raise ValueError(f"dimdiff must be 1 or 2, got {dimdiff}")

    # B = invX11 · Sigma_tilde · invX11
    B = invX11 @ sig_trunc @ invX11

    # M = Sigma_cond + X12·B·X12' where Sigma_cond = X22 - X12·invX11·X12'
    Sigma_cond = X22 - X12 @ invX11 @ X12.T
    M = Sigma_cond + X12 @ B @ X12.T

    # --- Layer 1: condcov = Y · M · Y' ---
    ggY, ggM = gbothxomegax(Y, M, x1_symmetric=True, x1_diagonal=True)
    # ggY: (dim1, dd1cov), ggM: (dd1cov, dd1cov)

    # --- Layer 2: M = X22 - P1 + P2, P1 = X12·invX11·X12', P2 = X12·B·X12' ---
    gX12_P1, ginvX11_P1 = gbothxomegax(X12, invX11)
    gX12_P2, gB_P2 = gbothxomegax(X12, B)

    # Combined X12 gradient (through -P1 and +P2)
    gX12_combined = (-gX12_P1 + gX12_P2) @ ggM   # (dim1*dimdiff, dd1cov)

    # --- Layer 3: B = invX11 · Sigma_tilde · invX11 ---
    ginvX11_B, gSigtilde_B = gbothxomegax(
        invX11, sig_trunc, x1_symmetric=True,
    )

    # Combined invX11 gradient
    ginvX11_combined = (-ginvX11_P1 + ginvX11_B @ gB_P2) @ ggM  # (ddcov, dd1cov)

    # Sigma_tilde gradient (through B -> P2 -> M -> condcov)
    gSigtilde = gSigtilde_B @ gB_P2 @ ggM   # (ddcov, dd1cov)

    # --- X11 gradient from inverse path ---
    gX11 = ginverse_sym(X11) @ ginvX11_combined   # (ddcov, dd1cov)

    # --- Truncation gradients (Sigma_tilde -> mu, X11, C) ---
    if dimdiff == 1:
        _, dsigtrunc = grad_univariate_normal_trunc(
            float(mu[0]), sig2, float(C[0]),
        )
        dsig_all = dsigtrunc.reshape(-1, 1) @ gSigtilde   # (3, dd1cov)
    elif dimdiff == 2:
        _, domgderiv = grad_bivariate_normal_trunc(
            mu[:dimdiff], X11, C,
        )
        dsig_all = domgderiv @ gSigtilde   # (7, dd1cov)

    # gmu: only mu[:dimdiff] affects condcov (through Sigma_tilde)
    gmu = np.zeros((dim2, dd1cov))
    gmu[:dimdiff, :] = dsig_all[:dimdiff, :]

    # X11 accumulation from truncation path
    gX11 += dsig_all[dimdiff:dimdiff + ddcov, :]

    # gC: truncation point gradient
    gC = dsig_all[dimdiff + ddcov:, :]

    # --- Reorder X12 from row-major to column-major ---
    gX12_colmaj = np.empty_like(gX12_combined)
    for r in range(dim1):
        for c in range(dimdiff):
            gX12_colmaj[c * dim1 + r, :] = gX12_combined[r * dimdiff + c, :]

    # --- Assemble into vech(X) ordering ---
    stacked = np.vstack([gX11, gX12_colmaj, ggM])
    perm = _block_to_vech_perm(dimdiff, dim1)
    ggX = stacked[perm, :]

    return ggY, gmu, ggX, gC
