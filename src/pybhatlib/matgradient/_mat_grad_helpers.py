"""Matrix gradient helpers for compound gradient operations.

Translates the following GAUSS procedures from BHATLIB's Matgradient.src:
- gasymtosym: convert Kronecker-product gradient to symmetric-matrix gradient
- ginverse: gradient of symmetric matrix inverse
- gaomegab: gradient of A'·Ω·b w.r.t. symmetric Ω
- gbothxomegax: gradient of X₁·X₂·X₁' w.r.t. both X₁ and X₂

All functions follow BHATLIB's row-based arrangement convention.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.vecup._vec_ops import vecdup, vecsymmetry


def gasymtosym(G: NDArray, k: int, n: int) -> NDArray:
    """Convert gradient from vec → vec to vech → vech for symmetric matrices.

    When computing d vec(A)/d vec(B) via Kronecker products, the result has
    shape (k², n²) in BHATLIB's (input, output) convention. If A is k×k
    symmetric (input) and B is n×n symmetric (output), we reduce to
    d vech(A)/d vech(B) with shape (k(k+1)/2, n(n+1)/2).

    This function:
    1. Selects columns for upper-triangular elements of the n×n output matrix
    2. Left-multiplies by vecsymmetry(k) to merge symmetric rows of the k×k input

    Parameters
    ----------
    G : ndarray, shape (k², n²)
        Full Kronecker-product gradient in (input_vec, output_vec) convention.
    k : int
        Dimension of the input symmetric matrix (rows of G are vec of k×k).
    n : int
        Dimension of the output symmetric matrix (cols of G are vec of n×n).

    Returns
    -------
    Gsym : ndarray, shape (k(k+1)/2, n(n+1)/2)
        Gradient in (input_vech, output_vech) form.

    Notes
    -----
    GAUSS ref: gasymtosym (line 3159).
    """
    # Step 1: Select columns for upper-triangular elements of B (n×n)
    # Upper-tri mask flattened to column selector
    col_mask = np.zeros(n * n, dtype=bool)
    for i in range(n):
        for j in range(i, n):
            col_mask[i * n + j] = True
    G_cols = G[:, col_mask]  # shape (k², n(n+1)/2)

    # Step 2: Left-multiply by vecsymmetry to merge symmetric rows
    # vecsymmetry(k) has shape (k(k+1)/2, k²)
    S = vecsymmetry(np.zeros((k, k)))  # shape (k(k+1)/2, k²)
    Gsym = S @ G_cols  # shape (k(k+1)/2, n(n+1)/2)

    return Gsym


def ginverse_sym(
    X: NDArray,
    *,
    diagonal: bool = False,
    correlation: bool = False,
) -> NDArray:
    """Gradient of symmetric matrix inverse: d vech(X⁻¹)/d vech(X).

    For symmetric X, the gradient of X⁻¹ w.r.t. X is:
        d vec(X⁻¹)/d vec(X) = -(X⁻ᵀ ⊗ X⁻¹)
    converted to vech form via gasymtosym.

    Parameters
    ----------
    X : ndarray, shape (K, K)
        Symmetric positive-definite matrix.
    diagonal : bool
        If True, only return rows/columns for diagonal elements.
    correlation : bool
        If True, only return rows for off-diagonal elements (correlation matrix).

    Returns
    -------
    G : ndarray
        Gradient matrix. Shape depends on flags:
        - Default: (K(K+1)/2, K(K+1)/2)
        - diagonal: (K, K)
        - correlation: (K(K-1)/2, K(K+1)/2)

    Notes
    -----
    GAUSS ref: ginverse (line 3385), with _xinvsymmetric=1.
    """
    K = X.shape[0]
    Xinv = np.linalg.inv(X)

    # Full Kronecker gradient: -(X⁻ᵀ ⊗ X⁻¹) = -(X⁻¹ ⊗ X⁻¹) for symmetric X
    G_full = -np.kron(Xinv, Xinv)  # shape (K², K²)

    # Convert to symmetric form
    G = gasymtosym(G_full, K, K)  # shape (K(K+1)/2, K(K+1)/2)

    if diagonal:
        # Select rows and columns corresponding to diagonal elements
        diag_mask = _diag_mask(K)
        G = G[np.ix_(diag_mask, diag_mask)]
    elif correlation:
        # Select rows for off-diagonal elements only
        diag_mask = _diag_mask(K)
        offdiag_mask = ~diag_mask
        G = G[offdiag_mask, :]

    return G


def gaomegab(
    X1: NDArray,
    X2: NDArray,
    *,
    symmetric: bool = True,
    diagonal: bool = False,
) -> NDArray:
    """Gradient of X₁' ⊗ X₂ w.r.t. symmetric Ω in the middle.

    Computes d(X₁'·Ω·X₂)/d vech(Ω) for the implicit product,
    where the gradient is X₁' ⊗ X₂ converted to symmetric form.

    More precisely: for the bilinear form trace(X₁'·Ω·X₂), this gives
    the gradient w.r.t. vech(Ω).

    Parameters
    ----------
    X1 : ndarray, shape (K, L)
        Left matrix.
    X2 : ndarray, shape (K, M) or (K,)
        Right matrix or vector.

    symmetric : bool
        If True (default), Ω is symmetric; use vecsymmetry to merge.
    diagonal : bool
        If True, only diagonal elements of Ω are free.

    Returns
    -------
    G : ndarray
        Gradient matrix.

    Notes
    -----
    GAUSS ref: gaomegab (line 3453).
    The core formula: G = X₁' ⊗ X₂ (before symmetry adjustment).
    """
    X2 = X2.reshape(-1, 1) if X2.ndim == 1 else X2

    K = X1.shape[0]
    # Row-major convention: G_{(i*K+j), (a*M+b)} = dC_{ab}/dΩ_{ij} = X1_{ia}*X2_{jb}
    # = kron(X1, X2) in NumPy (row-major), NOT kron(X1.T, X2) (GAUSS column-major)
    G = np.kron(X1, X2)  # shape (K², L*M)

    if symmetric:
        # Apply vecsymmetry to merge symmetric rows
        S = vecsymmetry(np.zeros((K, K)))  # shape (K(K+1)/2, K²)
        G = S @ G  # shape (K(K+1)/2, L*M)

        if diagonal:
            diag_mask = _diag_mask(K)
            G = G[diag_mask, :]

    return G


def gbothxomegax(
    X1: NDArray,
    X2: NDArray,
    *,
    x1_symmetric: bool = False,
    x2_symmetric: bool = True,
    x1_diagonal: bool = False,
    x2_diagonal: bool = False,
    x2_correlation: bool = False,
) -> tuple[NDArray, NDArray]:
    """Gradient of A = X₁·X₂·X₁' w.r.t. both X₁ and X₂.

    Returns:
    - gx1: d vech(A)/d(X₁ params) — how A changes when X₁ changes
    - gx2: d vech(A)/d(X₂ params) — how A changes when X₂ changes

    Parameters
    ----------
    X1 : ndarray, shape (N, K)
        Left matrix.
    X2 : ndarray, shape (K, K)
        Middle matrix.
    x1_symmetric : bool
        If True, X₁ is symmetric; merge symmetric elements.
    x2_symmetric : bool
        If True (default), X₂ is symmetric; use vech form.
    x1_diagonal : bool
        If True, X₁ is diagonal; only diagonal elements are free.
    x2_diagonal : bool
        If True, X₂ is diagonal; only diagonal elements are free.
    x2_correlation : bool
        If True, X₂ is a correlation matrix; only off-diagonal elements free.

    Returns
    -------
    gx1 : ndarray
        Gradient of vech(A) w.r.t. X₁ parameters.
    gx2 : ndarray
        Gradient of vech(A) w.r.t. X₂ parameters.

    Notes
    -----
    GAUSS ref: gbothxomegax (line 3342).

    Core formulas (before symmetry adjustments):
    - gx2_full = X₁' ⊗ X₁'  (Kronecker product, shape K²×N²)
    - gx1_full = (I_N ⊗ X₂·X₁') + T  where T[·,i] = I_N ⊗ (X₂'·X₁')[:,i]

    The gx1 formula comes from d vec(X₁·X₂·X₁')/d vec(X₁).
    """
    N, K = X1.shape

    # --- gx2: gradient w.r.t. X₂ ---
    # d vec(A)/d vec(X₂) = X₁' ⊗ X₁'
    gx2_full = np.kron(X1.T, X1.T)  # shape (K², N²)

    # --- gx1: gradient w.r.t. X₁ ---
    # A = X₁·X₂·X₁', so dA_{ij}/dX₁_{mn} = δ_{im}(X₂·X₁')_{n,j} + δ_{jm}(X₁·X₂)_{i,n}
    # In vec form: d vec(A)/d vec(X₁) = (I_N ⊗ (X₂·X₁')) + T
    # where T stacks: for each col i of (X₂'·X₁'), repeat via eye(N)
    X2X1t = X2 @ X1.T  # shape (K, N)
    # GAUSS: temp = x2'*x1' = X₂ᵀ·X₁ᵀ, but since X₂ symmetric, X₂ᵀ = X₂
    # gx1_full = I_N ⊗ (X₂·X₁') + T
    # T is built column-by-column: t1 = {} ; for i=1..N: t1 = t1 ~ (eye(N) ⊗ temp[:,i])
    # where temp = X₂ᵀ·X₁ᵀ = X₂·X₁ᵀ (symmetric X₂)
    temp = X2X1t  # (K, N)
    gx1_full = np.kron(np.eye(N), X2X1t)  # shape (N·K, N·N)

    # Build T: for each column i of temp, create eye(N) ⊗ temp[:,i]
    # temp[:,i] is a K-vector; eye(N) ⊗ v gives (N·K, N) matrix
    # Concatenate horizontally for i=1..N: T shape (N·K, N·N)
    T = np.zeros((N * K, N * N))
    for i in range(N):
        T[:, i * N:(i + 1) * N] = np.kron(np.eye(N), temp[:, i:i + 1])
    gx1_full = gx1_full + T  # shape (N·K, N·N)

    # --- Apply symmetry flags ---
    if x2_symmetric:
        # Select columns of gx1 for upper-tri of A (N×N symmetric)
        upper_mask_N = _upper_tri_mask(N)
        gx1 = gx1_full[:, upper_mask_N]  # (N·K, N(N+1)/2)

        # Convert gx2 from K² rows → K(K+1)/2 rows via gasymtosym
        gx2 = gasymtosym(gx2_full, K, N)  # (K(K+1)/2, N(N+1)/2)

        # Apply vecsymmetry to gx1 if X₁ is symmetric
        diag_mask_K = _diag_mask(K)
        if x1_symmetric:
            S_N = vecsymmetry(np.zeros((N, N)))  # (N(N+1)/2, N²)
            # gx1 has shape (N·K, N(N+1)/2) — rows indexed by vec(X₁)
            # We need to reshape: gx1 rows are (N×K) = vec(X₁ᵀ), apply vecsymmetry
            # Actually GAUSS: temp1 = vecsymmetry(n); gx1cov = temp1 * gx1cov;
            # This means rows of gx1 indexed by vec(X₁) (N×K if X₁ is N×K)
            # But vecsymmetry(N) merges N×N symmetric → needs X₁ to be N×N
            # When x1_symmetric=True, X₁ is N×N (e.g., eye(N)), K=N
            S = vecsymmetry(np.zeros((N, N)))  # (N(N+1)/2, N²)
            gx1 = S @ gx1  # (N(N+1)/2, N(N+1)/2)

        if not x2_diagonal:
            if x1_diagonal:
                # Select only diagonal rows of gx1
                if x1_symmetric:
                    diag_mask_N = _diag_mask(N)
                    gx1 = gx1[diag_mask_N, :]
                else:
                    # X₁ is N×K, diagonal means only K diagonal entries
                    diag_rows = [i * K + i for i in range(min(N, K))]
                    gx1 = gx1[diag_rows, :]
        elif x2_diagonal:
            # Select diagonal rows of gx2
            gx2 = gx2[diag_mask_K, :]
            if x1_diagonal:
                if x1_symmetric:
                    diag_mask_N = _diag_mask(N)
                    gx1 = gx1[diag_mask_N, :]
                    # Also select diagonal columns of gx1 and gx2
                    gx1 = gx1[:, _diag_mask(N)]
                    gx2 = gx2[:, _diag_mask(N)]

        if x2_correlation:
            if x2_diagonal:
                gx2 = np.zeros_like(gx2[:1, :1])  # scalar 0
            else:
                # Remove diagonal rows from gx2
                offdiag_mask = ~diag_mask_K
                gx2 = gx2[offdiag_mask, :]

    return gx1, gx2


def _diag_mask(K: int) -> NDArray:
    """Boolean mask for diagonal positions in vech ordering."""
    mask = np.zeros(K * (K + 1) // 2, dtype=bool)
    idx = 0
    for i in range(K):
        for j in range(i, K):
            if i == j:
                mask[idx] = True
            idx += 1
    return mask


def _upper_tri_mask(N: int) -> NDArray:
    """Boolean mask for upper-triangular positions in row-major vec ordering."""
    mask = np.zeros(N * N, dtype=bool)
    for i in range(N):
        for j in range(i, N):
            mask[i * N + j] = True
    return mask
