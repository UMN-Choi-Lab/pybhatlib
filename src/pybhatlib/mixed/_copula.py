"""Conditional-Gaussian (copula) mean/covariance and their gradients.

Ports the random-coefficient <-> kernel conditional-Gaussian machinery from
BHATLIB's ``Matgradient.src`` (Bhat/Clower/Haddad/Jones GAUSS source):

- :func:`condition`            -- ``condition`` (Matgradient.src line 3904):
  Schur-complement conditional Gaussian mean ``B`` and covariance ``COVB``.
- :func:`gcondnewcov`          -- ``gcondnewcov`` (line 3348), built on
  ``gcondcov`` (line 3267): gradient of ``COVB`` w.r.t. ``Y`` and the
  correlation elements of ``X``.
- :func:`gcondnewmean`         -- ``gcondnewmean`` (line 3579), built on
  ``gcondmean`` (line 3498): gradient of ``B`` w.r.t. ``Y``, ``mu``, ``X`` and
  the marginal draw ``g``.
- :func:`gcondspecialnewmean`  -- ``gcondspecialnewmean`` (line 4108): the
  "special" variant where the marginal draw is generated as ``g = L11' e`` from
  an i.i.d. draw ``e`` and the upper Cholesky ``L11`` of the marginal
  correlation block; returns ``B`` together with structured, Cholesky-
  parameterized gradients ``gY``, ``gmu``, ``gW`` (standard deviations) and
  ``gX`` (correlation).

Conventions (matching BHATLIB)
------------------------------
* **Row-based vech**: symmetric matrices are vectorized row by row over their
  upper triangle.  ``COVB`` gradients are ordered by ``vecdup`` (upper triangle
  *including* the diagonal); correlation gradients (``gX``) are ordered by
  ``vecndup`` (strict upper triangle) of the *full* ``K x K`` matrix.
* **No module-level state**: the GAUSS globals ``_cholesky`` and ``_condcov``
  become explicit keyword arguments ``cholesky=False`` and ``condcov=False`` --
  the exact values the MNP kernel driver sets right before every call
  (``MNPKERCP.gss`` line 882: ``_condcov=0; _cholesky=0;``).
* ``xp`` is accepted for backend selection; computation is performed in NumPy
  (analytic gradients), matching the house pattern in ``_spherical.py``.

Only the driver's ``cholesky=False`` path is implemented; passing
``cholesky=True`` raises :class:`NotImplementedError` (the Cholesky-parameter
chain is not exercised by the kernel driver).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace
from pybhatlib.utils._safe_reparam import safe_cholesky


# ---------------------------------------------------------------------------
# small internal helpers (row-based vech ordering)
# ---------------------------------------------------------------------------
def _split_index(indxmarg: NDArray) -> tuple[NDArray, NDArray]:
    """Return (indeq, indneq): positions where ``indxmarg`` is 1 / 0.

    Mirrors GAUSS ``selif(seqa(1,1,K), indxmarg)`` and its complement, so the
    conditioning (marginal) set ``indeq`` and the target set ``indneq`` keep
    their natural ascending order.
    """
    m = np.asarray(indxmarg, dtype=np.float64).ravel()
    indeq = np.where(m == 1)[0]
    indneq = np.where(m != 1)[0]
    return indeq, indneq


def _vecdup(mat: NDArray) -> NDArray:
    """Upper triangle incl. diagonal, row-based (GAUSS ``vecdup``)."""
    K = mat.shape[0]
    return np.array([mat[i, j] for i in range(K) for j in range(i, K)],
                    dtype=np.float64)


def _offdiag_pairs(K: int) -> list[tuple[int, int]]:
    """Strict-upper (p, q) pairs, row-based (GAUSS ``vecndup`` order)."""
    return [(p, q) for p in range(K) for q in range(p + 1, K)]


def _upper_pairs(K: int) -> list[tuple[int, int]]:
    """Upper (p, q) pairs incl. diagonal, row-based (GAUSS ``vecdup`` order)."""
    return [(p, q) for p in range(K) for q in range(p, K)]


def _upper_chol(A: NDArray) -> NDArray:
    """Upper-triangular Cholesky ``U`` with ``U.T @ U == A`` (GAUSS ``chol``)."""
    return safe_cholesky(np.asarray(A, dtype=np.float64))[0].T


def _chol_lower_deriv(A: NDArray, dA: NDArray) -> tuple[NDArray, NDArray]:
    """Derivative of the lower Cholesky factor ``L`` (``L @ L.T == A``).

    Given a symmetric perturbation ``dA`` returns ``(L, dL)`` with
    ``dA = dL @ L.T + L @ dL.T``.  Standard forward-mode Cholesky
    differentiation via ``dL = L * Phi_lower`` where
    ``Phi = L^{-1} dA L^{-T}`` and the diagonal of ``Phi`` is halved.
    """
    L = safe_cholesky(np.asarray(A, dtype=np.float64))[0]
    Linv = np.linalg.inv(L)
    Phi = Linv @ dA @ Linv.T
    n = A.shape[0]
    Mlow = np.tril(Phi)
    Mlow[np.diag_indices(n)] *= 0.5
    return L, L @ Mlow


def _check_cholesky(cholesky: bool, name: str) -> None:
    if cholesky:
        raise NotImplementedError(
            f"{name}: cholesky=True (gradients w.r.t. Cholesky elements) is not "
            "implemented; the MNP kernel driver sets _cholesky=0."
        )


def _check_condcov(condcov: bool, name: str) -> None:
    if condcov:
        raise NotImplementedError(
            f"{name}: condcov=True (X as covariance) is not implemented; the "
            "MNP kernel driver sets _condcov=0 (X is a correlation matrix)."
        )


# ---------------------------------------------------------------------------
# condition  (Matgradient.src line 3904)
# ---------------------------------------------------------------------------
def condition(
    Y: NDArray,
    mu: NDArray,
    X: NDArray,
    g: NDArray,
    indxmarg: NDArray,
    *,
    xp=None,
) -> tuple[NDArray, NDArray]:
    r"""Conditional Gaussian mean and covariance (Schur complement).

    For a jointly normal vector partitioned by ``indxmarg`` into a marginal
    (conditioning) block ``D1`` and a target block ``D2``, with covariance/
    correlation ``X``, the conditional distribution of ``D2`` given ``D1 = g``
    has mean ``F = X12 X11^{-1} (g - mu1) + mu2`` and covariance
    ``COV = X22 - X12 X11^{-1} X12'``.  This returns ``B = Y F`` and
    ``COVB = Y COV Y``.

    Parameters
    ----------
    Y : ndarray, shape (M, M)
        Diagonal matrix, ``M = |indneq|`` (the target block size).
    mu : ndarray, shape (K,)
        Mean vector of the full ``K``-dimensional vector.
    X : ndarray, shape (K, K)
        Symmetric covariance or correlation matrix.
    g : ndarray, shape (|indeq|,)
        Marginal draw at which the conditioning is evaluated (same size as the
        marginal/conditioning block).
    indxmarg : ndarray, shape (K,)
        Vector of 0/1; ``1`` marks members of the marginal (conditioning) set.
    xp : backend, optional
        Array backend (computation uses NumPy internally).

    Returns
    -------
    B : ndarray, shape (M, 1)
        Conditional kernel mean ``Y F``.
    COVB : ndarray, shape (M, M)
        Conditional kernel covariance ``Y COV Y``.

    Notes
    -----
    GAUSS ref: ``condition`` (Matgradient.src line 3904).  Usage in the MNP
    kernel driver (``MNPKERCP.gss``)::

        { B1subq, xi2subq } = condition(eye(nc-1), zeros(nrndtot,1),
                                        omegastar, errbeta3, indxrand);
    """
    if xp is None:
        xp = array_namespace(X)
    Y = np.asarray(Y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).reshape(-1)

    indeq, indneq = _split_index(indxmarg)
    X11 = X[np.ix_(indeq, indeq)]
    X12 = X[np.ix_(indneq, indeq)]
    X22 = X[np.ix_(indneq, indneq)]
    invX11 = np.linalg.inv(X11)

    B = Y @ (X12 @ invX11 @ (g - mu[indeq]) + mu[indneq])
    COVB = Y @ (X22 - X12 @ invX11 @ X12.T) @ Y
    return B.reshape(-1, 1), COVB


# ---------------------------------------------------------------------------
# gcondnewcov  (Matgradient.src line 3348 / gcondcov line 3267)
# ---------------------------------------------------------------------------
def gcondnewcov(
    Y: NDArray,
    X: NDArray,
    indxmarg: NDArray,
    *,
    cholesky: bool = False,
    condcov: bool = False,
    xp=None,
) -> tuple[NDArray, NDArray]:
    r"""Gradient of the conditional covariance ``COVB`` w.r.t. ``Y`` and ``X``.

    With ``C = X22 - X12 X11^{-1} X12'`` and ``COVB = Y C Y`` (see
    :func:`condition`), this returns the analytic gradients

    * ``gY``: ``d vech(COVB) / d diag(Y)`` -- one row per free diagonal element
      of ``Y``.
    * ``gX``: ``d vech(COVB) / d (free elements of X)`` -- one row per free
      element of the *full* ``K x K`` matrix, in row-based ``vecndup`` order
      (correlation, ``condcov=False``) or ``vecdup`` order (covariance,
      ``condcov=True``).

    Output columns follow ``vecdup(COVB)`` (upper triangle incl. diagonal,
    row-based), i.e. the GAUSS ordering ``{dB22, dB23, dB33, ...}``.

    Parameters
    ----------
    Y : ndarray, shape (M, M)
        Diagonal matrix (target block size ``M``).
    X : ndarray, shape (K, K)
        Correlation matrix (``condcov=False``) or covariance (``condcov=True``).
    indxmarg : ndarray, shape (K,)
        0/1 marginal-set indicator.
    cholesky : bool, default False
        GAUSS ``_cholesky``.  Only ``False`` is implemented (the kernel driver
        sets ``_cholesky=0``).
    condcov : bool, default False
        GAUSS ``_condcov``.  ``False`` = ``X`` is a correlation matrix (driver
        default); ``True`` = ``X`` is a covariance matrix.
    xp : backend, optional

    Returns
    -------
    gY : ndarray, shape (M, M*(M+1)//2)
    gX : ndarray, shape (n_free, M*(M+1)//2)
        ``n_free = K*(K-1)//2`` if ``condcov=False`` else ``K*(K+1)//2``.

    Notes
    -----
    GAUSS ref: ``gcondnewcov`` (line 3348), which reorders the output of
    ``gcondcov`` (line 3267) into full-matrix element order.  The kernel driver
    calls this with ``_condcov=0; _cholesky=0``.
    """
    _check_cholesky(cholesky, "gcondnewcov")
    if xp is None:
        xp = array_namespace(X)
    Y = np.asarray(Y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    K = X.shape[0]

    indeq, indneq = _split_index(indxmarg)
    X11 = X[np.ix_(indeq, indeq)]
    X12 = X[np.ix_(indneq, indeq)]
    X22 = X[np.ix_(indneq, indneq)]
    invX11 = np.linalg.inv(X11)
    M = X12 @ invX11                      # (dim1 x dimdiff)
    C = X22 - X12 @ invX11 @ X12.T        # conditional covariance
    dim1 = len(indneq)

    # --- gY: d vech(YCY)/d y_i ,  COVB_ab = y_a C_ab y_b ---
    ydiag = np.diag(Y)
    out_pairs = _upper_pairs(dim1)
    gY = np.zeros((dim1, len(out_pairs)), dtype=np.float64)
    for i in range(dim1):
        for col, (a, b) in enumerate(out_pairs):
            val = 0.0
            if i == a:
                val += C[a, b] * ydiag[b]
            if i == b:
                val += ydiag[a] * C[a, b]
            gY[i, col] = val

    # --- gX: elementary symmetric perturbation of the free elements of X ---
    free_pairs = _upper_pairs(K) if condcov else _offdiag_pairs(K)
    gX = np.zeros((len(free_pairs), len(out_pairs)), dtype=np.float64)
    for r, (p, q) in enumerate(free_pairs):
        dX = np.zeros((K, K), dtype=np.float64)
        dX[p, q] = 1.0
        dX[q, p] = 1.0  # p == q simply re-sets the single diagonal entry
        dX11 = dX[np.ix_(indeq, indeq)]
        dX12 = dX[np.ix_(indneq, indeq)]
        dX22 = dX[np.ix_(indneq, indneq)]
        dC = dX22 - dX12 @ M.T - M @ dX12.T + M @ dX11 @ M.T
        dCOVB = Y @ dC @ Y
        gX[r, :] = _vecdup(dCOVB)
    return gY, gX


# ---------------------------------------------------------------------------
# gcondnewmean  (Matgradient.src line 3579 / gcondmean line 3498)
# ---------------------------------------------------------------------------
def gcondnewmean(
    Y: NDArray,
    mu: NDArray,
    X: NDArray,
    g: NDArray,
    indxmarg: NDArray,
    *,
    cholesky: bool = False,
    condcov: bool = False,
    xp=None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    r"""Gradient of the conditional mean ``B`` w.r.t. ``Y``, ``mu``, ``X``, ``g``.

    With ``M = X12 X11^{-1}``, ``F = M (g_eq - mu_eq) + mu_neq`` and ``B = Y F``
    (see :func:`condition`), returns the four analytic gradients matching GAUSS
    ``gcondnewmean``:

    * ``gY``  : ``d B / d diag(Y)`` -- ``diag(F)``, shape ``(M, M)``.
    * ``gmu`` : ``d B / d mu`` -- shape ``(K, M)``, full-vector row order.
    * ``gX``  : ``d B / d (free elements of X)`` -- rows in ``vecndup`` order
      (correlation) / ``vecdup`` order (covariance) of the full matrix.
    * ``gg``  : ``d B / d g`` -- shape ``(K, M)``; non-zero only in the marginal
      (``indeq``) rows.

    ``g`` is the **full ``K``-vector** with the marginal draw values in the
    ``indeq`` positions and zeros in the ``indneq`` positions (GAUSS
    ``gcondnewmean`` convention).

    Notes
    -----
    GAUSS ref: ``gcondnewmean`` (line 3579).  Driver: ``_condcov=0; _cholesky=0``.
    """
    _check_cholesky(cholesky, "gcondnewmean")
    if xp is None:
        xp = array_namespace(X)
    Y = np.asarray(Y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64).reshape(-1)
    K = X.shape[0]

    indeq, indneq = _split_index(indxmarg)
    X11 = X[np.ix_(indeq, indeq)]
    X12 = X[np.ix_(indneq, indeq)]
    invX11 = np.linalg.inv(X11)
    M = X12 @ invX11
    dim1 = len(indneq)
    dimdiff = len(indeq)

    g_eq = g[indeq]
    F = M @ (g_eq - mu[indeq]) + mu[indneq]

    # gY = diag(F)
    gY = np.diagflat(F)

    # gmu : eq rows -> -(Y M)^T ; neq rows -> Y
    YM = Y @ M                                     # (dim1 x dimdiff)
    gmu = np.zeros((K, dim1), dtype=np.float64)
    for e_i, row in enumerate(indeq):
        gmu[row, :] = -YM[:, e_i]
    for t_i, row in enumerate(indneq):
        gmu[row, :] = Y[t_i, :]

    # gX : dB = Y dM (g_eq - mu_eq),  dM = dX12 invX11 - M dX11 invX11
    free_pairs = _upper_pairs(K) if condcov else _offdiag_pairs(K)
    gv = g_eq - mu[indeq]
    gX = np.zeros((len(free_pairs), dim1), dtype=np.float64)
    for r, (p, q) in enumerate(free_pairs):
        dX = np.zeros((K, K), dtype=np.float64)
        dX[p, q] = 1.0
        dX[q, p] = 1.0
        dX11 = dX[np.ix_(indeq, indeq)]
        dX12 = dX[np.ix_(indneq, indeq)]
        dM = dX12 @ invX11 - M @ dX11 @ invX11
        gX[r, :] = Y @ (dM @ gv)

    # gg : eq rows -> (Y M)^T ; neq rows -> 0
    gg = np.zeros((K, dim1), dtype=np.float64)
    for e_i, row in enumerate(indeq):
        gg[row, :] = YM[:, e_i]

    return gY, gmu, gX, gg


# ---------------------------------------------------------------------------
# gcondspecialnewmean  (Matgradient.src line 4108)
# ---------------------------------------------------------------------------
def gcondspecialnewmean(
    Y: NDArray,
    mu: NDArray,
    X: NDArray,
    e: NDArray,
    indxmarg: NDArray,
    *,
    cholesky: bool = False,
    condcov: bool = False,
    xp=None,
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    r""""Special" conditional mean with a Cholesky-generated marginal draw.

    The marginal draw is generated as ``g = L11' e`` from an i.i.d. draw ``e``
    and the upper Cholesky ``L11`` of the marginal correlation block
    ``X11 = X[indeq, indeq]`` (``L11' L11 = X11``).  Writing
    ``M = X12 X11^{-1}``, ``L12' = X12 L11^{-1}`` and the *effective draw*
    ``etilde = e - L11'^{-1} mu_eq`` (so ``B = Y (L12' etilde + mu_neq)``), this
    returns ``B`` together with the structured, Cholesky-parameterized
    gradients produced by GAUSS ``gcondspecialnewmean``:

    * ``B``   : ndarray, shape ``(M, 1)`` -- conditional kernel mean.
    * ``gY``  : ndarray, shape ``(M, M)`` -- ``diag(F)`` with ``F = B / diag(Y)``.
    * ``gmu`` : ndarray, shape ``(K, M)`` -- ``d B / d mu`` (eq rows ``-(Y M)^T``;
      neq rows ``Y``).
    * ``gW``  : ndarray, shape ``(K, M)`` -- gradient w.r.t. the ``K`` standard
      deviations (evaluated at unit scale): eq rows ``Y M[:,e] * mu_eq[e]``;
      neq rows ``diag(Y P)`` with ``P = M (g - mu_eq)``.
    * ``gX``  : ndarray, shape ``(K*(K-1)//2, M)`` -- structured gradient w.r.t.
      the correlation off-diagonals of the full matrix (``vecndup`` order):
      cross (marginal x target) rows carry ``etilde_i`` at the matching target
      column; marginal (eq x eq) rows carry the Cholesky-parameterized term
      ``Y L12' d(L11'^{-1})/dr mu_eq``; target (neq x neq) rows are zero.

    ``e`` is the ``|indeq|``-length i.i.d. draw for the marginal block.

    Notes
    -----
    GAUSS ref: ``gcondspecialnewmean`` (line 4108).  The ``gX`` and ``gW``
    outputs are *structured partials* in the Cholesky parameterization used by
    the kernel driver -- they are **not** the plain total derivative of ``B``
    w.r.t. the correlation (which would remix the marginal-draw generation).
    Verified element-for-element against live GAUSS 26.1.1.
    """
    _check_cholesky(cholesky, "gcondspecialnewmean")
    _check_condcov(condcov, "gcondspecialnewmean")
    if xp is None:
        xp = array_namespace(X)
    Y = np.asarray(Y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    X = np.asarray(X, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    K = X.shape[0]

    indeq, indneq = _split_index(indxmarg)
    dimdiff = len(indeq)
    dim1 = len(indneq)
    eq_pos = {int(v): i for i, v in enumerate(indeq)}
    neq_pos = {int(v): i for i, v in enumerate(indneq)}

    X11 = X[np.ix_(indeq, indeq)]
    X12 = X[np.ix_(indneq, indeq)]
    invX11 = np.linalg.inv(X11)
    M = X12 @ invX11                              # (dim1 x dimdiff)
    L11 = _upper_chol(X11)                        # upper, L11' L11 = X11
    L11inv = np.linalg.inv(L11)
    L11tinv = np.linalg.inv(L11.T)
    L12t = X12 @ L11inv                           # = L12'  (dim1 x dimdiff)

    mu_eq = mu[indeq]
    mu_neq = mu[indneq]
    gcov = L11.T @ e                              # marginal draw
    etilde = e - L11tinv @ mu_eq                  # effective draw
    ydiag = np.diag(Y)

    # B
    B = Y @ (L12t @ etilde + mu_neq)

    # gY = diag(F), F = B / diag(Y)
    F = M @ (gcov - mu_eq) + mu_neq
    gY = np.diagflat(F)

    # gmu : eq -> -(Y M)^T ; neq -> Y
    YM = Y @ M
    gmu = np.zeros((K, dim1), dtype=np.float64)
    for e_i, row in enumerate(indeq):
        gmu[row, :] = -YM[:, e_i]
    for t_i, row in enumerate(indneq):
        gmu[row, :] = Y[t_i, :]

    # gW : eq -> Y M[:,e] * mu_eq[e] ; neq -> diag(Y P), P = M (gcov - mu_eq)
    P = M @ (gcov - mu_eq)
    gW = np.zeros((K, dim1), dtype=np.float64)
    for e_i, row in enumerate(indeq):
        gW[row, :] = YM[:, e_i] * mu_eq[e_i]
    for t_i, row in enumerate(indneq):
        gW[row, t_i] = ydiag[t_i] * P[t_i]

    # gX : structured correlation gradient, vecndup(full X) row order
    free_pairs = _offdiag_pairs(K)
    gX = np.zeros((len(free_pairs), dim1), dtype=np.float64)
    for r, (p, q) in enumerate(free_pairs):
        p_eq = p in eq_pos
        q_eq = q in eq_pos
        if p_eq and q_eq:
            # marginal (eq x eq): Cholesky-parameterized term through etilde
            dX11 = np.zeros((dimdiff, dimdiff), dtype=np.float64)
            dX11[eq_pos[p], eq_pos[q]] = 1.0
            dX11[eq_pos[q], eq_pos[p]] = 1.0
            _, dLlow = _chol_lower_deriv(X11, dX11)   # L11 = R'; R=L11
            dR = dLlow.T                               # d(upper L11)
            dL11tinv = -L11tinv @ dR.T @ L11tinv       # d inv(L11')
            detilde = -dL11tinv @ mu_eq
            gX[r, :] = Y @ (L12t @ detilde)
        elif p_eq and not q_eq:
            # cross: eq_i = pos of p, target col = pos of q
            gX[r, neq_pos[q]] = ydiag[neq_pos[q]] * etilde[eq_pos[p]]
        elif q_eq and not p_eq:
            gX[r, neq_pos[p]] = ydiag[neq_pos[p]] * etilde[eq_pos[q]]
        else:
            pass  # target x target -> 0
    return B.reshape(-1, 1), gY, gmu, gW, gX
