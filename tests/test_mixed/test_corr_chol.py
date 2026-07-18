"""Tests for the scaled radial parameterization and Cholesky-of-correlation
gradients ported from GAUSS BHATLIB (matgradient.src, vecup.src).

Covers:
  (a) newcholparmscaled(theta, 1).T @ newcholparmscaled(theta, 1) has unit diag
  (b) revnewcholparmscaled round-trips theta
  (c) gnewcholparmcorscaled vs central-FD of correlation wrt theta
  (d) gcholeskycor vs FD of the correlation wrt Cholesky off-diagonal elements
  (e) ggradchol vs FD of (L.T @ e) wrt the elements of L
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.matgradient._radial import (
    gnewcholparmcorscaled,
    newcholparmscaled,
    revnewcholparmscaled,
)
from pybhatlib.matgradient._corr_chol import gcholeskycor, ggradchol


def _n_theta(K: int) -> int:
    return K * (K - 1) // 2


def _rand_theta(K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Keep magnitudes moderate so the correlation stays well inside the PD cone.
    return rng.uniform(-1.5, 1.5, size=_n_theta(K))


def _corr_from_theta(theta: np.ndarray, scal: float) -> np.ndarray:
    L = newcholparmscaled(theta, scal)
    return L.T @ L


def _vecndup(m: np.ndarray) -> np.ndarray:
    """Row-based upper off-diagonal elements (matches vecup.vecndup)."""
    K = m.shape[0]
    out = []
    for i in range(K):
        for j in range(i + 1, K):
            out.append(m[i, j])
    return np.asarray(out, dtype=np.float64)


@pytest.mark.parametrize("K", [3, 4])
def test_a_unit_diagonal(K: int) -> None:
    theta = _rand_theta(K, seed=10 + K)
    L = newcholparmscaled(theta, 1.0)
    corr = L.T @ L
    assert np.allclose(np.diag(corr), 1.0, atol=1e-12)
    # Off-diagonals within (-1, 1) and matrix positive definite.
    assert np.all(np.abs(corr - np.diag(np.diag(corr))) < 1.0 + 1e-9)
    assert np.all(np.linalg.eigvalsh(corr) > 0.0)


def test_extreme_optimizer_step_remains_finite() -> None:
    """Unconstrained trial values must not overflow before an iteration."""
    theta = np.array([1_000.0, -1_000.0, 750.0])
    chol = newcholparmscaled(theta, 1.0)
    corr = chol.T @ chol

    assert np.all(np.isfinite(chol))
    assert np.all(np.isfinite(corr))
    np.testing.assert_allclose(np.diag(corr), np.ones(3), atol=1e-12)


@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("scal", [1.0, 2.0])
def test_b_roundtrip(K: int, scal: float) -> None:
    theta = _rand_theta(K, seed=20 + K)
    L = newcholparmscaled(theta, scal)
    theta_rec = revnewcholparmscaled(L, scal)
    assert np.allclose(theta_rec, theta, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("K", [3, 4])
@pytest.mark.parametrize("scal", [1.0, 2.0])
def test_c_gnewcholparmcorscaled_vs_fd(K: int, scal: float) -> None:
    theta = _rand_theta(K, seed=30 + K)
    grad1, grad2 = gnewcholparmcorscaled(_corr_from_theta(theta, scal), scal)

    # (c) d(corr_offdiag)/d(theta) via central differences.
    h = 1e-6
    n = _n_theta(K)
    fd1 = np.zeros((n, n), dtype=np.float64)
    for p in range(n):
        tp = theta.copy()
        tp[p] += h
        tm = theta.copy()
        tm[p] -= h
        d = (_vecndup(_corr_from_theta(tp, scal))
             - _vecndup(_corr_from_theta(tm, scal))) / (2.0 * h)
        fd1[p, :] = d
    assert np.allclose(grad1, fd1, atol=1e-5, rtol=1e-4)

    # d(corr_offdiag)/d(scal) via central differences.
    ds = (_vecndup(_corr_from_theta(theta, scal + h))
          - _vecndup(_corr_from_theta(theta, scal - h))) / (2.0 * h)
    assert np.allclose(grad2, ds, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("K", [3, 4])
def test_d_gcholeskycor_vs_fd(K: int) -> None:
    theta = _rand_theta(K, seed=40 + K)
    corr = _corr_from_theta(theta, 1.0)
    gc = gcholeskycor(corr)  # [chol_offdiag (row), corr_offdiag (col)]

    # Upper Cholesky U with U.T @ U = corr; free = off-diagonal elements S.
    U0 = np.linalg.cholesky(corr).T
    S0 = _vecndup(U0)
    n = _n_theta(K)

    def corr_from_S(S: np.ndarray) -> np.ndarray:
        U = np.zeros((K, K), dtype=np.float64)
        idx = 0
        for i in range(K):
            for j in range(i + 1, K):
                U[i, j] = S[idx]
                idx += 1
        U[0, 0] = 1.0
        for j in range(1, K):
            U[j, j] = np.sqrt(1.0 - np.sum(U[:j, j] ** 2))
        return U.T @ U

    h = 1e-6
    fd = np.zeros((n, n), dtype=np.float64)  # [S (row), corr (col)]
    for i in range(n):
        sp = S0.copy()
        sp[i] += h
        sm = S0.copy()
        sm[i] -= h
        fd[i, :] = (_vecndup(corr_from_S(sp)) - _vecndup(corr_from_S(sm))) / (2.0 * h)

    assert np.allclose(gc, fd, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("K", [3, 4])
def test_e_ggradchol_vs_fd(K: int) -> None:
    rng = np.random.default_rng(50 + K)
    # Random upper-triangular covariance Cholesky (positive diagonal).
    L = np.triu(rng.uniform(-1.0, 1.0, size=(K, K)))
    L[np.diag_indices(K)] = rng.uniform(0.5, 1.5, size=K)
    e = rng.uniform(-1.0, 1.0, size=K)
    A = np.eye(K)

    gchol = ggradchol(A, L, e, cholcov=True)  # [vech_upper(L) (row), M (col)]

    # vech-upper ordering of L used by ggradchol.
    upper_idx = [(j, i) for j in range(K) for i in range(j, K)]

    def B_from_Lvec(vec: np.ndarray) -> np.ndarray:
        Lm = np.zeros((K, K), dtype=np.float64)
        for k, (j, i) in enumerate(upper_idx):
            Lm[j, i] = vec[k]
        return A @ Lm.T @ e

    Lvec0 = np.array([L[j, i] for (j, i) in upper_idx], dtype=np.float64)
    h = 1e-6
    n = len(upper_idx)
    fd = np.zeros((n, K), dtype=np.float64)
    for k in range(n):
        vp = Lvec0.copy()
        vp[k] += h
        vm = Lvec0.copy()
        vm[k] -= h
        fd[k, :] = (B_from_Lvec(vp) - B_from_Lvec(vm)) / (2.0 * h)

    assert np.allclose(gchol, fd, atol=1e-6, rtol=1e-5)
