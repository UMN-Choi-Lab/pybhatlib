"""FD verification tests for _mat_grad_helpers: gasymtosym, ginverse_sym, gaomegab, gbothxomegax."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pybhatlib.matgradient._mat_grad_helpers import (
    gaomegab,
    gasymtosym,
    gbothxomegax,
    ginverse_sym,
)
from pybhatlib.vecup._vec_ops import matdupfull, vecdup

EPS = 1e-6
ATOL = 1e-4
RTOL = 1e-3


def _random_pd(K: int, rng: np.random.Generator) -> np.ndarray:
    """Generate random K×K positive-definite symmetric matrix."""
    A = rng.standard_normal((K, K))
    return A @ A.T + np.eye(K) * 0.5


def _fd_jacobian_vech_to_vech(
    func, X_sym: np.ndarray, eps: float = EPS
) -> np.ndarray:
    """FD Jacobian of vech(f(X)) w.r.t. vech(X) for symmetric X.

    Returns shape (n_in, n_out) following BHATLIB convention:
    row = input vech element, col = output vech element.

    func: takes symmetric matrix, returns symmetric matrix.
    """
    K = X_sym.shape[0]
    v = vecdup(X_sym)
    n_in = len(v)
    f0 = vecdup(func(X_sym))
    n_out = len(f0)
    J = np.zeros((n_in, n_out))
    for j in range(n_in):
        vp = v.copy()
        vp[j] += eps
        Xp = matdupfull(vp)
        fp = vecdup(func(Xp))
        vm = v.copy()
        vm[j] -= eps
        Xm = matdupfull(vm)
        fm = vecdup(func(Xm))
        J[j, :] = (fp - fm) / (2 * eps)
    return J


# ============================================================
# ginverse_sym tests
# ============================================================

class TestGinverseSym:
    """FD verification of ginverse_sym: d vech(X⁻¹)/d vech(X)."""

    @pytest.mark.parametrize("K", [2, 3, 4])
    def test_full(self, K: int):
        rng = np.random.default_rng(42 + K)
        X = _random_pd(K, rng)
        G_analytic = ginverse_sym(X)
        G_fd = _fd_jacobian_vech_to_vech(np.linalg.inv, X)
        assert_allclose(G_analytic, G_fd, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("K", [2, 3])
    def test_diagonal(self, K: int):
        """Test diagonal=True: only diagonal elements of vech."""
        rng = np.random.default_rng(100 + K)
        X = _random_pd(K, rng)
        G_full = ginverse_sym(X)
        G_diag = ginverse_sym(X, diagonal=True)
        # Should match selecting diagonal rows/cols from full
        n_vech = K * (K + 1) // 2
        diag_idx = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    diag_idx.append(idx)
                idx += 1
        G_expected = G_full[np.ix_(diag_idx, diag_idx)]
        assert_allclose(G_diag, G_expected, atol=1e-12)

    @pytest.mark.parametrize("K", [2, 3])
    def test_correlation(self, K: int):
        """Test correlation=True: only off-diagonal output rows."""
        rng = np.random.default_rng(200 + K)
        X = _random_pd(K, rng)
        G_full = ginverse_sym(X)
        G_corr = ginverse_sym(X, correlation=True)
        # Off-diagonal rows of vech
        offdiag_idx = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i != j:
                    offdiag_idx.append(idx)
                idx += 1
        G_expected = G_full[offdiag_idx, :]
        assert_allclose(G_corr, G_expected, atol=1e-12)


# ============================================================
# gaomegab tests
# ============================================================

class TestGaomegab:
    """FD verification of gaomegab: gradient of X1'·Omega·X2 w.r.t. vech(Omega)."""

    @pytest.mark.parametrize("K,L,M", [(3, 2, 2), (4, 1, 3), (2, 3, 1)])
    def test_symmetric(self, K: int, L: int, M: int):
        """Test symmetric Omega: d vec(X1'·Omega·X2)/d vech(Omega)."""
        rng = np.random.default_rng(300 + K * 10 + L)
        X1 = rng.standard_normal((K, L))
        X2 = rng.standard_normal((K, M))
        Omega = _random_pd(K, rng)

        G_analytic = gaomegab(X1, X2, symmetric=True)
        # G_analytic shape: (K(K+1)/2, L*M) — but GAUSS convention
        # Actually gaomegab returns (n_omega_params, L*M) where the gradient is
        # d vec(product)/d vech(Omega). Let me compute FD.

        # product = X1' @ Omega @ X2, shape (L, M)
        v_omega = vecdup(Omega)
        n_omega = len(v_omega)
        f0 = (X1.T @ Omega @ X2).ravel()
        n_out = len(f0)

        J_fd = np.zeros((n_omega, n_out))
        for j in range(n_omega):
            vp = v_omega.copy()
            vp[j] += EPS
            Op = matdupfull(vp)
            fp = (X1.T @ Op @ X2).ravel()
            vm = v_omega.copy()
            vm[j] -= EPS
            Om = matdupfull(vm)
            fm = (X1.T @ Om @ X2).ravel()
            J_fd[j, :] = (fp - fm) / (2 * EPS)

        # gaomegab returns shape: the code does kron(X1.T, X2) then S @ G
        # where S is (K(K+1)/2, K²). The result is (K(K+1)/2, ?) ...
        # Let me check the actual shape. kron(X1.T, X2) has shape (L*M, K²)
        # After S @ (kron(X1.T, X2).T)? No, the code does:
        #   G = np.kron(X1.T, X2)  # shape (L*M, K²)?
        # Wait, kron(X1.T, X2): X1.T is (L,K), X2 is (K,M)
        # kron of (L,K) and (K,M) = (L*K, K*M) — that's not right for the formula
        # Actually I need to re-read the code more carefully.
        # The formula: d vec(X1'·Omega·X2)/d vec(Omega) = X1' ⊗ X2
        # where ⊗ is Kronecker. This gives shape (?, ?)
        # Actually: d vec(C)/d vec(Omega) where C = X1' Omega X2, shape (L, M)
        # Using the identity d vec(ABC)/d vec(B) = (C' ⊗ A)
        # So d vec(X1' Omega X2)/d vec(Omega) = (X2' ⊗ X1') — not kron(X1.T, X2)!
        # Wait, GAUSS code says: G = x1'.*.x2 which is kron(x1', x2)
        # This would give d vec(X1' Omega X2)/d vec(Omega)
        # Let me verify: vec(ABC) = (C' ⊗ A) vec(B)
        # So d vec(X1' Omega X2)/d vec(Omega) = X2' ⊗ X1' — but this is the transpose?
        # Actually the standard formula gives gradient as a matrix where
        # vec(f)_i = sum_j J_{ij} * vec(input)_j
        # So J = X2' ⊗ X1'? But GAUSS says x1' ⊗ x2...
        # The difference is column vs row ordering convention.
        # BHATLIB uses row-based vec. Let me just compare numerically.

        # gaomegab returns shape. Let me just check:
        assert_allclose(G_analytic, J_fd, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("K", [2, 3])
    def test_diagonal(self, K: int):
        """Test diagonal=True: only diagonal of Omega."""
        rng = np.random.default_rng(400 + K)
        X1 = rng.standard_normal((K, 2))
        X2 = rng.standard_normal((K, 2))
        G_full = gaomegab(X1, X2, symmetric=True, diagonal=False)
        G_diag = gaomegab(X1, X2, symmetric=True, diagonal=True)
        # G_diag should be G_full with only diagonal rows of vech
        diag_idx = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    diag_idx.append(idx)
                idx += 1
        assert_allclose(G_diag, G_full[diag_idx, :], atol=1e-12)

    def test_vector_x2(self):
        """Test with X2 as a 1D vector."""
        rng = np.random.default_rng(500)
        K = 3
        X1 = rng.standard_normal((K, 2))
        X2_vec = rng.standard_normal(K)
        X2_mat = X2_vec.reshape(-1, 1)
        G_vec = gaomegab(X1, X2_vec, symmetric=True)
        G_mat = gaomegab(X1, X2_mat, symmetric=True)
        assert_allclose(G_vec, G_mat, atol=1e-12)


# ============================================================
# gbothxomegax tests
# ============================================================

class TestGbothxomegax:
    """FD verification of gbothxomegax: d vech(X1·X2·X1')/d(X1, X2)."""

    def _fd_gbothxomegax(
        self, X1: np.ndarray, X2: np.ndarray,
        x2_symmetric: bool = True,
    ):
        """Compute FD Jacobians for A = X1·X2·X1'."""
        N, K = X1.shape

        def compute_A(x1, x2):
            return x1 @ x2 @ x1.T

        # Output: vech(A), A is N×N symmetric
        A0 = compute_A(X1, X2)
        v_A0 = vecdup(A0)
        n_out = len(v_A0)

        # FD w.r.t. vec(X1)
        x1_flat = X1.ravel()
        n_x1 = len(x1_flat)
        J_x1 = np.zeros((n_x1, n_out))
        for j in range(n_x1):
            xp = x1_flat.copy()
            xp[j] += EPS
            X1p = xp.reshape(N, K)
            fp = vecdup(compute_A(X1p, X2))
            xm = x1_flat.copy()
            xm[j] -= EPS
            X1m = xm.reshape(N, K)
            fm = vecdup(compute_A(X1m, X2))
            J_x1[j, :] = (fp - fm) / (2 * EPS)

        # FD w.r.t. X2
        if x2_symmetric:
            v_x2 = vecdup(X2)
            n_x2 = len(v_x2)
            J_x2 = np.zeros((n_x2, n_out))
            for j in range(n_x2):
                vp = v_x2.copy()
                vp[j] += EPS
                X2p = matdupfull(vp)
                fp = vecdup(compute_A(X1, X2p))
                vm = v_x2.copy()
                vm[j] -= EPS
                X2m = matdupfull(vm)
                fm = vecdup(compute_A(X1, X2m))
                J_x2[j, :] = (fp - fm) / (2 * EPS)
        else:
            x2_flat = X2.ravel()
            n_x2 = len(x2_flat)
            J_x2 = np.zeros((n_x2, n_out))
            for j in range(n_x2):
                xp = x2_flat.copy()
                xp[j] += EPS
                X2p = xp.reshape(K, K)
                fp = vecdup(compute_A(X1, X2p))
                xm = x2_flat.copy()
                xm[j] -= EPS
                X2m = xm.reshape(K, K)
                fm = vecdup(compute_A(X1, X2m))
                J_x2[j, :] = (fp - fm) / (2 * EPS)

        return J_x1, J_x2

    @pytest.mark.parametrize("N,K", [(2, 2), (3, 2), (2, 3)])
    def test_basic(self, N: int, K: int):
        """Basic test: non-symmetric X1, symmetric X2."""
        rng = np.random.default_rng(600 + N * 10 + K)
        X1 = rng.standard_normal((N, K))
        X2 = _random_pd(K, rng)

        gx1, gx2 = gbothxomegax(X1, X2)
        J_x1_fd, J_x2_fd = self._fd_gbothxomegax(X1, X2)

        assert_allclose(gx1, J_x1_fd, atol=ATOL, rtol=RTOL)
        assert_allclose(gx2, J_x2_fd, atol=ATOL, rtol=RTOL)

    def test_x2_diagonal(self):
        """Test x2_diagonal: only diagonal elements of X2 are free."""
        rng = np.random.default_rng(700)
        N, K = 3, 2
        X1 = rng.standard_normal((N, K))
        X2 = _random_pd(K, rng)

        gx1_full, gx2_full = gbothxomegax(X1, X2)
        gx1_diag, gx2_diag = gbothxomegax(X1, X2, x2_diagonal=True)

        # gx2_diag should be diagonal rows of gx2_full
        diag_idx = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i == j:
                    diag_idx.append(idx)
                idx += 1
        assert_allclose(gx2_diag, gx2_full[diag_idx, :], atol=1e-12)

    def test_x2_correlation(self):
        """Test x2_correlation: only off-diagonal elements of X2."""
        rng = np.random.default_rng(800)
        N, K = 2, 3
        X1 = rng.standard_normal((N, K))
        X2 = _random_pd(K, rng)

        gx1, gx2_full = gbothxomegax(X1, X2)
        _, gx2_corr = gbothxomegax(X1, X2, x2_correlation=True)

        # Off-diagonal rows of gx2_full
        offdiag_idx = []
        idx = 0
        for i in range(K):
            for j in range(i, K):
                if i != j:
                    offdiag_idx.append(idx)
                idx += 1
        assert_allclose(gx2_corr, gx2_full[offdiag_idx, :], atol=1e-12)


# ============================================================
# gasymtosym tests
# ============================================================

class TestGasymtosym:
    """Verify gasymtosym correctly reduces Kronecker gradients to vech form."""

    @pytest.mark.parametrize("K", [2, 3])
    def test_identity_gradient(self, K: int):
        """For f(X) = X (identity), d vec(X)/d vec(X) = I.
        gasymtosym(I, K, K) should give the reduction to vech-to-vech."""
        G_full = np.eye(K * K)
        G_sym = gasymtosym(G_full, K, K)
        n_vech = K * (K + 1) // 2
        assert G_sym.shape == (n_vech, n_vech)

        # For identity on symmetric matrices, the vech-to-vech Jacobian should
        # have entries that map each vech element to itself
        # Diagonal entries of G_sym should be positive
        assert np.all(np.diag(G_sym) > 0)

    @pytest.mark.parametrize("K", [2, 3])
    def test_consistency_with_ginverse(self, K: int):
        """gasymtosym applied to -(X⁻¹⊗X⁻¹) should match ginverse_sym."""
        rng = np.random.default_rng(900 + K)
        X = _random_pd(K, rng)
        Xinv = np.linalg.inv(X)

        G_kron = -np.kron(Xinv, Xinv)
        G_asym = gasymtosym(G_kron, K, K)
        G_ginv = ginverse_sym(X)

        assert_allclose(G_asym, G_ginv, atol=1e-12)
