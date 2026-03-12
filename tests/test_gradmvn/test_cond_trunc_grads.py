"""Tests for compound gradient helpers: gcondcov, gcondmeantrunc, gcondcovtrunc.

All tests use finite-difference verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from scipy.stats import norm

from pybhatlib.gradmvn._bivariate_trunc import (
    truncated_bivariate_cov,
    truncated_bivariate_mean,
)
from pybhatlib.gradmvn._cond_trunc_grads import (
    _block_to_vech_perm,
    _vech_index,
    gcondcov,
    gcondcovtrunc,
    gcondmeantrunc,
)
from pybhatlib.vecup._vec_ops import vecdup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pd(K: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random K×K positive definite symmetric matrix."""
    A = rng.standard_normal((K, K))
    return A @ A.T + np.eye(K) * 0.5


def _vech(M: np.ndarray) -> np.ndarray:
    """Extract upper-triangle row-by-row (row-major vech)."""
    return vecdup(M)


def _fd_grad(func, x0, eps=1e-6):
    """Central finite-difference gradient of a vector-valued function."""
    f0 = func(x0)
    n_out = len(f0)
    n_in = len(x0)
    G = np.zeros((n_in, n_out))
    for i in range(n_in):
        xp = x0.copy()
        xm = x0.copy()
        xp[i] += eps
        xm[i] -= eps
        G[i, :] = (func(xp) - func(xm)) / (2 * eps)
    return G


# ---------------------------------------------------------------------------
# Test _vech_index
# ---------------------------------------------------------------------------

class TestVechIndex:
    def test_3x3(self):
        """vech of 3×3: (0,0),(0,1),(0,2),(1,1),(1,2),(2,2) → indices 0..5."""
        K = 3
        expected = [(0, 0, 0), (0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4), (2, 2, 5)]
        for i, j, idx in expected:
            assert _vech_index(i, j, K) == idx

    def test_4x4(self):
        K = 4
        # Count: 4+3+2+1 = 10 elements
        idx = 0
        for i in range(K):
            for j in range(i, K):
                assert _vech_index(i, j, K) == idx
                idx += 1


# ---------------------------------------------------------------------------
# Test _block_to_vech_perm
# ---------------------------------------------------------------------------

class TestBlockToVechPerm:
    def test_perm_2_2(self):
        """Verify permutation for dimdiff=2, dim1=2 (dim2=4)."""
        perm = _block_to_vech_perm(2, 2)
        # dim2=4, vech has 10 elements
        assert len(perm) == 10
        # All indices should be unique and cover 0..9
        assert sorted(perm) == list(range(10))

    def test_perm_1_2(self):
        """Verify permutation for dimdiff=1, dim1=2 (dim2=3)."""
        perm = _block_to_vech_perm(1, 2)
        # dim2=3, vech has 6 elements
        assert len(perm) == 6
        assert sorted(perm) == list(range(6))

    def test_perm_reconstructs_vech(self):
        """Build X from blocks, verify perm maps stacked blocks to vech(X)."""
        rng = np.random.default_rng(42)
        dimdiff, dim1 = 2, 3
        dim2 = dimdiff + dim1

        X = _make_pd(dim2, rng)
        X11 = X[:dimdiff, :dimdiff]
        X12 = X[dimdiff:, :dimdiff]   # dim1 × dimdiff
        X22 = X[dimdiff:, dimdiff:]

        # Stack: [vech(X11); vec_colmajor(X12); vech(X22)]
        vech_X11 = _vech(X11)
        # Column-major vec of X12: column by column
        vec_X12_colmaj = X12.T.ravel()  # transpose then ravel = column-major
        vech_X22 = _vech(X22)

        stacked = np.concatenate([vech_X11, vec_X12_colmaj, vech_X22])
        perm = _block_to_vech_perm(dimdiff, dim1)
        reordered = stacked[perm]

        # Should equal vech(X)
        np.testing.assert_allclose(reordered, _vech(X), atol=1e-14)

    def test_perm_1_3(self):
        """Verify for dimdiff=1, dim1=3 (ME method conditioning 1 var at a time)."""
        rng = np.random.default_rng(123)
        dimdiff, dim1 = 1, 3
        dim2 = dimdiff + dim1
        X = _make_pd(dim2, rng)

        X11 = X[:dimdiff, :dimdiff]
        X12 = X[dimdiff:, :dimdiff]
        X22 = X[dimdiff:, dimdiff:]

        vech_X11 = _vech(X11)
        vec_X12_colmaj = X12.T.ravel()
        vech_X22 = _vech(X22)

        stacked = np.concatenate([vech_X11, vec_X12_colmaj, vech_X22])
        perm = _block_to_vech_perm(dimdiff, dim1)
        reordered = stacked[perm]

        np.testing.assert_allclose(reordered, _vech(X), atol=1e-14)


# ---------------------------------------------------------------------------
# Test gcondcov
# ---------------------------------------------------------------------------

class TestGcondcov:
    """Finite-difference verification of gcondcov."""

    @pytest.fixture(params=[
        (2, 2),   # dimdiff=0: A = Y·X·Y'
        (3, 2),   # dimdiff=1: ME-like conditioning
        (4, 2),   # dimdiff=2: OVUS-like conditioning
        (5, 3),   # dimdiff=2, larger
        (4, 3),   # dimdiff=1, larger
    ])
    def setup(self, request):
        dim2, dim1 = request.param
        rng = np.random.default_rng(42 + dim2 * 10 + dim1)
        X = _make_pd(dim2, rng)
        Y = np.diag(rng.uniform(0.5, 2.0, dim1))
        return Y, X, dim1, dim2

    def test_ggY_fd(self, setup):
        """Verify d vech(A)/d diag(Y) against finite differences."""
        Y, X, dim1, dim2 = setup
        ggY_analytic, _ = gcondcov(Y, X)

        def func_Y(y_diag):
            Yt = np.diag(y_diag)
            dimdiff = dim2 - dim1
            if dimdiff == 0:
                A = Yt @ X @ Yt.T
            else:
                X11 = X[:dimdiff, :dimdiff]
                X12 = X[dimdiff:, :dimdiff]
                X22 = X[dimdiff:, dimdiff:]
                Xcond = X22 - X12 @ np.linalg.inv(X11) @ X12.T
                A = Yt @ Xcond @ Yt.T
            return _vech(A)

        y_diag = np.diag(Y).copy()
        ggY_fd = _fd_grad(func_Y, y_diag)

        np.testing.assert_allclose(
            ggY_analytic, ggY_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggY mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_ggX_fd(self, setup):
        """Verify d vech(A)/d vech(X) against finite differences."""
        Y, X, dim1, dim2 = setup
        _, ggX_analytic = gcondcov(Y, X)

        def func_X(x_vech):
            # Reconstruct symmetric X from vech
            Xt = np.zeros((dim2, dim2))
            idx = 0
            for i in range(dim2):
                for j in range(i, dim2):
                    Xt[i, j] = x_vech[idx]
                    Xt[j, i] = x_vech[idx]
                    idx += 1

            dimdiff = dim2 - dim1
            if dimdiff == 0:
                A = Y @ Xt @ Y.T
            else:
                X11 = Xt[:dimdiff, :dimdiff]
                X12 = Xt[dimdiff:, :dimdiff]
                X22 = Xt[dimdiff:, dimdiff:]
                Xcond = X22 - X12 @ np.linalg.inv(X11) @ X12.T
                A = Y @ Xcond @ Y.T
            return _vech(A)

        x_vech = _vech(X)
        ggX_fd = _fd_grad(func_X, x_vech)

        np.testing.assert_allclose(
            ggX_analytic, ggX_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggX mismatch for dim2={dim2}, dim1={dim1}",
        )


# ---------------------------------------------------------------------------
# Helpers for gcondmeantrunc FD tests
# ---------------------------------------------------------------------------

def _univariate_trunc_mean(mu_val, sig2, C_val):
    """E[Z | Z <= C] for univariate Z ~ N(mu, sig2)."""
    sig1 = np.sqrt(sig2)
    w = (C_val - mu_val) / sig1
    phi_w = norm.pdf(w)
    Phi_w = max(norm.cdf(w), 1e-300)
    lam = -phi_w / Phi_w
    return mu_val + sig1 * lam


def _condmean_forward(Y, mu, X, C):
    """Compute condmean = mu2 + Y @ X12 @ inv(X11) @ (mutrunc - mu1)."""
    dim1 = Y.shape[0]
    dim2 = X.shape[0]
    dimdiff = dim2 - dim1

    X11 = X[:dimdiff, :dimdiff]
    X12 = X[dimdiff:, :dimdiff]
    invX11 = np.linalg.inv(X11)

    if dimdiff == 1:
        mutrunc = np.array([_univariate_trunc_mean(mu[0], X11[0, 0], C[0])])
    elif dimdiff == 2:
        mutrunc = truncated_bivariate_mean(mu[:dimdiff], X11, C)
    else:
        raise ValueError(f"dimdiff must be 1 or 2, got {dimdiff}")

    return mu[dimdiff:] + Y @ X12 @ invX11 @ (mutrunc - mu[:dimdiff])


def _reconstruct_sym(x_vech, dim):
    """Reconstruct symmetric matrix from vech."""
    Xt = np.zeros((dim, dim))
    idx = 0
    for i in range(dim):
        for j in range(i, dim):
            Xt[i, j] = x_vech[idx]
            Xt[j, i] = x_vech[idx]
            idx += 1
    return Xt


# ---------------------------------------------------------------------------
# Test gcondmeantrunc
# ---------------------------------------------------------------------------

class TestGcondmeantrunc:
    """Finite-difference verification of gcondmeantrunc."""

    @pytest.fixture(params=[
        (3, 2),   # dimdiff=1, dim1=2 (ME-like)
        (4, 2),   # dimdiff=2, dim1=2 (OVUS-like)
        (4, 3),   # dimdiff=1, dim1=3 (ME, larger)
        (5, 3),   # dimdiff=2, dim1=3 (OVUS, larger)
    ])
    def setup(self, request):
        dim2, dim1 = request.param
        dimdiff = dim2 - dim1
        rng = np.random.default_rng(99 + dim2 * 10 + dim1)
        X = _make_pd(dim2, rng)
        Y = np.diag(rng.uniform(0.5, 2.0, dim1))
        mu = rng.standard_normal(dim2)
        # Truncation points above mu to avoid extreme tail
        C = mu[:dimdiff] + rng.uniform(0.5, 2.0, dimdiff)
        return Y, mu, X, C, dim1, dim2, dimdiff

    def test_ggY_fd(self, setup):
        """Verify d condmean / d diag(Y) against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        ggY_analytic, _, _, _ = gcondmeantrunc(Y, mu, X, C)

        def func_Y(y_diag):
            Yt = np.diag(y_diag)
            return _condmean_forward(Yt, mu, X, C)

        y_diag = np.diag(Y).copy()
        ggY_fd = _fd_grad(func_Y, y_diag)

        np.testing.assert_allclose(
            ggY_analytic, ggY_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggY mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_gmu_fd(self, setup):
        """Verify d condmean / d mu against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, gmu_analytic, _, _ = gcondmeantrunc(Y, mu, X, C)

        def func_mu(mu_vec):
            return _condmean_forward(Y, mu_vec, X, C)

        gmu_fd = _fd_grad(func_mu, mu.copy())

        np.testing.assert_allclose(
            gmu_analytic, gmu_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"gmu mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_ggX_fd(self, setup):
        """Verify d condmean / d vech(X) against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, _, ggX_analytic, _ = gcondmeantrunc(Y, mu, X, C)

        def func_X(x_vech):
            Xt = _reconstruct_sym(x_vech, dim2)
            return _condmean_forward(Y, mu, Xt, C)

        x_vech = _vech(X)
        ggX_fd = _fd_grad(func_X, x_vech)

        np.testing.assert_allclose(
            ggX_analytic, ggX_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggX mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_gC_fd(self, setup):
        """Verify d condmean / d C against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, _, _, gC_analytic = gcondmeantrunc(Y, mu, X, C)

        def func_C(c_vec):
            return _condmean_forward(Y, mu, X, c_vec)

        gC_fd = _fd_grad(func_C, C.copy())

        np.testing.assert_allclose(
            gC_analytic, gC_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"gC mismatch for dim2={dim2}, dim1={dim1}",
        )


# ---------------------------------------------------------------------------
# Helpers for gcondcovtrunc FD tests
# ---------------------------------------------------------------------------

def _univariate_trunc_var(mu_val, sig2, C_val):
    """Var[Z | Z <= C] for univariate Z ~ N(mu, sig2)."""
    sig1 = np.sqrt(sig2)
    w = (C_val - mu_val) / sig1
    phi_w = norm.pdf(w)
    Phi_w = max(norm.cdf(w), 1e-300)
    lam = -phi_w / Phi_w
    return sig2 * (1.0 + lam * (w - lam))


def _condcov_forward(Y, mu, X, C):
    """Compute condcov = vech(Y · Var(Z2|Z1<=C) · Y')."""
    dim1 = Y.shape[0]
    dim2 = X.shape[0]
    dimdiff = dim2 - dim1

    X11 = X[:dimdiff, :dimdiff]
    X12 = X[dimdiff:, :dimdiff]
    X22 = X[dimdiff:, dimdiff:]
    invX11 = np.linalg.inv(X11)

    # Schur complement
    Sigma_cond = X22 - X12 @ invX11 @ X12.T

    # Truncated covariance
    if dimdiff == 1:
        sig_trunc = np.array([[_univariate_trunc_var(mu[0], X11[0, 0], C[0])]])
    elif dimdiff == 2:
        sig_trunc = truncated_bivariate_cov(mu[:dimdiff], X11, C)
    else:
        raise ValueError(f"dimdiff must be 1 or 2, got {dimdiff}")

    # Law of total variance
    B = invX11 @ sig_trunc @ invX11
    M = Sigma_cond + X12 @ B @ X12.T

    condcov = Y @ M @ Y.T
    return _vech(condcov)


# ---------------------------------------------------------------------------
# Test gcondcovtrunc
# ---------------------------------------------------------------------------

class TestGcondcovtrunc:
    """Finite-difference verification of gcondcovtrunc."""

    @pytest.fixture(params=[
        (3, 2),   # dimdiff=1, dim1=2 (ME-like)
        (4, 2),   # dimdiff=2, dim1=2 (OVUS-like)
        (4, 3),   # dimdiff=1, dim1=3 (ME, larger)
        (5, 3),   # dimdiff=2, dim1=3 (OVUS, larger)
    ])
    def setup(self, request):
        dim2, dim1 = request.param
        dimdiff = dim2 - dim1
        rng = np.random.default_rng(77 + dim2 * 10 + dim1)
        X = _make_pd(dim2, rng)
        Y = np.diag(rng.uniform(0.5, 2.0, dim1))
        mu = rng.standard_normal(dim2)
        # Truncation points above mu to avoid extreme tail
        C = mu[:dimdiff] + rng.uniform(0.5, 2.0, dimdiff)
        return Y, mu, X, C, dim1, dim2, dimdiff

    def test_ggY_fd(self, setup):
        """Verify d vech(condcov) / d diag(Y) against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        ggY_analytic, _, _, _ = gcondcovtrunc(Y, mu, X, C)

        def func_Y(y_diag):
            Yt = np.diag(y_diag)
            return _condcov_forward(Yt, mu, X, C)

        y_diag = np.diag(Y).copy()
        ggY_fd = _fd_grad(func_Y, y_diag)

        np.testing.assert_allclose(
            ggY_analytic, ggY_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggY mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_gmu_fd(self, setup):
        """Verify d vech(condcov) / d mu against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, gmu_analytic, _, _ = gcondcovtrunc(Y, mu, X, C)

        def func_mu(mu_vec):
            return _condcov_forward(Y, mu_vec, X, C)

        gmu_fd = _fd_grad(func_mu, mu.copy())

        np.testing.assert_allclose(
            gmu_analytic, gmu_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"gmu mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_ggX_fd(self, setup):
        """Verify d vech(condcov) / d vech(X) against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, _, ggX_analytic, _ = gcondcovtrunc(Y, mu, X, C)

        def func_X(x_vech):
            Xt = _reconstruct_sym(x_vech, dim2)
            return _condcov_forward(Y, mu, Xt, C)

        x_vech = _vech(X)
        ggX_fd = _fd_grad(func_X, x_vech)

        np.testing.assert_allclose(
            ggX_analytic, ggX_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"ggX mismatch for dim2={dim2}, dim1={dim1}",
        )

    def test_gC_fd(self, setup):
        """Verify d vech(condcov) / d C against finite differences."""
        Y, mu, X, C, dim1, dim2, dimdiff = setup
        _, _, _, gC_analytic = gcondcovtrunc(Y, mu, X, C)

        def func_C(c_vec):
            return _condcov_forward(Y, mu, X, c_vec)

        gC_fd = _fd_grad(func_C, C.copy())

        np.testing.assert_allclose(
            gC_analytic, gC_fd, atol=1e-5, rtol=1e-4,
            err_msg=f"gC mismatch for dim2={dim2}, dim1={dim1}",
        )
