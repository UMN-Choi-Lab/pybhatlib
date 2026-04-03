"""Tests for Phase 2 batch MVNCD functions.

Verifies that:
1. mvncd_log_batch matches sequential mvncd() for all supported cases
2. Batch functions handle K=1, K=2, and K>=3 correctly
3. Per-observation covariance path works correctly
4. Fallback path works when method != "me"
"""

import numpy as np
import pytest

from pybhatlib.gradmvn._mvncd import mvncd, mvncd_log_batch, HAS_NUMBA
from pybhatlib.backend._array_api import get_backend


@pytest.fixture
def xp():
    return get_backend("numpy")


class TestMvncdLogBatch:
    """Test mvncd_log_batch function."""

    def test_batch_k2_iid(self, xp):
        """K=2 with shared IID covariance."""
        K = 2
        N = 50
        rng = np.random.RandomState(42)

        Lambda_diff = np.ones((K, K)) + np.eye(K)
        a_all = rng.randn(N, K) * 1.5

        log_probs = mvncd_log_batch(a_all, Lambda_diff, method="me")

        # Compare with sequential
        for q in range(N):
            prob = mvncd(xp.array(a_all[q]), xp.array(Lambda_diff), method="me", xp=xp)
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10, (
                f"Mismatch at q={q}: {log_probs[q]} vs {expected}"
            )

    def test_batch_k3_iid(self, xp):
        """K=3 with shared IID covariance (ME path)."""
        K = 3
        N = 30
        rng = np.random.RandomState(123)

        Lambda_diff = np.ones((K, K)) + np.eye(K)
        a_all = rng.randn(N, K) * 1.0

        log_probs = mvncd_log_batch(a_all, Lambda_diff, method="me")

        for q in range(N):
            prob = mvncd(xp.array(a_all[q]), xp.array(Lambda_diff), method="me", xp=xp)
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_k1(self, xp):
        """K=1 univariate case."""
        K = 1
        N = 20
        rng = np.random.RandomState(7)

        sigma = np.array([[2.0]])
        a_all = rng.randn(N, K) * 1.5

        log_probs = mvncd_log_batch(a_all, sigma, method="me")

        for q in range(N):
            prob = mvncd(xp.array(a_all[q]), xp.array(sigma), method="me", xp=xp)
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_per_obs_sigma(self, xp):
        """Per-observation covariance matrices."""
        K = 2
        N = 15
        rng = np.random.RandomState(99)

        a_all = rng.randn(N, K) * 0.5
        sigma_all = np.zeros((N, K, K))
        for q in range(N):
            A = rng.randn(K, K) * 0.3 + np.eye(K)
            sigma_all[q] = A @ A.T + 0.5 * np.eye(K)

        log_probs = mvncd_log_batch(
            a_all, sigma_all[0], method="me", per_obs_sigma=sigma_all
        )

        for q in range(N):
            prob = mvncd(
                xp.array(a_all[q]), xp.array(sigma_all[q]), method="me", xp=xp
            )
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_flexible_cov(self, xp):
        """Non-IID shared covariance (full Sigma)."""
        K = 2
        N = 25
        rng = np.random.RandomState(55)

        # Flexible covariance (not I + 11^T)
        A = np.array([[1.5, 0.3], [0.3, 1.2]])
        sigma = A @ A.T
        a_all = rng.randn(N, K) * 1.0

        log_probs = mvncd_log_batch(a_all, sigma, method="me")

        for q in range(N):
            prob = mvncd(xp.array(a_all[q]), xp.array(sigma), method="me", xp=xp)
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_fallback_ovus(self, xp):
        """Fallback to sequential when method is not 'me'."""
        K = 2
        N = 10
        rng = np.random.RandomState(42)

        Lambda_diff = np.ones((K, K)) + np.eye(K)
        a_all = rng.randn(N, K) * 1.0

        log_probs = mvncd_log_batch(a_all, Lambda_diff, method="ovus")

        for q in range(N):
            prob = mvncd(
                xp.array(a_all[q]), xp.array(Lambda_diff), method="ovus", xp=xp
            )
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_extreme_values(self, xp):
        """Handle extreme limits (very negative a -> near-zero prob)."""
        K = 2
        N = 5

        Lambda_diff = np.ones((K, K)) + np.eye(K)
        a_all = np.array([
            [-5.0, -5.0],  # very low prob
            [5.0, 5.0],    # very high prob
            [0.0, 0.0],    # moderate
            [-10.0, 3.0],  # mixed
            [3.0, -10.0],  # mixed
        ])

        log_probs = mvncd_log_batch(a_all, Lambda_diff, method="me")

        for q in range(N):
            prob = mvncd(
                xp.array(a_all[q]), xp.array(Lambda_diff), method="me", xp=xp
            )
            expected = np.log(max(prob, 1e-300))
            assert abs(log_probs[q] - expected) < 1e-10

    def test_batch_single_observation(self, xp):
        """Single observation (N=1)."""
        K = 2
        Lambda_diff = np.ones((K, K)) + np.eye(K)
        a_all = np.array([[1.0, 2.0]])

        log_probs = mvncd_log_batch(a_all, Lambda_diff, method="me")
        assert log_probs.shape == (1,)

        prob = mvncd(xp.array(a_all[0]), xp.array(Lambda_diff), method="me", xp=xp)
        expected = np.log(max(prob, 1e-300))
        assert abs(log_probs[0] - expected) < 1e-10
