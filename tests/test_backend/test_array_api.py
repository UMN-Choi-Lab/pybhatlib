"""Tests for backend abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.backend import get_backend, set_backend, array_namespace


class TestGetBackend:
    def test_numpy_backend(self):
        xp = get_backend("numpy")
        assert xp.name == "numpy"

    def test_default_is_numpy(self):
        xp = get_backend()
        assert xp.name == "numpy"

    def test_invalid_backend(self):
        with pytest.raises(ValueError):
            get_backend("invalid")


class TestArrayNamespace:
    def test_infer_numpy(self):
        xp = array_namespace(np.array([1.0]))
        assert xp.name == "numpy"

    def test_infer_none_returns_default(self):
        xp = array_namespace(None)
        assert xp.name == "numpy"


class TestNumpyBackendOps:
    def test_zeros(self, xp_numpy):
        z = xp_numpy.zeros((3, 3))
        np.testing.assert_array_equal(z, np.zeros((3, 3)))

    def test_ones(self, xp_numpy):
        o = xp_numpy.ones((2,))
        np.testing.assert_array_equal(o, np.ones(2))

    def test_eye(self, xp_numpy):
        e = xp_numpy.eye(3)
        np.testing.assert_array_equal(e, np.eye(3))

    def test_matmul(self, xp_numpy):
        A = xp_numpy.array([[1.0, 2.0], [3.0, 4.0]])
        B = xp_numpy.array([[5.0, 6.0], [7.0, 8.0]])
        C = xp_numpy.matmul(A, B)
        np.testing.assert_allclose(C, A @ B)

    def test_solve(self, xp_numpy):
        A = xp_numpy.array([[2.0, 1.0], [1.0, 3.0]])
        b = xp_numpy.array([1.0, 2.0])
        x = xp_numpy.solve(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-12)

    def test_normal_cdf(self, xp_numpy):
        np.testing.assert_allclose(xp_numpy.normal_cdf(xp_numpy.array(0.0)), 0.5, atol=1e-10)
        assert xp_numpy.normal_cdf(xp_numpy.array(-10.0)) < 1e-10
        assert xp_numpy.normal_cdf(xp_numpy.array(10.0)) > 1.0 - 1e-10

    def test_normal_pdf(self, xp_numpy):
        # phi(0) = 1/sqrt(2*pi)
        np.testing.assert_allclose(
            xp_numpy.normal_pdf(xp_numpy.array(0.0)),
            1.0 / np.sqrt(2 * np.pi),
            atol=1e-10,
        )

    def test_cholesky(self, xp_numpy, pd_3x3):
        L = xp_numpy.cholesky(pd_3x3)
        np.testing.assert_allclose(L @ L.T, pd_3x3, atol=1e-10)
