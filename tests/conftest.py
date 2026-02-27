"""Shared test fixtures for pybhatlib."""

from __future__ import annotations

import os

import numpy as np
import pytest

from pybhatlib.backend import get_backend


@pytest.fixture
def xp_numpy():
    """NumPy backend fixture."""
    return get_backend("numpy")


@pytest.fixture
def xp_torch():
    """PyTorch backend fixture (skips if torch not installed)."""
    pytest.importorskip("torch")
    return get_backend("torch")


@pytest.fixture(params=["numpy"])
def xp(request):
    """Parametrized backend fixture (numpy only by default)."""
    if request.param == "torch":
        pytest.importorskip("torch")
    return get_backend(request.param)


@pytest.fixture
def sym_3x3():
    """3x3 symmetric test matrix from BHATLIB paper (p. 6)."""
    return np.array([[1.0, 2.0, 3.0],
                     [2.0, 4.0, 5.0],
                     [3.0, 5.0, 6.0]])


@pytest.fixture
def pd_3x3():
    """3x3 positive-definite symmetric matrix."""
    return np.array([[4.0, 2.0, 1.0],
                     [2.0, 5.0, 3.0],
                     [1.0, 3.0, 6.0]])


@pytest.fixture
def cov_3x3():
    """3x3 covariance matrix with known std devs and correlations."""
    # Omega = omega * Omega* * omega
    # omega = diag(1.0, 1.5, 2.0)
    # Omega* = [[1, 0.6, 0.3], [0.6, 1, 0.5], [0.3, 0.5, 1]]
    omega = np.diag([1.0, 1.5, 2.0])
    corr = np.array([[1.0, 0.6, 0.3],
                     [0.6, 1.0, 0.5],
                     [0.3, 0.5, 1.0]])
    return omega @ corr @ omega


@pytest.fixture
def travelmode_path():
    """Path to TRAVELMODE.csv test data."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base, "examples", "data", "TRAVELMODE.csv")
    if not os.path.exists(path):
        pytest.skip("TRAVELMODE.csv not found")
    return path


def numerical_gradient(f, x, eps=1e-7):
    """Compute numerical gradient via central finite differences.

    Parameters
    ----------
    f : callable
        Scalar-valued function f(x).
    x : ndarray
        Point at which to evaluate the gradient.
    eps : float
        Perturbation size.

    Returns
    -------
    grad : ndarray
        Numerical gradient, same shape as x.
    """
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    x_flat = x.ravel()
    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_minus = x_flat.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad.ravel()[i] = (f(x_plus.reshape(x.shape)) - f(x_minus.reshape(x.shape))) / (2 * eps)
    return grad
