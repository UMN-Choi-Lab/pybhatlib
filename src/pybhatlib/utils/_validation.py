"""Input validation utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def check_symmetric(A: NDArray, tol: float = 1e-10) -> bool:
    """Check if a matrix is symmetric within tolerance."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, A.T, atol=tol)


def check_positive_definite(A: NDArray) -> bool:
    """Check if a matrix is positive definite via eigenvalues."""
    if not check_symmetric(A):
        return False
    eigvals = np.linalg.eigvalsh(A)
    return bool(np.all(eigvals > 0))


def check_2d(A: NDArray, name: str = "A") -> None:
    """Raise ValueError if A is not 2-dimensional."""
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional, got shape {A.shape}")


def check_square(A: NDArray, name: str = "A") -> None:
    """Raise ValueError if A is not square."""
    check_2d(A, name)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"{name} must be square, got shape {A.shape}")
