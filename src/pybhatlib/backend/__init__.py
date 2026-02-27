"""Backend abstraction for NumPy/PyTorch array operations."""

from pybhatlib.backend._array_api import (
    array_namespace,
    get_backend,
    set_backend,
)

__all__ = ["get_backend", "set_backend", "array_namespace"]
