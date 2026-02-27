"""Backend abstraction: select NumPy or PyTorch array operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

BackendName = Literal["numpy", "torch"]

_DEFAULT_BACKEND: BackendName = "numpy"

# Cached backend instances
_backends: dict[str, Any] = {}


def get_backend(name: BackendName | None = None) -> Any:
    """Return a backend namespace providing array operations.

    Parameters
    ----------
    name : {"numpy", "torch"} or None
        Backend name. If None, returns the current default backend.

    Returns
    -------
    backend : NumpyBackend or TorchBackend
        Object exposing array creation/manipulation functions.
    """
    if name is None:
        name = _DEFAULT_BACKEND

    if name not in _backends:
        if name == "numpy":
            from pybhatlib.backend._numpy_backend import NumpyBackend
            _backends[name] = NumpyBackend()
        elif name == "torch":
            from pybhatlib.backend._torch_backend import TorchBackend
            _backends[name] = TorchBackend()
        else:
            raise ValueError(f"Unknown backend: {name!r}. Use 'numpy' or 'torch'.")

    return _backends[name]


def set_backend(name: BackendName) -> None:
    """Set the default backend globally.

    Parameters
    ----------
    name : {"numpy", "torch"}
        Backend to use by default when ``xp`` is not provided.
    """
    global _DEFAULT_BACKEND
    if name not in ("numpy", "torch"):
        raise ValueError(f"Unknown backend: {name!r}. Use 'numpy' or 'torch'.")
    _DEFAULT_BACKEND = name


def array_namespace(*arrays: Any) -> Any:
    """Infer the backend from the input arrays.

    If any array is a PyTorch tensor, returns the torch backend.
    Otherwise returns the numpy backend.

    Parameters
    ----------
    *arrays : array-like
        Input arrays to inspect.

    Returns
    -------
    backend : NumpyBackend or TorchBackend
    """
    for arr in arrays:
        if arr is None:
            continue
        cls_name = type(arr).__module__
        if cls_name.startswith("torch"):
            return get_backend("torch")
        if isinstance(arr, np.ndarray):
            return get_backend("numpy")

    return get_backend()
