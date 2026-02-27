"""PyTorch backend implementation (optional dependency)."""

from __future__ import annotations

import math
from typing import Any


def _import_torch():
    """Lazy import of torch."""
    try:
        import torch
        return torch
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for the torch backend. "
            "Install it with: pip install pybhatlib[torch]"
        ) from e


class TorchBackend:
    """Backend wrapping PyTorch for array operations with GPU support."""

    name = "torch"

    def __init__(self, device: str = "cpu", dtype: Any = None):
        self._torch = _import_torch()
        self.device = device
        self.float64 = self._torch.float64
        self.float32 = self._torch.float32
        self.int64 = self._torch.int64
        self._default_dtype = dtype or self._torch.float64

    # --- Array creation ---
    def array(self, data, dtype=None):
        dtype = dtype or self._default_dtype
        if isinstance(data, self._torch.Tensor):
            return data.to(device=self.device, dtype=dtype)
        return self._torch.tensor(data, dtype=dtype, device=self.device)

    def zeros(self, shape, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.zeros(shape, dtype=dtype, device=self.device)

    def ones(self, shape, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.ones(shape, dtype=dtype, device=self.device)

    def eye(self, n, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.eye(n, dtype=dtype, device=self.device)

    def arange(self, start, stop=None, step=1, dtype=None):
        dtype = dtype or self._default_dtype
        if stop is None:
            return self._torch.arange(start, dtype=dtype, device=self.device)
        return self._torch.arange(start, stop, step, dtype=dtype, device=self.device)

    def full(self, shape, fill_value, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.full(shape, fill_value, dtype=dtype, device=self.device)

    def empty(self, shape, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.empty(shape, dtype=dtype, device=self.device)

    def linspace(self, start, stop, num, dtype=None):
        dtype = dtype or self._default_dtype
        return self._torch.linspace(start, stop, num, dtype=dtype, device=self.device)

    # --- Array manipulation ---
    def concatenate(self, arrays, axis=0):
        return self._torch.cat(arrays, dim=axis)

    def stack(self, arrays, axis=0):
        return self._torch.stack(arrays, dim=axis)

    def vstack(self, arrays):
        return self._torch.vstack(arrays)

    def hstack(self, arrays):
        return self._torch.hstack(arrays)

    def reshape(self, a, shape):
        return a.reshape(shape)

    def ravel(self, a):
        return a.ravel()

    def squeeze(self, a, axis=None):
        if axis is None:
            return a.squeeze()
        return a.squeeze(dim=axis)

    def expand_dims(self, a, axis):
        return a.unsqueeze(dim=axis)

    def diag(self, v, k=0):
        return self._torch.diag(v, diagonal=k)

    def diagonal(self, a, offset=0):
        return self._torch.diagonal(a, offset=offset)

    def triu_indices(self, n, k=0):
        return self._torch.triu_indices(n, n, offset=k)

    def tril_indices(self, n, k=0):
        return self._torch.tril_indices(n, n, offset=k)

    def where(self, condition, x=None, y=None):
        if x is None and y is None:
            return self._torch.where(condition)
        return self._torch.where(condition, x, y)

    def clip(self, a, a_min, a_max):
        return self._torch.clamp(a, min=a_min, max=a_max)

    def copy(self, a):
        return a.clone()

    def sort(self, a, axis=-1):
        return self._torch.sort(a, dim=axis).values

    def argsort(self, a, axis=-1):
        return self._torch.argsort(a, dim=axis)

    def argmax(self, a, axis=None):
        if axis is None:
            return self._torch.argmax(a)
        return self._torch.argmax(a, dim=axis)

    # --- Math operations ---
    def sqrt(self, x):
        return self._torch.sqrt(x)

    def exp(self, x):
        return self._torch.exp(x)

    def log(self, x):
        return self._torch.log(x)

    def abs(self, x):
        return self._torch.abs(x)

    def sign(self, x):
        return self._torch.sign(x)

    def maximum(self, x1, x2):
        return self._torch.maximum(x1, x2)

    def minimum(self, x1, x2):
        return self._torch.minimum(x1, x2)

    def sum(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.sum()
        return a.sum(dim=axis, keepdim=keepdims)

    def prod(self, a, axis=None):
        if axis is None:
            return a.prod()
        return a.prod(dim=axis)

    def mean(self, a, axis=None, keepdims=False):
        if axis is None:
            return a.mean()
        return a.mean(dim=axis, keepdim=keepdims)

    def var(self, a, axis=None, ddof=0):
        if axis is None:
            return a.var(correction=ddof)
        return a.var(dim=axis, correction=ddof)

    def cumsum(self, a, axis=None):
        if axis is None:
            return a.ravel().cumsum(dim=0)
        return a.cumsum(dim=axis)

    def pi(self):
        return math.pi

    def inf(self):
        return float("inf")

    # --- Linear algebra ---
    def dot(self, a, b):
        if a.dim() == 1 and b.dim() == 1:
            return self._torch.dot(a, b)
        return a @ b

    def matmul(self, a, b):
        return a @ b

    def outer(self, a, b):
        return self._torch.outer(a, b)

    def transpose(self, a):
        if a.dim() < 2:
            return a
        return a.T

    def solve(self, A, b):
        return self._torch.linalg.solve(A, b)

    def cholesky(self, A):
        return self._torch.linalg.cholesky(A)

    def det(self, A):
        return self._torch.linalg.det(A)

    def inv(self, A):
        return self._torch.linalg.inv(A)

    def eigh(self, A):
        return self._torch.linalg.eigh(A)

    def norm(self, a, ord=None, axis=None):
        return self._torch.linalg.norm(a, ord=ord, dim=axis)

    def ldl(self, A):
        """LDL^T decomposition via torch.linalg.ldl_factor."""
        factors, pivots, info = self._torch.linalg.ldl_factor_ex(A)
        # Extract L and D from the factored form
        n = A.shape[0]
        L = self._torch.tril(factors, diagonal=-1) + self._torch.eye(n, dtype=A.dtype, device=A.device)
        D = self._torch.diag(self._torch.diagonal(factors))
        return L, D, pivots

    # --- Statistical distributions ---
    def normal_pdf(self, x):
        """Standard normal PDF."""
        return self._torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def normal_cdf(self, x):
        """Standard normal CDF using erfc for numerical stability."""
        return 0.5 * self._torch.erfc(-x / math.sqrt(2.0))

    def normal_ppf(self, p):
        """Standard normal quantile (inverse CDF)."""
        return self._torch.erfinv(2.0 * p - 1.0) * math.sqrt(2.0)

    def normal_logpdf(self, x):
        """Standard normal log-PDF."""
        return -0.5 * (x * x + math.log(2.0 * math.pi))

    # --- Type checking ---
    def is_array(self, x):
        return isinstance(x, self._torch.Tensor)

    def to_numpy(self, x):
        if isinstance(x, self._torch.Tensor):
            return x.detach().cpu().numpy()
        import numpy as np
        return np.asarray(x)

    def astype(self, x, dtype):
        return x.to(dtype=dtype)
