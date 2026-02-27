"""NumPy + SciPy backend implementation."""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.stats


class NumpyBackend:
    """Backend wrapping NumPy + SciPy for array operations."""

    name = "numpy"
    float64 = np.float64
    float32 = np.float32
    int64 = np.int64

    # --- Array creation ---
    @staticmethod
    def array(data, dtype=None):
        return np.asarray(data, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype=np.float64):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def ones(shape, dtype=np.float64):
        return np.ones(shape, dtype=dtype)

    @staticmethod
    def eye(n, dtype=np.float64):
        return np.eye(n, dtype=dtype)

    @staticmethod
    def arange(start, stop=None, step=1, dtype=None):
        if stop is None:
            return np.arange(start, dtype=dtype)
        return np.arange(start, stop, step, dtype=dtype)

    @staticmethod
    def full(shape, fill_value, dtype=np.float64):
        return np.full(shape, fill_value, dtype=dtype)

    @staticmethod
    def empty(shape, dtype=np.float64):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def linspace(start, stop, num, dtype=np.float64):
        return np.linspace(start, stop, num, dtype=dtype)

    # --- Array manipulation ---
    @staticmethod
    def concatenate(arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    @staticmethod
    def stack(arrays, axis=0):
        return np.stack(arrays, axis=axis)

    @staticmethod
    def vstack(arrays):
        return np.vstack(arrays)

    @staticmethod
    def hstack(arrays):
        return np.hstack(arrays)

    @staticmethod
    def reshape(a, shape):
        return np.reshape(a, shape)

    @staticmethod
    def ravel(a):
        return np.ravel(a)

    @staticmethod
    def squeeze(a, axis=None):
        return np.squeeze(a, axis=axis)

    @staticmethod
    def expand_dims(a, axis):
        return np.expand_dims(a, axis)

    @staticmethod
    def diag(v, k=0):
        return np.diag(v, k=k)

    @staticmethod
    def diagonal(a, offset=0):
        return np.diagonal(a, offset=offset)

    @staticmethod
    def triu_indices(n, k=0):
        return np.triu_indices(n, k=k)

    @staticmethod
    def tril_indices(n, k=0):
        return np.tril_indices(n, k=k)

    @staticmethod
    def where(condition, x=None, y=None):
        if x is None and y is None:
            return np.where(condition)
        return np.where(condition, x, y)

    @staticmethod
    def clip(a, a_min, a_max):
        return np.clip(a, a_min, a_max)

    @staticmethod
    def copy(a):
        return np.copy(a)

    @staticmethod
    def sort(a, axis=-1):
        return np.sort(a, axis=axis)

    @staticmethod
    def argsort(a, axis=-1):
        return np.argsort(a, axis=axis)

    @staticmethod
    def argmax(a, axis=None):
        return np.argmax(a, axis=axis)

    # --- Math operations ---
    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def exp(x):
        return np.exp(x)

    @staticmethod
    def log(x):
        return np.log(x)

    @staticmethod
    def abs(x):
        return np.abs(x)

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def maximum(x1, x2):
        return np.maximum(x1, x2)

    @staticmethod
    def minimum(x1, x2):
        return np.minimum(x1, x2)

    @staticmethod
    def sum(a, axis=None, keepdims=False):
        return np.sum(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def prod(a, axis=None):
        return np.prod(a, axis=axis)

    @staticmethod
    def mean(a, axis=None, keepdims=False):
        return np.mean(a, axis=axis, keepdims=keepdims)

    @staticmethod
    def var(a, axis=None, ddof=0):
        return np.var(a, axis=axis, ddof=ddof)

    @staticmethod
    def cumsum(a, axis=None):
        return np.cumsum(a, axis=axis)

    @staticmethod
    def pi(self=None):
        return np.pi

    @staticmethod
    def inf(self=None):
        return np.inf

    # --- Linear algebra ---
    @staticmethod
    def dot(a, b):
        return np.dot(a, b)

    @staticmethod
    def matmul(a, b):
        return a @ b

    @staticmethod
    def outer(a, b):
        return np.outer(a, b)

    @staticmethod
    def transpose(a):
        return a.T

    @staticmethod
    def solve(A, b):
        return scipy.linalg.solve(A, b)

    @staticmethod
    def cholesky(A):
        return scipy.linalg.cholesky(A, lower=True)

    @staticmethod
    def det(A):
        return scipy.linalg.det(A)

    @staticmethod
    def inv(A):
        return scipy.linalg.inv(A)

    @staticmethod
    def eigh(A):
        return scipy.linalg.eigh(A)

    @staticmethod
    def norm(a, ord=None, axis=None):
        return scipy.linalg.norm(a, ord=ord, axis=axis)

    @staticmethod
    def ldl(A):
        """LDL^T decomposition. Returns (L, D, perm)."""
        return scipy.linalg.ldl(A)

    # --- Statistical distributions ---
    @staticmethod
    def normal_pdf(x):
        """Standard normal PDF."""
        return scipy.stats.norm.pdf(x)

    @staticmethod
    def normal_cdf(x):
        """Standard normal CDF."""
        return scipy.stats.norm.cdf(x)

    @staticmethod
    def normal_ppf(p):
        """Standard normal quantile (inverse CDF)."""
        return scipy.stats.norm.ppf(p)

    @staticmethod
    def normal_logpdf(x):
        """Standard normal log-PDF."""
        return scipy.stats.norm.logpdf(x)

    # --- Type checking ---
    @staticmethod
    def is_array(x):
        return isinstance(x, np.ndarray)

    @staticmethod
    def to_numpy(x):
        return np.asarray(x)

    @staticmethod
    def astype(x, dtype):
        return x.astype(dtype)
