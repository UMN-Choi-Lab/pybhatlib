"""Logistic reparameterization helpers for unconstrained optimization.

Ports the logistic CDF/PDF utilities from BHATLIB (GAUSS ``cdlogit`` /
``pdlogit`` and friends, ``gradients mvn.src`` lines 6600-6622) used to map an
unconstrained real optimizer parameter onto a bounded interval, e.g. the
Yeo-Johnson power-transform parameter ``lambda in (0, 2)`` used by the MDCEV
linear (Bhat 2018) outside-good utility specification::

    lam = 2 * cdlogit(lamnew)

All functions here are elementwise, ``xp``-aware (defaulting to the NumPy
backend), and implemented with a numerically-stable logistic form (splitting
on the sign of ``x`` so that ``exp`` is never evaluated on a large positive
argument) so they stay finite for ``x`` of any magnitude and remain
Numba-JIT-friendly (plain arithmetic, no ``scipy.stats`` objects).

Notes
-----
The GAUSS source defines ``gradpdlogit`` (``gradients mvn.src:6620-6622``) as
``pdlogit(x) .* (1 - exp(-x)) / (1 + exp(-x))``. That expression is the
*negative* of the true derivative of the GAUSS ``pdlogit`` (verified against
central finite differences). :func:`gradpdlogit` here implements the
mathematically correct derivative ``d/dx pdlogit(x) = pdlogit(x) * (1 - 2 *
cdlogit(x))`` instead of the literal (sign-flipped) GAUSS expression.
"""

from __future__ import annotations

from numpy.typing import NDArray

from pybhatlib.backend._array_api import array_namespace

__all__ = [
    "cdlogit",
    "pdlogit",
    "gradpdlogit",
    "lam_from_lamnew",
    "d_lam_d_lamnew",
    "logitmod",
    "gradlogitmod",
]


def cdlogit(x: NDArray, *, xp=None) -> NDArray:
    """Standard logistic CDF, ``1 / (1 + exp(-x))``.

    Uses the numerically-stable split form (branching on the sign of ``x``)
    so that ``exp`` is only ever evaluated on a non-positive argument,
    keeping the result finite for arbitrarily large ``|x|``.

    Parameters
    ----------
    x : array_like
        Input values.
    xp : backend, optional
        Array backend. Inferred from ``x`` if not given.

    Returns
    -------
    ndarray
        ``cdlogit(x)``, elementwise, in ``(0, 1)``.
    """
    if xp is None:
        xp = array_namespace(x)

    x = xp.array(x, dtype=xp.float64)
    z = xp.exp(-xp.abs(x))
    pos = 1.0 / (1.0 + z)
    neg = z / (1.0 + z)
    return xp.where(x >= 0, pos, neg)


def pdlogit(x: NDArray, *, xp=None) -> NDArray:
    """Derivative of :func:`cdlogit`: the logistic PDF.

    ``pdlogit(x) = cdlogit(x) * (1 - cdlogit(x))``
    (GAUSS ref: ``gradients mvn.src:6616-6618``).

    Parameters
    ----------
    x : array_like
        Input values.
    xp : backend, optional
        Array backend. Inferred from ``x`` if not given.

    Returns
    -------
    ndarray
        ``pdlogit(x)``, elementwise, in ``(0, 0.25]``.
    """
    if xp is None:
        xp = array_namespace(x)

    p = cdlogit(x, xp=xp)
    return p * (1.0 - p)


def gradpdlogit(x: NDArray, *, xp=None) -> NDArray:
    """Derivative of :func:`pdlogit`, i.e. ``d^2/dx^2 cdlogit(x)``.

    ``gradpdlogit(x) = pdlogit(x) * (1 - 2 * cdlogit(x))``. See the module
    docstring for why this differs in sign from the literal GAUSS
    ``gradpdlogit`` expression.

    Parameters
    ----------
    x : array_like
        Input values.
    xp : backend, optional
        Array backend. Inferred from ``x`` if not given.

    Returns
    -------
    ndarray
        ``d/dx pdlogit(x)``, elementwise.
    """
    if xp is None:
        xp = array_namespace(x)

    c = cdlogit(x, xp=xp)
    p = c * (1.0 - c)
    return p * (1.0 - 2.0 * c)


def lam_from_lamnew(lamnew: NDArray, *, xp=None) -> NDArray:
    """Map an unconstrained real to the Yeo-Johnson ``lambda in (0, 2)``.

    ``lam = 2 * cdlogit(lamnew)`` (GAUSS ref: ``vecup.src:2221``, used in the
    MDCEV linear/Bhat-2018 outside-good power transform).

    Parameters
    ----------
    lamnew : array_like
        Unconstrained optimizer parameter.
    xp : backend, optional
        Array backend. Inferred from ``lamnew`` if not given.

    Returns
    -------
    ndarray
        ``lam``, elementwise, in ``(0, 2)``.
    """
    if xp is None:
        xp = array_namespace(lamnew)

    return 2.0 * cdlogit(lamnew, xp=xp)


def d_lam_d_lamnew(lamnew: NDArray, *, xp=None) -> NDArray:
    """Chain-rule derivative of :func:`lam_from_lamnew`.

    ``d(lam)/d(lamnew) = 2 * pdlogit(lamnew)`` (GAUSS ref: ``vecup.src:2268``).

    Parameters
    ----------
    lamnew : array_like
        Unconstrained optimizer parameter.
    xp : backend, optional
        Array backend. Inferred from ``lamnew`` if not given.

    Returns
    -------
    ndarray
        ``d(lam)/d(lamnew)``, elementwise.
    """
    if xp is None:
        xp = array_namespace(lamnew)

    return 2.0 * pdlogit(lamnew, xp=xp)


def logitmod(a: NDArray, *, xp=None) -> NDArray:
    """Multinomial-logit (softmax) probabilities ``exp(a) / sum(exp(a))``.

    Ports GAUSS ``logitmod`` (``gradients mvn.src:6761``). Given a vector of
    ``K`` utilities ``a``, returns the ``K`` softmax probabilities. Used by the
    MNP kernel-scale reparameterization, where the scale vector is
    ``wker = sqrt(logitmod(xscalker))[1:]`` (dropping the first entry, the
    sum-of-squares normalization).

    A numerically-stable form is used (subtracting ``max(a)`` before
    exponentiating) so ``exp`` is never evaluated on a large positive argument;
    this is mathematically identical to the literal GAUSS expression and keeps
    the result finite for ``a`` of any magnitude. Uses only plain arithmetic
    (no ``scipy.stats`` objects) so it stays Numba-JIT-friendly.

    Parameters
    ----------
    a : array_like, shape (K,)
        Utility values.
    xp : backend, optional
        Array backend. Inferred from ``a`` if not given.

    Returns
    -------
    ndarray, shape (K,)
        Softmax probabilities, non-negative and summing to 1.
    """
    if xp is None:
        xp = array_namespace(a)

    a = xp.array(a, dtype=xp.float64)
    e = xp.exp(a - a[xp.argmax(a)])
    return e / xp.sum(e)


def gradlogitmod(a: NDArray, *, xp=None):
    """Softmax probabilities and their Jacobian.

    Ports GAUSS ``gradlogitmod`` (``gradients mvn.src:6791``), which returns two
    outputs: the softmax probabilities ``F = logitmod(a)`` and the ``K x K``
    Jacobian ``ga`` with ``ga[i, j] = d(pi_j) / d(a_i)``:

    - diagonal ``ga[i, i] = pi_i * (1 - pi_i)``,
    - off-diagonal ``ga[i, j] = -pi_i * pi_j`` (for ``i != j``).

    Equivalently ``ga = diag(F) - outer(F, F)``. The Jacobian is symmetric, so
    the GAUSS "``d(pi_col) / d(a_row)``" layout and the transpose coincide.

    Parameters
    ----------
    a : array_like, shape (K,)
        Utility values.
    xp : backend, optional
        Array backend. Inferred from ``a`` if not given.

    Returns
    -------
    F : ndarray, shape (K,)
        Softmax probabilities (same as :func:`logitmod`).
    ga : ndarray, shape (K, K)
        Jacobian ``d(pi_j) / d(a_i)``.
    """
    if xp is None:
        xp = array_namespace(a)

    F = logitmod(a, xp=xp)
    ga = xp.diag(F) - xp.outer(F, F)
    return F, ga
