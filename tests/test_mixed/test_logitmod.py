"""Tests for :func:`logitmod` / :func:`gradlogitmod`.

Ports of GAUSS ``logitmod`` / ``gradlogitmod`` (``gradients mvn.src:6761,6791``),
the sum-of-squares softmax kernel-scale reparameterization used by the MNP
kernel scale ``wker = sqrt(logitmod(xscalker))[1:]``.

Validation:

1. ``logitmod`` is a valid probability vector (non-negative, sums to 1) and
   matches a live-GAUSS reference (GAUSS 26.1.1) to ~1e-9 for a fixed ``a``.
2. ``gradlogitmod`` returns that same probability vector plus its Jacobian,
   verified against central finite differences of ``logitmod`` (~1e-8) and
   against the same live-GAUSS reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pybhatlib.utils._logistic import gradlogitmod, logitmod

# Fixed input used to generate the live-GAUSS reference CSVs (see module docstr).
A_FIXED = np.array([0.3, -1.2, 2.5, 0.7, -0.45], dtype=np.float64)

# Live-GAUSS 26.1.1 reference outputs for A_FIXED (from tgauss -b ref_logitmod.gss,
# csvWriteM). w = logitmod(a); ga = second output of gradlogitmod(a).
GAUSS_W = np.array(
    [
        0.0818844231977322,
        0.0182708844617718,
        0.73900802475291,
        0.122157204878487,
        0.0386794627090992,
    ],
    dtype=np.float64,
)

GAUSS_GA = np.array(
    [
        [0.0751793644353069, -0.00149610083546459, -0.0605132458453875,
         -0.0100027722609221, -0.00316724549353278],
        [-0.00149610083546459, 0.0179370592427564, -0.0135023302365826,
         -0.00223192017650782, -0.000706707994201362],
        [-0.0605132458453875, -0.0135023302365826, 0.192875164103712,
         -0.090275154686587, -0.0285844333351552],
        [-0.0100027722609221, -0.00223192017650782, -0.090275154686587,
         0.107234822174762, -0.00472497505074521],
        [-0.00316724549353278, -0.000706707994201362, -0.0285844333351552,
         -0.00472497505074521, 0.0371833618736346],
    ],
    dtype=np.float64,
)


def test_logitmod_is_probability_vector():
    """logitmod is non-negative and sums to exactly 1."""
    for a in (A_FIXED, np.array([0.0, 0.0, 0.0]), np.array([100.0, 99.0, -50.0])):
        w = logitmod(a)
        assert np.all(w >= 0.0)
        assert w.sum() == pytest.approx(1.0, abs=1e-15)


def test_logitmod_matches_gauss():
    """logitmod matches live-GAUSS reference to ~1e-9."""
    w = logitmod(A_FIXED)
    np.testing.assert_allclose(w, GAUSS_W, atol=1e-9, rtol=0.0)


def test_gradlogitmod_returns_probabilities():
    """First output of gradlogitmod equals logitmod / the GAUSS reference."""
    F, _ = gradlogitmod(A_FIXED)
    np.testing.assert_allclose(F, logitmod(A_FIXED), atol=1e-15, rtol=0.0)
    np.testing.assert_allclose(F, GAUSS_W, atol=1e-9, rtol=0.0)


def test_gradlogitmod_jacobian_matches_gauss():
    """gradlogitmod Jacobian matches live-GAUSS reference to ~1e-9."""
    _, ga = gradlogitmod(A_FIXED)
    np.testing.assert_allclose(ga, GAUSS_GA, atol=1e-9, rtol=0.0)


def test_gradlogitmod_jacobian_vs_finite_difference():
    """ga[i, j] = d(pi_j)/d(a_i) matches central finite differences to ~1e-8."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=6)
        _, ga = gradlogitmod(a)

        h = 1e-6
        fd = np.zeros_like(ga)
        for i in range(a.size):
            ap = a.copy()
            am = a.copy()
            ap[i] += h
            am[i] -= h
            # column j = d(pi_j)/d(a_i) -> stored at ga[i, j]
            fd[i, :] = (logitmod(ap) - logitmod(am)) / (2.0 * h)

        np.testing.assert_allclose(ga, fd, atol=1e-8, rtol=1e-6)


def test_gradlogitmod_jacobian_symmetric():
    """The Jacobian is symmetric, so GAUSS layout == its transpose."""
    _, ga = gradlogitmod(A_FIXED)
    np.testing.assert_allclose(ga, ga.T, atol=1e-15, rtol=0.0)
