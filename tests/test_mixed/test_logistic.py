"""Tests for pybhatlib.utils._logistic (BHATLIB cdlogit/pdlogit family)."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.utils._logistic import (
    cdlogit,
    d_lam_d_lamnew,
    gradpdlogit,
    lam_from_lamnew,
    pdlogit,
)

_H = 1e-4  # central finite-difference step used throughout


def _central_fd(f, x, h=_H):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def test_cdlogit_at_zero():
    assert cdlogit(0.0) == pytest.approx(0.5)


def test_lam_from_lamnew_at_zero():
    assert lam_from_lamnew(0.0) == pytest.approx(1.0)


def test_pdlogit_matches_fd_of_cdlogit():
    x = np.linspace(-8.0, 8.0, 33)
    fd = _central_fd(cdlogit, x)
    np.testing.assert_allclose(pdlogit(x), fd, atol=1e-8)


def test_gradpdlogit_matches_fd_of_pdlogit():
    x = np.linspace(-8.0, 8.0, 33)
    fd = _central_fd(pdlogit, x)
    np.testing.assert_allclose(gradpdlogit(x), fd, atol=1e-7)


def test_d_lam_d_lamnew_matches_fd_of_lam_from_lamnew():
    x = np.linspace(-8.0, 8.0, 33)
    fd = _central_fd(lam_from_lamnew, x)
    np.testing.assert_allclose(d_lam_d_lamnew(x), fd, atol=1e-8)


@pytest.mark.parametrize("x", [1000.0, -1000.0])
def test_overflow_safe(x):
    assert np.isfinite(cdlogit(x))
    assert np.isfinite(pdlogit(x))
    assert np.isfinite(gradpdlogit(x))
    assert np.isfinite(lam_from_lamnew(x))
    assert np.isfinite(d_lam_d_lamnew(x))

    # Analytically expected saturation behavior.
    expected_cd = 1.0 if x > 0 else 0.0
    assert cdlogit(x) == pytest.approx(expected_cd, abs=1e-12)
    assert pdlogit(x) == pytest.approx(0.0, abs=1e-12)
    assert gradpdlogit(x) == pytest.approx(0.0, abs=1e-12)
    assert lam_from_lamnew(x) == pytest.approx(2.0 * expected_cd, abs=1e-12)
    assert d_lam_d_lamnew(x) == pytest.approx(0.0, abs=1e-12)


def test_vectorized_shape_preserved():
    x = np.linspace(-5.0, 5.0, 11)
    assert cdlogit(x).shape == x.shape
    assert pdlogit(x).shape == x.shape
    assert gradpdlogit(x).shape == x.shape
    assert lam_from_lamnew(x).shape == x.shape
    assert d_lam_d_lamnew(x).shape == x.shape
