"""Property and regression tests for centralized safe reparameterizations."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.utils._safe_reparam import (
    _CORR_LIMIT,
    corr_from_angle,
    nearest_pd_correlation,
    safe_cholesky,
    safe_exp,
)


def test_safe_exp_matches_numpy_exactly_in_band() -> None:
    x = np.linspace(-50.0, 50.0, 1001)
    np.testing.assert_array_equal(safe_exp(x), np.exp(x))


@pytest.mark.parametrize("value", [-1e6, -1e3, 1e3, 1e6])
def test_safe_exp_is_finite_for_extreme_inputs(value: float) -> None:
    assert np.isfinite(safe_exp(value))


def test_corr_from_angle_matches_naive_in_identifiable_band() -> None:
    theta = np.linspace(-20.0, 20.0, 1001)
    naive = (np.exp(theta) - 1.0) / (np.exp(theta) + 1.0)
    np.testing.assert_allclose(corr_from_angle(theta, 1.0), naive, rtol=2e-15, atol=0.0)


def test_corr_from_angle_stays_strictly_inside_unit_interval() -> None:
    corr = corr_from_angle(np.array([-1e6, 1e6]), 1.0)
    np.testing.assert_array_equal(np.abs(corr), np.full(2, _CORR_LIMIT))


def test_nearest_pd_correlation_repairs_indefinite_direct_entries() -> None:
    raw = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, -0.99], [0.99, -0.99, 1.0]])
    repaired = nearest_pd_correlation(raw)
    np.testing.assert_allclose(repaired, repaired.T, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(np.diag(repaired), np.ones(3))
    assert np.linalg.eigvalsh(repaired).min() > 0.0


def test_nearest_pd_correlation_is_noop_for_well_conditioned_input() -> None:
    corr = np.array([[1.0, 0.25], [0.25, 1.0]])
    np.testing.assert_array_equal(nearest_pd_correlation(corr), corr)


def test_safe_cholesky_reports_zero_jitter_for_pd_input() -> None:
    corr = np.array([[1.0, 0.25], [0.25, 1.0]])
    chol, jitter = safe_cholesky(corr)
    assert jitter == 0.0
    np.testing.assert_array_equal(chol, np.linalg.cholesky(corr))


def test_safe_cholesky_repairs_singular_input_and_reports_jitter() -> None:
    chol, jitter = safe_cholesky(np.ones((2, 2)))
    assert jitter > 0.0
    assert np.all(np.isfinite(chol))
    assert np.linalg.eigvalsh(chol @ chol.T).min() > 0.0


def test_safe_cholesky_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        safe_cholesky(np.array([[1.0, np.nan], [np.nan, 1.0]]))


@pytest.mark.parametrize("value", [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_safe_exp_central_difference_matches_analytic_gradient(value: float) -> None:
    step = 1e-5
    finite_difference = (safe_exp(value + step) - safe_exp(value - step)) / (2.0 * step)
    np.testing.assert_allclose(finite_difference, safe_exp(value), rtol=2e-10, atol=1e-11)


@pytest.mark.parametrize("value", [-10.0, -1.0, 0.0, 1.0, 10.0])
def test_corr_central_difference_matches_analytic_gradient(value: float) -> None:
    step = 1e-5
    corr = corr_from_angle(value, 1.0)
    finite_difference = (
        corr_from_angle(value + step, 1.0) - corr_from_angle(value - step, 1.0)
    ) / (2.0 * step)
    np.testing.assert_allclose(finite_difference, 0.5 * (1.0 - corr**2), rtol=2e-7)
