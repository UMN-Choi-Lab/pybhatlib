"""Tests for the DrawSource strategy (pybhatlib.mixed._draws).

Covers the fixed output contract ``(n_rep, n_ind, n_rnd)``, the GAUSS-order
panel reshape, the fixture round-trip, and scipy-Halton determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._draws import (
    DrawSource,
    FixtureDrawSource,
    ScipyHaltonDrawSource,
    panel_reshape_gauss,
)


# ---------------------------------------------------------------------------
# Hand-built GAUSS-order example
# ---------------------------------------------------------------------------

# Stored GAUSS ``ass`` block: n_rep=2, n_ind=3, n_rnd=2 -> 6 columns.
# Columns are coefficient-major, individual-minor:
#   [coef0_i0, coef0_i1, coef0_i2, coef1_i0, coef1_i1, coef1_i2]
# so that draws[r, i, k] == ass2d[r, k * n_ind + i].
_ASS2D = np.array(
    [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    ]
)
_N_REP, _N_IND, _N_RND = 2, 3, 2

# GAUSS errbeta1temp = reshape(ass[r,.], nrndcoef, nii)' , per replication.
_EXPECTED = np.array(
    [
        [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
    ]
)


def test_panel_reshape_matches_gauss_order():
    out = panel_reshape_gauss(_ASS2D, _N_IND, _N_RND)
    assert out.shape == (_N_REP, _N_IND, _N_RND)
    np.testing.assert_array_equal(out, _EXPECTED)


def test_panel_reshape_elementwise_formula():
    # draws[r, i, k] == ass2d[r, k * n_ind + i] for every element.
    out = panel_reshape_gauss(_ASS2D, _N_IND, _N_RND)
    for r in range(_N_REP):
        for i in range(_N_IND):
            for k in range(_N_RND):
                assert out[r, i, k] == _ASS2D[r, k * _N_IND + i]


def test_panel_reshape_bad_columns():
    with pytest.raises(ValueError):
        panel_reshape_gauss(_ASS2D, n_ind=4, n_rnd=2)  # 8 != 6


def test_panel_reshape_requires_2d():
    with pytest.raises(ValueError):
        panel_reshape_gauss(np.zeros(6), _N_IND, _N_RND)


# ---------------------------------------------------------------------------
# FixtureDrawSource
# ---------------------------------------------------------------------------

def test_fixture_from_array_contract_shape():
    src = FixtureDrawSource(_ASS2D)
    out = src.draws(_N_IND, _N_RND, _N_REP)
    assert out.shape == (_N_REP, _N_IND, _N_RND)
    np.testing.assert_array_equal(out, _EXPECTED)


def test_fixture_csv_roundtrip(tmp_path):
    # Hand-written ass CSV round-trips to the contract to 1e-12.
    csv_path = tmp_path / "ass.csv"
    ass = np.array(
        [
            [0.11, -0.22, 0.33, -0.44],   # n_ind=2, n_rnd=2 -> 4 cols
            [1.55, -2.66, 3.77, -4.88],
        ]
    )
    np.savetxt(csv_path, ass, delimiter=",")

    src = FixtureDrawSource(str(csv_path))
    out = src.draws(n_ind=2, n_rnd=2, n_rep=2)

    expected = panel_reshape_gauss(ass, 2, 2)
    assert out.shape == (2, 2, 2)
    np.testing.assert_allclose(out, expected, atol=1e-12, rtol=0.0)
    # And the raw stored values survive the CSV round-trip.
    np.testing.assert_allclose(
        np.sort(out.ravel()), np.sort(ass.ravel()), atol=1e-12, rtol=0.0
    )


def test_fixture_bad_nrep_raises():
    src = FixtureDrawSource(_ASS2D)
    with pytest.raises(ValueError):
        src.draws(_N_IND, _N_RND, n_rep=5)


def test_fixture_bad_dims_raises():
    src = FixtureDrawSource(_ASS2D)
    with pytest.raises(ValueError):
        src.draws(n_ind=4, n_rnd=2, n_rep=2)  # 8 != 6


def test_fixture_rejects_non_2d_array():
    with pytest.raises(ValueError):
        FixtureDrawSource(np.zeros((2, 2, 2)))


def test_fixture_run_is_ignored():
    src = FixtureDrawSource(_ASS2D)
    a = src.draws(_N_IND, _N_RND, _N_REP, run=1)
    b = src.draws(_N_IND, _N_RND, _N_REP, run=7)
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# ScipyHaltonDrawSource
# ---------------------------------------------------------------------------

def test_scipy_halton_contract_shape():
    src = ScipyHaltonDrawSource(seed=0)
    out = src.draws(n_ind=5, n_rnd=3, n_rep=4)
    assert out.shape == (4, 5, 3)
    assert np.all(np.isfinite(out))


def test_scipy_halton_deterministic_same_seed():
    a = ScipyHaltonDrawSource(seed=1234).draws(6, 2, 3)
    b = ScipyHaltonDrawSource(seed=1234).draws(6, 2, 3)
    np.testing.assert_array_equal(a, b)


def test_scipy_halton_differs_across_seeds():
    a = ScipyHaltonDrawSource(seed=1).draws(6, 2, 3)
    b = ScipyHaltonDrawSource(seed=2).draws(6, 2, 3)
    assert not np.allclose(a, b)


def test_scipy_halton_run_offsets_seed():
    src = ScipyHaltonDrawSource(seed=100)
    r1 = src.draws(6, 2, 3, run=1)
    r2 = src.draws(6, 2, 3, run=2)
    # run=1 reproduces the base seed exactly; run=2 gives different draws.
    np.testing.assert_array_equal(r1, ScipyHaltonDrawSource(seed=100).draws(6, 2, 3))
    assert not np.allclose(r1, r2)


def test_scipy_halton_run_must_be_positive():
    with pytest.raises(ValueError):
        ScipyHaltonDrawSource(seed=0).draws(6, 2, 3, run=0)


def test_scipy_halton_does_not_mutate_global_rng():
    before = np.random.get_state()
    ScipyHaltonDrawSource(seed=None).draws(4, 2, 2)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "src",
    [FixtureDrawSource(_ASS2D), ScipyHaltonDrawSource(seed=0)],
)
def test_sources_satisfy_protocol(src):
    assert isinstance(src, DrawSource)
