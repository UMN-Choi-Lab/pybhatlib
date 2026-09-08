"""Tests for the PanelIndex Dmask primitive (pybhatlib.vecup._panel)."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.vecup._panel import PanelIndex


@pytest.fixture
def toy_ids() -> np.ndarray:
    # 3 persons, 7 obs: person 0 has 3 rows, person 1 has 2 rows,
    # person 2 has 2 rows.
    return np.array([0, 0, 0, 1, 1, 2, 2])


@pytest.fixture
def toy_panel(toy_ids: np.ndarray) -> PanelIndex:
    return PanelIndex.from_ids(toy_ids)


def _hand_built_mask() -> np.ndarray:
    # rows = obs (7), cols = person (3)
    return np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def test_from_ids_shapes(toy_panel: PanelIndex):
    assert toy_panel.n_obs == 7
    assert toy_panel.n_ind == 3
    np.testing.assert_array_equal(toy_panel.ids, np.array([0, 1, 2]))


def test_dense_mask_matches_hand_built(toy_panel: PanelIndex):
    dmask = toy_panel.mask()
    np.testing.assert_array_equal(dmask, _hand_built_mask())


def test_broadcast_matches_dmask_matmul(toy_panel: PanelIndex):
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((toy_panel.n_ind, 4))
    dmask = toy_panel.mask()

    expected = dmask @ draws
    result = toy_panel.broadcast(draws)

    np.testing.assert_allclose(result, expected)
    assert result.shape == (toy_panel.n_obs, 4)


def test_broadcast_1d(toy_panel: PanelIndex):
    person_vals = np.array([10.0, 20.0, 30.0])
    result = toy_panel.broadcast(person_vals)
    expected = np.array([10.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0])
    np.testing.assert_allclose(result, expected)


def test_scatter_sum_matches_dmask_transpose_matmul(toy_panel: PanelIndex):
    rng = np.random.default_rng(1)
    obs_vals = rng.standard_normal((toy_panel.n_obs, 5))
    dmask = toy_panel.mask()

    expected = dmask.T @ obs_vals
    result = toy_panel.scatter_sum(obs_vals)

    np.testing.assert_allclose(result, expected)
    assert result.shape == (toy_panel.n_ind, 5)


def test_scatter_sum_1d(toy_panel: PanelIndex):
    obs_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = toy_panel.scatter_sum(obs_vals)
    # person0: 1+2+3=6, person1: 4+5=9, person2: 6+7=13
    np.testing.assert_allclose(result, np.array([6.0, 9.0, 13.0]))


def test_logprod_matches_manual_product(toy_panel: PanelIndex):
    rng = np.random.default_rng(2)
    p_obs = rng.uniform(0.1, 0.9, size=toy_panel.n_obs)
    log_p_obs = np.log(p_obs)

    result = toy_panel.logprod(log_p_obs)

    expected = np.array(
        [
            p_obs[0] * p_obs[1] * p_obs[2],
            p_obs[3] * p_obs[4],
            p_obs[5] * p_obs[6],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_logprod_with_draws_axis(toy_panel: PanelIndex):
    rng = np.random.default_rng(3)
    ndraw = 6
    p_obs = rng.uniform(0.1, 0.9, size=(toy_panel.n_obs, ndraw))
    log_p_obs = np.log(p_obs)

    result = toy_panel.logprod(log_p_obs)
    assert result.shape == (toy_panel.n_ind, ndraw)

    expected = np.stack(
        [
            np.prod(p_obs[0:3, :], axis=0),
            np.prod(p_obs[3:5, :], axis=0),
            np.prod(p_obs[5:7, :], axis=0),
        ],
        axis=0,
    )
    np.testing.assert_allclose(result, expected)


def test_weightind_matches_manual_mean(toy_panel: PanelIndex):
    weights = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 100.0, 200.0])
    result = toy_panel.weightind(weights)
    expected = np.array([2.0, 15.0, 150.0])  # means within each person
    np.testing.assert_allclose(result, expected)


def test_degenerate_cross_sectional_one_obs_per_person():
    # One observation per person -> Dmask is a permuted identity matrix.
    ids = np.array([2, 0, 1])
    panel = PanelIndex.from_ids(ids)

    assert panel.n_obs == 3
    assert panel.n_ind == 3

    dmask = panel.mask()
    # each row and column has exactly one 1
    np.testing.assert_array_equal(dmask.sum(axis=0), np.ones(3))
    np.testing.assert_array_equal(dmask.sum(axis=1), np.ones(3))
    # it is a permutation matrix: dmask @ dmask.T == I (n_obs, n_obs)
    np.testing.assert_allclose(dmask @ dmask.T, np.eye(3))

    p_obs = np.array([0.3, 0.5, 0.8])
    log_p_obs = np.log(p_obs)
    result = panel.logprod(log_p_obs)
    # sorted ids are [0, 1, 2] -> person 0 is obs index 1 (id=0),
    # person 1 is obs index 2 (id=1), person 2 is obs index 0 (id=2).
    expected = np.array([p_obs[1], p_obs[2], p_obs[0]])
    np.testing.assert_allclose(result, expected)


def test_non_contiguous_rows_same_person():
    # Person rows interleaved (not contiguous) -> must still group correctly.
    ids = np.array([0, 1, 0, 1, 0])
    panel = PanelIndex.from_ids(ids)

    obs_vals = np.array([1.0, 10.0, 2.0, 20.0, 3.0])
    result = panel.scatter_sum(obs_vals)
    # person 0 (id=0): rows 0,2,4 -> 1+2+3=6; person 1 (id=1): rows 1,3 -> 30
    np.testing.assert_allclose(result, np.array([6.0, 30.0]))


def test_from_ids_rejects_empty():
    with pytest.raises(ValueError):
        PanelIndex.from_ids(np.array([]))


def test_from_ids_rejects_non_1d():
    with pytest.raises(ValueError):
        PanelIndex.from_ids(np.array([[0, 1], [2, 3]]))
