"""Gate for the MNP differencing (``dM``) matrix port (no GAUSS required).

Ports the behaviour of GAUSS ``proc dM`` (MNPKERCP.gss ~line 1365): the full map
holds ``-1`` in the chosen alternative's column and identity ``+1`` elsewhere, so
each row differences a non-chosen alternative against the chosen one
(``M @ V = V_other - V_chosen``), restricted to available non-chosen rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.models.mnpkercp._mnpkercp_diff import dm_matrix


def _hand_built(chosen_idx: int, nc: int) -> np.ndarray:
    """Reference all-available ``(nc-1) x nc`` map: -1 in chosen col, +1 else.

    Row ``k`` corresponds to the ``k``-th non-chosen alternative and equals
    ``e_{other_k} - e_{chosen}`` so that ``(M @ V)[k] = V_other_k - V_chosen``.
    """
    nonchosen = [j for j in range(nc) if j != chosen_idx]
    M = np.zeros((nc - 1, nc), dtype=float)
    for k, alt in enumerate(nonchosen):
        M[k, alt] = 1.0
        M[k, chosen_idx] = -1.0
    return M


@pytest.mark.parametrize("chosen_idx", [0, 1, 2])
def test_all_available_matches_hand_built(chosen_idx: int) -> None:
    nc = 3
    avail = np.ones(nc)
    M = dm_matrix(chosen_idx, avail, nc)
    assert M.shape == (nc - 1, nc)
    np.testing.assert_array_equal(M, _hand_built(chosen_idx, nc))


@pytest.mark.parametrize("chosen_idx", [0, 1, 2])
def test_all_available_differences_against_chosen(chosen_idx: int) -> None:
    nc = 3
    avail = np.ones(nc)
    V = np.array([3.0, 7.0, 2.0])
    M = dm_matrix(chosen_idx, avail, nc)
    diff = M @ V
    nonchosen = [j for j in range(nc) if j != chosen_idx]
    expected = np.array([V[alt] - V[chosen_idx] for alt in nonchosen])
    np.testing.assert_allclose(diff, expected)


def test_row_has_neg_one_in_chosen_column() -> None:
    nc = 3
    avail = np.ones(nc)
    for chosen_idx in range(nc):
        M = dm_matrix(chosen_idx, avail, nc)
        assert np.all(M[:, chosen_idx] == -1.0)
        # every other column across all rows sums to a single +1 (identity block)
        others = [j for j in range(nc) if j != chosen_idx]
        assert np.array_equal(M[:, others].sum(axis=0), np.ones(nc - 1))


def test_availability_restriction_drops_rows() -> None:
    # nc=4, chosen=0 available; one non-chosen alt (index 2) unavailable.
    nc = 4
    avail = np.array([1, 1, 0, 1])
    M = dm_matrix(0, avail, nc)
    # A = (#available - 1) = 3 - 1 = 2 rows.
    assert M.shape == (2, nc)
    # Retained rows correspond to available non-chosen alts 1 and 3.
    expected = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],  # alt 1 - chosen 0
            [-1.0, 0.0, 0.0, 1.0],  # alt 3 - chosen 0
        ]
    )
    np.testing.assert_array_equal(M, expected)


def test_availability_restriction_middle_chosen() -> None:
    # nc=4, chosen=2; drop non-chosen alt 0 (unavailable).
    nc = 4
    avail = np.array([0, 1, 1, 1])
    M = dm_matrix(2, avail, nc)
    assert M.shape == (2, nc)  # available = {1,2,3}, A = 2
    expected = np.array(
        [
            [0.0, 1.0, -1.0, 0.0],  # alt 1 - chosen 2
            [0.0, 0.0, -1.0, 1.0],  # alt 3 - chosen 2
        ]
    )
    np.testing.assert_array_equal(M, expected)


def test_row_count_equals_num_available_minus_one() -> None:
    nc = 5
    avail = np.array([1, 0, 1, 1, 0])  # 3 available, chosen must be available
    M = dm_matrix(0, avail, nc)
    assert M.shape[0] == int(avail.sum()) - 1


def test_rejects_unavailable_chosen() -> None:
    with pytest.raises(ValueError):
        dm_matrix(1, np.array([1, 0, 1]), 3)


def test_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError):
        dm_matrix(0, np.ones(4), 3)  # avail_mask wrong length
    with pytest.raises(ValueError):
        dm_matrix(5, np.ones(3), 3)  # chosen_idx out of range
    with pytest.raises(ValueError):
        dm_matrix(0, np.ones(1), 1)  # nc < 2


def test_xp_kwarg_backend() -> None:
    nc = 3
    M_default = dm_matrix(1, np.ones(nc), nc)
    M_xp = dm_matrix(1, np.ones(nc), nc, xp=np)
    np.testing.assert_array_equal(M_default, M_xp)
