"""Tests for vecup vector/matrix operations against BHATLIB paper examples."""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.vecup import (
    vecdup,
    vecndup,
    matdupfull,
    matdupdiagonefull,
    vecsymmetry,
    nondiag,
)


class TestVecdup:
    def test_paper_example(self, sym_3x3):
        """BHATLIB paper p.6: vecdup([[1,2,3],[2,4,5],[3,5,6]]) -> [1,2,3,4,5,6]"""
        result = vecdup(sym_3x3)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5, 6])

    def test_2x2(self):
        M = np.array([[1.0, 0.5], [0.5, 2.0]])
        np.testing.assert_array_equal(vecdup(M), [1.0, 0.5, 2.0])

    def test_1x1(self):
        M = np.array([[5.0]])
        np.testing.assert_array_equal(vecdup(M), [5.0])


class TestVecndup:
    def test_paper_example(self, sym_3x3):
        """BHATLIB paper p.6: vecndup -> [2, 3, 5]"""
        result = vecndup(sym_3x3)
        np.testing.assert_array_equal(result, [2, 3, 5])

    def test_2x2(self):
        M = np.array([[1.0, 0.7], [0.7, 2.0]])
        np.testing.assert_array_equal(vecndup(M), [0.7])


class TestMatdupfull:
    def test_paper_example(self):
        """BHATLIB paper p.6: [1,2,3,4,5,6] -> [[1,2,3],[2,4,5],[3,5,6]]"""
        result = matdupfull(np.array([1, 2, 3, 4, 5, 6], dtype=float))
        expected = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_roundtrip(self, sym_3x3):
        """matdupfull(vecdup(A)) should recover A for symmetric A."""
        v = vecdup(sym_3x3)
        recovered = matdupfull(v)
        np.testing.assert_array_equal(recovered, sym_3x3)

    def test_2x2(self):
        result = matdupfull(np.array([1.0, 0.5, 2.0]))
        expected = np.array([[1.0, 0.5], [0.5, 2.0]])
        np.testing.assert_array_equal(result, expected)


class TestMatdupdiagonefull:
    def test_paper_example(self):
        """BHATLIB paper p.6: [0.6, 0.5, 0.5] -> [[1,0.6,0.5],[0.6,1,0.5],[0.5,0.5,1]]"""
        result = matdupdiagonefull(np.array([0.6, 0.5, 0.5]))
        expected = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.5], [0.5, 0.5, 1.0]])
        np.testing.assert_allclose(result, expected)

    def test_unit_diagonal(self):
        result = matdupdiagonefull(np.array([0.3]))
        np.testing.assert_allclose(result, [[1.0, 0.3], [0.3, 1.0]])
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0


class TestVecsymmetry:
    def test_shape(self, sym_3x3):
        S = vecsymmetry(sym_3x3)
        assert S.shape == (6, 9)  # 3*(3+1)/2 x 3*3

    def test_diagonal_element(self):
        M = np.eye(2)
        S = vecsymmetry(M)
        # S should be (3, 4): 3 upper-tri elements, 4 full elements
        assert S.shape == (3, 4)
        # First row (element [0,0]): only position 0 in flattened
        np.testing.assert_array_equal(S[0], [1, 0, 0, 0])


class TestNondiag:
    def test_3x3(self):
        M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        result = nondiag(M)
        np.testing.assert_array_equal(result, [2, 3, 4, 6, 7, 8])

    def test_2x2(self):
        M = np.array([[1, 2], [3, 4]], dtype=float)
        result = nondiag(M)
        np.testing.assert_array_equal(result, [2, 3])
