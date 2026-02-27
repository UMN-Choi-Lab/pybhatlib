"""Tests for MNPControl field alignment."""

from __future__ import annotations

import pytest

from pybhatlib.models.mnp import MNPControl


class TestMNPControlDefaults:
    def test_default_method_is_ovus(self):
        ctrl = MNPControl()
        assert ctrl.method == "ovus"

    def test_randdiag_field_exists(self):
        ctrl = MNPControl(randdiag=True)
        assert ctrl.randdiag is True

    def test_default_randdiag_is_false(self):
        ctrl = MNPControl()
        assert ctrl.randdiag is False

    def test_new_fields_have_defaults(self):
        ctrl = MNPControl()
        assert ctrl.spherical is True
        assert ctrl.scal == 1.0
        assert ctrl.IID_first is True
        assert ctrl.want_covariance is True
        assert ctrl.seed10 == 1234
        assert ctrl.perms == 0

    def test_rannddiag_removed(self):
        """Old field name 'rannddiag' should not exist."""
        ctrl = MNPControl()
        assert not hasattr(ctrl, "rannddiag")

    def test_method_can_be_set(self):
        ctrl = MNPControl(method="tvbs")
        assert ctrl.method == "tvbs"
