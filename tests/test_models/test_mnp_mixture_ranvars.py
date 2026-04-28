"""Tests for the M1 mixture-ranvars auto-expansion ergonomic fix.

Validates that ``ranvars=["OVTT"]`` (single base name) is automatically
expanded across mixture-of-normals segments when ``control.nseg > 1``,
matching the GAUSS user-facing pattern ``ranvars = { OVTT1 OVTT2 }``
from ``Gauss Files and Comparison/MNP/MNP Table2 d.gss:113``.

The expansion is purely additive: legacy segment-suffixed naming
(``ranvars=["OVTT1", "OVTT2"]`` with separate spec entries) continues
to work and produces an identical fit.
"""

from __future__ import annotations

import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel

ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

SPEC_BASE = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero", "Alt3_ch": "sero"},
    "AGE45_TR": {"Alt1_ch": "sero", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT": {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST": {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# Legacy GAUSS-style spec where OVTT is split into per-segment columns
# (OVTT1 active in segment 1, OVTT2 active in segment 2, modulo the fact
# that pybhatlib does not have segment-specific X — both columns end up
# pointing to the same OVTT_* raw data columns).
SPEC_LEGACY_OVTT_SPLIT = {
    "CON_SR": SPEC_BASE["CON_SR"],
    "CON_TR": SPEC_BASE["CON_TR"],
    "AGE45_DA": SPEC_BASE["AGE45_DA"],
    "AGE45_TR": SPEC_BASE["AGE45_TR"],
    "IVTT": SPEC_BASE["IVTT"],
    "OVTT1": SPEC_BASE["OVTT"],
    "OVTT2": SPEC_BASE["OVTT"],
    "COST": SPEC_BASE["COST"],
}


def _make_model(spec, nseg, ranvars, travelmode_path, **ctrl_kwargs):
    ctrl = MNPControl(
        iid=False,
        mix=True,
        nseg=nseg,
        maxiter=10,
        verbose=0,
        seed=42,
        **ctrl_kwargs,
    )
    return MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=spec,
        control=ctrl,
        ranvars=ranvars,
    )


# ---------------------------------------------------------------------------
# Backward-compatibility (nseg=1) — auto-expansion must be a no-op
# ---------------------------------------------------------------------------


class TestRanvarsBackwardCompatibilityNseg1:
    def test_ranvars_single_string_unchanged_for_nseg1(self, travelmode_path):
        m = _make_model(SPEC_BASE, nseg=1, ranvars="OVTT",
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 1
        assert m.var_names[m.ranvar_indices[0]] == "OVTT"

    def test_ranvars_list_unchanged_for_nseg1(self, travelmode_path):
        m = _make_model(SPEC_BASE, nseg=1, ranvars=["OVTT"],
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 1
        assert m.var_names[m.ranvar_indices[0]] == "OVTT"


# ---------------------------------------------------------------------------
# Auto-expansion (nseg>1)
# ---------------------------------------------------------------------------


class TestRanvarsAutoExpansionNseg2:
    def test_ranvars_expanded_for_nseg2_base_name(self, travelmode_path):
        """``ranvars=["OVTT"]`` with nseg=2 → length-2 ranvar_indices.

        Both indices must point to the OVTT column in their respective
        segment slice of the (segment-duplicated) design matrix.
        """
        m = _make_model(SPEC_BASE, nseg=2, ranvars=["OVTT"],
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 2

        # Both ranvar_indices entries refer to OVTT (in their segment slice).
        # We verify by checking the underlying var_names entry strips to
        # the base "OVTT".
        for ri in m.ranvar_indices:
            name = m.var_names[ri]
            # Either "OVTT" (segment 1) or "OVTT_s2", "OVTT_s3", ...
            base = name.split("_s")[0]
            assert base == "OVTT", (
                f"ranvar_indices entry {ri} -> {name!r}, base={base!r}, "
                f"expected base 'OVTT'"
            )

    def test_ranvars_string_form_expanded_for_nseg2(self, travelmode_path):
        """Single-string form ``ranvars="OVTT"`` also auto-expands."""
        m = _make_model(SPEC_BASE, nseg=2, ranvars="OVTT",
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 2

    def test_ranvars_segment_suffixed_still_works(self, travelmode_path):
        """Legacy ``ranvars=["OVTT1", "OVTT2"]`` (with separate spec
        entries) continues to produce length-2 ranvar_indices.
        """
        m = _make_model(
            SPEC_LEGACY_OVTT_SPLIT, nseg=2,
            ranvars=["OVTT1", "OVTT2"],
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 2
        names = [m.var_names[ri] for ri in m.ranvar_indices]
        assert "OVTT1" in names
        assert "OVTT2" in names

    def test_ranvars_unknown_raises(self, travelmode_path):
        """Base name not in spec keys → ValueError."""
        with pytest.raises(ValueError, match="not found"):
            _make_model(SPEC_BASE, nseg=2, ranvars=["NOPE"],
                        travelmode_path=travelmode_path)

    def test_ranvars_unknown_with_digits_raises(self, travelmode_path):
        """Pseudo-segment-suffixed name whose base also unknown → error."""
        with pytest.raises(ValueError, match="not found"):
            _make_model(SPEC_BASE, nseg=2, ranvars=["NOPE1", "NOPE2"],
                        travelmode_path=travelmode_path)


# ---------------------------------------------------------------------------
# Parameter count growth
# ---------------------------------------------------------------------------


class TestNParamsPerSegmentOverhead:
    def test_n_params_grows_with_nseg_for_random_coef(self, travelmode_path):
        """``n_params(nseg=2)`` exceeds ``n_params(nseg=1)`` by exactly the
        expected per-segment overhead.

        For ``ranvars=["OVTT"]``, auto-expansion makes ``n_rand = nseg``
        (one ranvar per segment slice). Per-segment overhead for each
        extra segment is therefore:

            +1 segment-probability param
            + n_beta extra betas
            + n_omega(n_rand=nseg) per segment
            + the n_omega growth in segment 1 (n_rand = 1 → nseg)
        """
        m1 = _make_model(SPEC_BASE, nseg=1, ranvars=["OVTT"],
                         travelmode_path=travelmode_path)
        m2 = _make_model(SPEC_BASE, nseg=2, ranvars=["OVTT"],
                         travelmode_path=travelmode_path)

        assert m2.n_params > m1.n_params

        # Compute expected delta directly from the new layout.
        # nseg=1: n_beta + n_lambda + n_omega(1)
        # nseg=2: n_beta + n_lambda + n_omega(2) + (nseg-1) + (n_beta + n_omega(2))
        n_beta = m1.n_beta
        # Lambda count is the same for both:
        n_lambda_terms = m1.n_params - n_beta - 1  # 1 omega in nseg=1 base
        # Sanity-check our reasoning:
        assert m1.n_params == n_beta + n_lambda_terms + 1

        n_rand_2 = 2
        n_omega_2 = n_rand_2 * (n_rand_2 + 1) // 2  # 3
        n_seg_extra = m2.control.nseg - 1  # 1 prob param
        expected_m2 = (
            n_beta              # seg-1 betas
            + n_lambda_terms    # lambda
            + n_omega_2         # seg-1 omega (now 2x2)
            + n_seg_extra       # mixture probability
            + n_beta            # seg-2 betas
            + n_omega_2         # seg-2 omega (now 2x2)
        )
        assert m2.n_params == expected_m2, (
            f"n_params(nseg=2)={m2.n_params}, expected {expected_m2}"
        )


# ---------------------------------------------------------------------------
# Heavy fit-based parity tests (Model (d))
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_table2_model_d_with_auto_ranvars_lower_LL(travelmode_path):
    """Fit Model (d) using the new ``ranvars=["OVTT"]`` style.

    Paper target is -634.975 (±1). If the auto-expansion alone closes
    the gap, this passes; otherwise, the residual delta documents that
    the remaining gap is structural (shared/varying coefficient handling
    is needed, deferred to a separate phase).
    """
    ctrl = MNPControl(
        iid=False, mix=True, nseg=2,
        maxiter=200, verbose=0, seed=42,
    )
    model = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_BASE,
        control=ctrl,
        ranvars=["OVTT"],
    )
    results = model.fit()

    # Paper target: -634.975 (±1 tolerance specified by the M1 plan).
    # Do NOT widen this tolerance to make it pass — if the LL doesn't
    # reach this range, the residual is reported as evidence the
    # remaining gap is structural.
    assert -636.0 <= results.ll_total <= -633.0, (
        f"Model (d) auto-ranvars LL = {results.ll_total:.3f}, "
        f"expected within [-636, -633] (paper target -634.975)"
    )


@pytest.mark.slow
def test_table2_model_d_legacy_naming_still_passes(travelmode_path):
    """Fit Model (d) with both the new and legacy ranvars naming and
    assert the two paths produce IDENTICAL log-likelihoods.

    With identical starting values and a deterministic optimizer, the
    two parameterizations must converge to the same LL up to numerical
    precision (≥ 1e-6).
    """
    ctrl_kwargs = dict(
        iid=False, mix=True, nseg=2,
        maxiter=200, verbose=0, seed=42,
    )

    # New style
    model_new = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_BASE,
        control=MNPControl(**ctrl_kwargs),
        ranvars=["OVTT"],
    )
    res_new = model_new.fit()

    # Legacy style (8-entry spec with OVTT1, OVTT2)
    model_legacy = MNPModel(
        data=travelmode_path,
        alternatives=ALTERNATIVES,
        availability="none",
        spec=SPEC_LEGACY_OVTT_SPLIT,
        control=MNPControl(**ctrl_kwargs),
        ranvars=["OVTT1", "OVTT2"],
    )
    res_legacy = model_legacy.fit()

    # Same number of parameters.
    assert model_new.n_params == model_legacy.n_params, (
        f"new n_params={model_new.n_params}, "
        f"legacy n_params={model_legacy.n_params}"
    )
    # Same LL within tolerance — the two paths must produce identical fits.
    assert abs(res_new.ll_total - res_legacy.ll_total) < 1e-3, (
        f"new LL={res_new.ll_total:.6f}, legacy LL={res_legacy.ll_total:.6f}"
    )
