"""Tests for mixture-of-normals ``ranvars`` handling (shared coefficients).

Under the shared/varying layout (docs/plans/MIXTURE_SHARED_COEFFICIENTS_PLAN.md)
a single base name ``ranvars=["OVTT"]`` with ``control.nseg > 1`` gives OVTT
one segment-specific coefficient/Omega slot per segment while every other
coefficient is shared across segments; the model resolves the name to ONE
column index (no auto-expansion). Legacy segment-suffixed naming
(``ranvars=["OVTT1", "OVTT2"]`` with separate spec entries) still resolves,
but now means two distinct random coefficients.
"""

from __future__ import annotations

import warnings

import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel

ALTERNATIVES = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

SPEC_BASE = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero", "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
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
    "AGE45_SR": SPEC_BASE["AGE45_SR"],
    "IVTT": SPEC_BASE["IVTT"],
    "OVTT1": SPEC_BASE["OVTT"],
    "OVTT2": SPEC_BASE["OVTT"],
    "COST": SPEC_BASE["COST"],
}

# Spec that uses ``AGE45`` directly as a coefficient key (not ``AGE45_DA``).
# Used to verify that ``_strip_segment_suffix`` does not silently strip
# ``"AGE45"`` → ``"AGE4"`` when ``"AGE4"`` is absent from var_names.
SPEC_WITH_AGE45_KEY = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "AGE45": {"Alt1_ch": "AGE45", "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT": SPEC_BASE["IVTT"],
    "OVTT": SPEC_BASE["OVTT"],
    "COST": SPEC_BASE["COST"],
}

# Explicit per-segment suffix form: OVTT_seg1 and OVTT_seg2 as separate keys
# both pointing to the same OVTT_* raw data columns.
SPEC_SEG_SUFFIX = {
    "CON_SR": SPEC_BASE["CON_SR"],
    "CON_TR": SPEC_BASE["CON_TR"],
    "AGE45_DA": SPEC_BASE["AGE45_DA"],
    "AGE45_SR": SPEC_BASE["AGE45_SR"],
    "IVTT": SPEC_BASE["IVTT"],
    "OVTT_seg1": SPEC_BASE["OVTT"],
    "OVTT_seg2": SPEC_BASE["OVTT"],
    "COST": SPEC_BASE["COST"],
}


def _make_model(spec, nseg, ranvars, travelmode_path, mix=True, **ctrl_kwargs):
    ctrl = MNPControl(
        iid=False,
        mix=mix,
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


class TestRanvarsSingleSlotNseg2:
    """Shared-coefficients layout: a ranvar name resolves to ONE column index.

    ``_unpack_params`` gives every name in ``ranvars`` its own segment-specific
    coefficient/Omega slot per segment, so the model must NOT pre-duplicate the
    index (that was the old MNP-006 auto-expansion, which made the per-segment
    Omega rank-deficient).
    """

    def test_ranvars_single_index_for_nseg2_base_name(self, travelmode_path):
        m = _make_model(SPEC_BASE, nseg=2, ranvars=["OVTT"],
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices == [m.var_names.index("OVTT")]

    def test_ranvars_string_form_single_index_for_nseg2(self, travelmode_path):
        """Single-string form ``ranvars="OVTT"`` resolves the same way."""
        m = _make_model(SPEC_BASE, nseg=2, ranvars="OVTT",
                        travelmode_path=travelmode_path)
        assert m.ranvar_indices == [m.var_names.index("OVTT")]

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
        shared-coefficients per-segment overhead.

        For ``ranvars=["OVTT"]`` (``n_rand = 1``) each extra segment adds:

            +1 segment-probability param
            + n_rand segment-specific betas (only the ranvars vary)
            + n_omega(n_rand) for that segment's Omega

        Every other coefficient is shared with segment 1 and costs nothing.
        """
        m1 = _make_model(SPEC_BASE, nseg=1, ranvars=["OVTT"],
                         travelmode_path=travelmode_path)
        m2 = _make_model(SPEC_BASE, nseg=2, ranvars=["OVTT"],
                         travelmode_path=travelmode_path)

        assert m2.n_params > m1.n_params

        n_beta = m1.n_beta
        n_rand = 1
        n_omega = n_rand * (n_rand + 1) // 2  # 1
        # nseg=1: n_beta + n_lambda + n_omega
        n_lambda_terms = m1.n_params - n_beta - n_omega
        assert m1.n_params == n_beta + n_lambda_terms + n_omega

        n_seg_extra = m2.control.nseg - 1  # 1 prob param
        expected_m2 = m1.n_params + n_seg_extra * (1 + n_rand + n_omega)
        assert m2.n_params == expected_m2, (
            f"n_params(nseg=2)={m2.n_params}, expected {expected_m2}"
        )


# ---------------------------------------------------------------------------
# Heavy fit-based parity tests (Model (d))
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_table2_model_d_shared_coefficients_hit_paper_LL(travelmode_path):
    """Fit Model (d) with ``ranvars=["OVTT"]`` under the shared-coefficients
    layout and assert the published BHATLIB LL (-634.975) is reached.

    Before the shared/varying refactor every beta was duplicated per segment
    and the fit stalled at ~-624.4 (degenerate segment); the refactor closes
    the gap (LL -634.938 on TRAVELMODE).
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

    # Paper target: -634.975 (±1). Do NOT widen this tolerance.
    assert -636.0 <= results.loglik * results.n_obs <= -633.0, (
        f"Model (d) LL = {results.loglik * results.n_obs:.3f}, "
        f"expected within [-636, -633] (paper target -634.975)"
    )


@pytest.mark.slow
def test_table2_model_d_legacy_naming_still_passes(travelmode_path):
    """Fit Model (d) with both the new and legacy ranvars naming.

    Under the shared-coefficients layout the legacy 8-entry spec
    (``OVTT1``/``OVTT2`` separate, both in ``ranvars``) is a genuinely
    different model: two random coefficients over duplicate data columns
    (rank-deficient Omega, warned about) instead of one. It still resolves
    and fits; this test pins its parameter count relative to the new form
    and checks that the two land in the same LL neighbourhood.
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

    # Legacy form = TWO random coefficients (OVTT1, OVTT2) instead of one:
    #   +1 beta column (OVTT2) shared across segments
    #   +(n_omega(2) - n_omega(1)) = +2 in the segment-1 Omega
    #   per extra segment: +1 segment-specific beta, +2 Omega entries
    nseg = model_new.control.nseg
    expected_extra = 1 + 2 + (nseg - 1) * (1 + 2)
    assert model_legacy.n_params == model_new.n_params + expected_extra, (
        f"legacy n_params={model_legacy.n_params}, "
        f"new n_params={model_new.n_params}, "
        f"expected delta={expected_extra}"
    )

    # LL parity — the two paths target the same global optimum but
    # may stop at slightly different local optima due to the legacy
    # form's extra (redundant) betas (amplified under the first-diff-var=1
    # kernel; empirical gap ~3.2 units, see docstring).
    assert abs(res_new.loglik * res_new.n_obs - res_legacy.loglik * res_legacy.n_obs) < 3.5, (
        f"new LL={res_new.loglik * res_new.n_obs:.6f}, legacy LL={res_legacy.loglik * res_legacy.n_obs:.6f}, "
        f"delta={res_new.loglik * res_new.n_obs - res_legacy.loglik * res_legacy.n_obs:+.4f}"
    )


# ---------------------------------------------------------------------------
# Fix 1 — _strip_segment_suffix tightening (MNP-006 pre-merge fixes)
# ---------------------------------------------------------------------------


class TestStripSegmentSuffixFix:
    """Verify that _strip_segment_suffix no longer mishandles ``"AGE45"``."""

    def test_strip_suffix_does_not_affect_age45(self, travelmode_path):
        """``ranvars=["AGE45"]`` must resolve to the AGE45 column index.

        Bug: the old implementation stripped any trailing digit, so
        ``"AGE45"`` → ``"AGE4"``.  With the fix, the strip is only applied
        when the base (``"AGE4"``) also appears in var_names — which it
        doesn't here — so ``"AGE45"`` is resolved directly.
        """
        # SPEC_WITH_AGE45_KEY has "AGE45" as a var_name, NOT "AGE4".
        m = _make_model(
            SPEC_WITH_AGE45_KEY, nseg=1,
            ranvars=["AGE45"],
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 1
        resolved_name = m.var_names[m.ranvar_indices[0]]
        assert resolved_name == "AGE45", (
            f"Expected 'AGE45' but got {resolved_name!r} — "
            "digit strip is too permissive."
        )

    def test_strip_suffix_age45_not_in_var_names_raises(self, travelmode_path):
        """If ``"AGE45"`` is absent from spec and spec has no ``"AGE4"``
        either, a ``ValueError`` must be raised — not a silent wrong-variable
        resolution.
        """
        # SPEC_BASE has "AGE45_DA" / "AGE45_SR" as keys, NOT "AGE45".
        with pytest.raises(ValueError, match="not found"):
            _make_model(
                SPEC_BASE, nseg=2,
                ranvars=["AGE45"],
                travelmode_path=travelmode_path,
            )

    def test_strip_suffix_recognizes_seg_form(self, travelmode_path):
        """``ranvars=["OVTT_seg1", "OVTT_seg2"]`` must resolve to the two
        explicit per-segment OVTT spec entries when SPEC_SEG_SUFFIX is used.
        """
        m = _make_model(
            SPEC_SEG_SUFFIX, nseg=2,
            ranvars=["OVTT_seg1", "OVTT_seg2"],
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 2
        names = [m.var_names[ri] for ri in m.ranvar_indices]
        assert "OVTT_seg1" in names, f"Expected OVTT_seg1 in {names}"
        assert "OVTT_seg2" in names, f"Expected OVTT_seg2 in {names}"

    def test_strip_suffix_legacy_ovtt1_form_still_works(self, travelmode_path):
        """``ranvars=["OVTT1", "OVTT2"]`` with both names literally in
        var_names must resolve to the OVTT1 and OVTT2 column indices
        (no auto-expansion, no strip).

        Reference: GAUSS ``ranvars = { OVTT1 OVTT2 }`` from
        ``Gauss Files and Comparison/MNP/MNP Table2 d.gss:113``.
        """
        m = _make_model(
            SPEC_LEGACY_OVTT_SPLIT, nseg=2,
            ranvars=["OVTT1", "OVTT2"],
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is not None
        assert len(m.ranvar_indices) == 2
        names = [m.var_names[ri] for ri in m.ranvar_indices]
        assert "OVTT1" in names, f"Expected OVTT1 in {names}"
        assert "OVTT2" in names, f"Expected OVTT2 in {names}"

    def test_duplicate_ranvar_indices_emits_warning(self, travelmode_path):
        """Listing the same variable twice in ``ranvars`` resolves to duplicate
        column indices (a rank-deficient Omega) and must emit a
        ``RuntimeWarning``.
        """
        with pytest.warns(RuntimeWarning, match="duplicate column indices"):
            _make_model(
                SPEC_BASE, nseg=2,
                ranvars=["OVTT", "OVTT"],
                travelmode_path=travelmode_path,
            )


# ---------------------------------------------------------------------------
# Pre-merge polish (post-Opus-review): warning attribution, edge-case tests,
# direct _strip_segment_suffix coverage, mix-flag normalization.
# ---------------------------------------------------------------------------


class TestStripSegmentSuffixDirect:
    """Direct unit tests for the static ``_strip_segment_suffix`` method.

    Covers the docstring guarantee that legacy plain-digit form is **not**
    recognised when ``var_names=None`` — the static method is reachable
    without instantiating an MNPModel.
    """

    def test_explicit_seg_form_recognised_without_var_names(self):
        assert MNPModel._strip_segment_suffix("OVTT_seg1") == "OVTT"
        assert MNPModel._strip_segment_suffix("OVTT_s2") == "OVTT"

    def test_legacy_digit_form_not_recognised_without_var_names(self):
        # Without var_names, the digit-strip must NOT fire — even for
        # cases where there is no false-strip risk like AGE45.
        assert MNPModel._strip_segment_suffix("OVTT1") is None
        assert MNPModel._strip_segment_suffix("AGE45") is None

    def test_legacy_digit_form_with_var_names_requires_both(self):
        # Strip fires only when BOTH name and base appear in var_names.
        assert (
            MNPModel._strip_segment_suffix("OVTT1", var_names=["OVTT", "OVTT1"])
            == "OVTT"
        )
        # Base missing — no strip.
        assert (
            MNPModel._strip_segment_suffix("OVTT1", var_names=["OVTT1"])
            is None
        )
        # Name missing from var_names — no strip (prevents AGE45→AGE4 case).
        assert (
            MNPModel._strip_segment_suffix("AGE45", var_names=["AGE4"])
            is None
        )

    def test_no_match_returns_none(self):
        assert MNPModel._strip_segment_suffix("OVTT") is None
        assert MNPModel._strip_segment_suffix("") is None


class TestMixFlagNormalization:
    """Regression tests for the ``mix=True, ranvar_indices=None`` normalization."""

    def test_mix_true_with_none_ranvars_normalizes_mix_to_false(
        self, travelmode_path
    ):
        """``MNPControl(mix=True)`` with ``ranvars=None`` should not leave
        the model in the meaningless ``mix=True, ranvar_indices=None``
        state — ``self.control.mix`` is normalized to ``False``.
        """
        m = _make_model(
            SPEC_BASE, nseg=1, ranvars=None,
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is None
        assert m.control.mix is False

    def test_mix_true_with_empty_ranvars_normalizes_mix_to_false(
        self, travelmode_path
    ):
        """Same normalization for ``ranvars=[]`` (empty list)."""
        m = _make_model(
            SPEC_BASE, nseg=1, ranvars=[],
            travelmode_path=travelmode_path,
        )
        assert m.ranvar_indices is None
        assert m.control.mix is False


class TestDuplicateWarningAttribution:
    """The duplicate-index warning fires on the duplicate itself, not on nseg.

    Auto-expansion no longer exists, so the only way to get duplicate
    ``ranvar_indices`` is to list a variable twice. That is rank-deficient
    regardless of ``nseg`` and always warns; the single-name mixture form
    (``ranvars=["OVTT"]`` with ``nseg=2``) never does.
    """

    def test_user_duplicates_warn_even_for_nseg1(self, travelmode_path):
        with pytest.warns(RuntimeWarning, match="duplicate column indices"):
            m = _make_model(
                SPEC_BASE, nseg=1,
                ranvars=["OVTT", "OVTT"],
                travelmode_path=travelmode_path,
            )
        # the user's choice is preserved (no dedup)
        assert m.ranvar_indices == [
            m.var_names.index("OVTT"),
            m.var_names.index("OVTT"),
        ]

    def test_single_name_nseg2_does_not_warn(self, travelmode_path):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            m = _make_model(
                SPEC_BASE, nseg=2,
                ranvars=["OVTT"],
                travelmode_path=travelmode_path,
            )
        assert m.ranvar_indices == [m.var_names.index("OVTT")]


class TestNseg1ExplicitSuffixHelpfulError:
    """Pre-merge UX fix: nseg=1 + explicit ``_seg``/``_s`` suffix gives a
    targeted hint pointing at the nseg=1 limitation.

    Note: the legacy plain-digit form (e.g. ``"OVTT1"``) only triggers
    ``_strip_segment_suffix`` when both ``"OVTT1"`` and ``"OVTT"`` appear
    in ``var_names``; with nseg=1 + SPEC_BASE that condition cannot be
    met, so the hint is reachable only via the explicit form.
    """

    def test_nseg1_explicit_seg_suffix_error_mentions_nseg(self, travelmode_path):
        # ``ranvars=["OVTT_seg1"]`` with nseg=1 and only "OVTT" in var_names —
        # the error should hint at the nseg=1 limitation.
        with pytest.raises(ValueError, match="nseg=1"):
            _make_model(
                SPEC_BASE, nseg=1,
                ranvars=["OVTT_seg1"],
                travelmode_path=travelmode_path,
            )
