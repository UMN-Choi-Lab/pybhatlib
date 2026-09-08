"""Self-consistency gate for MixingSpec, ParamLayout, and ParamSpace.unpack.

Ports the MIXMNL.gss spec-setup expectations (no live GAUSS required):

    var_unordnames = { UNO1 UNO2 UNO3 FTIME1 FTIME2 FTIME3 CONTINC1 }
    normvar        = { UNO1 UNO2 }
    logvar         = { }
    yjvar          = { CONTINC1 }
    varneg = varpos = { }

and asserts the counts, the index masks, and the ParamLayout pack/slice
round-trip / coverage documented in MIXED_PANEL_MODELS_PLAN.md T0.10.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._reparam import (
    EstimationSpace,
    ParamLayout,
    ReportingSpace,
)
from pybhatlib.mixed._spec import MixingSpec

VAR_NAMES = ["UNO1", "UNO2", "UNO3", "FTIME1", "FTIME2", "FTIME3", "CONTINC1"]
NORMVAR = ["UNO1", "UNO2"]
YJVAR = ["CONTINC1"]


@pytest.fixture
def spec() -> MixingSpec:
    return MixingSpec.from_var_names(
        VAR_NAMES, normvar=NORMVAR, logvar=[], yjvar=YJVAR, varneg=[], varpos=[]
    )


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

def test_counts(spec: MixingSpec) -> None:
    assert spec.n_beta == 7
    assert spec.nrndnor == 2
    assert spec.nrndlog == 0
    assert spec.nrndyj == 1
    assert spec.nrndcoef == 3
    assert spec.nrndtot == 3
    assert spec.nrndtcor == 3  # 3*(3-1)/2
    assert spec.nscale == 3
    assert spec.numlam == 3
    assert spec.nvarneg == 0
    assert spec.nvarpos == 0


def test_nrndtcor_formula() -> None:
    for k in range(1, 8):
        names = [f"V{i}" for i in range(k)]
        s = MixingSpec.from_var_names(names, normvar=names)
        assert s.nrndtcor == k * (k - 1) // 2
        assert s.nrndcoef == k
        assert s.numlam == k
        assert s.nscale == k


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_mixpos(spec: MixingSpec) -> None:
    # normvar UNO1,UNO2 -> positions 0,1 ; yjvar CONTINC1 -> position 6
    np.testing.assert_array_equal(spec.mixposn, [0, 1])
    np.testing.assert_array_equal(spec.mixposlg, [])
    np.testing.assert_array_equal(spec.mixposyj, [6])
    np.testing.assert_array_equal(spec.mixpos, [0, 1, 6])


def test_indxrndvar(spec: MixingSpec) -> None:
    # (n_beta, nrndcoef) selection: col j selects var mixpos[j]
    expected = np.zeros((7, 3))
    expected[0, 0] = 1.0
    expected[1, 1] = 1.0
    expected[6, 2] = 1.0
    np.testing.assert_array_equal(spec.indxrndvar, expected)
    # indxrndvar maps a random-coef vector into the correct var positions
    f1 = np.array([10.0, 20.0, 30.0])
    placed = spec.indxrndvar @ f1
    np.testing.assert_array_equal(
        placed, [10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 30.0]
    )


def test_injection_masks_additive(spec: MixingSpec) -> None:
    # No neg/pos/log vars -> purely additive injection
    np.testing.assert_array_equal(spec.indxvarnonegposlog, np.ones(7))
    np.testing.assert_array_equal(spec.indxvarnegposlog, np.zeros(7))


def test_sign_masks_empty(spec: MixingSpec) -> None:
    np.testing.assert_array_equal(spec.indxvarneg, np.zeros(7))
    np.testing.assert_array_equal(spec.indxvarpos, np.zeros(7))
    assert spec.posvarneg.shape == (7, 0)
    assert spec.posvarpos.shape == (7, 0)


def test_poslog_empty_when_no_log(spec: MixingSpec) -> None:
    assert spec.poslog.shape == (3, 0)
    assert spec.posnolog.shape == (3, 0)


def test_actlam(spec: MixingSpec) -> None:
    # normal coefs -> 0, yj coef -> 1, ordered [nor|log|yj]
    np.testing.assert_array_equal(spec.actlam, [0.0, 0.0, 1.0])


def test_fix_location_zero_requires_additive_random_coefficient() -> None:
    with pytest.raises(ValueError, match="random coefficient"):
        MixingSpec.from_var_names(["A", "B"], fix_location_zero=["B"])

    with pytest.raises(ValueError, match="additive"):
        MixingSpec.from_var_names(
            ["A"], logvar=["A"], varneg=["A"], fix_location_zero=["A"]
        )


# ---------------------------------------------------------------------------
# Sign / log masks on a richer spec (exercise the neg/pos/log branches)
# ---------------------------------------------------------------------------

def test_varneg_varpos_masks() -> None:
    names = ["A", "B", "C", "D"]
    s = MixingSpec.from_var_names(
        names,
        normvar=["A"],
        logvar=["B"],
        yjvar=["C"],
        varneg=["D"],
        varpos=["B"],
    )
    # B in varpos, D in varneg
    np.testing.assert_array_equal(s.indxvarneg, [0, 0, 0, 1])
    np.testing.assert_array_equal(s.indxvarpos, [0, 1, 0, 0])
    np.testing.assert_array_equal(s.indxvarnegpos, [0, 1, 0, 1])
    np.testing.assert_array_equal(s.indxvarnonegpos, [1, 0, 1, 0])
    # logvar B is in varpos -> multiplicative injection at position B
    np.testing.assert_array_equal(s.indxvarnegposlog, [0, 1, 0, 0])
    np.testing.assert_array_equal(s.indxvarnonegposlog, [1, 0, 1, 1])
    # rc ordering [nor(A) | log(B) | yj(C)]
    np.testing.assert_array_equal(s.mixpos, [0, 1, 2])
    # poslog selects the log coefficient (rc index 1); posnolog the rest
    assert s.poslog.shape == (3, 1)
    assert s.posnolog.shape == (3, 2)
    np.testing.assert_array_equal(s.poslog[:, 0], [0, 1, 0])


def test_missing_name_raises() -> None:
    with pytest.raises(ValueError):
        MixingSpec.from_var_names(["A", "B"], normvar=["Z"])


# ---------------------------------------------------------------------------
# ParamLayout
# ---------------------------------------------------------------------------

def layout_for(spec: MixingSpec, n_kern: int = 0) -> ParamLayout:
    return ParamLayout(
        n_beta=spec.n_beta,
        n_rcor=spec.nrndtcor,
        n_scal=spec.nscale,
        n_lam=spec.numlam,
        n_kern=n_kern,
    )


def test_layout_n_theta(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    # 7 + 3 + 3 + 3 + 0 = 16
    assert lay.n_theta == 16


def test_slices_cover_no_gap_no_overlap(spec: MixingSpec) -> None:
    for n_kern in (0, 2):
        lay = layout_for(spec, n_kern=n_kern)
        sl = lay.slices()
        assert list(sl.keys()) == ["beta", "rcor", "scal", "lam", "kern"]
        covered = np.zeros(lay.n_theta, dtype=int)
        prev_stop = 0
        for name in ["beta", "rcor", "scal", "lam", "kern"]:
            s = sl[name]
            assert s.start == prev_stop  # contiguous, no gap
            prev_stop = s.stop
            covered[s] += 1
        assert prev_stop == lay.n_theta  # covers the whole vector
        np.testing.assert_array_equal(covered, np.ones(lay.n_theta, dtype=int))


def test_pack_slices_roundtrip(spec: MixingSpec) -> None:
    lay = layout_for(spec, n_kern=2)
    rng = np.random.default_rng(0)
    dbeta = rng.standard_normal(lay.n_beta)
    drcor = rng.standard_normal(lay.n_rcor)
    dscal = rng.standard_normal(lay.n_scal)
    dlam = rng.standard_normal(lay.n_lam)
    dkern = rng.standard_normal(lay.n_kern)

    theta = lay.pack(dbeta, drcor, dscal, dlam, dkern)
    assert theta.shape == (lay.n_theta,)

    sl = lay.slices()
    np.testing.assert_array_equal(theta[sl["beta"]], dbeta)
    np.testing.assert_array_equal(theta[sl["rcor"]], drcor)
    np.testing.assert_array_equal(theta[sl["scal"]], dscal)
    np.testing.assert_array_equal(theta[sl["lam"]], dlam)
    np.testing.assert_array_equal(theta[sl["kern"]], dkern)


def test_pack_empty_kern(spec: MixingSpec) -> None:
    lay = layout_for(spec, n_kern=0)
    theta = lay.pack(
        np.zeros(7), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(0)
    )
    assert theta.shape == (16,)


def test_pack_wrong_length_raises(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    with pytest.raises(ValueError):
        lay.pack(np.zeros(6), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(0))


# ---------------------------------------------------------------------------
# ParamSpace.unpack — estimation vs reporting
# ---------------------------------------------------------------------------

def test_estimation_space_unpack_shapes(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    space = EstimationSpace(lay, scal=1.0, intordn1=20)
    rng = np.random.default_rng(1)
    theta = rng.standard_normal(lay.n_theta) * 0.3

    rc = space.unpack(theta, spec, want_grad=True)
    k = spec.nrndcoef
    assert rc.xmu.shape == (spec.n_beta,)
    assert rc.x11chol.shape == (k, k)
    assert rc.omegastar.shape == (k, k)
    # correlation matrix: unit diagonal, symmetric
    np.testing.assert_allclose(np.diag(rc.omegastar), np.ones(k), atol=1e-10)
    np.testing.assert_allclose(rc.omegastar, rc.omegastar.T, atol=1e-12)
    # x11chol' x11chol == omegastar
    np.testing.assert_allclose(rc.x11chol.T @ rc.x11chol, rc.omegastar, atol=1e-10)
    # scale = exp(xscalrand) > 0
    assert np.all(rc.wscalrand > 0)
    # yj lambda in (0, 2)
    assert np.all((rc.xlamrnd > 0) & (rc.xlamrnd < 2))
    assert rc.xlamrnd.shape == (k,)
    assert rc.mulamrnd.shape == (k,)
    assert rc.siglamrnd.shape == (k,)
    # gradient blocks present
    assert rc.dxmudxmu1 is not None and rc.dxmudxmu1.shape == (spec.n_beta,)
    assert rc.gtempstar is not None
    assert rc.dxlamrnddxlam is not None and rc.dxlamrnddxlam.shape == (k, k)
    np.testing.assert_array_equal(rc.dxmudxmu1, np.ones(spec.n_beta))


def test_estimation_space_no_grad(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    space = EstimationSpace(lay)
    theta = np.zeros(lay.n_theta)
    rc = space.unpack(theta, spec, want_grad=False)
    assert rc.dxmudxmu1 is None
    assert rc.gtempstar is None
    assert rc.dxlamrnddxlam is None
    # xscalrand=0 -> scale exp(0)=1 ; xlam=0 -> lam=2*cdlogit(0)=1
    np.testing.assert_allclose(rc.wscalrand, np.ones(spec.nrndcoef))
    np.testing.assert_allclose(rc.xlamrnd, np.ones(spec.nrndcoef))
    # lam=1 -> identity YJ -> meanyj = (0, 1)
    np.testing.assert_allclose(rc.mulamrnd, np.zeros(spec.nrndcoef), atol=1e-9)
    np.testing.assert_allclose(rc.siglamrnd, np.ones(spec.nrndcoef), atol=1e-9)


def test_fix_location_zero_pins_mean_but_keeps_random_scale() -> None:
    fixed_spec = MixingSpec.from_var_names(
        ["A", "B"], normvar=["A"], fix_location_zero=["A"]
    )
    lay = layout_for(fixed_spec)
    theta = np.zeros(lay.n_theta)
    theta[lay.slices()["beta"]] = [3.0, 2.0]
    theta[lay.slices()["scal"]] = np.log(1.7)

    rc = EstimationSpace(lay).unpack(theta, fixed_spec, want_grad=True)

    np.testing.assert_array_equal(rc.xmu, [0.0, 2.0])
    np.testing.assert_array_equal(rc.dxmudxmu1, [0.0, 1.0])
    assert rc.wscalrand[0] == pytest.approx(1.7)


def test_fix_location_zero_empty_default_is_noop() -> None:
    default_spec = MixingSpec.from_var_names(["A", "B"], normvar=["A"])
    lay = layout_for(default_spec)
    theta = np.zeros(lay.n_theta)
    theta[lay.slices()["beta"]] = [3.0, 2.0]

    rc = EstimationSpace(lay).unpack(theta, default_spec, want_grad=True)

    np.testing.assert_array_equal(rc.xmu, [3.0, 2.0])
    np.testing.assert_array_equal(rc.dxmudxmu1, [1.0, 1.0])


def test_actlam_pins_non_yj_lambdas_in_estimation_space(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    theta = np.zeros(lay.n_theta)
    theta[lay.slices()["lam"]] = [20.0, -20.0, 0.7]

    rc = EstimationSpace(lay).unpack(theta, spec, want_grad=True)

    np.testing.assert_array_equal(rc.xlamrnd[:2], [1.0, 1.0])
    assert rc.xlamrnd[2] != pytest.approx(1.0)
    np.testing.assert_array_equal(rc.dxlamrnddxlam[:2], np.zeros((2, 3)))
    assert rc.dxlamrnddxlam[2, 2] > 0.0


def test_actlam_pins_non_yj_lambdas_in_reporting_space(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    theta = np.zeros(lay.n_theta)
    theta[lay.slices()["scal"]] = 1.0
    theta[lay.slices()["lam"]] = [0.2, 1.8, 0.7]

    rc = ReportingSpace(lay).unpack(theta, spec)

    np.testing.assert_array_equal(rc.xlamrnd, [1.0, 1.0, 0.7])


def test_estimation_space_extreme_sign_and_scale_blocks_remain_finite() -> None:
    rich_spec = MixingSpec.from_var_names(
        ["A", "B", "C"], normvar=["A", "B", "C"], varneg=["A"], varpos=["B"]
    )
    lay = layout_for(rich_spec)
    theta = np.zeros(lay.n_theta)
    slices = lay.slices()
    theta[slices["beta"]] = [1e6, 1e6, 1e6]
    theta[slices["scal"]] = 1e6

    rc = EstimationSpace(lay).unpack(theta, rich_spec, want_grad=True)

    assert np.all(np.isfinite(rc.xmu))
    assert np.all(np.isfinite(rc.wscalrand))
    assert np.all(np.isfinite(rc.dxmudxmu1))


def test_reporting_space_direct(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    space = ReportingSpace(lay, scal=1.0, intordn1=20)
    sl = lay.slices()

    # Build a reporting theta: correlation entries in [-0.5, 0.5], direct
    # scale and lambda values.
    theta = np.zeros(lay.n_theta)
    theta[sl["beta"]] = np.arange(1.0, 8.0)
    theta[sl["rcor"]] = [0.2, -0.1, 0.3]
    theta[sl["scal"]] = [1.5, 0.8, 2.0]
    theta[sl["lam"]] = [1.0, 1.0, 0.7]

    rc = space.unpack(theta, spec, want_grad=False)
    # beta entered directly (no exp reparam)
    np.testing.assert_array_equal(rc.xmu, np.arange(1.0, 8.0))
    # scale entered directly
    np.testing.assert_array_equal(rc.wscalrand, [1.5, 0.8, 2.0])
    # lambda entered directly
    np.testing.assert_array_equal(rc.xlamrnd, [1.0, 1.0, 0.7])
    # omegastar reconstructed from direct correlation entries, unit diagonal
    np.testing.assert_allclose(np.diag(rc.omegastar), np.ones(spec.nrndcoef))
    np.testing.assert_allclose(
        rc.x11chol.T @ rc.x11chol, rc.omegastar, atol=1e-10
    )
    # reporting space carries no reparam gradients
    assert rc.dxmudxmu1 is None
    assert rc.gtempstar is None


def test_reporting_space_repairs_indefinite_direct_correlation(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    theta = np.zeros(lay.n_theta)
    slices = lay.slices()
    theta[slices["rcor"]] = [0.99, 0.99, -0.99]
    theta[slices["scal"]] = 1.0
    theta[slices["lam"]] = 1.0

    rc = ReportingSpace(lay).unpack(theta, spec)

    assert np.linalg.eigvalsh(rc.omegastar).min() > 0.0
    np.testing.assert_allclose(rc.x11chol.T @ rc.x11chol, rc.omegastar, atol=1e-12)


def test_spher_not_implemented(spec: MixingSpec) -> None:
    lay = layout_for(spec)
    with pytest.raises(NotImplementedError):
        EstimationSpace(lay, spher=True)
