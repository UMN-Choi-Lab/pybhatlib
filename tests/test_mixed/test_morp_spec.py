"""Gate for the MORP kernel-aware MixingSpec / ParamLayout / threshold reparam.

No live GAUSS required. Encodes the MORP (joint ordered-YJ) parameter layout
from ``Joint Ordered YJ with Cross-Sectional or Panel Random Coefficients.gss``
(b vector line 477::

    b = thresh | b1 | startrandker | scalcoef | blamrnd | blamker

which maps, block-for-block, to::

    theta = thresholds(nthresh) | ordered-coeffs(ncoeford)
          | joint-correlation(nrndtcor over nrndtot = nrndcoef + nord)
          | rc-scale(nrndcoef) | rc-lam(numlam)
          | kernel-lam(nord, only if _normker == 0)

Key MORP-vs-MNP differences exercised here:

1. a THRESHOLD block LEADS the vector -- ``nthresh = sum(n_categories[d] - 1)``
   cut points, one increment sub-block per ordinal dimension, parameterized so
   the realized cut points stay strictly ordered;
2. the joint correlation is over ``nrndtot = nrndcoef + nord`` (the ``nord``
   ordinal kernel-error dimensions, not ``nc - 1``);
3. MORP fixes ``wker = ones`` so there is NO kernel-scale block
   (``n_kern == 0``); instead an optional YJ kernel-lam block of ``nord`` params
   TRAILS the vector when ``normker`` is False (GAUSS ``_normker == 0``).

Gate configuration: ``nord = 2``, ``nrndcoef = 1``, YJ kernel (``normker=False``).
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout, thresh_reparam
from pybhatlib.mixed._spec import MixingSpec

# 3 exogenous vars; 1 normal random coefficient; 2 ordinal outcome dimensions
# with 3 and 4 observed categories -> (3-1) + (4-1) = 5 threshold params.
VAR_NAMES = ["X0", "X1", "X2"]
NORMVAR = ["X0"]
NORD = 2
N_CATEGORIES = [3, 4]  # numthresh = (2, 3), nthresh = 5


@pytest.fixture
def spec() -> MixingSpec:
    return MixingSpec.from_var_names(
        VAR_NAMES,
        normvar=NORMVAR,
        nord=NORD,
        n_categories=N_CATEGORIES,
        normker=False,  # _normker == 0 -> YJ kernel-lam block of nord params
    )


@pytest.fixture
def layout(spec: MixingSpec) -> ParamLayout:
    return ParamLayout(
        n_beta=spec.n_beta,
        n_rcor=spec.nrndtcor,
        n_scal=spec.nscale,
        n_lam=spec.numlam,
        n_kern=spec.n_kern,
        kern_before_lam=True,
        n_thresh=spec.nthresh,
        n_kernlam=spec.nkernlam,
    )


# ---------------------------------------------------------------------------
# Joint sizing (nrndtot = nrndcoef + nord) + threshold / kernel-lam counts
# ---------------------------------------------------------------------------

def test_joint_sizing(spec: MixingSpec) -> None:
    assert spec.n_beta == 3
    assert spec.nrndcoef == 1
    assert spec.nord == 2
    # JOINT rc + ordinal-kernel correlation space
    assert spec.nrndtot == 3  # nrndcoef(1) + nord(2)
    assert spec.nrndtcor == 3  # 3*2//2
    # MORP fixes wker = ones -> NO kernel-scale block
    assert spec.n_kern == 0
    assert spec.nscale == 1
    assert spec.numlam == 1
    # threshold + kernel-lam blocks
    assert spec.numthresh == (2, 3)
    assert spec.nthresh == 5  # (3-1) + (4-1)
    assert spec.nkernlam == 2  # nord, since normker=False


def test_nrndtot_morp_formula() -> None:
    # nrndtot = nrndcoef + nord across a small grid; n_kern stays 0 for MORP.
    for nrnd in range(1, 4):
        for nord in range(1, 4):
            names = [f"V{i}" for i in range(nrnd)]
            s = MixingSpec.from_var_names(
                names,
                normvar=names,
                nord=nord,
                n_categories=[3] * nord,
            )
            assert s.nrndtot == nrnd + nord
            assert s.nrndtcor == (nrnd + nord) * (nrnd + nord - 1) // 2
            assert s.n_kern == 0
            assert s.nthresh == 2 * nord  # each (3-1)


def test_normker_true_no_kernlam() -> None:
    s = MixingSpec.from_var_names(
        VAR_NAMES, normvar=NORMVAR, nord=NORD, n_categories=N_CATEGORIES,
    )  # normker defaults True
    assert s.nkernlam == 0
    assert s.normker is True


def test_mnp_and_morp_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        MixingSpec.from_var_names(
            VAR_NAMES, normvar=NORMVAR, kernel_dim=2, nord=2, n_categories=[3, 3]
        )


def test_n_categories_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        MixingSpec.from_var_names(
            VAR_NAMES, normvar=NORMVAR, nord=2, n_categories=[3]
        )


def test_mnl_default_unchanged() -> None:
    # nord/kernel_dim omitted -> plain MNL facade, MORP blocks empty.
    s = MixingSpec.from_var_names(VAR_NAMES, normvar=NORMVAR)
    assert s.nord == 0
    assert s.nthresh == 0
    assert s.numthresh == ()
    assert s.nkernlam == 0
    assert s.nrndtot == s.nrndcoef == 1


# ---------------------------------------------------------------------------
# ParamLayout: MORP ordering (thresh leads, kernlam trails)
# ---------------------------------------------------------------------------

def test_total_n_theta(layout: ParamLayout) -> None:
    # 5 thresh + 3 beta + 3 rcor + 1 scal + 0 kern + 1 lam + 2 kernlam
    assert layout.n_theta == 15


def test_morp_slices(layout: ParamLayout) -> None:
    sl = layout.slices()
    assert sl["thresh"] == slice(0, 5)  # LEADS
    assert sl["beta"] == slice(5, 8)  # ordered coeffs
    assert sl["rcor"] == slice(8, 11)  # joint correlation
    assert sl["scal"] == slice(11, 12)  # rc-scale
    assert sl["lam"] == slice(12, 13)  # rc-lam
    assert sl["kernlam"] == slice(13, 15)  # kernel-lam TRAILS


def test_morp_slices_tile_no_gap_no_overlap(layout: ParamLayout) -> None:
    sl = layout.slices()
    # every named block (including the empty kern) tiles [0, n_theta) exactly.
    covered = np.zeros(layout.n_theta, dtype=int)
    for s in sl.values():
        covered[s] += 1
    np.testing.assert_array_equal(covered, np.ones(layout.n_theta, dtype=int))
    # named blocks the gate cares about are all present
    for key in ("thresh", "beta", "rcor", "scal", "lam", "kernlam"):
        assert key in sl
    # physical order: thresh first, kernlam last
    keys = list(sl.keys())
    assert keys[0] == "thresh"
    assert keys[-1] == "kernlam"


def test_pack_slices_roundtrip(layout: ParamLayout) -> None:
    rng = np.random.default_rng(0)
    dthresh = rng.standard_normal(layout.n_thresh)
    dbeta = rng.standard_normal(layout.n_beta)
    drcor = rng.standard_normal(layout.n_rcor)
    dscal = rng.standard_normal(layout.n_scal)
    dlam = rng.standard_normal(layout.n_lam)
    dkern = rng.standard_normal(layout.n_kern)
    dkernlam = rng.standard_normal(layout.n_kernlam)

    theta = layout.pack(
        dbeta, drcor, dscal, dlam, dkern, dthresh=dthresh, dkernlam=dkernlam
    )
    assert theta.shape == (15,)

    sl = layout.slices()
    np.testing.assert_array_equal(theta[sl["thresh"]], dthresh)
    np.testing.assert_array_equal(theta[sl["beta"]], dbeta)
    np.testing.assert_array_equal(theta[sl["rcor"]], drcor)
    np.testing.assert_array_equal(theta[sl["scal"]], dscal)
    np.testing.assert_array_equal(theta[sl["lam"]], dlam)
    np.testing.assert_array_equal(theta[sl["kernlam"]], dkernlam)


def test_pack_places_thresh_first_kernlam_last(layout: ParamLayout) -> None:
    theta = layout.pack(
        np.full(3, 2.0),  # beta
        np.full(3, 3.0),  # rcor
        np.full(1, 4.0),  # scal
        np.full(1, 6.0),  # lam
        np.zeros(0),      # kern (empty)
        dthresh=np.full(5, 1.0),
        dkernlam=np.full(2, 7.0),
    )
    expected = np.array(
        [1.0] * 5 + [2.0] * 3 + [3.0] * 3 + [4.0] * 1 + [6.0] * 1 + [7.0] * 2
    )
    np.testing.assert_array_equal(theta, expected)


def test_pack_wrong_length_raises(layout: ParamLayout) -> None:
    with pytest.raises(ValueError):
        layout.pack(
            np.zeros(3), np.zeros(3), np.zeros(1), np.zeros(1), np.zeros(0),
            dthresh=np.zeros(4),  # wrong: expected 5
            dkernlam=np.zeros(2),
        )


# ---------------------------------------------------------------------------
# Backward compatibility: MNL / MNP layouts unchanged by the additive blocks
# ---------------------------------------------------------------------------

def test_mnl_layout_keys_unchanged() -> None:
    lay = ParamLayout(n_beta=7, n_rcor=1, n_scal=2, n_lam=2, n_kern=1)
    assert list(lay.slices().keys()) == ["beta", "rcor", "scal", "lam", "kern"]


def test_mnp_layout_keys_unchanged() -> None:
    lay = ParamLayout(
        n_beta=7, n_rcor=6, n_scal=2, n_lam=2, n_kern=1, kern_before_lam=True
    )
    assert list(lay.slices().keys()) == ["beta", "rcor", "scal", "kern", "lam"]


# ---------------------------------------------------------------------------
# Threshold reparam: increments -> ordered cut points, with gradient
# ---------------------------------------------------------------------------

def _thresh_ref(bthresh: np.ndarray, numthresh) -> np.ndarray:
    """Reference: per block tau[0]=b0, tau[k]=tau[k-1]+exp(b_k)."""
    out = []
    off = 0
    for m in numthresh:
        blk = bthresh[off : off + m]
        tau = np.empty(m)
        if m:
            tau[0] = blk[0]
            for k in range(1, m):
                tau[k] = tau[k - 1] + np.exp(blk[k])
        out.append(tau)
        off += m
    return np.concatenate(out) if out else np.zeros(0)


def test_thresh_increments_are_ordered(spec: MixingSpec) -> None:
    rng = np.random.default_rng(1)
    bthresh = rng.standard_normal(spec.nthresh)
    tau, _ = thresh_reparam(bthresh, spec.numthresh)
    np.testing.assert_allclose(tau, _thresh_ref(bthresh, spec.numthresh))
    # within each ordinal block, cut points strictly increase
    off = 0
    for m in spec.numthresh:
        blk = tau[off : off + m]
        assert np.all(np.diff(blk) > 0.0)
        off += m


def test_thresh_matches_fixed_morp_form() -> None:
    # arbitrary sizes / values, including a single-threshold block
    bthresh = np.array([0.3, -0.5, 1.2, -2.0, 0.0, 0.7])
    numthresh = (3, 1, 2)
    tau, _ = thresh_reparam(bthresh, numthresh)
    np.testing.assert_allclose(tau, _thresh_ref(bthresh, numthresh))


def test_thresh_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        thresh_reparam(np.zeros(4), (2, 3))  # sum 5 != 4


def test_thresh_gradient_vs_fd() -> None:
    rng = np.random.default_rng(2)
    eps = 1e-6
    numthresh = (2, 3)
    n = sum(numthresh)
    x0 = rng.standard_normal(n)
    _, dtau = thresh_reparam(x0, numthresh, want_grad=True)
    assert dtau.shape == (n, n)
    fd = np.zeros((n, n))
    for j in range(n):
        xp_ = x0.copy()
        xm = x0.copy()
        xp_[j] += eps
        xm[j] -= eps
        tp, _ = thresh_reparam(xp_, numthresh)
        tm, _ = thresh_reparam(xm, numthresh)
        fd[:, j] = (tp - tm) / (2 * eps)
    np.testing.assert_allclose(dtau, fd, atol=1e-6)


def test_thresh_gradient_block_diagonal() -> None:
    # cross-block partials must be exactly zero (block-diagonal Jacobian).
    numthresh = (2, 3)
    x0 = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    _, dtau = thresh_reparam(x0, numthresh, want_grad=True)
    # block 0 rows (0:2) must not depend on block 1 cols (2:5) and vice versa
    np.testing.assert_array_equal(dtau[0:2, 2:5], np.zeros((2, 3)))
    np.testing.assert_array_equal(dtau[2:5, 0:2], np.zeros((3, 2)))


def test_thresh_empty() -> None:
    tau, dtau = thresh_reparam(np.zeros(0), (), want_grad=True)
    assert tau.shape == (0,)
    assert dtau.shape == (0, 0)
