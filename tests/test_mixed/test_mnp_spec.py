"""Gate for the MNP kernel-aware MixingSpec / ParamLayout / wker reparam.

No live GAUSS required. Encodes the 18-parameter TRAVELMODE MNP layout
(confirmed from ``MNPKERCP.gss`` and
``tests/fixtures/mixed/mnp/travelmode_nocopula/b.csv``)::

    b = means(nvar=7) | corr(nrndtcor=6) | rc-scale(nrndcoef=2)
        | kernel-scale(nc-2=1) | lam(numlam=2)

Key MNP-vs-MNL differences exercised here:

1. the correlation is over the JOINT rc + differenced-kernel space,
   ``nrndtot = nrndcoef + (nc - 1)`` so ``nrndtcor = nrndtot*(nrndtot-1)//2``;
2. a kernel-scale block of ``n_kern = nc - 2`` params sits BETWEEN rc-scale
   and lam (``ParamLayout(kern_before_lam=True)``);
3. the kernel scale uses the sum-of-squares reparam
   ``wker = sqrt(logitmod([0|xscalker]))[1:]``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.mixed._reparam import ParamLayout, wker_reparam
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.utils._logistic import logitmod

# 7 exogenous vars; 2 normal random coefficients; nc = 3 alternatives so
# kernel_dim = nc - 1 = 2 differenced kernel errors.
VAR_NAMES = ["V0", "V1", "V2", "V3", "V4", "V5", "V6"]
NORMVAR = ["V0", "V1"]
NC = 3
KERNEL_DIM = NC - 1  # 2


@pytest.fixture
def spec() -> MixingSpec:
    return MixingSpec.from_var_names(
        VAR_NAMES, normvar=NORMVAR, kernel_dim=KERNEL_DIM
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
    )


# ---------------------------------------------------------------------------
# Joint-omegastar sizing
# ---------------------------------------------------------------------------

def test_joint_sizing(spec: MixingSpec) -> None:
    assert spec.n_beta == 7
    assert spec.nrndcoef == 2
    assert spec.kernel_dim == 2
    # JOINT rc + differenced-kernel correlation space
    assert spec.nrndtot == 4  # nrndcoef(2) + (nc-1)(2)
    assert spec.nrndtcor == 6  # 4*3//2
    assert spec.n_kern == 1  # nc - 2
    assert spec.nscale == 2
    assert spec.numlam == 2


def test_nrndtcor_joint_formula() -> None:
    # nrndtot = nrndcoef + kernel_dim, nrndtcor = nrndtot*(nrndtot-1)//2
    for nrnd in range(1, 5):
        for kdim in range(0, 5):
            names = [f"V{i}" for i in range(nrnd)]
            s = MixingSpec.from_var_names(
                names, normvar=names, kernel_dim=kdim
            )
            assert s.nrndtot == nrnd + kdim
            assert s.nrndtcor == (nrnd + kdim) * (nrnd + kdim - 1) // 2
            assert s.n_kern == max(kdim - 1, 0)


def test_mnl_default_unchanged() -> None:
    # kernel_dim omitted -> plain MNL facade (nrndtot == nrndcoef, n_kern == 0)
    s = MixingSpec.from_var_names(VAR_NAMES, normvar=NORMVAR)
    assert s.nrndtot == s.nrndcoef == 2
    assert s.nrndtcor == 1  # 2*1//2
    assert s.n_kern == 0
    assert s.kernel_dim == 0


def test_kernel_dim_negative_raises() -> None:
    with pytest.raises(ValueError):
        MixingSpec.from_var_names(VAR_NAMES, normvar=NORMVAR, kernel_dim=-1)


# ---------------------------------------------------------------------------
# ParamLayout: MNP ordering (kern between scal and lam)
# ---------------------------------------------------------------------------

def test_total_n_theta(layout: ParamLayout) -> None:
    assert layout.n_theta == 18  # 7 + 6 + 2 + 1 + 2


def test_mnp_slices(layout: ParamLayout) -> None:
    sl = layout.slices()
    assert sl["beta"] == slice(0, 7)  # means
    assert sl["rcor"] == slice(7, 13)  # joint correlation
    assert sl["scal"] == slice(13, 15)  # rc-scale
    assert sl["kern"] == slice(15, 16)  # kernel-scale (BETWEEN scal and lam)
    assert sl["lam"] == slice(16, 18)  # lam


def test_mnp_slices_cover_no_gap_no_overlap(layout: ParamLayout) -> None:
    sl = layout.slices()
    # physical order places kern between scal and lam
    order = ["beta", "rcor", "scal", "kern", "lam"]
    assert list(sl.keys()) == order
    covered = np.zeros(layout.n_theta, dtype=int)
    prev_stop = 0
    for name in order:
        s = sl[name]
        assert s.start == prev_stop
        prev_stop = s.stop
        covered[s] += 1
    assert prev_stop == layout.n_theta
    np.testing.assert_array_equal(covered, np.ones(layout.n_theta, dtype=int))


def test_pack_slices_roundtrip(layout: ParamLayout) -> None:
    rng = np.random.default_rng(0)
    dbeta = rng.standard_normal(layout.n_beta)
    drcor = rng.standard_normal(layout.n_rcor)
    dscal = rng.standard_normal(layout.n_scal)
    dlam = rng.standard_normal(layout.n_lam)
    dkern = rng.standard_normal(layout.n_kern)

    theta = layout.pack(dbeta, drcor, dscal, dlam, dkern)
    assert theta.shape == (18,)

    sl = layout.slices()
    np.testing.assert_array_equal(theta[sl["beta"]], dbeta)
    np.testing.assert_array_equal(theta[sl["rcor"]], drcor)
    np.testing.assert_array_equal(theta[sl["scal"]], dscal)
    np.testing.assert_array_equal(theta[sl["kern"]], dkern)
    np.testing.assert_array_equal(theta[sl["lam"]], dlam)


def test_pack_places_kern_before_lam(layout: ParamLayout) -> None:
    # Distinct sentinel values per block so ordering is unambiguous.
    theta = layout.pack(
        np.full(7, 1.0),
        np.full(6, 2.0),
        np.full(2, 3.0),
        np.full(2, 5.0),  # lam
        np.full(1, 4.0),  # kern
    )
    expected = np.array(
        [1.0] * 7 + [2.0] * 6 + [3.0] * 2 + [4.0] * 1 + [5.0] * 2
    )
    np.testing.assert_array_equal(theta, expected)


def test_mnl_layout_ordering_unchanged() -> None:
    # Default (kern last) ordering must be preserved for the MNL path.
    lay = ParamLayout(n_beta=7, n_rcor=1, n_scal=2, n_lam=2, n_kern=1)
    assert list(lay.slices().keys()) == ["beta", "rcor", "scal", "lam", "kern"]
    theta = lay.pack(
        np.full(7, 1.0),
        np.full(1, 2.0),
        np.full(2, 3.0),
        np.full(2, 5.0),  # lam
        np.full(1, 4.0),  # kern (last)
    )
    expected = np.array([1.0] * 7 + [2.0] * 1 + [3.0] * 2 + [5.0] * 2 + [4.0] * 1)
    np.testing.assert_array_equal(theta, expected)


def test_pack_wrong_length_raises(layout: ParamLayout) -> None:
    with pytest.raises(ValueError):
        layout.pack(
            np.zeros(6), np.zeros(6), np.zeros(2), np.zeros(2), np.zeros(1)
        )


# ---------------------------------------------------------------------------
# wker sum-of-squares reparameterization
# ---------------------------------------------------------------------------

def _wker_ref(xscalker: np.ndarray) -> np.ndarray:
    """Literal GAUSS form: sqrt(logitmod([0|xscalker]))[1:]."""
    full = np.concatenate([[0.0], np.asarray(xscalker, dtype=float)])
    return np.sqrt(logitmod(full))[1:]


def test_wker_matches_gauss_form() -> None:
    rng = np.random.default_rng(3)
    for n_kern in (1, 2, 3):
        xscalker = rng.standard_normal(n_kern)
        wker, _ = wker_reparam(xscalker)
        np.testing.assert_allclose(wker, _wker_ref(xscalker), atol=1e-12)


def test_wker_sum_of_squares_normalized() -> None:
    # Including the dropped reference entry, the full scale vector has
    # unit sum of squares (GAUSS "SUM OF SQUARES ... NORMALIZED TO ONE").
    rng = np.random.default_rng(7)
    xscalker = rng.standard_normal(1)
    wker, _ = wker_reparam(xscalker)
    full = np.concatenate([[0.0], xscalker])
    wfull = np.sqrt(logitmod(full))
    np.testing.assert_allclose(np.sum(wfull**2), 1.0, atol=1e-12)
    # dropped-reference part equals wker
    np.testing.assert_allclose(wfull[1:], wker, atol=1e-12)


def test_wker_gate_case() -> None:
    # nc = 3 -> n_kern = 1; the single stored kernel-scale param.
    xscalker = np.array([0.75])
    wker, dwker = wker_reparam(xscalker, want_grad=True)
    assert wker.shape == (1,)
    np.testing.assert_allclose(wker, _wker_ref(xscalker), atol=1e-12)
    assert dwker.shape == (1, 1)


def test_wker_gradient_vs_fd() -> None:
    rng = np.random.default_rng(11)
    eps = 1e-6
    for n_kern in (1, 2, 3):
        x0 = rng.standard_normal(n_kern)
        _, dwker = wker_reparam(x0, want_grad=True)
        assert dwker.shape == (n_kern, n_kern)
        fd = np.zeros((n_kern, n_kern))
        for j in range(n_kern):
            xp_ = x0.copy()
            xm = x0.copy()
            xp_[j] += eps
            xm[j] -= eps
            wp, _ = wker_reparam(xp_)
            wm, _ = wker_reparam(xm)
            fd[:, j] = (wp - wm) / (2 * eps)
        np.testing.assert_allclose(dwker, fd, atol=1e-6)


def test_wker_empty_kern() -> None:
    wker, dwker = wker_reparam(np.zeros(0), want_grad=True)
    assert wker.shape == (0,)
    assert dwker.shape == (0, 0)
