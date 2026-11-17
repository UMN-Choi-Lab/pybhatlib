"""MixMNL random-coefficient summary offsets match the reporting vector.

The reporting vector is ``[beta | corr | sd | lam]`` with the correlation and
lambda blocks always present (fixed entries for ``randdiag`` / non-YJ
coefficients), independent of the smaller estimation-space ``ParamLayout``.
"""

from __future__ import annotations

import numpy as np
import pytest

from pybhatlib.models.mixmnl._mixmnl_control import MixMNLControl
from pybhatlib.models.mixmnl._mixmnl_model import MixMNLModel
from tests.test_mixed.test_mixmnl_facade import (  # noqa: F401
    _ALTS,
    _SPEC,
    synthetic_mnl_data,
)


@pytest.mark.parametrize("randdiag", [True, False])
def test_rc_summary_matches_reporting_vector(synthetic_mnl_data, randdiag):
    ctrl = MixMNLControl(
        normvar=("X1",), yjvar=("X2",), randdiag=randdiag,
        n_rep=4, verbose=0, maxiter=5, tol=1e-6,
    )
    res = MixMNLModel(synthetic_mnl_data, _ALTS, spec=_SPEC, control=ctrl).fit()
    names, p = res.param_names, res.params

    np.testing.assert_allclose(
        res.random_coef_scales, [p[names.index("sd[X1]")], p[names.index("sd[X2]")]]
    )
    np.testing.assert_allclose(
        res.random_coef_shapes, [p[names.index("lam[X1]")], p[names.index("lam[X2]")]]
    )
    assert res.random_coef_shapes[0] == pytest.approx(1.0)  # normal: fixed
    rho = p[names.index("corr[X1,X2]")]
    if randdiag:
        assert rho == 0.0
    np.testing.assert_allclose(res.random_coef_corr_matrix[0, 1], rho)
