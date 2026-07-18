"""FACADE GATE (no GAUSS): MixMNL interface conformance + fixed-coef collapse.

Covers plan tasks T0.15 / T0.16:

* :class:`MixMNLControl` round-trips through ``dataclasses.asdict``.
* :class:`MixMNLResults` exposes the canonical harmonized fields
  (``params`` / ``loglik`` / ``n_iter``) and its deprecated aliases
  (``b`` / ``ll`` / ``n_iterations`` / ``ll_total``) still resolve but warn.
* With an **empty** random-coefficient spec (``normvar = logvar = yjvar = {}``),
  :meth:`MixMNLModel.fit` converges to the SAME log-likelihood and coefficients
  as the shipped fixed-coefficient :class:`MNLModel` on a small dataset (1e-6).

Imports use the private module paths (the family ``__init__`` is wired
separately), mirroring ``tests/test_mixed/test_mixmnl_collapse.py``.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._reparam import EstimationSpace
from pybhatlib.models.mixmnl._mixmnl_control import MixMNLControl
from pybhatlib.models.mixmnl._mixmnl_model import MixMNLModel
from pybhatlib.models.mixmnl._mixmnl_results import MixMNLResults
from pybhatlib.models.mnl._mnl_control import MNLControl
from pybhatlib.models.mnl._mnl_model import MNLModel
from pybhatlib.vecup._panel import PanelIndex

_ALTS = ["Alt1_ch", "Alt2_ch"]
_SPEC = {
    "X1": {"Alt1_ch": "x1_a1", "Alt2_ch": "x1_a2"},
    "X2": {"Alt1_ch": "x2_a1", "Alt2_ch": "x2_a2"},
}


@pytest.fixture
def synthetic_mnl_data() -> pd.DataFrame:
    """Small binary-choice dataset (same construction as tests/test_models)."""
    rng = np.random.default_rng(42)
    n_obs = 200

    x = rng.standard_normal(n_obs)
    z = rng.standard_normal(n_obs)
    x_a1 = x + rng.standard_normal(n_obs) * 0.1
    x_a2 = x + rng.standard_normal(n_obs) * 0.1
    z_a1 = z + rng.standard_normal(n_obs) * 0.1
    z_a2 = z + rng.standard_normal(n_obs) * 0.1

    beta_true = np.array([1.0, -1.0, 0.5, -0.5])
    V1 = beta_true[0] * x_a1 + beta_true[2] * z_a1
    V2 = beta_true[1] * x_a2 + beta_true[3] * z_a2
    probs = np.exp(np.column_stack([V1, V2]))
    probs = probs / probs.sum(axis=1, keepdims=True)
    draws = rng.uniform(size=n_obs)
    choices = (draws > probs[:, 0]).astype(int)

    return pd.DataFrame(
        {
            "Alt1_ch": 1 - choices,
            "Alt2_ch": choices,
            "x1_a1": x_a1,
            "x1_a2": x_a2,
            "x2_a1": z_a1,
            "x2_a2": z_a2,
        }
    )


# ---------------------------------------------------------------------------
# T0.15 -- control round-trip
# ---------------------------------------------------------------------------

class TestMixMNLControlRoundTrip:
    def test_defaults_construct(self):
        ctrl = MixMNLControl()
        assert ctrl.n_rep == 125
        assert ctrl.randdiag is False
        assert ctrl.intordn1 == 20
        assert ctrl.spher is False
        assert ctrl.scal == 1.0
        assert ctrl.person_id is None
        assert ctrl.weight_var is None
        assert tuple(ctrl.fix_location_zero) == ()
        assert tuple(ctrl.normvar) == ()

    def test_asdict_roundtrip(self):
        ctrl = MixMNLControl(
            normvar=("X1",),
            yjvar=("X2",),
            varneg=("X1",),
            person_id="pid",
            n_rep=25,
            intordn1=30,
            spher=False,
            scal=2.0,
            optimizer="lbfgsb",
            tol=1e-9,
            verbose=0,
        )
        rebuilt = MixMNLControl(**dataclasses.asdict(ctrl))
        assert rebuilt == ctrl
        # spot-check a representative slice of fields survived verbatim
        assert tuple(rebuilt.normvar) == ("X1",)
        assert tuple(rebuilt.yjvar) == ("X2",)
        assert rebuilt.person_id == "pid"
        assert rebuilt.n_rep == 25
        assert rebuilt.intordn1 == 30
        assert rebuilt.scal == 2.0
        assert rebuilt.optimizer == "lbfgsb"

    def test_randdiag_removes_correlation_parameters(self, synthetic_mnl_data):
        ctrl = MixMNLControl(normvar=("X1", "X2"), randdiag=True, n_rep=2)
        model = MixMNLModel(synthetic_mnl_data, _ALTS, spec=_SPEC, control=ctrl)
        spec, layout = model._build_spec_layout()
        state = EstimationSpace(layout).unpack(
            np.zeros(layout.n_theta), spec
        )

        assert layout.n_rcor == 0
        np.testing.assert_array_equal(state.omegastar, np.eye(2))

        estimator = model._build_estimator(
            spec, layout, PanelIndex.from_ids(model.person_ids)
        )
        loglik, score = estimator.simulated_loglik(
            np.zeros(layout.n_theta), want_grad=True
        )
        assert np.all(np.isfinite(loglik))
        assert score.shape == (len(synthetic_mnl_data), layout.n_theta)

    def test_observation_weights_are_averaged_by_person(self, synthetic_mnl_data):
        data = synthetic_mnl_data.iloc[:6].copy()
        data["pid"] = [0, 0, 1, 1, 1, 2]
        data["weight"] = [1.0, 3.0, 2.0, 4.0, 6.0, 5.0]
        control = MixMNLControl(
            person_id="pid", weight_var="weight", n_rep=1
        )
        model = MixMNLModel(data, _ALTS, spec=_SPEC, control=control)
        spec, layout = model._build_spec_layout()
        panel = PanelIndex.from_ids(model.person_ids)
        estimator = model._build_estimator(spec, layout, panel)

        np.testing.assert_allclose(estimator.weightind, [2.0, 4.0, 5.0])

    def test_no_weight_column_defaults_to_person_weights_of_one(
        self, synthetic_mnl_data
    ):
        control = MixMNLControl(n_rep=1)
        model = MixMNLModel(synthetic_mnl_data, _ALTS, spec=_SPEC, control=control)
        spec, layout = model._build_spec_layout()
        panel = PanelIndex.from_ids(model.person_ids)

        estimator = model._build_estimator(spec, layout, panel)
        np.testing.assert_array_equal(estimator.weightind, np.ones(panel.n_ind))

    def test_replace_is_nondestructive(self):
        ctrl = MixMNLControl(n_rep=10)
        ctrl2 = dataclasses.replace(ctrl, n_rep=99)
        assert ctrl.n_rep == 10
        assert ctrl2.n_rep == 99


# ---------------------------------------------------------------------------
# T0.15 -- results canonical fields + deprecated aliases warn
# ---------------------------------------------------------------------------

def _minimal_results(**overrides) -> MixMNLResults:
    kw = dict(
        params=np.array([0.1, -0.2]),
        se=np.array([0.05, 0.06]),
        t_stat=np.array([2.0, -3.3]),
        p_value=np.array([0.04, 0.001]),
        gradient=np.array([1e-7, -1e-7]),
        loglik=-1.2345,
        n_obs=200,
        n_ind=200,
        param_names=["X1", "X2"],
        corr_matrix=np.eye(2),
        cov_matrix=np.eye(2) * 0.01,
        n_iter=7,
        convergence_time=0.01,
        converged=True,
        return_code=0,
    )
    kw.update(overrides)
    return MixMNLResults(**kw)


class TestMixMNLResultsInterface:
    def test_canonical_fields(self):
        r = _minimal_results()
        assert np.allclose(r.params, [0.1, -0.2])
        assert r.loglik == pytest.approx(-1.2345)
        assert r.n_iter == 7

    def test_aliases_warn_and_forward(self):
        r = _minimal_results()
        with pytest.warns(DeprecationWarning):
            assert np.allclose(r.b, r.params)
        with pytest.warns(DeprecationWarning):
            assert r.ll == pytest.approx(r.loglik)
        with pytest.warns(DeprecationWarning):
            assert r.n_iterations == r.n_iter
        with pytest.warns(DeprecationWarning):
            assert r.ll_total == pytest.approx(r.loglik * r.n_obs)

    def test_legacy_construction_warns(self):
        # build the kwargs without ``params`` and inject the legacy ``b=``
        kw = dict(
            se=np.array([0.05, 0.06]),
            t_stat=np.array([2.0, -3.3]),
            p_value=np.array([0.04, 0.001]),
            gradient=np.array([1e-7, -1e-7]),
            loglik=-1.2345,
            n_obs=200,
            n_ind=200,
            param_names=["X1", "X2"],
            corr_matrix=np.eye(2),
            cov_matrix=np.eye(2) * 0.01,
            n_iter=7,
            convergence_time=0.01,
            converged=True,
            return_code=0,
            b=np.array([0.3, 0.4]),
        )
        with pytest.warns(DeprecationWarning):
            r = MixMNLResults(**kw)
        # legacy ``b=`` mapped onto canonical ``params``
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert np.allclose(r.params, [0.3, 0.4])

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(TypeError):
            _minimal_results(not_a_field=123)


# ---------------------------------------------------------------------------
# T0.16 -- empty rc spec reproduces the fixed-coefficient MNL
# ---------------------------------------------------------------------------

class TestMixMNLCollapsesToMNL:
    def test_empty_rc_matches_mnl(self, synthetic_mnl_data: pd.DataFrame):
        # --- fixed-coefficient MNL reference -------------------------------
        mnl = MNLModel(
            data=synthetic_mnl_data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=MNLControl(verbose=0, maxiter=200),
        ).fit()

        # --- MixMNL with an EMPTY random-coefficient spec ------------------
        ctrl = MixMNLControl(
            normvar=(), logvar=(), yjvar=(),   # no random coefficients
            n_rep=1, verbose=0, optimizer="bfgs", tol=1e-10,
        )
        mix = MixMNLModel(
            data=synthetic_mnl_data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=ctrl,
        ).fit()

        # cross-sectional -> one individual per observation
        assert mix.n_ind == mix.n_obs == len(synthetic_mnl_data)

        # coefficients agree to 1e-6
        np.testing.assert_allclose(mix.params, mnl.params, rtol=0, atol=1e-6)

        # log-likelihood agrees (mean and total) to 1e-6
        np.testing.assert_allclose(mix.loglik, mnl.loglik, rtol=0, atol=1e-6)
        np.testing.assert_allclose(
            mix.loglik * mix.n_ind, mnl.loglik * mnl.n_obs, rtol=0, atol=1e-6
        )

    def test_empty_rc_theta_is_beta_only(self, synthetic_mnl_data: pd.DataFrame):
        """No random coefficients -> theta has exactly the fixed-coef block."""
        mix_model = MixMNLModel(
            data=synthetic_mnl_data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=MixMNLControl(normvar=(), logvar=(), yjvar=(), verbose=0),
        )
        spec, layout = mix_model._build_spec_layout()
        assert spec.nrndcoef == 0
        assert layout.n_theta == len(_SPEC)
        assert layout.n_rcor == layout.n_scal == layout.n_lam == layout.n_kern == 0
