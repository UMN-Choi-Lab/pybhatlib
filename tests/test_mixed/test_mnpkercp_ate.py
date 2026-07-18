"""GATE (no GAUSS): MNPKerCP draw-integrated predict / ATE / forecast.

The mixed-panel MNP (MNPKerCP) family wires ``.predict()`` / ``.forecast`` /
``.ate()`` to the shared mixed-prediction machinery
(:mod:`pybhatlib.mixed._predict`) by lifting the shipped fixed-coefficient MNP
choice-probability formulation over the mixing draws.  Validation (per Dale's
guidance -- NO GAUSS forecast oracle) is by:

1. **COLLAPSE** -- with an empty random-coefficient spec (``nrndcoef = 0``) the
   family's draw-integrated ``predict`` / ``ate`` reproduce the shipped
   fixed-coefficient MNP (:func:`pybhatlib.models.mnp._mnp_ate.mnp_ate` /
   ``mnp_ate_from_params``) on the same data, at the *same* fitted coefficients,
   to ``1e-6``.  With two alternatives the MVNCD reduces to the univariate normal
   CDF, so the two independent MNP kernels agree to machine precision.
2. **Mixing active** -- with a normal random coefficient, ``predict`` returns
   valid market shares (non-negative, summing to one) and ``ate(scenarios=...)``
   returns a well-formed result whose per-scenario shares are valid.
3. **Interface conformance** -- the result is a
   :class:`~pybhatlib.mixed._predict.MixedATEResult`
   (:class:`~pybhatlib.models._ate_common.ATEResultMixin`) exposing the
   harmonized ``predicted_shares`` / ``shares_per_scenario`` /
   ``alternative_names`` / ``n_obs`` fields and the ``.comparison()`` /
   ``.summary()`` surface; scenarios are normalised via
   :func:`~pybhatlib.models._ate_common.scenarios_to_dict`.

Imports use the private module paths (the family ``__init__`` is wired
separately), mirroring the sibling ``tests/test_mixed`` gates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._predict import MixedATEResult
from pybhatlib.models._ate_common import ATEResultMixin
from pybhatlib.models.mnp._mnp_ate import mnp_ate_from_params
from pybhatlib.models.mnpkercp._mnpkercp_ate import mnpkercp_ate_from_params
from pybhatlib.models.mnpkercp._mnpkercp_control import MNPKerCPControl
from pybhatlib.models.mnpkercp._mnpkercp_model import MNPKerCPModel

_ALTS = ["Alt1_ch", "Alt2_ch"]
_SPEC = {
    "X": {"Alt1_ch": "x_a1", "Alt2_ch": "x_a2"},
    "Z": {"Alt1_ch": "z_a1", "Alt2_ch": "z_a2"},
}


@pytest.fixture
def binary_probit_data() -> pd.DataFrame:
    """Small two-alternative probit dataset (one observation per person)."""
    rng = np.random.default_rng(0)
    n_obs = 120
    x_a1 = rng.standard_normal(n_obs)
    x_a2 = rng.standard_normal(n_obs)
    z_a1 = rng.standard_normal(n_obs)
    z_a2 = rng.standard_normal(n_obs)
    latent = (
        0.8 * (x_a1 - x_a2)
        - 0.6 * (z_a1 - z_a2)
        + rng.standard_normal(n_obs)
    )
    alt1 = (latent > 0).astype(int)
    return pd.DataFrame(
        {
            "Alt1_ch": alt1,
            "Alt2_ch": 1 - alt1,
            "x_a1": x_a1,
            "x_a2": x_a2,
            "z_a1": z_a1,
            "z_a2": z_a2,
        }
    )


# ---------------------------------------------------------------------------
# (1) COLLAPSE -- nrndcoef == 0 reproduces the shipped fixed-coefficient MNP
# ---------------------------------------------------------------------------

class TestCollapseToFixedCoefMNP:
    def _fit_no_mixing(self, data: pd.DataFrame) -> MNPKerCPModel:
        model = MNPKerCPModel(
            data=data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=MNPKerCPControl(
                normvar=(), logvar=(), yjvar=(),   # no random coefficients
                n_rep=1, copula=False, verbose=0, maxiter=400,
            ),
        )
        model.fit()
        return model

    def test_empty_rc_theta_is_beta_only(self, binary_probit_data):
        """No random coefficients -> theta has exactly the fixed-coef block."""
        model = MNPKerCPModel(
            data=binary_probit_data, alternatives=_ALTS, spec=_SPEC,
            control=MNPKerCPControl(normvar=(), logvar=(), yjvar=(), verbose=0),
        )
        spec, layout = model._build_spec_layout()
        assert spec.nrndcoef == 0
        assert layout.n_theta == len(_SPEC)
        assert layout.n_rcor == layout.n_scal == layout.n_lam == 0

    def test_predict_collapse(self, binary_probit_data):
        model = self._fit_no_mixing(binary_probit_data)
        beta = np.asarray(model.results_.params, dtype=np.float64)[: len(_SPEC)]

        # draw-integrated (nrndcoef == 0) mixed prediction
        shares = model.predict()

        # shipped fixed-coefficient MNP at the SAME coefficients (IID kernel)
        ref = mnp_ate_from_params(
            beta, kernel_cov=None,
            data=binary_probit_data, spec=_SPEC, alternatives=_ALTS,
        )

        assert shares.shape == (len(_ALTS),)
        np.testing.assert_allclose(
            shares, ref.predicted_shares, rtol=0, atol=1e-6
        )
        # forecast is the same share vector.
        np.testing.assert_allclose(model.forecast(), shares, rtol=0, atol=1e-12)

    def test_ate_collapse(self, binary_probit_data):
        model = self._fit_no_mixing(binary_probit_data)
        beta = np.asarray(model.results_.params, dtype=np.float64)[: len(_SPEC)]

        scenarios = {
            "base": {"x_a1": 0.0, "x_a2": 0.0},
            "treat": {"x_a1": 1.0, "x_a2": 0.0},
        }

        mixed = model.ate(scenarios=scenarios)
        ref = mnp_ate_from_params(
            beta, kernel_cov=None,
            data=binary_probit_data, spec=_SPEC, alternatives=_ALTS,
            scenarios=scenarios,
        )

        # baseline shares collapse
        np.testing.assert_allclose(
            mixed.predicted_shares, ref.predicted_shares, rtol=0, atol=1e-6
        )
        # per-scenario shares collapse
        assert set(mixed.shares_per_scenario) == set(ref.shares_per_scenario)
        for name in scenarios:
            np.testing.assert_allclose(
                mixed.shares_per_scenario[name],
                ref.shares_per_scenario[name],
                rtol=0, atol=1e-6,
            )

    def test_ate_from_params_collapse(self, binary_probit_data):
        """mnpkercp_ate_from_params (no re-fit) collapses to the fixed-coef MNP."""
        model = self._fit_no_mixing(binary_probit_data)
        beta = np.asarray(model.results_.params, dtype=np.float64)[: len(_SPEC)]

        scenarios = {
            "base": {"x_a1": 0.0, "x_a2": 0.0},
            "treat": {"x_a1": 1.0, "x_a2": 0.0},
        }

        # estimation-space theta == reporting beta when there is no mixing.
        from_params = mnpkercp_ate_from_params(
            beta,
            data=binary_probit_data,
            alternatives=_ALTS,
            spec=_SPEC,
            scenarios=scenarios,
            control=MNPKerCPControl(
                normvar=(), logvar=(), yjvar=(),
                n_rep=1, copula=False, verbose=0,
            ),
        )
        ref = mnp_ate_from_params(
            beta, kernel_cov=None,
            data=binary_probit_data, spec=_SPEC, alternatives=_ALTS,
            scenarios=scenarios,
        )

        assert isinstance(from_params, MixedATEResult)
        np.testing.assert_allclose(
            from_params.predicted_shares, ref.predicted_shares, rtol=0, atol=1e-6
        )
        for name in scenarios:
            np.testing.assert_allclose(
                from_params.shares_per_scenario[name],
                ref.shares_per_scenario[name],
                rtol=0, atol=1e-6,
            )


# ---------------------------------------------------------------------------
# (1b) ROUND-TRIP -- fitted reporting params reproduce the fitted .ate()
# ---------------------------------------------------------------------------

class TestATEFromParamsRoundTrip:
    """mnpkercp_ate_from_params consumes the fitted reporting-space params.

    CAVEAT-1 gate: the fitted ``results_.params`` are documented as
    reporting-space; feeding them to :func:`mnpkercp_ate_from_params` (which now
    builds the estimator with a ``ReportingSpace``) must reproduce the fitted
    :meth:`MNPKerCPModel.ate` bit-for-bit -- no reporting -> estimation inversion
    is performed, and the deterministic scrambled-Halton source (fixed
    ``draw_seed``) is reproduced by the rebuild so the draws are identical.
    """

    def _control(self, **overrides) -> MNPKerCPControl:
        base = dict(
            normvar=("X",), n_rep=6, copula=False, draw_seed=3,
            verbose=0, maxiter=60,
        )
        base.update(overrides)
        return MNPKerCPControl(**base)

    def test_reporting_params_reproduce_fitted_ate(self, binary_probit_data):
        model = MNPKerCPModel(
            data=binary_probit_data, alternatives=_ALTS, spec=_SPEC,
            control=self._control(),
        )
        model.fit()

        scenarios = {
            "base": {"x_a1": 0.0, "x_a2": 0.0},
            "treat": {"x_a1": 1.0, "x_a2": 0.0},
        }
        fitted = model.ate(scenarios=scenarios)

        # fitted params are reporting-space (natural) -> plug in verbatim.
        params = np.asarray(model.results_.params, dtype=np.float64)
        from_params = mnpkercp_ate_from_params(
            params,
            data=binary_probit_data,
            alternatives=_ALTS,
            spec=_SPEC,
            scenarios=scenarios,
            control=self._control(),   # same draw_seed / n_rep -> identical draws
        )

        assert isinstance(from_params, MixedATEResult)
        np.testing.assert_allclose(
            from_params.predicted_shares, fitted.predicted_shares,
            rtol=0, atol=1e-6,
        )
        assert set(from_params.shares_per_scenario) == set(
            fitted.shares_per_scenario
        )
        for name in scenarios:
            np.testing.assert_allclose(
                from_params.shares_per_scenario[name],
                fitted.shares_per_scenario[name],
                rtol=0, atol=1e-6,
            )


# ---------------------------------------------------------------------------
# (2) Mixing active -- valid shares + well-formed ATE
# ---------------------------------------------------------------------------

class TestMixingActive:
    def _fit_mixing(self, data: pd.DataFrame) -> MNPKerCPModel:
        model = MNPKerCPModel(
            data=data,
            alternatives=_ALTS,
            spec=_SPEC,
            control=MNPKerCPControl(
                normvar=("X",),           # one normal random coefficient
                n_rep=6, copula=False, draw_seed=3, verbose=0, maxiter=60,
            ),
        )
        model.fit()
        return model

    def test_predict_shares_valid(self, binary_probit_data):
        model = self._fit_mixing(binary_probit_data)
        shares = model.predict()
        assert shares.shape == (len(_ALTS),)
        assert np.all(shares >= 0.0)
        np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-10)

    def test_ate_well_formed(self, binary_probit_data):
        model = self._fit_mixing(binary_probit_data)
        scenarios = {"base": {"x_a1": 0.0}, "treat": {"x_a1": 1.0}}
        result = model.ate(scenarios=scenarios)

        assert isinstance(result, MixedATEResult)
        assert result.n_obs == len(binary_probit_data)
        assert set(result.shares_per_scenario) == {"base", "treat"}
        for s in result.shares_per_scenario.values():
            assert s.shape == (len(_ALTS),)
            assert np.all(s >= 0.0)
            np.testing.assert_allclose(s.sum(), 1.0, atol=1e-10)
        np.testing.assert_allclose(result.predicted_shares.sum(), 1.0, atol=1e-10)

    def test_scenario_moves_shares(self, binary_probit_data):
        """A strong covariate shift changes the draw-integrated shares."""
        model = self._fit_mixing(binary_probit_data)
        base = model.predict(scenario={"x_a1": -2.0})
        treat = model.predict(scenario={"x_a1": 2.0})
        assert not np.allclose(base, treat)


# ---------------------------------------------------------------------------
# (3) Interface conformance -- harmonized ATEResultMixin surface
# ---------------------------------------------------------------------------

class TestInterfaceConformance:
    def test_result_is_harmonized(self, binary_probit_data):
        model = MNPKerCPModel(
            data=binary_probit_data, alternatives=_ALTS, spec=_SPEC,
            control=MNPKerCPControl(
                normvar=("X",), n_rep=4, copula=False, draw_seed=1,
                verbose=0, maxiter=40,
            ),
        )
        model.fit()

        scenarios = {"lo": {"x_a1": 0.0}, "hi": {"x_a1": 1.0}}
        result = model.ate(scenarios=scenarios)

        # inherits the shared ATE mixin (comparison / summary surface)
        assert isinstance(result, ATEResultMixin)

        # harmonized fields
        assert result.alternative_names == _ALTS
        assert result.predicted_shares.shape == (len(_ALTS),)

        # .comparison(base, treatment) -> finite per-alternative pct change
        pct = result.comparison("lo", "hi")
        assert pct.shape == (len(_ALTS),)
        assert np.all(np.isfinite(pct))

        # .summary() renders without error
        text = result.summary()
        assert isinstance(text, str) and len(text) > 0

    def test_ate_requires_scenarios(self, binary_probit_data):
        model = MNPKerCPModel(
            data=binary_probit_data, alternatives=_ALTS, spec=_SPEC,
            control=MNPKerCPControl(verbose=0, n_rep=1),
        )
        model.fit()
        with pytest.raises(ValueError):
            model.ate()

    def test_predict_before_fit_raises(self, binary_probit_data):
        model = MNPKerCPModel(
            data=binary_probit_data, alternatives=_ALTS, spec=_SPEC,
            control=MNPKerCPControl(verbose=0),
        )
        with pytest.raises(RuntimeError):
            model.predict()
