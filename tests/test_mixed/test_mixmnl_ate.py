"""GATE: MixMNL .predict() / .ate() / forecast wiring (Phase 4, self-contained).

NO GAUSS oracle exists for the mixed forecast (Dale's guidance) -- validation is
by **collapse** to the shipped fixed-coefficient MNL forecast / ATE plus
**interface conformance**, not GAUSS parity.

Checks
------
1. **COLLAPSE** -- with an empty random-coefficient spec (``nrndcoef == 0``) the
   MixMNL family's draw-integrated ``predict`` / ``ate`` equal the shipped
   fixed-coefficient :func:`~pybhatlib.models.mnl.mnl_predict` /
   :func:`~pybhatlib.models.mnl.mnl_ate` on the same small data, at the *same*
   coefficients, to ``1e-6``.
2. **MIXING ACTIVE** -- with a normal random coefficient, ``predict`` returns
   valid per-observation probabilities (rows sum to one), ``predict_shares``
   returns valid market shares, and ``ate(scenarios=...)`` returns a well-formed
   result.
3. **INTERFACE CONFORMANCE** -- the ATE result exposes the harmonized
   :class:`~pybhatlib.models._ate_common.ATEResultMixin` surface
   (``predicted_shares`` / ``shares_per_scenario`` normalised via
   ``scenarios_to_dict`` / ``.comparison()`` / ``.summary()``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._ate_common import ATEResultMixin
from pybhatlib.models.mixmnl import MixMNLControl, MixMNLModel, MixMNLResults
from pybhatlib.models.mixmnl._mixmnl_ate import mixmnl_ate, mixmnl_ate_from_params
from pybhatlib.models.mixmnl._mixmnl_forecast import (
    mixmnl_predict,
    mixmnl_predict_choice,
)
from pybhatlib.mixed._predict import MixedATEResult
from pybhatlib.models.mnl._mnl_ate import mnl_ate_from_params
from pybhatlib.models.mnl._mnl_forecast import mnl_predict
from pybhatlib.models.mnl._mnl_results import MNLResults

NC = 3                                   # alternatives
NVAR = 2                                 # random/fixed coefficients
ALTS = [f"ch{k}" for k in range(NC)]
VAR_NAMES = [f"x{v}" for v in range(NVAR)]
SPEC = {f"x{v}": {f"ch{k}": f"x{v}_{k}" for k in range(NC)} for v in range(NVAR)}


def _make_data(seed: int, occ: list[int]) -> pd.DataFrame:
    """Synthetic long-format panel frame: one row per (person, occasion)."""
    rng = np.random.default_rng(seed)
    pid = np.concatenate([np.full(c, i) for i, c in enumerate(occ)])
    n = pid.shape[0]
    cols: dict[str, np.ndarray] = {"pid": pid}
    for v in range(NVAR):
        for k in range(NC):
            cols[f"x{v}_{k}"] = 0.5 * rng.standard_normal(n)
    chosen = np.zeros((n, NC), dtype=np.float64)
    chosen[np.arange(n), rng.integers(0, NC, size=n)] = 1.0
    for k in range(NC):
        cols[f"ch{k}"] = chosen[:, k]
    return pd.DataFrame(cols)


def _mixmnl(frame: pd.DataFrame, *, normvar=(), n_rep=3, draw_seed=7) -> MixMNLModel:
    return MixMNLModel(
        data=frame,
        alternatives=ALTS,
        availability="none",
        spec=SPEC,
        control=MixMNLControl(
            normvar=tuple(normvar),
            person_id="pid",
            n_rep=n_rep,
            draw_seed=draw_seed,
            verbose=0,
        ),
    )


# ---------------------------------------------------------------------------
# (1) COLLAPSE: nrndcoef == 0 -> fixed-coef MNL predict / ate, same beta
# ---------------------------------------------------------------------------

def test_collapse_predict_equals_fixed_mnl():
    frame = _make_data(seed=11, occ=[1, 1, 2, 1, 3, 1, 2])
    rng = np.random.default_rng(3)
    beta = 0.6 * rng.standard_normal(NVAR)

    model = _mixmnl(frame, normvar=())              # nrndcoef == 0
    comp = model._predict_components(beta)
    assert comp.pipeline.n_rnd == 0

    mix_pred = mixmnl_predict(comp, frame, SPEC)     # (n_obs, NC)

    X, _ = parse_spec(SPEC, frame, ALTS, nseg=1)
    mnl_res = MNLResults.from_estimates(beta, param_names=VAR_NAMES)
    ref_pred = mnl_predict(mnl_res, np.asarray(X, dtype=np.float64))

    assert mix_pred.shape == ref_pred.shape == (len(frame), NC)
    np.testing.assert_allclose(mix_pred, ref_pred, rtol=0, atol=1e-6)


def test_collapse_predict_shares_equals_fixed_mnl():
    frame = _make_data(seed=12, occ=[2, 1, 1, 3, 1, 2])
    rng = np.random.default_rng(4)
    beta = 0.6 * rng.standard_normal(NVAR)

    model = _mixmnl(frame, normvar=())
    comp = model._predict_components(beta)
    # Market shares = sample mean of the per-observation draw-integrated probs.
    mix_shares = mixmnl_predict(comp, frame, SPEC).mean(axis=0)      # (NC,)

    X, _ = parse_spec(SPEC, frame, ALTS, nseg=1)
    mnl_res = MNLResults.from_estimates(beta, param_names=VAR_NAMES)
    ref_shares = mnl_predict(mnl_res, np.asarray(X, dtype=np.float64)).mean(axis=0)

    np.testing.assert_allclose(mix_shares, ref_shares, rtol=0, atol=1e-6)


def test_collapse_ate_equals_fixed_mnl():
    frame = _make_data(seed=13, occ=[1, 2, 1, 1, 3, 2, 1])
    rng = np.random.default_rng(5)
    beta = 0.6 * rng.standard_normal(NVAR)

    scenarios = {
        "base": {"x0_0": 0.0},
        "treat": {"x0_0": 1.0},
    }

    model = _mixmnl(frame, normvar=())
    mix_res = mixmnl_ate(model, beta, data=frame, spec=SPEC, scenarios=scenarios)

    ref_res = mnl_ate_from_params(
        beta,
        param_names=VAR_NAMES,
        data=frame,
        spec=SPEC,
        alternatives=ALTS,
        scenarios=scenarios,
    )

    np.testing.assert_allclose(
        mix_res.predicted_shares, ref_res.predicted_shares, rtol=0, atol=1e-6
    )
    assert set(mix_res.shares_per_scenario) == set(ref_res.shares_per_scenario)
    for name in ref_res.shares_per_scenario:
        np.testing.assert_allclose(
            mix_res.shares_per_scenario[name],
            ref_res.shares_per_scenario[name],
            rtol=0,
            atol=1e-6,
        )

    # the collapsed ATE comparison matches the fixed-coef comparison too.
    np.testing.assert_allclose(
        mix_res.comparison("base", "treat"),
        ref_res.comparison("base", "treat"),
        rtol=0,
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# (2) MIXING ACTIVE: valid shares / quantities, well-formed ATE
# ---------------------------------------------------------------------------

def test_mixing_predict_valid():
    frame = _make_data(seed=21, occ=[3, 1, 2, 2, 1, 3])
    model = _mixmnl(frame, normvar=(VAR_NAMES[0],), n_rep=8)
    spec = model._build_spec_layout()[0]
    assert spec.nrndcoef == 1

    rng = np.random.default_rng(6)
    theta = 0.3 * rng.standard_normal(model._build_spec_layout()[1].n_theta)
    comp = model._predict_components(theta)

    per_obs = mixmnl_predict(comp, frame, SPEC)
    assert per_obs.shape == (len(frame), NC)
    assert np.all(per_obs >= 0.0)
    np.testing.assert_allclose(per_obs.sum(axis=1), 1.0, atol=1e-10)

    choice = mixmnl_predict_choice(comp, frame, SPEC)
    assert choice.shape == (len(frame),)
    assert np.all((choice >= 0) & (choice < NC))
    np.testing.assert_array_equal(choice, np.argmax(per_obs, axis=1))

    shares = mixmnl_predict(comp, frame, SPEC).mean(axis=0)
    assert shares.shape == (NC,)
    assert np.all(shares >= 0.0)
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-10)


def test_mixing_ate_well_formed():
    frame = _make_data(seed=22, occ=[2, 2, 1, 3, 1, 2])
    model = _mixmnl(frame, normvar=(VAR_NAMES[0],), n_rep=6)
    layout = model._build_spec_layout()[1]

    rng = np.random.default_rng(9)
    theta = 0.3 * rng.standard_normal(layout.n_theta)

    scenarios = {"base": {"x0_0": 0.0}, "treat": {"x0_0": 1.5}}
    res = mixmnl_ate(model, theta, data=frame, spec=SPEC, scenarios=scenarios)

    assert isinstance(res, MixedATEResult)
    assert res.n_obs == len(frame)
    assert set(res.shares_per_scenario) == {"base", "treat"}
    for s in res.shares_per_scenario.values():
        assert s.shape == (NC,)
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-10)

    pct = res.comparison("base", "treat")
    assert pct.shape == (NC,)
    assert np.all(np.isfinite(pct))
    np.testing.assert_allclose(res.predicted_shares.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# (3) INTERFACE CONFORMANCE: harmonized ATEResultMixin surface + facade
# ---------------------------------------------------------------------------

def test_result_interface_conformance():
    frame = _make_data(seed=31, occ=[2, 1, 2, 1])
    model = _mixmnl(frame, normvar=())
    layout = model._build_spec_layout()[1]
    beta = np.zeros(layout.n_theta)

    # DataFrame-form scenarios exercise scenarios_to_dict normalisation.
    scenarios = pd.DataFrame(
        {"x0_0": [0.0, 2.0]}, index=["lo", "hi"]
    )
    res = mixmnl_ate(model, beta, data=frame, spec=SPEC, scenarios=scenarios)

    assert isinstance(res, ATEResultMixin)
    assert hasattr(res, "predicted_shares")
    assert set(res.shares_per_scenario) == {"lo", "hi"}       # scenarios_to_dict
    assert callable(res.comparison)
    assert isinstance(res.summary(), str)                     # ATESummaryMixin


def test_facade_predict_ate_after_fit():
    """The model facade wires .predict()/.ate() to the shared machinery."""
    frame = _make_data(seed=41, occ=[2, 1, 3, 1, 2])
    model = _mixmnl(frame, normvar=(VAR_NAMES[0],), n_rep=5)

    # calling before fit is guarded by the harmonized _require_results.
    with pytest.raises(RuntimeError):
        model.predict()
    with pytest.raises(RuntimeError):
        model.ate(scenarios={"base": {"x0_0": 0.0}})

    model.control.maxiter = 50
    model.fit()

    per_obs = model.predict()
    assert per_obs.shape == (len(frame), NC)
    np.testing.assert_allclose(per_obs.sum(axis=1), 1.0, atol=1e-10)

    choice = model.predict_choice()
    assert choice.shape == (len(frame),)
    np.testing.assert_array_equal(choice, np.argmax(per_obs, axis=1))

    shares = model.predict().mean(axis=0)
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-10)

    res = model.ate(scenarios={"base": {"x0_0": 0.0}, "treat": {"x0_0": 1.0}})
    assert isinstance(res, MixedATEResult)
    assert set(res.shares_per_scenario) == {"base", "treat"}
    assert np.all(np.isfinite(res.comparison("base", "treat")))


# ---------------------------------------------------------------------------
# (4) HARMONIZED SURFACE: mixmnl_ate_from_params + MixMNLResults.from_estimates
# ---------------------------------------------------------------------------

def test_ate_from_params_matches_model_ate():
    """mixmnl_ate_from_params mirrors a model-built ATE at the same params."""
    frame = _make_data(seed=51, occ=[1, 2, 1, 3, 1, 2])
    rng = np.random.default_rng(7)
    beta = 0.6 * rng.standard_normal(NVAR)                 # nrndcoef == 0 -> theta

    scenarios = {"base": {"x0_0": 0.0}, "treat": {"x0_0": 1.0}}

    model = _mixmnl(frame, normvar=())
    ref = mixmnl_ate(model, beta, data=frame, spec=SPEC, scenarios=scenarios)

    res = mixmnl_ate_from_params(
        beta,
        data=frame,
        spec=SPEC,
        alternatives=ALTS,
        control=MixMNLControl(person_id="pid", verbose=0),
        scenarios=scenarios,
    )

    assert isinstance(res, MixedATEResult)
    np.testing.assert_allclose(
        res.predicted_shares, ref.predicted_shares, rtol=0, atol=1e-6
    )
    for name in scenarios:
        np.testing.assert_allclose(
            res.shares_per_scenario[name],
            ref.shares_per_scenario[name],
            rtol=0,
            atol=1e-6,
        )


def test_ate_from_params_reporting_roundtrip():
    """Fitted reporting-space params reproduce the model's own .ate() exactly.

    CAVEAT-1 round trip: a fitted ``results.params`` is in *reporting* space;
    feeding it straight to ``mixmnl_ate_from_params`` (which builds the context
    with :class:`~pybhatlib.mixed._reparam.ReportingSpace`) must reproduce the
    draw-integrated ATE that the fitted model's :meth:`ate` computes from its
    estimation-space ``_theta_hat`` (via ``EstimationSpace``).  Uses two random
    normal coefficients so the ``rcor`` correlation block is non-trivial.
    """
    frame = _make_data(seed=61, occ=[2, 1, 3, 1, 2, 2])
    control_kwargs = dict(
        normvar=(VAR_NAMES[0], VAR_NAMES[1]),      # nrndcoef == 2 -> rcor active
        person_id="pid",
        n_rep=6,
        draw_seed=7,
        verbose=0,
    )
    model = MixMNLModel(
        data=frame,
        alternatives=ALTS,
        availability="none",
        spec=SPEC,
        control=MixMNLControl(maxiter=40, **control_kwargs),
    )
    spec = model._build_spec_layout()[0]
    assert spec.nrndcoef == 2 and spec.nrndtcor == 1

    res_fit = model.fit()

    scenarios = {"base": {"x0_0": 0.0}, "treat": {"x0_0": 1.0}}
    ref = model.ate(scenarios=scenarios)                    # estimation-space path

    # results.params is reporting-space; feed it straight back through the
    # ReportingSpace path (no inversion).
    got = mixmnl_ate_from_params(
        res_fit.params,
        data=frame,
        spec=SPEC,
        alternatives=ALTS,
        control=MixMNLControl(**control_kwargs),
        scenarios=scenarios,
    )

    assert isinstance(got, MixedATEResult)
    np.testing.assert_allclose(
        got.predicted_shares, ref.predicted_shares, rtol=0, atol=1e-6
    )
    for name in scenarios:
        np.testing.assert_allclose(
            got.shares_per_scenario[name],
            ref.shares_per_scenario[name],
            rtol=0,
            atol=1e-6,
        )
    np.testing.assert_allclose(
        got.comparison("base", "treat"),
        ref.comparison("base", "treat"),
        rtol=0,
        atol=1e-6,
    )


def test_results_from_estimates_minimal():
    """MixMNLResults.from_estimates holds externally-supplied params."""
    params = np.array([0.3, -0.5])
    res = MixMNLResults.from_estimates(params, param_names=list(VAR_NAMES))

    np.testing.assert_array_equal(res.params, params)
    assert res.param_names == list(VAR_NAMES)
    assert res.converged is True
    assert res.return_code == 0
    assert np.isnan(res.loglik)
    assert np.all(np.isnan(res.se))
