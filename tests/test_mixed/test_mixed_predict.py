"""GATE: shared mixed prediction / ATE machinery (Phase 4, self-contained).

NO GAUSS oracle -- validation is by COLLAPSE + interface conformance:

1. **collapse** -- with ``nrndcoef = 0`` (no mixing) the draw-integrated
   :func:`mixed_predict_shares` equals a single-draw kernel prediction
   (``softmax(X @ beta)`` averaged over the sample);
2. **validity** -- discrete shares are non-negative and sum to one;
3. **ATE well-formedness** -- a two-scenario :func:`mixed_ate` returns a result
   whose ``.comparison(base, treatment)`` is a finite per-alternative vector.

A fourth check exercises the actual mixing layer (``nrndcoef > 0``): the
draw-integrated shares are still valid, and the collapse identity is a special
case of the same code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pybhatlib.mixed._draws import FixtureDrawSource, ScipyHaltonDrawSource
from pybhatlib.mixed._engine import DesignData, MSLConfig
from pybhatlib.mixed._predict import (
    MixedATEResult,
    MixedPredictComponents,
    mixed_ate,
    mixed_predict_shares,
)
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.vecup._panel import PanelIndex

NC = 3            # alternatives
NVAR = 2          # variables per alternative == n_beta
VAR_NAMES = [f"v{i}" for i in range(NVAR)]


class _Obs:
    def __init__(self, avail: np.ndarray, chosen: np.ndarray) -> None:
        self.avail = avail
        self.chosen = chosen


def _make_frame(occ, seed):
    """Synthetic long-format frame: one row per (person, occasion, alt-var)."""
    rng = np.random.default_rng(seed)
    person_ids = np.concatenate([np.full(c, i) for i, c in enumerate(occ)])
    n_obs = person_ids.shape[0]
    cols = {"pid": person_ids}
    # column layout: iv_<alt>_<var>
    for k in range(NC):
        for v in range(NVAR):
            cols[f"iv_{k}_{v}"] = 0.5 * rng.standard_normal(n_obs)
    chosen = np.zeros((n_obs, NC), dtype=np.float64)
    chosen[np.arange(n_obs), rng.integers(0, NC, size=n_obs)] = 1.0
    for k in range(NC):
        cols[f"chosen_{k}"] = chosen[:, k]
    return pd.DataFrame(cols)


def _build_design(frame: pd.DataFrame, spec) -> DesignData:
    """Rebuild the (n_obs, NC, NVAR) design + obs bundle from a frame.

    Kernel-agnostic build_design closure for the gate: ``spec`` is ignored (the
    synthetic layout is fixed); a real family closes over ``parse_spec``.
    """
    n_obs = len(frame)
    X = np.empty((n_obs, NC, NVAR), dtype=np.float64)
    for k in range(NC):
        for v in range(NVAR):
            X[:, k, v] = frame[f"iv_{k}_{v}"].to_numpy(dtype=np.float64)
    avail = np.ones((n_obs, NC), dtype=np.float64)
    chosen = frame[[f"chosen_{k}" for k in range(NC)]].to_numpy(dtype=np.float64)
    return DesignData(X=X, obs=_Obs(avail, chosen))


def _softmax(V, avail):
    vmax = np.max(np.where(avail > 0, V, -np.inf), axis=1, keepdims=True)
    p1 = np.exp(V - vmax) * avail
    return p1 / p1.sum(axis=1, keepdims=True)


def _components(frame, *, normvar=(), n_rep=1, draw_seed=7):
    """Assemble a MixedPredictComponents over the mixmnl softmax kernel."""
    spec = MixingSpec.from_var_names(var_names=VAR_NAMES, normvar=tuple(normvar))
    layout = ParamLayout(
        n_beta=spec.n_beta,
        n_rcor=spec.nrndtcor,
        n_scal=spec.nscale,
        n_lam=spec.numlam,
        n_kern=0,
    )
    panel = PanelIndex.from_ids(frame["pid"].to_numpy())
    space = EstimationSpace(layout, scal=1.0, intordn1=20)
    pipeline = RandomCoefPipeline(spec, layout, scal=1.0, intordn1=20)
    kernel = SoftmaxKernel(NC)
    cfg = MSLConfig(n_rep=n_rep)

    if spec.nrndcoef == 0:
        draws = FixtureDrawSource(
            np.zeros((n_rep, panel.n_ind * spec.nrndcoef))
        )
    else:
        draws = ScipyHaltonDrawSource(seed=draw_seed)

    return spec, layout, MixedPredictComponents(
        theta=np.zeros(layout.n_theta),   # placeholder; caller sets theta
        panel=panel,
        draws=draws,
        pipeline=pipeline,
        space=space,
        kernel=kernel,
        layout=layout,
        config=cfg,
        build_design=_build_design,
        alternative_names=[f"alt{k}" for k in range(NC)],
    )


# ---------------------------------------------------------------------------
# (1) collapse: nrndcoef == 0 mixed prediction == single-draw softmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_rep", [1, 5])
def test_collapse_no_mixing(n_rep):
    frame = _make_frame([1, 1, 1, 1, 2, 3], seed=11)
    spec, layout, comp = _components(frame, normvar=(), n_rep=n_rep)
    assert spec.nrndcoef == 0

    rng = np.random.default_rng(3)
    beta = 0.5 * rng.standard_normal(NVAR)
    comp.theta = beta.copy()   # theta == beta when there are no rnd blocks

    shares = mixed_predict_shares(comp, frame, spec=None)

    # reference: fixed-coef single-draw softmax averaged over the sample.
    design = _build_design(frame, None)
    V = np.einsum("qcv,qv->qc", design.X, np.broadcast_to(beta, (len(frame), NVAR)))
    ref = _softmax(V, design.obs.avail).mean(axis=0)

    np.testing.assert_allclose(shares, ref, rtol=0, atol=1e-12)


# ---------------------------------------------------------------------------
# (2) validity: non-negative, sums to one (discrete)
# ---------------------------------------------------------------------------

def test_shares_valid_with_mixing():
    frame = _make_frame([3, 1, 2, 2, 1, 3], seed=21)
    spec, layout, comp = _components(frame, normvar=(VAR_NAMES[0],), n_rep=8)
    assert spec.nrndcoef == 1

    rng = np.random.default_rng(5)
    theta = 0.3 * rng.standard_normal(layout.n_theta)
    comp.theta = theta

    shares = mixed_predict_shares(comp, frame, spec=None)

    assert shares.shape == (NC,)
    assert np.all(shares >= 0.0)
    np.testing.assert_allclose(shares.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# (3) two-scenario ATE returns a well-formed comparison
# ---------------------------------------------------------------------------

def test_two_scenario_ate_comparison():
    frame = _make_frame([2, 2, 1, 3, 1, 2], seed=31)
    spec, layout, comp = _components(frame, normvar=(VAR_NAMES[0],), n_rep=6)

    rng = np.random.default_rng(9)
    comp.theta = 0.3 * rng.standard_normal(layout.n_theta)

    scenarios = {
        "base": {"iv_0_0": 0.0},
        "treat": {"iv_0_0": 1.0},
    }
    result = mixed_ate(comp, frame, spec=None, scenarios=scenarios)

    assert isinstance(result, MixedATEResult)
    assert result.n_obs == len(frame)
    assert set(result.shares_per_scenario) == {"base", "treat"}
    for s in result.shares_per_scenario.values():
        assert s.shape == (NC,)
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-10)

    pct = result.comparison("base", "treat")
    assert pct.shape == (NC,)
    assert np.all(np.isfinite(pct))

    # baseline shares are valid too.
    np.testing.assert_allclose(result.predicted_shares.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# (extra) scenario overrides move shares; draws are shared across scenarios
# ---------------------------------------------------------------------------

def test_scenario_override_changes_shares():
    frame = _make_frame([2, 2, 2, 2], seed=41)
    spec, layout, comp = _components(frame, normvar=(), n_rep=1)

    rng = np.random.default_rng(2)
    comp.theta = 0.5 * rng.standard_normal(layout.n_theta)

    base = mixed_predict_shares(comp, frame, spec=None, scenario={"iv_0_0": -1.0})
    treat = mixed_predict_shares(comp, frame, spec=None, scenario={"iv_0_0": 2.0})

    # a strong shift in alt-0's first covariate must change alt-0's share.
    assert not np.allclose(base, treat)
