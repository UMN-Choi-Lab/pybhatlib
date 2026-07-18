"""Shared mixed / panel prediction + ATE machinery (Phase 4).

The four mixed-model families (``mixmnl``, ``mnpkercp``, ``morp_flex``,
``mdcev_mixed``) share one MSL simulation engine
(:class:`~pybhatlib.mixed._engine.MixedMSLEstimator`); post-estimation
prediction and ATE are equally shared here.  The formulation lifts each
family's *shipped fixed-coefficient* ATE / forecast
(``models/{mnl,mnp,morp,mdcev}/_*_ate.py`` / ``_*_forecast.py``) over the
mixing draws:

    for each MSL replication r (random coefficients drawn from the mixing
    distribution, held FIXED per individual -- the same Halton draws as the
    fit):
        realize the random-coefficient pipeline -> xmunew per observation
        form the systematic utilities   Vsub = X @ xmunew
        ask the kernel for the per-observation prediction
            (choice probabilities for the discrete kernels;
             consumption for MDCEV)
    average over draws (integrating out the mixing distribution) -> per-obs
    prediction, then average over the sample -> per-alternative (or per-outcome
    / per-good) predicted shares.

An ATE is the difference / ratio of the averaged quantity between two
counterfactual scenarios (each scenario's exogenous covariates injected via
:func:`~pybhatlib.models._ate_common.apply_scenario_overrides`, the design
matrix rebuilt, and the *same* mixing draws reused so the only thing that moves
is the covariate).  This mirrors the fixed-coefficient ATE with the extra
draw-integration layer.

Kernel-agnostic
---------------
The engine and this module never inspect which kernel they hold.  A family
plugs its kernel's per-draw prediction in one of two ways:

* implement a ``predict_shares(Vsub, obs, kstate, *, rc_draw) -> (n_obs,
  n_out)`` method on its kernel object (preferred), or
* pass a ``kernel_predict`` callable in :class:`MixedPredictComponents`.

If neither is supplied the discrete default
(:func:`default_kernel_predict`, availability-masked softmax over ``Vsub``) is
used -- correct for the logit-form families (MixMNL) and a documented default
for the others to override.

Validation is by *collapse* (``nrndcoef = 0`` mixed prediction equals the
single-draw fixed-coefficient kernel prediction) and interface conformance, not
GAUSS parity -- no GAUSS forecast oracle exists for the mixed drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd  # noqa: F401  (string type annotations)
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._engine import DesignData, MSLConfig
from pybhatlib.mixed._kernel import MixedKernel
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import ParamLayout, ParamSpace
from pybhatlib.models._ate_common import (
    ATEResultMixin,
    ScenarioSpec,
    apply_scenario_overrides,
    scenarios_to_dict,
)
from pybhatlib.vecup._panel import PanelIndex

# A design builder rebuilds the per-observation design (``DesignData``) from a
# (possibly scenario-overridden) DataFrame and the variable-spec mapping.  The
# family supplies it (typically a thin closure over ``parse_spec`` + the
# alternative / availability columns); this module never parses a spec itself.
DesignBuilder = Callable[["pd.DataFrame", Any], DesignData]

# A kernel-prediction hook returns the per-observation, per-output prediction
# for one MSL replication.  Signature mirrors ``MixedKernel.probability`` but
# returns the full ``(n_obs, n_out)`` per-alternative / per-good matrix rather
# than only the observed-outcome probability.
KernelPredict = Callable[[MixedKernel, NDArray, Any, Any, NDArray], NDArray]


# ---------------------------------------------------------------------------
# Kernel-prediction dispatch
# ---------------------------------------------------------------------------

def _softmax_avail(Vsub: NDArray, avail: NDArray) -> NDArray:
    """Availability-masked softmax over alternatives (batched over obs).

    Numerically guarded (per-row max shift over available alternatives; zero
    denominator guard), matching ``models/mnl`` and the mixmnl softmax kernel.

    Parameters
    ----------
    Vsub : NDArray, shape (n_obs, nc)
        Systematic utilities per observation and alternative.
    avail : NDArray, shape (n_obs, nc)
        Availability mask (``> 0`` where the alternative is available).

    Returns
    -------
    NDArray, shape (n_obs, nc)
        Normalised choice probabilities.
    """
    Vsub = np.asarray(Vsub, dtype=np.float64)
    avail = np.asarray(avail, dtype=np.float64)
    v_max = np.max(np.where(avail > 0, Vsub, -np.inf), axis=1, keepdims=True)
    v_max = np.where(np.isfinite(v_max), v_max, 0.0)
    # Mask *inside* the exponent (exp(-inf) == 0): an unavailable alternative
    # with utility > v_max + 709 would otherwise overflow exp then hit inf*0=NaN.
    p1 = np.exp(np.where(avail > 0, Vsub - v_max, -np.inf))
    denom = p1.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return p1 / denom


def default_kernel_predict(
    kernel: MixedKernel,
    Vsub: NDArray,
    obs: Any,
    kstate: Any,
    rc_draw: NDArray,
) -> NDArray:
    """Per-observation kernel prediction for one MSL replication.

    Dispatch order:

    1. if ``kernel`` implements ``predict_shares(Vsub, obs, kstate, *,
       rc_draw)``, delegate to it (the family owns the formulation, e.g. MNP
       differenced-MVNCD choice probabilities, MDCEV expected consumption);
    2. otherwise fall back to the availability-masked softmax over ``Vsub``
       (``obs.avail`` if present, else all-ones) -- correct for logit-form
       kernels (MixMNL) and a documented default the other families override.

    Parameters
    ----------
    kernel : MixedKernel
        The fitted model's inner kernel.
    Vsub : NDArray, shape (n_obs, nc)
        Systematic utilities for this replication (``X @ xmunew``).
    obs : Any
        The kernel ``obs`` bundle (softmax reads ``obs.avail``).
    kstate : Any
        The kernel state from :meth:`MixedKernel.prepare`.
    rc_draw : NDArray, shape (n_obs, n_rnd)
        The correlated random-coefficient draw for this replication (``errbeta3``).

    Returns
    -------
    NDArray, shape (n_obs, n_out)
        Per-observation, per-output prediction (choice probabilities /
        consumption) for this draw.
    """
    predict_shares = getattr(kernel, "predict_shares", None)
    if callable(predict_shares):
        pred = predict_shares(Vsub, obs, kstate, rc_draw=rc_draw)
        return np.asarray(pred, dtype=np.float64)
    avail = getattr(obs, "avail", None)
    if avail is None:
        avail = np.ones_like(np.asarray(Vsub, dtype=np.float64))
    return _softmax_avail(Vsub, avail)


# ---------------------------------------------------------------------------
# Components bundle
# ---------------------------------------------------------------------------

@dataclass
class MixedPredictComponents:
    """Everything the shared predictor needs from a *fitted* mixed model.

    A family's ``.ate()`` / ``.predict()`` facade builds this bundle from its
    fitted estimator: the estimation-space parameter vector, the mixing
    pipeline / reparameterization space / kernel / layout / config, the panel
    index, the (fit-identical) draw source, and a ``build_design`` closure that
    rebuilds the design tensor for a scenario-overridden DataFrame.  A
    :class:`~pybhatlib.mixed._engine.MixedMSLEstimator` may be passed to the
    prediction functions directly instead, provided it additionally carries a
    ``theta`` attribute (and a ``build_design`` closure whenever scenarios are
    used).

    Attributes
    ----------
    theta : NDArray, shape (n_theta,)
        Fitted parameter vector in **estimation space** (``[beta | rcor | scal
        | lam | kern]`` order), as consumed by ``space.unpack`` /
        ``pipeline.realize``.  (Not the natural reporting-space vector stored on
        the ``*Results`` object.)
    panel : PanelIndex
        Person-index mapping (draws held fixed per individual across occasions).
    draws : DrawSource
        The *same* draw strategy used at fit time, so predictions integrate over
        the identical mixing draws and scenario differences are pure covariate
        effects.
    pipeline : RandomCoefPipeline
        Random-coefficient realization (correlate / YJ / scale / inject).
    space : ParamSpace
        Reparameterization strategy compatible with ``pipeline``.
    kernel : MixedKernel
        The fitted inner kernel; queried for the per-draw prediction.
    layout : ParamLayout
        Parameter-block partition (threaded to ``kernel.prepare``).
    config : MSLConfig
        Engine configuration; only ``n_rep`` is used for prediction.
    build_design : DesignBuilder or None
        ``(data_frame, spec) -> DesignData``.  Required when a scenario override
        is applied (the design must be rebuilt).  ``None`` is allowed only when
        ``design`` is supplied and no scenario is used.
    design : DesignData or None
        A pre-built design for the plain (no-scenario) path when
        ``build_design`` is not supplied.
    kernel_predict : KernelPredict or None
        Explicit per-draw prediction hook; overrides
        :func:`default_kernel_predict` when supplied.
    alternative_names : list of str or None
        Output labels (alternatives / outcomes / goods), forwarded to the ATE
        result.
    """

    theta: NDArray
    panel: PanelIndex
    draws: DrawSource
    pipeline: RandomCoefPipeline
    space: ParamSpace
    kernel: MixedKernel
    layout: ParamLayout
    config: MSLConfig
    build_design: Optional[DesignBuilder] = None
    design: Optional[DesignData] = None
    kernel_predict: Optional[KernelPredict] = None
    alternative_names: Optional[list[str]] = None


@dataclass
class _ResolvedComponents:
    """Normalised view over a components bundle or an annotated estimator."""

    theta: NDArray
    panel: PanelIndex
    draws: DrawSource
    pipeline: RandomCoefPipeline
    space: ParamSpace
    kernel: MixedKernel
    layout: ParamLayout
    config: MSLConfig
    build_design: Optional[DesignBuilder]
    design: Optional[DesignData]
    kernel_predict: KernelPredict
    alternative_names: Optional[list[str]]


def _resolve(obj: Any) -> _ResolvedComponents:
    """Coerce a components bundle or annotated estimator to a normalised view.

    Parameters
    ----------
    obj : MixedPredictComponents or MixedMSLEstimator
        The prediction context.  An estimator must additionally carry a
        ``theta`` attribute (the estimation-space fitted vector); it may carry
        ``build_design`` / ``kernel_predict`` / ``alternative_names`` too.

    Returns
    -------
    _ResolvedComponents

    Raises
    ------
    TypeError
        If ``obj`` lacks the required attributes (notably ``theta``).
    """
    def _req(name: str) -> Any:
        if not hasattr(obj, name):
            raise TypeError(
                f"prediction context is missing required attribute {name!r}; "
                "pass a MixedPredictComponents, or a MixedMSLEstimator "
                "annotated with .theta"
            )
        return getattr(obj, name)

    theta = np.asarray(_req("theta"), dtype=np.float64)
    kp = getattr(obj, "kernel_predict", None) or default_kernel_predict
    return _ResolvedComponents(
        theta=theta,
        panel=_req("panel"),
        draws=_req("draws"),
        pipeline=_req("pipeline"),
        space=_req("space"),
        kernel=_req("kernel"),
        layout=_req("layout"),
        config=_req("config"),
        build_design=getattr(obj, "build_design", None),
        design=getattr(obj, "design", None),
        kernel_predict=kp,
        alternative_names=getattr(obj, "alternative_names", None),
    )


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_per_obs(
    rc_ctx: _ResolvedComponents,
    design: DesignData,
    draws: DrawSource,
    *,
    xp=None,
) -> NDArray:
    """Draw-integrated per-observation prediction (batched over obs).

    For each MSL replication, realise the random-coefficient pipeline, form the
    systematic utilities, and query the kernel for the per-observation
    prediction; average over draws.

    Parameters
    ----------
    rc_ctx : _ResolvedComponents
        Normalised prediction context.
    design : DesignData
        Per-observation design tensor ``X`` and kernel ``obs`` bundle for the
        (possibly scenario-overridden) covariates.
    draws : DrawSource
        Draw strategy (fit-identical); ``draws.draws(n_ind, n_rnd, n_rep)`` must
        return ``(n_rep, n_ind, n_rnd)``.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs, n_out)
        Per-observation prediction averaged over the mixing draws.
    """
    panel = rc_ctx.panel
    pipeline = rc_ctx.pipeline
    n_rep = rc_ctx.config.n_rep

    rc = rc_ctx.space.unpack(rc_ctx.theta, pipeline.spec, want_grad=False)
    kstate = rc_ctx.kernel.prepare(rc_ctx.theta, rc_ctx.layout)
    ass = draws.draws(panel.n_ind, pipeline.n_rnd, n_rep)

    X = np.asarray(design.X, dtype=np.float64)
    obs = design.obs

    acc: Optional[NDArray] = None
    for r in range(n_rep):
        errbeta1 = panel.broadcast(ass[r])                      # (n_obs, n_rnd)
        cache = pipeline.realize(errbeta1, rc, want_grad=False)
        Vsub = np.einsum("qcv,qv->qc", X, cache.xmunew)         # (n_obs, nc)
        pred = rc_ctx.kernel_predict(
            rc_ctx.kernel, Vsub, obs, kstate, cache.errbeta3
        )
        pred = np.asarray(pred, dtype=np.float64)
        acc = pred if acc is None else acc + pred

    per_obs = acc / float(n_rep)                                # (n_obs, n_out)
    if xp is not None:
        per_obs = xp.asarray(per_obs)
    return per_obs


def _design_for(
    rc_ctx: _ResolvedComponents,
    data: "pd.DataFrame | None",
    spec: Any,
    scenario: Optional[dict[str, float | str]],
) -> DesignData:
    """Build the design for a (possibly scenario-overridden) DataFrame.

    Applies ``scenario`` via :func:`apply_scenario_overrides`, then rebuilds the
    design via ``rc_ctx.build_design``; falls back to the pre-built
    ``rc_ctx.design`` only in the plain (no-scenario) path.

    Raises
    ------
    ValueError
        If a scenario override is requested but no ``build_design`` closure is
        available, or if neither ``build_design`` nor a pre-built ``design`` is
        available.
    """
    if scenario is not None:
        if rc_ctx.build_design is None:
            raise ValueError(
                "a scenario override requires a build_design closure on the "
                "prediction context to rebuild the design matrix"
            )
        if data is None:
            raise ValueError("data is required to apply a scenario override")
        data_use = apply_scenario_overrides(data, scenario)
        return rc_ctx.build_design(data_use, spec)

    if rc_ctx.build_design is not None:
        if data is None:
            raise ValueError("data is required when build_design is used")
        return rc_ctx.build_design(data, spec)
    if rc_ctx.design is not None:
        return rc_ctx.design
    raise ValueError(
        "prediction context has neither a build_design closure nor a "
        "pre-built design"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mixed_predict_shares(
    estimator_or_components: Any,
    data: "pd.DataFrame | None" = None,
    spec: Any = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Draw-integrated, sample-averaged predicted shares for a mixed model.

    Simulates over the mixing distribution -- drawing random coefficients (held
    fixed per individual, exactly as at fit time), evaluating the kernel's
    per-observation prediction for each draw, averaging over draws, then over
    the sample.  Kernel-agnostic: the per-draw prediction is obtained from the
    kernel (``predict_shares`` method) or an explicit ``kernel_predict`` hook,
    defaulting to the availability-masked softmax.

    Parameters
    ----------
    estimator_or_components : MixedPredictComponents or MixedMSLEstimator
        The fitted-model prediction context.  A raw estimator must carry a
        ``theta`` attribute (estimation-space fitted vector) and, when a
        scenario is used, a ``build_design`` closure.
    data : pd.DataFrame, optional
        Dataset.  Required when the design is rebuilt (``build_design`` present
        or a scenario override is applied).
    spec : Any, optional
        Variable specification forwarded verbatim to ``build_design``.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides,
        applied via :func:`apply_scenario_overrides` before the design is
        rebuilt.  ``None`` (default) predicts at the observed covariates.
    draws : DrawSource, optional
        Override the context's draw strategy (e.g. more replications for a
        smoother forecast).  Defaults to the fit-identical source so scenarios
        share draws.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_out,)
        Sample-averaged, draw-integrated prediction per alternative / outcome /
        good.  For discrete kernels these are market shares (non-negative,
        summing to one over available alternatives).
    """
    rc_ctx = _resolve(estimator_or_components)
    draw_src = draws if draws is not None else rc_ctx.draws
    design = _design_for(rc_ctx, data, spec, scenario)
    per_obs = _simulate_per_obs(rc_ctx, design, draw_src)
    shares = per_obs.mean(axis=0)                               # (n_out,)
    if xp is not None:
        shares = xp.asarray(shares)
    return shares


@dataclass
class MixedATEResult(ATEResultMixin):
    """ATE result for a mixed / panel discrete-choice model.

    Carries the draw-integrated baseline shares plus per-scenario shares, and
    inherits :meth:`~pybhatlib.models._ate_common.ATEComparisonMixin.comparison`
    (pairwise percentage change) and
    :meth:`~pybhatlib.models._ate_common.ATESummaryMixin.summary`.

    Attributes
    ----------
    n_obs : int
        Number of observation rows.
    predicted_shares : NDArray, shape (n_out,)
        Baseline (observed-covariate) draw-integrated shares.
    shares_per_scenario : dict[str, NDArray] or None
        Per-scenario draw-integrated shares (``(n_out,)`` each); keys are
        scenario names.
    alternative_names : list of str or None
        Output labels.
    base_shares, treatment_shares, pct_ate : NDArray or None
        Populated by :meth:`comparison` callers or left ``None`` (the summary
        omits the base/treatment columns when absent).
    """

    _model_label = "Mixed"
    _ate_func_name = "mixed_ate"

    n_obs: int
    predicted_shares: NDArray
    shares_per_scenario: Optional[dict[str, NDArray]] = None
    alternative_names: Optional[list[str]] = None
    base_shares: Optional[NDArray] = None
    treatment_shares: Optional[NDArray] = None
    pct_ate: Optional[NDArray] = None


def mixed_ate(
    estimator_or_components: Any,
    data: "pd.DataFrame | None" = None,
    spec: Any = None,
    *,
    scenarios: ScenarioSpec,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    xp=None,
) -> MixedATEResult:
    """Draw-integrated ATE across counterfactual scenarios for a mixed model.

    Computes baseline shares (observed covariates) and, for each scenario, the
    draw-integrated shares under that scenario's covariate overrides -- reusing
    the *same* mixing draws throughout so the ATE isolates the covariate effect.
    Use :meth:`MixedATEResult.comparison` for pairwise percentage changes
    between any two scenarios.

    Parameters
    ----------
    estimator_or_components : MixedPredictComponents or MixedMSLEstimator
        The fitted-model prediction context (must supply ``build_design`` so
        each scenario's design can be rebuilt).
    data : pd.DataFrame
        Dataset the scenarios override.  Required.
    spec : Any, optional
        Variable specification forwarded to ``build_design``.
    scenarios : dict or pd.DataFrame
        Scenario specification, normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`:

        - dict form ``{name: {col: scalar | source_col, ...}, ...}``
        - DataFrame form (rows = scenarios, columns = variables).
    draws : DrawSource, optional
        Override the context's draw strategy.  Defaults to the fit-identical
        source (shared across scenarios).
    alternative_names : list of str, optional
        Output labels; overrides the context's own.
    xp : backend, optional
        Array backend used to wrap the result arrays. Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus ``shares_per_scenario`` and the
        ``.comparison()`` / ``.summary()`` surface.

    Raises
    ------
    ValueError
        If ``data`` is ``None`` or the context cannot rebuild a scenario design.
    """
    if data is None:
        raise ValueError("mixed_ate requires data for scenario overrides")

    rc_ctx = _resolve(estimator_or_components)
    draw_src = draws if draws is not None else rc_ctx.draws
    names = alternative_names or rc_ctx.alternative_names

    # baseline: observed covariates, no override.
    base_design = _design_for(rc_ctx, data, spec, None)
    baseline = _simulate_per_obs(rc_ctx, base_design, draw_src)
    n_obs = int(baseline.shape[0])
    predicted_shares = baseline.mean(axis=0)

    scenario_dict = scenarios_to_dict(scenarios)
    shares_per_scenario: dict[str, NDArray] = {}
    for name, overrides in scenario_dict.items():
        design = _design_for(rc_ctx, data, spec, overrides)
        per_obs = _simulate_per_obs(rc_ctx, design, draw_src)
        s = per_obs.mean(axis=0)
        shares_per_scenario[str(name)] = xp.asarray(s) if xp is not None else s

    if xp is not None:
        predicted_shares = xp.asarray(predicted_shares)

    return MixedATEResult(
        n_obs=n_obs,
        predicted_shares=predicted_shares,
        shares_per_scenario=shares_per_scenario,
        alternative_names=list(names) if names is not None else None,
    )


__all__ = [
    "MixedPredictComponents",
    "MixedATEResult",
    "mixed_predict_shares",
    "mixed_ate",
    "default_kernel_predict",
    "DesignBuilder",
    "KernelPredict",
]
