"""MORPFlex (mixed-panel MORP) prediction / forecasting over the mixing draws.

Wires the shipped fixed-coefficient MORP prediction formulation
(:func:`pybhatlib.models.morp._morp_forecast.morp_predict`) into the shared
mixed predictor (:mod:`pybhatlib.mixed._predict`).  The kernel's
:meth:`~pybhatlib.models.morp_flex._morp_flex_kernel.RectMvncdKernel.predict_shares`
returns the per-observation marginal ordinal-category probabilities for one MSL
replication; the shared engine draws the random coefficients (held fixed per
individual, exactly as at fit time), evaluates that per-draw prediction and
averages over draws -- integrating out the mixing distribution.

Because a marginal of a multivariate normal ignores off-diagonal correlation,
with ``copula=False`` and no random coefficients this collapses *exactly* to the
fixed-coefficient MORP marginal (the collapse gate); with mixing active it is
the draw-integrated ordinal marginal.  Validation is by collapse + interface
conformance, not GAUSS parity -- no GAUSS forecast oracle exists for the mixed
drivers (see ``docs/plans/MIXED_PANEL_MODELS_PLAN.md`` Phase 4).

The result mirrors the *per-outcome* shape of the fixed MORP: a list with one
``(N, n_categories[d])`` (prediction) or ``(n_categories[d],)`` (share) array
per ordinal dimension.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._engine import DesignData
from pybhatlib.mixed._predict import (
    MixedPredictComponents,
    _design_for,
    _resolve,
    _simulate_per_obs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pybhatlib.models.morp_flex._morp_flex_model import MORPFlexModel


# ---------------------------------------------------------------------------
# Prediction context assembly
# ---------------------------------------------------------------------------

def _fit_ctx(model: "MORPFlexModel") -> SimpleNamespace:
    """Return the cached post-fit context or raise if the model is unfit."""
    ctx = getattr(model, "_fit_ctx", None)
    if ctx is None:
        raise RuntimeError(
            f"{type(model).__name__} has not been fit yet; call .fit() first."
        )
    return ctx


def _make_build_design(model: "MORPFlexModel"):
    """Build a ``(DataFrame, spec) -> DesignData`` closure for scenario designs.

    The ordinal outcomes are irrelevant to prediction (``predict_shares`` returns
    every category's probability, not only the observed one), so the design's
    ``obs`` bundle carries the model's original ``y_ord`` unchanged.
    """
    dep_vars = model.dep_vars
    y_ord = model.y_ord

    def build(data_frame, spec) -> DesignData:
        X, _ = parse_spec(spec, data_frame, dep_vars, nseg=1)
        return DesignData(
            X=np.asarray(X, dtype=np.float64),
            obs=SimpleNamespace(y_ord=y_ord),
        )

    return build


def build_components(
    model: "MORPFlexModel",
    *,
    design: Optional[DesignData] = None,
    build_design=None,
) -> MixedPredictComponents:
    """Assemble the shared :class:`MixedPredictComponents` from a fitted model.

    Parameters
    ----------
    model : MORPFlexModel
        A fitted model (``.fit()`` populated ``_fit_ctx``).
    design : DesignData, optional
        Pre-built design for the plain (no-scenario) path; defaults to the
        fitted estimator's training design.
    build_design : callable, optional
        ``(DataFrame, spec) -> DesignData`` closure required for scenario
        overrides.

    Returns
    -------
    MixedPredictComponents
        Bundle carrying the estimation-space parameter vector, the mixing
        pipeline / space / kernel / layout / config, the fit panel and
        fit-identical draws.
    """
    ctx = _fit_ctx(model)
    est = ctx.est
    return MixedPredictComponents(
        theta=ctx.theta,
        panel=est.panel,
        draws=est.draws,
        pipeline=est.pipeline,
        space=est.space,
        kernel=est.kernel,
        layout=est.layout,
        config=est.config,
        build_design=build_design,
        design=design if design is not None else est.design,
        alternative_names=None,
    )


def draw_integrated_per_obs(
    components: MixedPredictComponents,
    *,
    data=None,
    spec=None,
    scenario: Optional[dict] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Draw-integrated per-observation marginal category probabilities.

    Thin wrapper over the shared engine helpers
    (:func:`pybhatlib.mixed._predict._resolve` /
    :func:`~pybhatlib.mixed._predict._design_for` /
    :func:`~pybhatlib.mixed._predict._simulate_per_obs`): resolves the context,
    (re)builds the design for an optional scenario override, and simulates the
    draw-integrated per-observation prediction.

    Returns
    -------
    NDArray, shape (n_obs, sum(n_categories))
        Per-observation marginal category probabilities, concatenated across
        ordinal dimensions, averaged over the mixing draws.
    """
    rc_ctx = _resolve(components)
    draw_src = draws if draws is not None else rc_ctx.draws
    design = _design_for(rc_ctx, data, spec, scenario)
    return _simulate_per_obs(rc_ctx, design, draw_src, xp=xp)


def split_by_dim(
    shares_2d: NDArray, n_categories: Sequence[int]
) -> list[NDArray]:
    """Split a concatenated ``(..., sum(n_categories))`` block per dimension.

    Returns
    -------
    list of NDArray
        ``out[d]`` is the ``n_categories[d]``-wide slice for ordinal dimension
        ``d`` (keeping every leading axis).
    """
    out: list[NDArray] = []
    col = 0
    for ncat in n_categories:
        ncat = int(ncat)
        out.append(np.asarray(shares_2d[..., col:col + ncat], dtype=np.float64))
        col += ncat
    return out


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def morp_flex_predict(
    model: "MORPFlexModel",
    data=None,
    *,
    scenario: Optional[dict] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> list[NDArray]:
    """Draw-integrated per-observation ordinal category probabilities.

    Mirrors :func:`pybhatlib.models.morp._morp_forecast.morp_predict` (a list
    with one ``(N, n_categories[d])`` array per ordinal dimension), adding the
    integration over the mixing distribution via the shared MSL engine.
    Signature normalized to the common mixed-model convention
    ``(data=None, *, scenario=None, draws=None, xp=None)``.

    Parameters
    ----------
    model : MORPFlexModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset to predict on; the design is rebuilt from the frame via the
        model's spec.  Defaults to the fitted training frame.  Must have the
        same number of rows as the fit sample (the mixing draws are held per fit
        individual).
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides,
        applied before the design is rebuilt.  ``None`` predicts at the observed
        covariates.
    draws : DrawSource, optional
        Override the fit-identical draw strategy (e.g. more replications for a
        smoother forecast).
    xp : module, optional
        Array backend used to wrap the per-dimension arrays. Defaults to NumPy.

    Returns
    -------
    list of NDArray
        ``probs[d]`` has shape ``(N, n_categories[d])``; each row is a valid
        (normalised) category distribution.
    """
    if data is None and scenario is None:
        # Plain path: reuse the fitted training design directly.
        components = build_components(model)
        per_obs = draw_integrated_per_obs(components, draws=draws)
    else:
        # Rebuild the design from the (optionally scenario-overridden) frame.
        build_design = _make_build_design(model)
        components = build_components(model, build_design=build_design)
        data_use = model.data if data is None else data
        per_obs = draw_integrated_per_obs(
            components, data=data_use, spec=model.spec_dict,
            scenario=scenario, draws=draws,
        )
    probs = split_by_dim(per_obs, model.n_categories)
    for d, block in enumerate(probs):
        row_sum = block.sum(axis=1, keepdims=True)
        block = block / np.where(row_sum > 0, row_sum, 1.0)
        probs[d] = xp.asarray(block) if xp is not None else block
    return probs


def morp_flex_predict_category(
    model: "MORPFlexModel",
    data=None,
    *,
    scenario: Optional[dict] = None,
    draws: Optional[DrawSource] = None,
) -> NDArray:
    """Most-likely ordinal category per observation and dimension.

    Mirrors :func:`pybhatlib.models.morp._morp_forecast.morp_predict_category`
    over the draw-integrated marginals.  Signature normalized to the common
    mixed-model convention (``data`` / ``scenario`` overrides).

    Returns
    -------
    NDArray, shape (N, nord)
        Predicted 0-based category for each observation and ordinal dimension.
    """
    probs = morp_flex_predict(model, data, scenario=scenario, draws=draws)
    n = probs[0].shape[0]
    nord = len(probs)
    categories = np.zeros((n, nord), dtype=np.int64)
    for d in range(nord):
        categories[:, d] = np.argmax(probs[d], axis=1)
    return categories
