"""MixMNL prediction / forecasting over the mixing distribution.

The mixed-logit analogue of :func:`pybhatlib.models.mnl.mnl_predict`.  The
shipped fixed-coefficient MNL forms per-observation choice probabilities as an
availability-masked softmax of ``X @ beta``; the mixed family lifts that exact
formulation over the Halton mixing draws using the shared engine
(:mod:`pybhatlib.mixed._predict`): for each MSL replication a random-coefficient
vector is drawn (held fixed per individual, exactly as at fit time), the
systematic utilities ``Vsub = X @ xmunew`` are formed, the softmax kernel's
per-observation probabilities are evaluated, and the result is averaged over the
draws.

With an empty random-coefficient spec (``nrndcoef == 0``) the single-draw
pipeline reproduces ``X @ beta`` verbatim, so :func:`mixmnl_predict` collapses to
:func:`~pybhatlib.models.mnl.mnl_predict` value-for-value -- the collapse gate.

There is **no** GAUSS forecast oracle for the mixed drivers (Dale's guidance):
validation is by collapse to the fixed-coefficient forecast plus interface
conformance, not GAUSS parity.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd  # noqa: F401  (string type annotations)
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._predict import (
    MixedPredictComponents,
    _design_for,
    _resolve,
    _simulate_per_obs,
)


def mixmnl_predict(
    components: MixedPredictComponents | Any,
    data: "pd.DataFrame | None" = None,
    spec: Any = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Per-observation draw-integrated choice probabilities for a MixMNL model.

    Mirrors :func:`pybhatlib.models.mnl.mnl_predict` (output shape ``(n_obs,
    nc)``) with the extra draw-integration layer: the softmax kernel's
    per-observation probabilities are averaged over the mixing draws rather than
    evaluated at a single coefficient vector.

    Parameters
    ----------
    components : MixedPredictComponents or MixedMSLEstimator
        The fitted-model prediction context (estimation-space ``theta``, panel,
        draw source, pipeline, reparam space, softmax kernel, layout, config and
        a ``build_design`` closure).  Built by
        :meth:`~pybhatlib.models.mixmnl.MixMNLModel._predict_components`.
    data : pd.DataFrame, optional
        Dataset the design is rebuilt from.  Defaults to the training frame the
        ``build_design`` closure was created over when ``None``.
    spec : Any, optional
        Variable specification forwarded to ``build_design`` (the model's spec
        dict when the family default is used).
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides,
        applied via
        :func:`~pybhatlib.models._ate_common.apply_scenario_overrides` before the
        design is rebuilt.  ``None`` (default) predicts at the observed
        covariates.
    draws : DrawSource, optional
        Override the context's draw strategy (e.g. more replications for a
        smoother forecast).  Defaults to the fit-identical source.
    xp : backend, optional
        Array backend used to wrap the result.  Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs, nc)
        Per-observation, draw-integrated choice probabilities (each row
        non-negative and summing to one over available alternatives).
    """
    rc_ctx = _resolve(components)
    draw_src = draws if draws is not None else rc_ctx.draws
    design = _design_for(rc_ctx, data, spec, scenario)
    return _simulate_per_obs(rc_ctx, design, draw_src, xp=xp)


def mixmnl_predict_choice(
    components: MixedPredictComponents | Any,
    data: "pd.DataFrame | None" = None,
    spec: Any = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Most-likely predicted alternative for each observation of a MixMNL model.

    The mixed-logit analogue of
    :func:`pybhatlib.models.mnl.mnl_predict_choice`: takes the ``argmax`` over
    the draw-integrated per-observation choice probabilities from
    :func:`mixmnl_predict`, returning the 0-based index of the modal
    alternative.

    Parameters
    ----------
    components : MixedPredictComponents or MixedMSLEstimator
        The fitted-model prediction context (see :func:`mixmnl_predict`).
    data : pd.DataFrame, optional
        Dataset the design is rebuilt from.  Defaults to the training frame.
    spec : Any, optional
        Variable specification forwarded to ``build_design``.
    scenario : dict, optional
        Single-scenario covariate overrides (see :func:`mixmnl_predict`).
    draws : DrawSource, optional
        Override the context's draw strategy.
    xp : backend, optional
        Array backend used to wrap the intermediate probabilities.

    Returns
    -------
    NDArray, shape (n_obs,)
        The 0-based index of the most probable alternative per observation.
    """
    probs = mixmnl_predict(
        components, data, spec, scenario=scenario, draws=draws, xp=xp
    )
    return np.argmax(np.asarray(probs), axis=1)


__all__ = ["mixmnl_predict", "mixmnl_predict_choice"]
