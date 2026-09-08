"""Average Treatment Effect (ATE) for the mixed MDCEV family (Phase 4).

Thin wrapper over the shared mixed / panel ATE machinery
(:func:`pybhatlib.mixed._predict.mixed_ate`).  The mixed MDCEV ATE lifts the
shipped fixed-coefficient MDCEV participation prediction
(:func:`pybhatlib.models.mdcev._mdcev_forecast.mdcev_predict`) over the mixing
draws and differences the sample-averaged shares between counterfactual
scenarios, reusing the *same* mixing draws throughout so the ATE isolates the
covariate effect.

The result is a :class:`~pybhatlib.mixed._predict.MixedATEResult`, which is an
:class:`~pybhatlib.models._ate_common.ATEResultMixin` exposing the harmonized
``predicted_shares`` / ``shares_per_scenario`` fields plus the ``.comparison()``
and ``.summary()`` surface shared across MNP / MORP / MDCEV / MNL.

Validation is by *collapse* (``nrndcoef == 0`` mixed ATE equals the shipped
fixed-coefficient :func:`pybhatlib.models.mdcev._mdcev_ate.mdcev_ate` on the same
data) and interface conformance, not GAUSS parity (no GAUSS forecast oracle
exists for the mixed drivers).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._predict import MixedATEResult, mixed_ate
from pybhatlib.mixed._reparam import ReportingSpace
from pybhatlib.models._ate_common import ScenarioSpec
from pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast import (
    build_mdcev_mixed_components,
)
from pybhatlib.vecup._panel import PanelIndex


def mdcev_mixed_ate(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenarios: ScenarioSpec,
    n_draws: int = 1000,
    seed: int = 1234,
    alternative_names: Optional[list[str]] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> MixedATEResult:
    """Draw-integrated ATE across counterfactual scenarios for a mixed MDCEV model.

    Computes baseline participation shares (observed covariates) and, for each
    scenario, the draw-integrated shares under that scenario's covariate
    overrides, delegating to the shared
    :func:`~pybhatlib.mixed._predict.mixed_ate`.

    Parameters
    ----------
    model : MDCEVMixedModel
        A fitted model (carries the design specification and cached fitted
        estimator / estimation-space ``theta``).
    data : pd.DataFrame, optional
        Dataset the scenarios override; defaults to the model's own data.
    scenarios : dict or pd.DataFrame
        Scenario specification, normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`.
    n_draws : int, default 1000
        Monte-Carlo error draws per observation.
    seed : int, default 1234
        Monte-Carlo seed (shared across scenarios and MSL replications).
    alternative_names : list of str, optional
        Output labels; defaults to the model's ``alternatives``.
    draws : DrawSource, optional
        Override the fit-time mixing-draw source (shared across scenarios).
    xp : backend, optional
        Array backend used to wrap the result arrays. Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus ``shares_per_scenario`` and the
        ``.comparison()`` / ``.summary()`` surface.
    """
    components = build_mdcev_mixed_components(
        model, n_draws=n_draws, seed=seed, alternative_names=alternative_names
    )
    data_use = model.data if data is None else data
    return mixed_ate(
        components,
        data_use,
        None,
        scenarios=scenarios,
        draws=draws,
        alternative_names=alternative_names,
        xp=xp,
    )


def _prime_model_at_theta(
    model: Any,
    params: NDArray,
    *,
    draws: Optional[DrawSource] = None,
) -> None:
    """Assemble the MSL engine and pin the model's fitted state at reporting params.

    Bypasses the optimiser: builds the mixing spec, parameter layout, panel
    index and :class:`~pybhatlib.mixed._engine.MixedMSLEstimator` from the
    model's control / design, then writes the ``_fitted_spec`` /
    ``_fitted_layout`` / ``_fitted_est`` / ``_fitted_theta`` attributes that
    :func:`build_mdcev_mixed_components` reads.  Mirrors
    :meth:`MDCEVMixedModel._fit` up to (but excluding) the optimisation.

    The estimator's reparameterization space is swapped to
    :class:`~pybhatlib.mixed._reparam.ReportingSpace`, which consumes
    reporting-space (natural) parameters directly: ``xmu`` is the reported
    coefficient (no ``exp`` / sign reparam), the ``rcor`` block *is* the
    off-diagonal correlation (``matndupdiagonefull`` -> Cholesky), and the scale
    / Yeo-Johnson ``lambda`` blocks enter directly (no ``exp`` / no ``cdlogit``).
    No reporting -> estimation inversion is therefore performed -- ``params`` is
    fed to the engine as-is and ``predict`` / ``ate`` run at the natural
    parameters.  ``ReportingSpace`` emits no reparam gradients, which is fine
    because ATE / predict need none.

    Parameters
    ----------
    model : MDCEVMixedModel
        The (unfitted) model to prime in place.
    params : NDArray, shape (n_theta,)
        Reporting-space (natural) parameter vector in
        ``[beta | gamma | rcor | kern | scal | lam]`` order, matching the fitted
        :attr:`MDCEVMixedResults.params` layout.
    draws : DrawSource, optional
        Draw-source override forwarded to ``build_estimator``.
    """
    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    est = model.build_estimator(spec, layout, panel, draws=draws)
    ctrl = model.control
    est.space = ReportingSpace(
        layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
    )
    model._fitted_spec = spec
    model._fitted_layout = layout
    model._fitted_est = est
    model._fitted_theta = np.asarray(params, dtype=np.float64).reshape(-1)


def mdcev_mixed_ate_from_params(
    model: Any,
    params: NDArray,
    *,
    data: "pd.DataFrame | None" = None,
    scenarios: ScenarioSpec,
    n_draws: int = 1000,
    seed: int = 1234,
    alternative_names: Optional[list[str]] = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> MixedATEResult:
    """Draw-integrated mixed-MDCEV ATE from externally supplied reporting params.

    The mixed-family analogue of
    :func:`pybhatlib.models.mnl.mnl_ate_from_params`: it lets ATEs be computed
    from a manually entered (e.g. GAUSS) parameter vector without re-running the
    optimiser.  The model's MSL engine is assembled and pinned at ``params`` via
    :func:`_prime_model_at_theta` (which swaps the estimator's reparameterization
    to :class:`~pybhatlib.mixed._reparam.ReportingSpace`), then the call is
    dispatched to :func:`mdcev_mixed_ate`, so the return value is the **same**
    :class:`~pybhatlib.mixed._predict.MixedATEResult` type produced by the
    fitted :meth:`MDCEVMixedModel.ate`.

    Parameters
    ----------
    model : MDCEVMixedModel
        A model instance (need not be fitted); primed in place at ``params``.
    params : NDArray, shape (n_theta,)
        Reporting-space (natural) parameter vector in
        ``[beta | gamma | rcor | kern | scal | lam]`` order -- the same layout and
        semantics as the fitted :attr:`MDCEVMixedResults.params` and the fixed-
        coefficient ``mdcev_ate_from_params`` (which likewise accepts *reported*
        coefficients).  Because the engine is run under
        :class:`~pybhatlib.mixed._reparam.ReportingSpace`, these natural
        parameters are consumed directly: ``beta`` is the reported coefficient,
        the ``rcor`` block is the off-diagonal correlation, and the scale /
        Yeo-Johnson ``lambda`` blocks enter as-is (no reporting -> estimation
        inversion).  The kernel-owned ``gamma`` and MDCEV kernel scale pass
        through identically in both spaces.
    data : pd.DataFrame, optional
        Dataset the scenarios override; defaults to the model's own data.
    scenarios : dict or pd.DataFrame
        Scenario specification, normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`.
    n_draws : int, default 1000
        Monte-Carlo error draws per observation.
    seed : int, default 1234
        Monte-Carlo seed (shared across scenarios and MSL replications).
    alternative_names : list of str, optional
        Output labels; defaults to the model's ``alternatives``.
    draws : DrawSource, optional
        Override the mixing-draw source (shared across scenarios).
    xp : backend, optional
        Array backend used to wrap the result arrays. Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus ``shares_per_scenario`` and the
        ``.comparison()`` / ``.summary()`` surface.
    """
    _prime_model_at_theta(model, params, draws=draws)
    return mdcev_mixed_ate(
        model,
        data,
        scenarios=scenarios,
        n_draws=n_draws,
        seed=seed,
        alternative_names=alternative_names,
        draws=draws,
        xp=xp,
    )


__all__ = ["mdcev_mixed_ate", "mdcev_mixed_ate_from_params", "MixedATEResult"]
