"""Average Treatment Effect (ATE) post-estimation for the MixMNL model.

The mixed-logit analogue of :func:`pybhatlib.models.mnl.mnl_ate`.  The shipped
fixed-coefficient MNL ATE compares mean predicted shares between counterfactual
covariate scenarios; the mixed family lifts that formulation over the Halton
mixing draws using the shared engine (:func:`pybhatlib.mixed._predict.mixed_ate`):
baseline and per-scenario shares are draw-integrated (random coefficients drawn
per individual, exactly as at fit time) and the *same* mixing draws are reused
across scenarios so the ATE isolates the covariate effect.

The result is the harmonized :class:`~pybhatlib.mixed._predict.MixedATEResult`
(an :class:`~pybhatlib.models._ate_common.ATEResultMixin`), exposing
``predicted_shares`` / ``shares_per_scenario`` plus ``.comparison()`` /
``.summary()``.  Scenario specifications are normalised through
:func:`~pybhatlib.models._ate_common.scenarios_to_dict` inside the shared
engine.

With an empty random-coefficient spec (``nrndcoef == 0``) the draw-integrated
shares collapse to the fixed-coefficient MNL shares value-for-value -- the
collapse gate.  There is **no** GAUSS ATE oracle for the mixed drivers (Dale's
guidance): validation is by collapse plus interface conformance, not GAUSS
parity.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd  # noqa: F401  (string type annotations)
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._predict import MixedATEResult, mixed_ate
from pybhatlib.models._ate_common import ScenarioSpec


def mixmnl_ate(
    model: Any,
    theta: NDArray,
    *,
    data: "pd.DataFrame | None" = None,
    spec: Any = None,
    scenarios: "ScenarioSpec | None" = None,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    reporting: bool = False,
    xp=None,
) -> MixedATEResult:
    """Draw-integrated ATE across counterfactual scenarios for a MixMNL model.

    Builds the shared prediction context from ``model`` at the parameter vector
    ``theta`` (via
    :meth:`~pybhatlib.models.mixmnl.MixMNLModel._predict_components`) and
    dispatches to :func:`pybhatlib.mixed._predict.mixed_ate`, which computes the
    draw-integrated baseline shares and, for each scenario, the shares under that
    scenario's covariate overrides -- reusing the *same* mixing draws throughout.

    Parameters
    ----------
    model : MixMNLModel
        The MixMNL facade (supplies the mixing spec / layout, panel index,
        random-coefficient pipeline, softmax kernel, draw source and the
        ``build_design`` closure through ``_predict_components``).  Need not be
        fitted -- ``theta`` is passed explicitly.
    theta : NDArray, shape (n_theta,)
        Parameter vector in the space selected by ``reporting``: **estimation
        space** (``[beta | rcor | scal | lam]`` order, ``reporting=False``, the
        fit path) or **reporting space** (natural parameters, ``reporting=True``).
        With ``nrndcoef == 0`` both spaces coincide and this is simply ``beta``.
    data : pd.DataFrame, optional
        Dataset the scenarios override.  Defaults to the model's training frame.
    spec : Any, optional
        Variable specification forwarded to ``build_design``.  Defaults to the
        model's spec dict.
    scenarios : dict or pd.DataFrame, optional
        Scenario specification, normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`:

        - dict form ``{name: {col: scalar | source_col, ...}, ...}``
        - DataFrame form (rows = scenarios, columns = variables).

        ``None`` (default) yields baseline ``predicted_shares`` only with an
        empty ``shares_per_scenario``.
    draws : DrawSource, optional
        Override the context's draw strategy.  Defaults to the fit-identical
        source (shared across scenarios).
    alternative_names : list of str, optional
        Output labels for the result.  Defaults to the model's ``alternatives``.
    reporting : bool, default False
        Interpret ``theta`` in reporting space (natural parameters) rather than
        estimation space; forwarded to
        :meth:`~pybhatlib.models.mixmnl.MixMNLModel._predict_components` to select
        :class:`~pybhatlib.mixed._reparam.ReportingSpace`.
    xp : backend, optional
        Array backend used to wrap the result arrays.  Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus ``shares_per_scenario`` and the
        ``.comparison()`` / ``.summary()`` surface.
    """
    components = model._predict_components(
        theta,
        draws=draws,
        alternative_names=alternative_names,
        reporting=reporting,
    )
    data_use = data if data is not None else model.data
    spec_use = spec if spec is not None else model.spec_dict
    # An empty spec still exercises the shared engine's baseline computation.
    scen = scenarios if scenarios is not None else {}
    return mixed_ate(
        components,
        data_use,
        spec_use,
        scenarios=scen,
        draws=draws,
        alternative_names=alternative_names,
        xp=xp,
    )


def mixmnl_ate_from_params(
    params: NDArray,
    *,
    data: "pd.DataFrame",
    spec: Any,
    alternatives: list[str],
    control: Any = None,
    availability: str | list[str] = "none",
    var_names: Optional[list[str]] = None,
    scenarios: "ScenarioSpec | None" = None,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    xp=None,
) -> MixedATEResult:
    """Draw-integrated ATE directly from externally supplied *reporting* params.

    The mixed-logit analogue of
    :func:`pybhatlib.models.mnl.mnl_ate_from_params`: it assembles a
    :class:`~pybhatlib.models.mixmnl.MixMNLModel` over *data* / *spec* /
    *alternatives* / *control*, then dispatches to :func:`mixmnl_ate` at the
    supplied parameters, so ATEs can be computed from manually entered (e.g.
    GAUSS) estimates without re-fitting.  The returned object is the same
    :class:`~pybhatlib.mixed._predict.MixedATEResult` produced by a fitted
    model's :meth:`~pybhatlib.models.mixmnl.MixMNLModel.ate`.

    Parameters
    ----------
    params : ndarray, shape (n_params,)
        Parameter vector in **reporting space** (natural parameters), same order
        and layout as a fitted :class:`MixMNLResults.params`:
        ``[beta | rcor | scal | lam]`` (the softmax kernel owns no ``kern``
        block) -- fixed coefficients entered directly, ``rcor`` the off-diagonal
        random-coefficient correlations, ``scal`` the std-dev (scale) vector, and
        ``lam`` the Yeo-Johnson powers in ``(0, 2)``.  The context is built with
        :class:`~pybhatlib.mixed._reparam.ReportingSpace`, which consumes these
        natural parameters directly -- no reporting->estimation inversion is
        performed.  This matches the fixed-coefficient
        :func:`~pybhatlib.models.mnl.mnl_ate_from_params` (reported coefficients)
        and a fitted ``results.params``.  With an empty random-coefficient spec
        (``nrndcoef == 0``) this is simply ``beta`` and coincides with the
        fixed-coefficient vector.
    data : pd.DataFrame
        Dataset the model and scenarios are built over.
    spec : Any
        Variable specification mapping (as for :class:`MixMNLModel`).
    alternatives : list of str
        Choice-indicator column names.
    control : MixMNLControl or None
        Estimation control carrying the random-coefficient spec / MSL knobs.
        Defaults to :class:`MixMNLControl` (which yields a fixed-coefficient
        MNL, ``nrndcoef == 0``).
    availability : str or list of str, default "none"
        Availability specification forwarded to :class:`MixMNLModel`.
    var_names : list of str or None
        Coefficient names forwarded to :class:`MixMNLModel`.
    scenarios : dict or pd.DataFrame, optional
        Scenario specification (see :func:`mixmnl_ate`).
    draws : DrawSource, optional
        Override the draw strategy (shared across scenarios).
    alternative_names : list of str, optional
        Output labels; defaults to ``alternatives``.
    xp : backend, optional
        Array backend used to wrap the result arrays.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus ``shares_per_scenario`` and the
        ``.comparison()`` / ``.summary()`` surface.
    """
    # Local import avoids a model <-> ate import cycle at module load.
    from pybhatlib.models.mixmnl._mixmnl_model import MixMNLModel

    model = MixMNLModel(
        data=data,
        alternatives=alternatives,
        availability=availability,
        spec=spec,
        var_names=var_names,
        control=control,
    )
    return mixmnl_ate(
        model,
        np.asarray(params, dtype=np.float64),
        data=data,
        spec=spec,
        scenarios=scenarios,
        draws=draws,
        alternative_names=alternative_names,
        reporting=True,
        xp=xp,
    )


__all__ = ["mixmnl_ate", "mixmnl_ate_from_params", "MixedATEResult"]
