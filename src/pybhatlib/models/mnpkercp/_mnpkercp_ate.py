"""Draw-integrated Average Treatment Effect (ATE) for the mixed-panel MNP.

Lifts the shipped fixed-coefficient MNP ATE
(:func:`pybhatlib.models.mnp._mnp_ate.mnp_ate`) over the mixing draws via the
shared machinery (:func:`pybhatlib.mixed._predict.mixed_ate`): baseline shares
are the draw-integrated MVNCD choice probabilities at the observed covariates,
and each scenario's shares reuse the *same* mixing draws under that scenario's
covariate overrides, so the ATE isolates the covariate effect.

The per-alternative choice-probability formulation is supplied by
:func:`pybhatlib.models.mnpkercp._mnpkercp_forecast._mnpkercp_kernel_predict`; the
scenario overrides / normalisation reuse
:mod:`pybhatlib.models._ate_common` (``ScenarioSpec`` /
``apply_scenario_overrides`` / ``scenarios_to_dict``) through the shared engine.
The returned :class:`~pybhatlib.mixed._predict.MixedATEResult` inherits the
harmonized ``.comparison()`` / ``.summary()`` surface
(:class:`~pybhatlib.models._ate_common.ATEResultMixin`).

Validation is by *collapse* (``nrndcoef = 0`` reproduces the shipped
fixed-coefficient MNP ATE) and interface conformance, not GAUSS parity.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd  # noqa: F401  (string type annotations)
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._predict import MixedATEResult, mixed_ate
from pybhatlib.mixed._reparam import ReportingSpace
from pybhatlib.models._ate_common import ScenarioSpec
from pybhatlib.models.mnpkercp._mnpkercp_control import MNPKerCPControl
from pybhatlib.models.mnpkercp._mnpkercp_forecast import build_predict_components


def mnpkercp_ate(
    model: Any,
    *,
    scenarios: ScenarioSpec,
    data: "pd.DataFrame | None" = None,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    xp: Any = None,
) -> MixedATEResult:
    """Draw-integrated ATE across counterfactual scenarios for a fitted MNPKerCP.

    Computes baseline shares (observed covariates) and, for each scenario, the
    draw-integrated shares under that scenario's covariate overrides, reusing the
    fit-identical mixing draws throughout.

    Parameters
    ----------
    model : MNPKerCPModel
        A fitted model (``model.fit()`` already called).
    scenarios : dict or pd.DataFrame
        Scenario specification, normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`:

        - dict form ``{name: {col: scalar | source_col, ...}, ...}``
        - DataFrame form (rows = scenarios, columns = variables).
    data : pd.DataFrame, optional
        Dataset the scenarios override; defaults to the model's fitted data.
    draws : DrawSource, optional
        Override the fit-identical draw strategy (shared across scenarios).
    alternative_names : list of str, optional
        Output labels; defaults to ``model.alternatives``.
    xp : module, optional
        Array backend used to wrap the result arrays. Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus per-scenario ``shares_per_scenario``
        and the ``.comparison()`` / ``.summary()`` surface.
    """
    names = alternative_names if alternative_names is not None else list(model.alternatives)
    comp = build_predict_components(model, draws=draws, alternative_names=names)
    frame = data if data is not None else model.data
    return mixed_ate(
        comp,
        frame,
        spec=model.spec_dict,
        scenarios=scenarios,
        draws=draws,
        alternative_names=names,
        xp=xp,
    )


def _prepare_at_params(
    model: Any,
    params: NDArray,
    *,
    draws: Optional[DrawSource] = None,
) -> None:
    """Assemble the MSL estimator on ``model`` at externally supplied params.

    Mirrors the engine-assembly half of
    :meth:`~pybhatlib.models.mnpkercp._mnpkercp_model.MNPKerCPModel._fit` but
    skips optimisation: it builds the mixing spec / parameter layout, panel
    index and MSL estimator, then persists the fit-context attributes
    (``_theta_hat`` / ``_est`` / ``_mixing_spec`` / ``_layout``) that the shared
    prediction machinery reads via
    :func:`~pybhatlib.models.mnpkercp._mnpkercp_forecast.build_predict_components`.

    The estimator is built with a
    :class:`~pybhatlib.mixed._reparam.ReportingSpace` (not the optimizer's
    ``EstimationSpace``), so the supplied *reporting-space* (natural) params are
    consumed **directly**: ``xmu`` is the reported coefficient (no ``exp``), the
    ``rcor`` block is the off-diagonal correlation (``matndupdiagonefull`` ->
    Cholesky), and the scale / ``lambda`` blocks are entered as-is (no ``exp`` /
    no ``cdlogit``). No reporting -> estimation inversion is performed. The
    reporting space emits no reparameterization gradients, which is fine because
    ATE / predict need none.

    Parameters
    ----------
    model : MNPKerCPModel
        A constructed (not necessarily fitted) model carrying the data / spec /
        control.
    params : NDArray, shape (n_theta,)
        **Reporting-space** (natural) parameter vector in the GAUSS ``b`` block
        order ``[beta | rcor | scal | kern | lam]`` -- the same layout and
        semantics as the fitted ``MNPKerCPResults.params``.
    draws : DrawSource, optional
        Draw source for the MSL integration.  Defaults to the model's runtime
        scrambled-Halton source (or zeros when there are no random coefficients).

    Raises
    ------
    ValueError
        If ``params`` does not match the model's parameter-layout length.
    """
    from pybhatlib.vecup._panel import PanelIndex

    spec, layout = model._build_spec_layout()
    panel = PanelIndex.from_ids(model.person_ids)
    ctrl = model.control
    space = ReportingSpace(
        layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
    )
    est = model._build_estimator(spec, layout, panel, draws=draws, space=space)

    params_r = np.asarray(params, dtype=np.float64).ravel()
    if params_r.shape[0] != layout.n_theta:
        raise ValueError(
            f"params has length {params_r.shape[0]}, expected {layout.n_theta} "
            f"for the reporting-space [beta | rcor | scal | kern | lam] layout"
        )

    model._theta_hat = params_r
    model._est = est
    model._mixing_spec = spec
    model._layout = layout


def mnpkercp_ate_from_params(
    params: NDArray,
    *,
    data: "pd.DataFrame",
    alternatives: list[str],
    spec: dict,
    scenarios: ScenarioSpec,
    control: MNPKerCPControl | None = None,
    availability: str | list[str] = "none",
    var_names: list[str] | None = None,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    xp: Any = None,
) -> MixedATEResult:
    """Draw-integrated MNPKerCP ATE directly from externally supplied params.

    Convenience wrapper mirroring
    :func:`pybhatlib.models.mnl._mnl_ate.mnl_ate_from_params`: it constructs a
    :class:`~pybhatlib.models.mnpkercp._mnpkercp_model.MNPKerCPModel`, assembles
    the MSL estimator *at the supplied parameters* (without re-fitting), and
    dispatches to :func:`mnpkercp_ate` so a draw-integrated ATE can be computed
    from manually entered (e.g. GAUSS) or fitted estimates.  The returned type is
    the same :class:`~pybhatlib.mixed._predict.MixedATEResult` as the fitted
    :meth:`MNPKerCPModel.ate`.

    The supplied params are consumed in **reporting space** via a
    :class:`~pybhatlib.mixed._reparam.ReportingSpace`, so the fitted
    ``MNPKerCPResults.params`` (documented as reporting-space) plug in verbatim
    and reproduce the fitted :meth:`MNPKerCPModel.ate`. No reporting ->
    estimation inversion is needed: ``xmu`` is the reported coefficient, the
    ``rcor`` block is the off-diagonal correlation, and the scale / ``lambda``
    blocks are entered directly.

    Parameters
    ----------
    params : NDArray, shape (n_theta,)
        **Reporting-space** (natural) parameter vector, in GAUSS ``b`` block
        order ``[beta | rcor | scal | kern | lam]`` -- the same layout and
        semantics as the fitted ``MNPKerCPResults.params`` (fixed coefficients,
        the joint random-coefficient + differenced-kernel correlation
        off-diagonal entries, the random-coefficient scale/std-dev vector, the
        kernel-scale parameters, and the Yeo-Johnson ``lambda`` values in
        ``(0, 2)``).  When there are no random coefficients
        (``normvar``/``logvar``/``yjvar`` empty) and no sign constraints, the
        vector reduces to the ``beta`` block, which coincides with the reported
        coefficients accepted by the fixed-coefficient
        :func:`pybhatlib.models.mnp._mnp_ate.mnp_ate_from_params`.
    data : pd.DataFrame
        Dataset the scenarios override.
    alternatives : list of str
        Column names holding the 0/1 choice indicators.
    spec : dict
        Variable specification mapping (as for ``MNPKerCPModel``).
    scenarios : dict or pd.DataFrame
        Scenario specification (see :func:`mnpkercp_ate`).
    control : MNPKerCPControl, optional
        Control structure (random-coefficient spec, copula flag, MSL knobs).
        Defaults to :class:`MNPKerCPControl`.
    availability : str or list of str, default "none"
        Availability specification forwarded to ``MNPKerCPModel``.
    var_names : list of str, optional
        Coefficient names forwarded to ``MNPKerCPModel``.
    draws : DrawSource, optional
        Draw source for the MSL integration (shared across scenarios).
    alternative_names : list of str, optional
        Output labels; defaults to ``alternatives``.
    xp : module, optional
        Array backend used to wrap the result arrays. Defaults to NumPy.

    Returns
    -------
    MixedATEResult
        Baseline ``predicted_shares`` plus per-scenario shares and the
        ``.comparison()`` / ``.summary()`` surface.
    """
    from pybhatlib.models.mnpkercp._mnpkercp_model import MNPKerCPModel

    model = MNPKerCPModel(
        data=data,
        alternatives=alternatives,
        availability=availability,
        spec=spec,
        var_names=var_names,
        control=control,
    )
    _prepare_at_params(model, params, draws=draws)
    return mnpkercp_ate(
        model,
        scenarios=scenarios,
        data=data,
        draws=draws,
        alternative_names=alternative_names,
        xp=xp,
    )


__all__ = ["mnpkercp_ate", "mnpkercp_ate_from_params", "MixedATEResult"]
