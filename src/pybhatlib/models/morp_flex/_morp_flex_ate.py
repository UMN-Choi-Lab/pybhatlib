"""Average Treatment Effect (ATE) post-estimation for MORPFlex (mixed MORP).

Draw-integrated ordinal-probability predictions under base and counterfactual
scenarios for the mixed-panel MORP model.  Lifts the shipped fixed-coefficient
MORP ATE (:func:`pybhatlib.models.morp._morp_ate.morp_ate`) over the mixing
draws through the shared engine: each scenario's covariate overrides are applied
via :func:`pybhatlib.models._ate_common.apply_scenario_overrides`, the design is
rebuilt, and the *same* mixing draws are reused so the ATE isolates the covariate
effect.

Per-outcome (ATE opt-out)
-------------------------
Like the fixed :class:`~pybhatlib.models.morp._morp_ate.MORPATEResult`, MORPFlex
predicts one ordinal-category distribution per outcome dimension, so
:class:`MORPFlexATEResult` deliberately does **not** inherit the single-vector
:class:`~pybhatlib.models._ate_common.ATEResultMixin`; ``predicted_probs`` /
``shares_per_scenario`` carry one array per dimension and ``comparison`` returns
one percentage-change array per dimension.

Validation is by collapse (``nrndcoef = 0`` mixed ATE equals the fixed-coef MORP
ATE) + interface conformance, not GAUSS parity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._engine import DesignData
from pybhatlib.models._ate_common import (
    ScenarioSpec,
    scenarios_to_dict as _scenarios_to_dict,
)
from pybhatlib.models.morp_flex._morp_flex_forecast import (
    _make_build_design,
    build_components,
    draw_integrated_per_obs,
    split_by_dim,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pybhatlib.models.morp_flex._morp_flex_model import MORPFlexModel


@dataclass
class MORPFlexATEResult:
    """MORPFlex (mixed-panel MORP) ATE analysis results.

    Per-outcome: each field carries one array per ordinal dimension.  Not an
    :class:`~pybhatlib.models._ate_common.ATEResultMixin` (see the module
    docstring).

    Attributes
    ----------
    n_obs : int
        Number of observations averaged over.
    predicted_probs : list of NDArray
        Baseline (observed-covariate) draw-integrated mean category
        probabilities; ``predicted_probs[d]`` has shape ``(n_categories[d],)``.
    base_probs, treatment_probs : list of NDArray or None
        Kept for symmetry with the fixed MORP ATE result; populated by callers
        that difference two scenarios directly (``None`` otherwise).
    shares_per_scenario : dict[str, list[NDArray]] or None
        Per-scenario draw-integrated mean category probabilities (one list per
        scenario name, one array per dimension).
    """

    n_obs: int
    predicted_probs: list[NDArray]
    base_probs: Optional[list[NDArray]] = None
    treatment_probs: Optional[list[NDArray]] = None
    shares_per_scenario: Optional[dict[str, list[NDArray]]] = None

    def comparison(self, base: str, treatment: str) -> list[NDArray]:
        """Percentage change between two scenarios, per outcome dimension.

        Mirrors :meth:`pybhatlib.models.morp._morp_ate.MORPATEResult.comparison`:
        one array per dimension ``d`` of shape ``(n_categories[d],)`` equal to
        ``100 * (treatment[d] - base[d]) / base[d]`` (``NaN`` where the base
        probability is zero).

        Raises
        ------
        ValueError
            If ``shares_per_scenario`` is ``None`` or a scenario name is missing.
        """
        if self.shares_per_scenario is None:
            raise ValueError(
                "comparison() requires shares_per_scenario; run morp_flex_ate "
                "with scenarios=..."
            )
        if base not in self.shares_per_scenario:
            raise ValueError(
                f"Scenario '{base}' not found in shares_per_scenario"
            )
        if treatment not in self.shares_per_scenario:
            raise ValueError(
                f"Scenario '{treatment}' not found in shares_per_scenario"
            )
        b_list = self.shares_per_scenario[base]
        t_list = self.shares_per_scenario[treatment]
        out: list[NDArray] = []
        for b, t in zip(b_list, t_list):
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(np.where(b > 0, 100.0 * (t - b) / b, np.nan))
        return out


def _mean_split(per_obs: NDArray, n_categories) -> list[NDArray]:
    """Mean over observations, then split into per-dimension category vectors."""
    mean = np.asarray(per_obs, dtype=np.float64).mean(axis=0)      # (sum_ncat,)
    return [np.asarray(b, dtype=np.float64) for b in split_by_dim(mean, n_categories)]


def morp_flex_ate(
    model: "MORPFlexModel",
    X: Optional[NDArray] = None,
    *,
    scenarios: "ScenarioSpec | None" = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> MORPFlexATEResult:
    """Draw-integrated predicted ordinal probabilities / ATE for MORPFlex.

    Mirrors :func:`pybhatlib.models.morp._morp_ate.morp_ate` (per-outcome mean
    category probabilities; ``scenarios=`` for counterfactuals) with the added
    integration over the mixing distribution via the shared MSL engine.

    Parameters
    ----------
    model : MORPFlexModel
        A fitted model.
    X : NDArray, shape (N, nord, n_beta), optional
        Pre-built design for the single-design (no-scenario) path.  Defaults to
        the fitted training design.  Must have the same number of rows as the
        fit sample (the mixing draws are held per fit individual).
    scenarios : dict or pd.DataFrame, optional
        Scenario specification (``{name: {col: scalar | source_col, ...}, ...}``
        or a DataFrame), normalised via
        :func:`~pybhatlib.models._ate_common.scenarios_to_dict`.  When supplied,
        the design is rebuilt per scenario from the model's data / spec and the
        results are returned in ``shares_per_scenario`` (with ``.comparison``).
    draws : DrawSource, optional
        Override the fit-identical draw strategy (shared across scenarios).
    xp : module, optional
        Array backend used to wrap the per-dimension arrays. Defaults to NumPy.

    Returns
    -------
    MORPFlexATEResult

    Raises
    ------
    ValueError
        If ``scenarios`` is given but the model carries no data / spec to rebuild
        the design.
    """
    n_categories = model.n_categories

    def _wrap(list_of_arr: list[NDArray]) -> list[NDArray]:
        if xp is None:
            return list_of_arr
        return [xp.asarray(a) for a in list_of_arr]

    # --- Scenario mode ---
    if scenarios is not None:
        if getattr(model, "data", None) is None or getattr(model, "spec_dict", None) is None:
            raise ValueError(
                "scenarios= requires the model to carry data and spec"
            )
        build_design = _make_build_design(model)
        components = build_components(model, build_design=build_design)
        data = model.data
        spec = model.spec_dict

        base_per_obs = draw_integrated_per_obs(
            components, data=data, spec=spec, scenario=None, draws=draws
        )
        baseline = _mean_split(base_per_obs, n_categories)

        shares_per_scenario: dict[str, list[NDArray]] = {}
        for name, overrides in _scenarios_to_dict(scenarios).items():
            per_obs = draw_integrated_per_obs(
                components, data=data, spec=spec, scenario=overrides, draws=draws
            )
            shares_per_scenario[str(name)] = _wrap(
                _mean_split(per_obs, n_categories)
            )

        return MORPFlexATEResult(
            n_obs=int(base_per_obs.shape[0]),
            predicted_probs=_wrap(baseline),
            shares_per_scenario=shares_per_scenario,
        )

    # --- Single-design mode ---
    if X is None:
        components = build_components(model)
    else:
        from types import SimpleNamespace

        design = DesignData(
            X=np.asarray(X, dtype=np.float64),
            obs=SimpleNamespace(y_ord=model.y_ord),
        )
        components = build_components(model, design=design)

    per_obs = draw_integrated_per_obs(components, draws=draws)
    return MORPFlexATEResult(
        n_obs=int(per_obs.shape[0]),
        predicted_probs=_wrap(_mean_split(per_obs, n_categories)),
    )


def morp_flex_ate_from_params(
    params: NDArray,
    *,
    data,
    dep_vars: list[str],
    spec: dict,
    n_categories: list[int],
    control: "MORPFlexControl | None" = None,
    param_names: Optional[list[str]] = None,
    scenarios: "ScenarioSpec | None" = None,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> MORPFlexATEResult:
    """Compute MORPFlex draw-integrated ATE directly from supplied parameters.

    Mirrors :func:`pybhatlib.models.mnl._mnl_ate.mnl_ate_from_params` and the
    fixed-coefficient :func:`pybhatlib.models.morp._morp_ate.morp_ate_from_params`
    (which accept the *reported* coefficients): it builds the mixed estimator at
    externally-supplied parameters (e.g. the fitted
    :attr:`MORPFlexResults.params`, or GAUSS estimates) and dispatches to
    :func:`morp_flex_ate`, so ATEs can be computed without re-fitting.

    Unlike the fixed-coefficient models the ATE is *simulated* — the supplied
    parameters are used to assemble the shared MSL engine (draws, mixing pipeline
    and rectangle-MVNCD kernel), and the per-outcome ordinal probabilities are
    integrated over the mixing draws.

    Parameter space
    ---------------
    ``params`` is the **reporting-space** (natural) parameter vector, identical
    in order and meaning to :attr:`MORPFlexResults.params`: ordered threshold cut
    points, fixed coefficients, the joint off-diagonal correlation entries, the
    scale (std-dev) vector, the Yeo-Johnson powers in ``(0, 2)`` and — for the YJ
    kernel — the ordinal-kernel powers.  No estimation<->reporting inversion is
    performed: the engine is assembled in reporting space
    (:class:`~pybhatlib.mixed._reparam.ReportingSpace` plus a reporting-mode
    kernel, via ``MORPFlexModel._build_estimator(reporting=True)``), which
    consumes these natural parameters directly (``xmu`` = the reported
    coefficient, the ``rcor`` block *is* the off-diagonal correlation, scale and
    lambda entered directly).  The reporting path is gradient-free, which is
    sufficient for prediction / ATE.

    Parameters
    ----------
    params : ndarray, shape (n_theta,)
        Reporting-space (natural) parameter vector in the physical block order
        ``[thresh | beta | rcor | scal | lam | kernlam]`` — the same layout as
        :attr:`MORPFlexResults.params`.
    data : str, os.PathLike, or pd.DataFrame
        Dataset used to rebuild the design (and, for ``scenarios=``, to apply
        covariate overrides).
    dep_vars : list of str
        Ordinal outcome column names (``nord = len(dep_vars)``).
    spec : dict
        Variable specification mapping, as for :class:`MORPFlexModel`.
    n_categories : list of int
        Number of ordinal categories per dimension.
    control : MORPFlexControl or None
        Estimation control (random-coefficient spec, copula / yj_kernel flags,
        MSL knobs).  Defaults to :class:`MORPFlexControl`.
    param_names : list of str or None
        Reporting-parameter names for the constructed results object.  Defaults
        to the model's derived names.
    scenarios : dict or pd.DataFrame, optional
        Scenario specification forwarded to :func:`morp_flex_ate`.
    draws : DrawSource, optional
        Override the runtime draw strategy (shared across scenarios and used when
        assembling the estimator).
    xp : module, optional
        Array backend used to wrap the per-dimension arrays.

    Returns
    -------
    MORPFlexATEResult
        The same per-outcome ATE result type returned by :meth:`MORPFlexModel.ate`.

    Raises
    ------
    ValueError
        If ``params`` length does not match the model's parameter count.
    """
    from types import SimpleNamespace

    from pybhatlib.vecup._panel import PanelIndex
    from pybhatlib.models.morp_flex._morp_flex_control import MORPFlexControl
    from pybhatlib.models.morp_flex._morp_flex_model import MORPFlexModel
    from pybhatlib.models.morp_flex._morp_flex_results import MORPFlexResults

    ctrl = control if control is not None else MORPFlexControl()
    model = MORPFlexModel(
        data=data,
        dep_vars=dep_vars,
        spec=spec,
        n_categories=n_categories,
        control=ctrl,
    )

    spec_mix, layout = model._build_spec_layout()
    params_arr = np.asarray(params, dtype=np.float64).reshape(-1)
    if params_arr.shape[0] != layout.n_theta:
        raise ValueError(
            f"params has length {params_arr.shape[0]}, expected {layout.n_theta}"
        )

    panel = PanelIndex.from_ids(model.person_ids)
    # Reporting-space engine: ReportingSpace + reporting-mode kernel consume the
    # natural parameters directly (no estimation<->reporting inversion).
    est = model._build_estimator(
        spec_mix, layout, panel, draws=draws, reporting=True
    )

    # Prime the post-fit context / results so the shared ATE path can drive the
    # mixed predictor without an optimisation pass.  ``_fit_ctx.theta`` is the
    # reporting-space vector, matching ``est.space`` / ``est.kernel``.
    model._fit_ctx = SimpleNamespace(
        theta=params_arr, spec=spec_mix, layout=layout, panel=panel, est=est
    )
    names = (
        param_names
        if param_names is not None
        else model._param_names(spec_mix, layout)
    )
    model.results_ = MORPFlexResults.from_estimates(
        params_arr, param_names=names, control=ctrl
    )

    return morp_flex_ate(model, scenarios=scenarios, draws=draws, xp=xp)
