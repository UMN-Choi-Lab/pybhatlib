"""Draw-integrated prediction / forecast for the mixed MDCEV family (Phase 4).

The mixed MDCEV model mixes over the baseline-utility coefficients only; the
translation (``gamma``) parameters and the MDCEV kernel error scale are
kernel-owned, non-mixed parameters, and there is no copula.  Post-estimation
prediction therefore lifts the *shipped fixed-coefficient* MDCEV formulation
(:func:`pybhatlib.models.mdcev._mdcev_forecast.mdcev_predict` for participation
shares, :func:`~pybhatlib.models.mdcev._mdcev_forecast.mdcev_forecast` for
expenditure allocation) over the mixing draws, exactly as the shared mixed
prediction machinery (:mod:`pybhatlib.mixed._predict`) prescribes.

Exact reuse of the shipped forecast
-----------------------------------
The shipped ``mdcev_predict`` / ``mdcev_forecast`` compute the baseline utility
``v = X_new @ beta`` *internally* from a design ``X_new`` and a shared
``beta``.  In the mixed model the per-observation baseline utility ``Vsub``
already carries the drawn random coefficients (``Vsub = X @ xmunew`` for one
MSL replication), so we feed it through the shipped evaluator with an
**identity one-column design** ``X_new = Vsub[:, :, None]`` and ``beta = [1.0]``.
Then ``v_q == Vsub[q]`` and the shipped code path is reproduced verbatim -- no
duplication of the MDCEV consumption / allocation logic.

Collapse
--------
With an empty random-coefficient spec (``nrndcoef == 0``) the drawn ``xmunew``
equals the fixed ``beta`` for every draw, so ``Vsub == X @ beta`` is constant
across the MSL replications; feeding it through ``mdcev_predict`` /
``mdcev_forecast`` with the **same Monte-Carlo seed** reproduces the shipped
fixed-coefficient prediction value-for-value (the RNG path is identical:
``default_rng(seed)`` then one ``rng.integers`` per observation).  Validation is
by this collapse plus interface conformance, not GAUSS parity (no GAUSS forecast
oracle exists for the mixed drivers).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._engine import DesignData
from pybhatlib.mixed._predict import (
    KernelPredict,
    MixedPredictComponents,
    _design_for,
    _resolve,
    _simulate_per_obs,
    mixed_predict_shares,
)
from pybhatlib.models._ate_common import apply_scenario_overrides
from pybhatlib.models.mdcev._mdcev_forecast import mdcev_forecast, mdcev_predict
from pybhatlib.models.mdcev._mdcev_model import _ensure_special_cols


# ---------------------------------------------------------------------------
# per-draw kernel-prediction hooks (reuse the shipped MDCEV formulation)
# ---------------------------------------------------------------------------

def make_mdcev_mixed_share_predict(
    *,
    n_draws: int = 1000,
    seed: int = 1234,
    outside_good_gamma: float = -1000.0,
) -> KernelPredict:
    """Build a per-draw *participation-share* prediction hook for the MSL engine.

    The returned callable matches the shared
    :data:`~pybhatlib.mixed._predict.KernelPredict` signature
    ``(kernel, Vsub, obs, kstate, rc_draw) -> (n_obs, nc)`` and delegates to the
    shipped :func:`~pybhatlib.models.mdcev._mdcev_forecast.mdcev_predict`, feeding
    the drawn baseline utilities ``Vsub`` through an identity one-column design
    (``X_new = Vsub[:, :, None]``, ``beta = [1.0]``) so the shipped MDCEV
    consumption-probability code is reproduced verbatim per MSL replication.

    Parameters
    ----------
    n_draws : int, default 1000
        Monte-Carlo error draws per observation inside ``mdcev_predict``.
    seed : int, default 1234
        Monte-Carlo seed; held fixed across MSL replications so a
        ``nrndcoef == 0`` mixed prediction reproduces the fixed-coefficient
        prediction value-for-value.
    outside_good_gamma : float, default -1000.0
        Fixed satiation forced on the outside good (GAUSS ``u[.,1] = -1000``).

    Returns
    -------
    KernelPredict
        Hook returning per-observation participation shares ``(n_obs, nc)`` for
        one MSL replication.
    """

    def _predict(
        kernel: Any,
        Vsub: NDArray,
        obs: Any,
        kstate: Any,
        rc_draw: NDArray,
    ) -> NDArray:
        Vsub = np.asarray(Vsub, dtype=np.float64)
        gamma_raw = np.asarray(kstate.gamma_raw, dtype=np.float64)
        sigma = float(np.exp(kstate.log_sigma))
        gamma_design = np.asarray(obs.gamma_design, dtype=np.float64)  # (n_obs, nc, ng)
        price = np.asarray(obs.price, dtype=np.float64)                # (n_obs, nc)

        x_eff = Vsub[:, :, None]                                       # (n_obs, nc, 1)
        b_rep = np.concatenate([[1.0], gamma_raw])                    # nvarm==1
        shares = mdcev_predict(
            None,
            x_eff,
            gamma_design,
            price,
            n_draws=n_draws,
            seed=seed,
            b_reported=b_rep,
            sigma=sigma,
            outside_good_gamma=outside_good_gamma,
        )
        return np.asarray(shares, dtype=np.float64)

    return _predict


def _forecast_per_draw(
    Vsub: NDArray,
    obs: Any,
    kstate: Any,
    budget: NDArray,
    *,
    n_replications: int,
    seed: int,
    num_outside: int,
    outside_good_gamma: float,
) -> NDArray:
    """Mean expenditure allocation ``(n_obs, nc)`` for one MSL replication.

    Delegates to the shipped
    :func:`~pybhatlib.models.mdcev._mdcev_forecast.mdcev_forecast` via the same
    identity one-column design trick, then averages the stacked
    ``(n_replications * n_obs, nc)`` allocation over the replications.
    """
    Vsub = np.asarray(Vsub, dtype=np.float64)
    n_obs, nc = Vsub.shape
    gamma_raw = np.asarray(kstate.gamma_raw, dtype=np.float64)
    sigma = float(np.exp(kstate.log_sigma))
    gamma_design = np.asarray(obs.gamma_design, dtype=np.float64)
    price = np.asarray(obs.price, dtype=np.float64)

    x_eff = Vsub[:, :, None]
    b_rep = np.concatenate([[1.0], gamma_raw])
    stacked = mdcev_forecast(
        None,
        x_eff,
        gamma_design,
        price,
        budget,
        n_replications=n_replications,
        seed=seed,
        num_outside=num_outside,
        b_reported=b_rep,
        sigma=sigma,
        outside_good_gamma=outside_good_gamma,
    )
    # stacked is (n_replications * n_obs, nc), replication-major.
    return stacked.reshape(n_replications, n_obs, nc).mean(axis=0)


# ---------------------------------------------------------------------------
# design rebuild + component assembly
# ---------------------------------------------------------------------------

def make_mdcev_mixed_design_builder(model: Any):
    """Return a ``(data_frame, spec) -> DesignData`` closure for a mixed MDCEV model.

    Rebuilds the baseline / satiation design tensors and consumption / price
    matrices for a (possibly scenario-overridden) DataFrame from the fitted
    model's ``utility_spec`` / ``gamma_spec`` / ``alternatives`` / ``price_cols``,
    mirroring :meth:`MDCEVMixedModel._build_design`.  The ``spec`` argument is
    accepted for signature compatibility with the shared machinery and ignored.

    Parameters
    ----------
    model : MDCEVMixedModel
        The fitted model carrying the design specification.

    Returns
    -------
    Callable[[pd.DataFrame, Any], DesignData]
    """
    nc = model.nc
    nvarm = model.nvarm
    nvargam = model.nvargam
    utility_spec = model.utility_spec
    gamma_spec = model.gamma_spec
    alternatives = list(model.alternatives)
    price_cols = list(model.price_cols)

    def build(df: pd.DataFrame, spec: Any = None) -> DesignData:
        df = df.copy()
        _ensure_special_cols(df)

        def col(name: str) -> NDArray:
            return df[name].to_numpy(dtype=np.float64)

        n_obs = len(df)
        X = np.zeros((n_obs, nc, nvarm), dtype=np.float64)
        for k in range(nc):
            for v in range(nvarm):
                X[:, k, v] = col(utility_spec[k, v])
        gd = np.zeros((n_obs, nc, nvargam), dtype=np.float64)
        for k in range(nc):
            for j in range(nvargam):
                gd[:, k, j] = col(gamma_spec[k, j])
        consumption = np.column_stack([col(a) for a in alternatives])
        price = np.column_stack([col(p) for p in price_cols])
        obs = SimpleNamespace(
            consumption=consumption, price=price, gamma_design=gd
        )
        return DesignData(X=X, obs=obs)

    return build


def build_mdcev_mixed_components(
    model: Any,
    *,
    n_draws: int = 1000,
    seed: int = 1234,
    alternative_names: Optional[list[str]] = None,
    draws: Optional[DrawSource] = None,
) -> MixedPredictComponents:
    """Assemble a :class:`~pybhatlib.mixed._predict.MixedPredictComponents` bundle.

    Reads the fitted estimation-space ``theta`` and the fitted MSL estimator
    (panel / draws / pipeline / space / kernel / layout / config) cached on the
    model by :meth:`MDCEVMixedModel._fit`, and wires the MDCEV participation-share
    hook plus a scenario design-rebuild closure.

    Parameters
    ----------
    model : MDCEVMixedModel
        A fitted model (``_fit`` must have run) *or* a model whose
        ``_fitted_theta`` / ``_fitted_est`` attributes have been set directly.
    n_draws : int, default 1000
        Monte-Carlo draws per observation for the participation-share hook.
    seed : int, default 1234
        Monte-Carlo seed (shared across MSL replications).
    alternative_names : list of str, optional
        Output labels; defaults to the model's ``alternatives``.
    draws : DrawSource, optional
        Override the fit-time draw source (e.g. more replications).

    Returns
    -------
    MixedPredictComponents

    Raises
    ------
    RuntimeError
        If the model has not been fit and no fitted state has been supplied.
    """
    if getattr(model, "_fitted_est", None) is None or getattr(
        model, "_fitted_theta", None
    ) is None:
        raise RuntimeError(
            "MDCEVMixedModel must be fit before prediction/ATE; call .fit() "
            "first (or set _fitted_theta / _fitted_est / _fitted_spec / "
            "_fitted_layout for a fixed-theta evaluation)."
        )
    est = model._fitted_est
    hook = make_mdcev_mixed_share_predict(
        n_draws=n_draws,
        seed=seed,
        outside_good_gamma=model.control.outside_good_gamma,
    )
    names = alternative_names or list(model.alternatives)
    return MixedPredictComponents(
        theta=np.asarray(model._fitted_theta, dtype=np.float64),
        panel=est.panel,
        draws=draws if draws is not None else est.draws,
        pipeline=est.pipeline,
        space=est.space,
        kernel=est.kernel,
        layout=est.layout,
        config=est.config,
        build_design=make_mdcev_mixed_design_builder(model),
        design=est.design,
        kernel_predict=hook,
        alternative_names=names,
    )


# ---------------------------------------------------------------------------
# public prediction / forecast entry points
# ---------------------------------------------------------------------------

def mdcev_mixed_predict(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    n_draws: int = 1000,
    seed: int = 1234,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Draw-integrated, sample-averaged predicted participation shares.

    Lifts the shipped fixed-coefficient MDCEV participation prediction over the
    mixing draws via the shared
    :func:`~pybhatlib.mixed._predict.mixed_predict_shares` machinery: random
    coefficients are drawn (held fixed per individual, exactly as at fit time),
    the shipped ``mdcev_predict`` formulation is evaluated per draw, and the
    result is averaged over draws and then over the sample.

    Parameters
    ----------
    model : MDCEVMixedModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset the prediction is built from; defaults to the model's own data.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides.
    n_draws : int, default 1000
        Monte-Carlo error draws per observation.
    seed : int, default 1234
        Monte-Carlo seed.
    draws : DrawSource, optional
        Override the fit-time mixing-draw source.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (nc,)
        Sample-averaged, draw-integrated participation share per alternative.
    """
    components = build_mdcev_mixed_components(model, n_draws=n_draws, seed=seed)
    data_use = model.data if data is None else data
    return mixed_predict_shares(
        components,
        data_use,
        None,
        scenario=scenario,
        draws=draws,
        xp=xp,
    )


def mdcev_mixed_predict_choice(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    n_draws: int = 1000,
    seed: int = 1234,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Most-likely consumed alternative per observation (draw-integrated).

    The mixed-MDCEV analogue of
    :func:`pybhatlib.models.mdcev.mdcev_predict_choice`: it forms the
    *per-observation* draw-integrated participation shares ``(n_obs, nc)`` -- the
    shipped MDCEV participation formulation lifted over the mixing draws via the
    shared :func:`pybhatlib.mixed._predict._simulate_per_obs` machinery -- and
    returns the index of the highest-share alternative for each observation.
    Unlike :func:`mdcev_mixed_predict` (which averages over the sample to a
    ``(nc,)`` share vector), this keeps the per-observation resolution needed for
    a discrete choice.

    Parameters
    ----------
    model : MDCEVMixedModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset the prediction is built from; defaults to the model's own data.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides.
    n_draws : int, default 1000
        Monte-Carlo error draws per observation.
    seed : int, default 1234
        Monte-Carlo seed.
    draws : DrawSource, optional
        Override the fit-time mixing-draw source.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs,)
        Zero-based index of the most likely consumed alternative for each
        observation.
    """
    components = build_mdcev_mixed_components(model, n_draws=n_draws, seed=seed)
    data_use = model.data if data is None else data
    rc_ctx = _resolve(components)
    draw_src = draws if draws is not None else rc_ctx.draws
    design = _design_for(rc_ctx, data_use, None, scenario)
    per_obs = _simulate_per_obs(rc_ctx, design, draw_src)
    choices = np.argmax(np.asarray(per_obs, dtype=np.float64), axis=1)
    if xp is not None:
        choices = xp.asarray(choices)
    return choices


def mdcev_mixed_forecast(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    budget: "NDArray | None" = None,
    budget_col: str = "tot",
    n_replications: int = 200,
    seed: int = 1234,
    num_outside: int = 1,
    draws: Optional[DrawSource] = None,
    xp=None,
) -> NDArray:
    """Draw-integrated expenditure-allocation forecast (per observation).

    Lifts the shipped fixed-coefficient
    :func:`~pybhatlib.models.mdcev._mdcev_forecast.mdcev_forecast` over the mixing
    draws: for each MSL replication the drawn baseline utilities are fed through
    the shipped allocation simulator (identity one-column design), the stacked
    replication allocations are averaged, and the per-observation mean allocation
    is then averaged over the mixing draws.

    Parameters
    ----------
    model : MDCEVMixedModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset the forecast is built from; defaults to the model's own data.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides.
    budget : NDArray, optional
        Per-observation budget ``(n_obs,)``; when ``None`` it is read from
        ``data[budget_col]`` if present, else set to ones.
    budget_col : str, default "tot"
        Budget column name used when ``budget`` is not given.
    n_replications : int, default 200
        Allocation replications per observation inside ``mdcev_forecast``.
    seed : int, default 1234
        Monte-Carlo seed (shared across MSL replications).
    num_outside : int, default 1
        Number of outside goods.
    draws : DrawSource, optional
        Override the fit-time mixing-draw source.
    xp : backend, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs, nc)
        Draw-integrated mean expenditure allocation per observation and good.
    """
    components = build_mdcev_mixed_components(model, seed=seed)
    est = model._fitted_est
    draw_src = draws if draws is not None else est.draws

    data_use = model.data if data is None else data
    if scenario is not None:
        data_use = apply_scenario_overrides(data_use, scenario)
    design = components.build_design(data_use, None)

    obs = design.obs
    n_obs = design.X.shape[0]
    if budget is not None:
        budget_arr = np.asarray(budget, dtype=np.float64).reshape(-1)
    elif budget_col in data_use.columns:
        budget_arr = data_use[budget_col].to_numpy(dtype=np.float64)
    else:
        budget_arr = np.ones(n_obs, dtype=np.float64)

    panel = est.panel
    pipeline = est.pipeline
    n_rep = est.config.n_rep
    rc = est.space.unpack(components.theta, pipeline.spec, want_grad=False)
    kstate = est.kernel.prepare(components.theta, est.layout)
    ass = draw_src.draws(panel.n_ind, pipeline.n_rnd, n_rep)

    X = np.asarray(design.X, dtype=np.float64)
    acc: Optional[NDArray] = None
    for r in range(n_rep):
        errbeta1 = panel.broadcast(ass[r])
        cache = pipeline.realize(errbeta1, rc, want_grad=False)
        Vsub = np.einsum("qcv,qv->qc", X, cache.xmunew)
        alloc = _forecast_per_draw(
            Vsub, obs, kstate, budget_arr,
            n_replications=n_replications, seed=seed,
            num_outside=num_outside,
            outside_good_gamma=model.control.outside_good_gamma,
        )
        acc = alloc if acc is None else acc + alloc

    per_obs = acc / float(n_rep)
    if xp is not None:
        per_obs = xp.asarray(per_obs)
    return per_obs


__all__ = [
    "make_mdcev_mixed_share_predict",
    "make_mdcev_mixed_design_builder",
    "build_mdcev_mixed_components",
    "mdcev_mixed_predict",
    "mdcev_mixed_predict_choice",
    "mdcev_mixed_forecast",
]
