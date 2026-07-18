"""Draw-integrated prediction / forecast for the mixed-panel MNP (MNPKerCP).

Wires :class:`~pybhatlib.models.mnpkercp._mnpkercp_model.MNPKerCPModel` to the
shared mixed prediction machinery (:mod:`pybhatlib.mixed._predict`). The
formulation *lifts the shipped fixed-coefficient MNP over the mixing draws*: for
each MSL replication the random-coefficient pipeline realises ``xmunew`` per
observation, the systematic utilities ``Vsub = X @ xmunew`` are formed, and the
MVNCD (OVUS) kernel is queried for the per-alternative choice probabilities; the
result is averaged over draws (integrating out the mixing distribution) and then
over the sample.

Kernel-prediction hook
----------------------
The MVNCD kernel's :meth:`~pybhatlib.models.mnpkercp._mnpkercp_kernel.MvncdKernel.probability`
returns only the *observed*-choice probability. :func:`_mnpkercp_kernel_predict`
lifts it to the full ``(n_obs, nc)`` per-alternative matrix by evaluating, for
each available candidate alternative ``j``, the probability that ``j`` is the
utility maximiser (designating ``j`` as chosen and reusing the exact same kernel
covariance / conditional mean), then row-normalising over the available
alternatives -- exactly mirroring the shipped fixed-coefficient
:func:`pybhatlib.models.mnp._mnp_ate._compute_predicted_shares` inner loop, with
the extra draw-integration supplied by the shared engine.

Validation is by *collapse* (``nrndcoef = 0`` reproduces the shipped
fixed-coefficient MNP prediction) and interface conformance, not GAUSS parity --
no GAUSS forecast oracle exists for the mixed driver.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.mixed._draws import DrawSource
from pybhatlib.mixed._engine import DesignData
from pybhatlib.mixed._kernel import MixedKernel
from pybhatlib.mixed._predict import (
    MixedPredictComponents,
    _design_for,
    _resolve,
    _simulate_per_obs,
    mixed_predict_shares,
)


class _CandidateObs:
    """Per-observation availability / chosen bundle for one candidate alternative.

    Mirrors :class:`~pybhatlib.models.mnpkercp._mnpkercp_model._MnpObs`; used only
    inside :func:`_mnpkercp_kernel_predict` to designate a candidate alternative
    as the (counterfactually) chosen one when querying the kernel.
    """

    __slots__ = ("avail", "chosen")

    def __init__(self, avail: NDArray, chosen: NDArray) -> None:
        self.avail = avail
        self.chosen = chosen


def _mnpkercp_kernel_predict(
    kernel: MixedKernel,
    Vsub: NDArray,
    obs: Any,
    kstate: Any,
    rc_draw: NDArray,
    *,
    xp: Any = None,
) -> NDArray:
    """Per-observation, per-alternative MVNCD choice probabilities for one draw.

    Lifts the MVNCD kernel's observed-choice probability to the full
    ``(n_obs, nc)`` per-alternative matrix by asking, for every *available*
    candidate alternative ``j``, for the probability that ``j`` is the utility
    maximiser (the MVNCD of the utilities differenced against ``j``), then
    row-normalising over the available alternatives.

    Parameters
    ----------
    kernel : MixedKernel
        The fitted MVNCD (OVUS) kernel
        (:class:`~pybhatlib.models.mnpkercp._mnpkercp_kernel.MvncdKernel`).
    Vsub : NDArray, shape (n_obs, nc)
        Systematic utilities for this replication (``X @ xmunew``).
    obs : Any
        Kernel ``obs`` bundle; only ``obs.avail`` (``(n_obs, nc)``) is read.
    kstate : Any
        Kernel state from :meth:`MvncdKernel.prepare` (draw-independent).
    rc_draw : NDArray, shape (n_obs, nrndcoef)
        Correlated random-coefficient draw for this replication (``errbeta3``);
        enters the copula conditional mean and is ignored when ``copula=False``.
    xp : module, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs, nc)
        Choice probabilities; each row is non-negative and sums to one over the
        observation's available alternatives (unavailable alternatives are 0).
    """
    Vsub = np.asarray(Vsub, dtype=np.float64)
    avail = np.asarray(obs.avail, dtype=np.float64)
    n_obs, nc = Vsub.shape

    rc = np.asarray(rc_draw, dtype=np.float64)
    if rc.ndim == 1:
        rc = rc.reshape(n_obs, -1)

    probs = np.zeros((n_obs, nc), dtype=np.float64)
    for j in range(nc):
        mask = avail[:, j] > 0.5
        if not np.any(mask):
            continue
        idx = np.nonzero(mask)[0]
        chosen_j = np.zeros((idx.size, nc), dtype=np.float64)
        chosen_j[:, j] = 1.0
        obs_j = _CandidateObs(avail[idx], chosen_j)
        res = kernel.probability(
            Vsub[idx], obs_j, kstate, rc_draw=rc[idx], want_grad=False,
        )
        probs[idx, j] = np.asarray(res.p_obs, dtype=np.float64)

    # Row-normalise over the available alternatives (matches the shipped
    # fixed-coefficient MNP ``_compute_predicted_shares`` normalisation).
    row = probs.sum(axis=1, keepdims=True)
    row = np.where(row <= 0.0, 1.0, row)
    out = probs / row
    if xp is not None:
        out = xp.asarray(out)
    return out


def build_predict_components(
    model: Any,
    *,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
) -> MixedPredictComponents:
    """Assemble a :class:`MixedPredictComponents` bundle from a fitted model.

    Reads the estimation-space parameter vector and the MSL engine components
    (panel, mixing draws, random-coefficient pipeline, reparameterization space,
    MVNCD kernel, parameter layout, and engine config) persisted on ``model`` by
    :meth:`MNPKerCPModel._fit`, and closes over :func:`parse_spec` so a scenario
    design can be rebuilt.

    Parameters
    ----------
    model : MNPKerCPModel
        A fitted model (``model.fit()`` already called).
    draws : DrawSource, optional
        Override the fit-identical draw strategy (e.g. more replications for a
        smoother forecast). Defaults to the source used at fit time.
    alternative_names : list of str, optional
        Output labels; defaults to ``model.alternatives``.

    Returns
    -------
    MixedPredictComponents
        Prediction context for :func:`mixed_predict_shares` /
        :func:`pybhatlib.mixed._predict.mixed_ate`.

    Raises
    ------
    RuntimeError
        If the model has not been fit.
    """
    est = _require_fit_context(model)
    names = alternative_names if alternative_names is not None else list(model.alternatives)

    def _build_design(data_frame: "pd.DataFrame", spec: Any) -> DesignData:
        """Rebuild the ``(n_obs, nc, n_var)`` design + obs bundle for a frame."""
        X, _ = parse_spec(spec, data_frame, model.alternatives, nseg=1)
        n = len(data_frame)
        if model.avail_cols is not None:
            avail = data_frame[model.avail_cols].to_numpy(dtype=np.float64)
        else:
            avail = np.ones((n, model.n_alts), dtype=np.float64)
        if all(c in data_frame.columns for c in model.alternatives):
            chosen = data_frame[model.alternatives].to_numpy(dtype=np.float64)
        else:
            chosen = np.zeros((n, model.n_alts), dtype=np.float64)
        return DesignData(X=X, obs=_CandidateObs(avail, chosen))

    return MixedPredictComponents(
        theta=np.asarray(model._theta_hat, dtype=np.float64),
        panel=est.panel,
        draws=draws if draws is not None else est.draws,
        pipeline=est.pipeline,
        space=est.space,
        kernel=est.kernel,
        layout=est.layout,
        config=est.config,
        build_design=_build_design,
        kernel_predict=_mnpkercp_kernel_predict,
        alternative_names=names,
    )


def mnpkercp_predict(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    draws: Optional[DrawSource] = None,
    alternative_names: Optional[list[str]] = None,
    xp: Any = None,
) -> NDArray:
    """Draw-integrated, sample-averaged predicted shares for a fitted MNPKerCP.

    Integrates the MVNCD per-alternative choice probabilities over the mixing
    distribution (drawing random coefficients held fixed per individual, exactly
    as at fit time) and averages over the sample.

    Parameters
    ----------
    model : MNPKerCPModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset to predict on; defaults to the model's fitted data.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides,
        applied before the design is rebuilt. ``None`` predicts at the observed
        covariates.
    draws : DrawSource, optional
        Override the fit-identical draw strategy.
    alternative_names : list of str, optional
        Output labels (unused for the bare share vector; kept for symmetry).
    xp : module, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (nc,)
        Sample-averaged, draw-integrated market shares (non-negative, summing to
        one over the available alternatives).
    """
    comp = build_predict_components(
        model, draws=draws, alternative_names=alternative_names
    )
    frame = data if data is not None else model.data
    return mixed_predict_shares(
        comp, frame, spec=model.spec_dict, scenario=scenario, draws=draws, xp=xp,
    )


def mnpkercp_predict_choice(
    model: Any,
    data: "pd.DataFrame | None" = None,
    *,
    scenario: Optional[dict[str, float | str]] = None,
    draws: Optional[DrawSource] = None,
    xp: Any = None,
) -> NDArray:
    """Per-observation most-likely predicted alternative for a fitted MNPKerCP.

    Discrete-family choice helper mirroring
    :func:`pybhatlib.models.mnp._mnp_forecast.mnp_predict_choice`: it forms the
    draw-integrated per-observation choice-probability matrix (integrating the
    MVNCD per-alternative probabilities over the mixing distribution, exactly as
    at fit time) and returns the argmax alternative index for each observation.

    Parameters
    ----------
    model : MNPKerCPModel
        A fitted model.
    data : pd.DataFrame, optional
        Dataset to predict on; defaults to the model's fitted data.
    scenario : dict, optional
        A single scenario's ``{column: scalar | source_column}`` overrides,
        applied before the design is rebuilt. ``None`` predicts at the observed
        covariates.
    draws : DrawSource, optional
        Override the fit-identical draw strategy.
    xp : module, optional
        Array backend used to wrap the result. Defaults to NumPy.

    Returns
    -------
    NDArray, shape (n_obs,)
        Predicted choice index (0-based) for each observation.
    """
    comp = build_predict_components(model, draws=draws)
    frame = data if data is not None else model.data
    rc_ctx = _resolve(comp)
    draw_src = draws if draws is not None else rc_ctx.draws
    design = _design_for(rc_ctx, frame, model.spec_dict, scenario)
    per_obs = _simulate_per_obs(rc_ctx, design, draw_src)          # (n_obs, nc)
    choices = np.argmax(np.asarray(per_obs, dtype=np.float64), axis=1)
    if xp is not None:
        choices = xp.asarray(choices)
    return choices


# ``forecast`` is the market-share forecast; alias of the share predictor so the
# family exposes the harmonized ``.forecast`` surface without a second code path.
mnpkercp_forecast = mnpkercp_predict


def _require_fit_context(model: Any):
    """Return the fitted MSL estimator persisted on ``model`` or raise.

    Parameters
    ----------
    model : MNPKerCPModel
        The model whose fit context is required.

    Returns
    -------
    MixedMSLEstimator
        The estimator built at fit time (carries panel / draws / pipeline /
        space / kernel / layout / config).

    Raises
    ------
    RuntimeError
        If the model has not been fit (``_fit`` has not run).
    """
    est = getattr(model, "_est", None)
    if est is None or getattr(model, "_theta_hat", None) is None:
        raise RuntimeError(
            f"{type(model).__name__} has not been fit yet; call .fit() first."
        )
    return est


__all__ = [
    "build_predict_components",
    "mnpkercp_predict",
    "mnpkercp_predict_choice",
    "mnpkercp_forecast",
]
