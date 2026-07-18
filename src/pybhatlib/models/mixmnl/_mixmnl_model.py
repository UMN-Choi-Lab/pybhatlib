"""MixMNL model estimation on the shared mixed/panel MSL engine.

The Mixed Multinomial Logit facade wires the harmonized model interface
(:class:`~pybhatlib.models._base.BaseModel`) to the softmax kernel + shared MSL
engine (:mod:`pybhatlib.mixed`).  ``_fit`` assembles the panel index, mixing
spec, parameter layout, random-coefficient pipeline, softmax kernel and
:class:`~pybhatlib.mixed._engine.MixedMSLEstimator`; packs the estimation-space
parameter vector ``theta = [beta | rcor | scal | lam]`` (the softmax kernel owns
no ``kern`` block); runs ``scipy.optimize.minimize(jac=True)`` against the
engine objective; and performs a reporting-space BHHH standard-error pass
(``cov = inv(S'S)`` from the per-individual score, delta-transformed into the
natural reporting parameters).

The random-coefficient specification collapses to the shipped fixed-coefficient
MNL when ``normvar = logvar = yjvar = {}`` (the plan T0.14 collapse gate): the
engine reproduces :func:`pybhatlib.models.mnl.mnl_loglik` value-for-value, so
``fit`` recovers the same optimum as :class:`~pybhatlib.models.mnl.MNLModel`.

``predict`` / ``ate`` are wired to the shared mixed prediction machinery
(:mod:`pybhatlib.mixed._predict`): each lifts the shipped fixed-coefficient MNL
forecast / ATE formulation over the mixing draws and collapses to it when the
random-coefficient spec is empty (``nrndcoef == 0``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import scipy.optimize as sopt
from numpy.typing import NDArray
from scipy.special import ndtr

from pybhatlib.io._data_loader import load_data, used_columns_selector
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.mixed._draws import DrawSource, FixtureDrawSource, ScipyHaltonDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._predict import MixedPredictComponents
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import (
    EstimationSpace,
    ParamLayout,
    ParamSpace,
    ReportingSpace,
)
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.mixmnl._mixmnl_control import MixMNLControl
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.models.mixmnl._mixmnl_results import MixMNLResults
from pybhatlib.vecup._panel import PanelIndex
from pybhatlib.vecup._vec_ops import vecndup


class _SoftmaxObs:
    """Per-observation availability / chosen bundle consumed by SoftmaxKernel.

    Mirrors the GAUSS ``davunord`` / ``dvunord`` design columns as dense
    ``(n_obs, nc)`` arrays.
    """

    def __init__(self, avail: NDArray, chosen: NDArray) -> None:
        self.avail = avail
        self.chosen = chosen


class MixMNLModel(BaseModel):
    """Mixed Multinomial Logit (MixMNL) model.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a data file or an in-memory DataFrame.
    alternatives : list of str
        Column names holding the 0/1 choice indicators.
    availability : str or list of str, default "none"
        ``"none"`` if every alternative is always available, otherwise the list
        of availability column names aligned with ``alternatives``.
    spec : dict or None
        Variable specification mapping variable names to alternative-specific
        column names / keywords (``"sero"`` / ``"uno"``), as in
        :class:`~pybhatlib.models.mnl.MNLModel`.
    var_names : list of str or None
        Coefficient names.  Inferred from ``spec`` keys when ``None``.
    control : MixMNLControl or None
        Estimation control structure (random-coefficient spec, MSL knobs,
        optimizer settings).

    Notes
    -----
    The random-coefficient specification lives on ``control``
    (``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``); an empty
    spec yields a fixed-coefficient MNL through the shared engine.
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        alternatives: list[str],
        availability: str | list[str] = "none",
        spec: dict | None = None,
        var_names: list[str] | None = None,
        control: MixMNLControl | None = None,
    ) -> None:
        self.control = control or MixMNLControl()

        if isinstance(availability, str) and availability.lower() == "none":
            _prune_avail = None
        else:
            _prune_avail = (
                availability if isinstance(availability, list) else [availability]
            )

        if isinstance(data, str):
            self.data_path = data
            # Load only the columns this model references (a wide panel can carry
            # 1000+ unused columns); over-inclusive by design, the ``uno``/``sero``
            # keywords are dropped and the parity tests guard the collector.
            usecols = used_columns_selector(
                value_cols=alternatives,
                avail_cols=_prune_avail,
                id_cols=[self.control.person_id],
                specs=[spec] if spec is not None else [],
            )
            self.data = load_data(data, usecols=usecols)
        else:
            self.data_path = "<DataFrame>"
            self.data = data

        self.alternatives = alternatives
        self.n_alts = len(alternatives)

        # --- availability --------------------------------------------------
        if isinstance(availability, str) and availability.lower() == "none":
            self.avail_cols = None
            self.avail = np.ones((len(self.data), self.n_alts), dtype=np.float64)
        else:
            avail_cols = (
                availability if isinstance(availability, list) else [availability]
            )
            self.avail_cols = avail_cols
            self.avail = self.data[avail_cols].to_numpy(dtype=np.float64)

        # --- design tensor -------------------------------------------------
        if spec is None:
            raise ValueError("spec is required")
        self.spec_dict = spec
        self.X, self._var_names = parse_spec(
            spec, self.data, self.alternatives, nseg=1
        )
        if var_names is not None:
            self._var_names = var_names
        self.var_names = list(self._var_names)
        self.n_beta = len(self.var_names)

        # --- chosen one-hot ------------------------------------------------
        choice_data = self.data[self.alternatives].to_numpy(dtype=np.float64)
        self.chosen = choice_data
        self.y = np.argmax(choice_data, axis=1).astype(np.int64)
        self.N = len(self.y)

        # --- panel person ids ---------------------------------------------
        if self.control.person_id is not None:
            self.person_ids = self.data[self.control.person_id].to_numpy()
        else:
            # cross-sectional: one observation per person (Dmask == I).
            self.person_ids = np.arange(self.N)

    # ------------------------------------------------------------------
    # engine assembly
    # ------------------------------------------------------------------

    def _build_spec_layout(self) -> tuple[MixingSpec, ParamLayout]:
        ctrl = self.control
        spec = MixingSpec.from_var_names(
            var_names=self.var_names,
            normvar=tuple(ctrl.normvar),
            logvar=tuple(ctrl.logvar),
            yjvar=tuple(ctrl.yjvar),
            varneg=tuple(ctrl.varneg),
            varpos=tuple(ctrl.varpos),
            randdiag=ctrl.randdiag,
        )
        layout = ParamLayout(
            n_beta=spec.n_beta,
            n_rcor=spec.nrndtcor,
            n_scal=spec.nscale,
            n_lam=spec.numlam,
            n_kern=0,                      # softmax kernel owns no parameters
        )
        return spec, layout

    def _make_draw_source(self, n_rnd: int, n_rep: int, n_ind: int):
        """Runtime draw source (zeros when there are no random coefficients)."""
        if n_rnd == 0:
            # scipy.qmc.Halton(d=0) is undefined; a zero-width block suffices
            # since the pipeline broadcasts a (n_obs, 0) draw matrix.
            return FixtureDrawSource(np.zeros((n_rep, n_ind * n_rnd)))
        return ScipyHaltonDrawSource(seed=self.control.draw_seed)

    def _build_estimator(
        self, spec: MixingSpec, layout: ParamLayout, panel: PanelIndex
    ) -> MixedMSLEstimator:
        ctrl = self.control
        design = DesignData(
            X=self.X, obs=_SoftmaxObs(self.avail, self.chosen)
        )
        space = EstimationSpace(
            layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
        )
        pipeline = RandomCoefPipeline(
            spec, layout, spher=ctrl.spher, scal=ctrl.scal, intordn1=ctrl.intordn1
        )
        kernel = SoftmaxKernel(self.n_alts)
        cfg = MSLConfig(
            n_rep=ctrl.n_rep,
            spher=ctrl.spher,
            scal=ctrl.scal,
            intordn1=ctrl.intordn1,
            floor_pcomp=ctrl.floor_pcomp,
            floor_z=ctrl.floor_z,
            score_convention="mask",
        )
        draws = self._make_draw_source(spec.nrndcoef, ctrl.n_rep, panel.n_ind)
        weightind = np.ones(panel.n_ind, dtype=np.float64)
        return MixedMSLEstimator(
            panel=panel,
            draws=draws,
            pipeline=pipeline,
            kernel=kernel,
            layout=layout,
            space=space,
            design=design,
            weightind=weightind,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # reporting-space transform (estimation space -> natural parameters)
    # ------------------------------------------------------------------

    def _to_reporting(
        self,
        theta: NDArray,
        spec: MixingSpec,
        layout: ParamLayout,
        space: ParamSpace,
    ) -> NDArray:
        """Map an estimation-space ``theta`` to natural reporting parameters.

        ``beta`` becomes the sign-reparameterized coefficients, ``rcor`` the
        off-diagonal correlation entries (row-based upper-triangular), ``scal``
        the scale (std-dev) vector ``exp(xscalrand)``, and ``lam`` the
        Yeo-Johnson powers ``2 cdlogit(xlam)`` in ``(0, 2)``.  Kernel-only
        parameters pass through unchanged.
        """
        rc = space.unpack(theta, spec, want_grad=False)
        sl = layout.slices()

        beta_r = np.asarray(rc.xmu, dtype=np.float64).reshape(-1)
        if layout.n_rcor > 0:
            rcor_r = vecndup(np.asarray(rc.omegastar, dtype=np.float64))
        else:
            rcor_r = np.zeros(0, dtype=np.float64)
        scal_r = np.asarray(rc.wscalrand, dtype=np.float64).reshape(-1)
        lam_r = np.asarray(rc.xlamrnd, dtype=np.float64).reshape(-1)
        kern_r = np.asarray(theta[sl["kern"]], dtype=np.float64).reshape(-1)
        return np.concatenate([beta_r, rcor_r, scal_r, lam_r, kern_r])

    def _reporting_jacobian(
        self,
        theta: NDArray,
        spec: MixingSpec,
        layout: ParamLayout,
        space: EstimationSpace,
        *,
        eps: float = 1e-6,
    ) -> NDArray:
        """Central finite-difference Jacobian ``d report / d theta``.

        Small ``n_theta`` makes the finite-difference cost negligible; the
        identity block for the ``beta`` slice (no sign reparameterization)
        recovers exactly the fixed-coefficient BHHH covariance.
        """
        theta = np.asarray(theta, dtype=np.float64)
        n = theta.shape[0]
        base = self._to_reporting(theta, spec, layout, space)
        jac = np.zeros((base.shape[0], n), dtype=np.float64)
        for j in range(n):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += eps
            tm[j] -= eps
            rp = self._to_reporting(tp, spec, layout, space)
            rm = self._to_reporting(tm, spec, layout, space)
            jac[:, j] = (rp - rm) / (2.0 * eps)
        return jac

    def _param_names(self, spec: MixingSpec) -> list[str]:
        """Reporting-parameter names in ``[beta | rcor | scal | lam]`` order."""
        names = list(self.var_names)
        rc_names = [self.var_names[int(p)] for p in spec.mixpos]
        # correlation names: row-based upper-triangular pairs
        for i in range(spec.nrndcoef):
            for j in range(i + 1, spec.nrndcoef):
                names.append(f"corr[{rc_names[i]},{rc_names[j]}]")
        names.extend(f"sd[{nm}]" for nm in rc_names)
        names.extend(f"lam[{nm}]" for nm in rc_names)
        return names

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(self) -> MixMNLResults:
        """Estimate the MixMNL model and return :class:`MixMNLResults`."""
        ctrl = self.control
        t_start = time.time()

        spec, layout = self._build_spec_layout()
        panel = PanelIndex.from_ids(self.person_ids)
        est = self._build_estimator(spec, layout, panel)

        n_theta = layout.n_theta
        if ctrl.startb is not None:
            theta0 = np.asarray(ctrl.startb, dtype=np.float64)
            if theta0.shape[0] != n_theta:
                raise ValueError(
                    f"startb has length {theta0.shape[0]}, expected {n_theta}"
                )
        else:
            theta0 = np.zeros(n_theta, dtype=np.float64)

        if ctrl.verbose >= 1:
            print(
                f"  MixMNL estimation: {panel.n_obs} obs, {panel.n_ind} "
                f"individuals, {n_theta} parameters, {self.n_alts} alternatives, "
                f"{spec.nrndcoef} random coef(s), nrep={ctrl.n_rep}"
            )

        # ---- optimize (jac=True: objective returns (neg_ll, neg_grad)) ----
        method_map = {"bfgs": "BFGS", "lbfgsb": "L-BFGS-B", "newton": "BFGS"}
        method = method_map.get(ctrl.optimizer, "BFGS")
        options: dict = {"maxiter": ctrl.maxiter, "gtol": ctrl.tol,
                         "disp": ctrl.verbose >= 2}
        if method == "L-BFGS-B":
            options["ftol"] = ctrl.tol

        res = sopt.minimize(
            est.objective, theta0, jac=True, method=method, options=options,
        )
        theta_hat = np.asarray(res.x, dtype=np.float64)
        # Retain the estimation-space optimum for post-estimation predict / ate
        # (the shared engine consumes estimation space; results store reporting
        # space).
        self._theta_hat = theta_hat

        # ---- final LL and per-individual score ----------------------------
        ll_pi, score = est.simulated_loglik(theta_hat, want_grad=True)
        ll_total = float(ll_pi.sum())
        ll_mean = ll_total / panel.n_ind
        g_sum = np.asarray(score, dtype=np.float64).sum(0)
        grad_norm = float(np.linalg.norm(g_sum))
        converged = bool(res.success) or grad_norm < ctrl.tol_check

        # ---- reporting-space BHHH covariance ------------------------------
        params_report = self._to_reporting(theta_hat, spec, layout, est.space)
        if ctrl.want_covariance:
            if ctrl.se_method.lower() != "bhhh":
                raise ValueError(
                    "MixMNL currently supports only se_method='bhhh'"
                )
            S = np.asarray(score, dtype=np.float64)          # (n_ind, n_theta)
            B = S.T @ S
            try:
                cov_est = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                cov_est = np.linalg.pinv(B)
            J = self._reporting_jacobian(theta_hat, spec, layout, est.space)
            cov = J @ cov_est @ J.T
            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            with np.errstate(invalid="ignore"):
                corr = cov / np.outer(se, se)
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            m = params_report.shape[0]
            cov = np.zeros((m, m))
            se = np.zeros(m)
            corr = np.zeros((m, m))

        with np.errstate(invalid="ignore", divide="ignore"):
            t_stat = np.where(se > 0, params_report / se, 0.0)
        p_value = 2.0 * (1.0 - ndtr(np.abs(t_stat)))

        t_elapsed = (time.time() - t_start) / 60.0
        n_iter = int(getattr(res, "nit", 0))

        if ctrl.verbose >= 1:
            msg = "converged" if converged else "NOT converged"
            print(
                f"  Optimisation {msg} in {n_iter} iterations "
                f"({t_elapsed:.4f} min).  LL = {ll_total:.4f}"
            )

        return MixMNLResults(
            params=params_report,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_sum,
            loglik=ll_mean,
            n_obs=panel.n_obs,
            n_ind=panel.n_ind,
            param_names=self._param_names(spec),
            corr_matrix=corr,
            cov_matrix=cov,
            n_iter=n_iter,
            convergence_time=t_elapsed,
            converged=converged,
            return_code=0 if converged else 1,
            control=ctrl,
            data_path=self.data_path,
            message=getattr(res, "message", None),
        )

    # ------------------------------------------------------------------
    # post-estimation context assembly (shared with predict / ate)
    # ------------------------------------------------------------------

    def _make_design_builder(self):
        """Return a ``(frame, spec) -> DesignData`` closure over this model.

        Rebuilds the ``(n_obs, nc, n_beta)`` design tensor and the softmax
        ``obs`` bundle (availability + chosen one-hot) from a possibly
        scenario-overridden frame, so the shared predictor can re-parse the
        design for each counterfactual.  The closure captures only immutable
        model attributes (no module-level state); the chosen one-hot falls back
        to zeros when a supplied frame omits the alternative columns (prediction
        does not use it).

        Returns
        -------
        Callable[[pd.DataFrame, Any], DesignData]
        """
        alternatives = self.alternatives
        avail_cols = self.avail_cols
        n_alts = self.n_alts
        default_spec = self.spec_dict

        def build_design(frame: pd.DataFrame, spec: "dict | None") -> DesignData:
            spec_use = spec if spec is not None else default_spec
            X, _ = parse_spec(spec_use, frame, alternatives, nseg=1)
            X = np.asarray(X, dtype=np.float64)
            if avail_cols is None:
                avail = np.ones((len(frame), n_alts), dtype=np.float64)
            else:
                avail = frame[avail_cols].to_numpy(dtype=np.float64)
            if all(a in frame.columns for a in alternatives):
                chosen = frame[alternatives].to_numpy(dtype=np.float64)
            else:
                chosen = np.zeros((len(frame), n_alts), dtype=np.float64)
            return DesignData(X=X, obs=_SoftmaxObs(avail, chosen))

        return build_design

    def _predict_components(
        self,
        theta: NDArray,
        *,
        draws: DrawSource | None = None,
        alternative_names: list[str] | None = None,
        reporting: bool = False,
    ) -> MixedPredictComponents:
        """Assemble a :class:`MixedPredictComponents` for post-estimation use.

        Rebuilds the mixing spec / layout, panel index, random-coefficient
        pipeline, reparameterization space, softmax kernel and MSL config exactly
        as :meth:`_build_estimator` does, and pairs them with a ``build_design``
        closure so the shared predictor can rebuild scenario designs.  The draw
        source defaults to the fit-identical strategy (deterministic given
        ``control.draw_seed``), so scenarios share draws.

        Parameters
        ----------
        theta : NDArray, shape (n_theta,)
            Parameter vector consumed by ``space.unpack``.  Its interpretation
            follows ``reporting``: **estimation-space** (``[beta | rcor | scal |
            lam]``, the ``lpr`` parameterization) when ``reporting`` is ``False``
            (the fit path, ``theta = model._theta_hat``); **reporting-space**
            natural parameters (the ``lpr1`` parameterization -- coefficients
            entered directly, ``rcor`` the off-diagonal correlations, ``scal``
            the std-dev vector, ``lam`` the Yeo-Johnson powers) when ``reporting``
            is ``True`` (matching a fitted :class:`MixMNLResults.params`).
        draws : DrawSource, optional
            Override the draw strategy; defaults to :meth:`_make_draw_source`.
        alternative_names : list of str, optional
            Output labels; defaults to the model's ``alternatives``.
        reporting : bool, default False
            Select the reparameterization space: :class:`EstimationSpace`
            (``False``) or :class:`ReportingSpace` (``True``).  ``ReportingSpace``
            consumes the natural reporting parameters directly (no
            reporting->estimation inversion), so a fitted ``results.params`` or an
            externally supplied natural vector drives predict / ATE unchanged.

        Returns
        -------
        MixedPredictComponents
        """
        ctrl = self.control
        spec, layout = self._build_spec_layout()
        panel = PanelIndex.from_ids(self.person_ids)
        space_cls = ReportingSpace if reporting else EstimationSpace
        space = space_cls(
            layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
        )
        pipeline = RandomCoefPipeline(
            spec, layout, spher=ctrl.spher, scal=ctrl.scal, intordn1=ctrl.intordn1
        )
        kernel = SoftmaxKernel(self.n_alts)
        cfg = MSLConfig(
            n_rep=ctrl.n_rep,
            spher=ctrl.spher,
            scal=ctrl.scal,
            intordn1=ctrl.intordn1,
            floor_pcomp=ctrl.floor_pcomp,
            floor_z=ctrl.floor_z,
            score_convention="mask",
        )
        draw_src = (
            draws
            if draws is not None
            else self._make_draw_source(spec.nrndcoef, ctrl.n_rep, panel.n_ind)
        )
        names = (
            list(alternative_names)
            if alternative_names is not None
            else list(self.alternatives)
        )
        return MixedPredictComponents(
            theta=np.asarray(theta, dtype=np.float64),
            panel=panel,
            draws=draw_src,
            pipeline=pipeline,
            space=space,
            kernel=kernel,
            layout=layout,
            config=cfg,
            build_design=self._make_design_builder(),
            alternative_names=names,
        )

    # ------------------------------------------------------------------
    # post-estimation surface (shared mixed prediction machinery)
    # ------------------------------------------------------------------

    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        scenario: dict[str, float | str] | None = None,
        draws: DrawSource | None = None,
        xp=None,
    ) -> NDArray:
        """Per-observation, draw-integrated choice probabilities.

        Mirrors :meth:`~pybhatlib.models.mnl.MNLModel.predict` (output shape
        ``(n_obs, nc)``) with the mixing-draw integration layer; see
        :func:`pybhatlib.models.mixmnl._mixmnl_forecast.mixmnl_predict`.  The
        signature follows the common mixed-family convention
        ``predict(data=None, *, scenario=None, draws=None, xp=None)``.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to predict on (the mixed facade rebuilds the design from a
            frame).  Defaults to the training frame.
        scenario : dict, optional
            A single scenario's ``{column: scalar | source_column}`` covariate
            overrides, applied before the design is rebuilt.  ``None`` (default)
            predicts at the observed covariates.
        draws : DrawSource, optional
            Override the fit-identical draw strategy.
        xp : backend, optional
            Array backend used to wrap the result.

        Returns
        -------
        NDArray, shape (n_obs, nc)
            Draw-integrated per-observation choice probabilities.
        """
        from pybhatlib.models.mixmnl._mixmnl_forecast import mixmnl_predict

        self._require_results()
        components = self._predict_components(self._theta_hat, draws=draws)
        data_use = self.data if data is None else data
        return mixmnl_predict(
            components, data_use, self.spec_dict, scenario=scenario, xp=xp
        )

    def predict_choice(
        self,
        data: pd.DataFrame | None = None,
        *,
        scenario: dict[str, float | str] | None = None,
        draws: DrawSource | None = None,
        xp=None,
    ) -> NDArray:
        """Most-likely predicted alternative per observation.

        Signature follows the common mixed-family convention
        ``(data=None, *, scenario=None, draws=None, xp=None)``.  See
        :func:`pybhatlib.models.mixmnl._mixmnl_forecast.mixmnl_predict_choice`.
        """
        from pybhatlib.models.mixmnl._mixmnl_forecast import mixmnl_predict_choice

        self._require_results()
        components = self._predict_components(self._theta_hat, draws=draws)
        data_use = self.data if data is None else data
        return mixmnl_predict_choice(
            components, data_use, self.spec_dict, scenario=scenario, xp=xp
        )

    def ate(
        self,
        *,
        scenarios=None,
        draws: DrawSource | None = None,
        xp=None,
        **kwargs,
    ):
        """Draw-integrated predicted shares / ATE across scenarios.

        Keyword-only, mirroring the fixed-coefficient facade.  ``data`` / ``spec``
        default to the model's own (override via ``kwargs``); pass ``scenarios=``
        for counterfactuals.  Returns a
        :class:`~pybhatlib.mixed._predict.MixedATEResult` (harmonized
        ``ATEResultMixin`` surface).  See
        :func:`pybhatlib.models.mixmnl._mixmnl_ate.mixmnl_ate`.

        Parameters
        ----------
        scenarios : dict or pd.DataFrame, optional
            Scenario specification (see :func:`mixmnl_ate`).
        draws : DrawSource, optional
            Override the fit-identical draw strategy.
        xp : backend, optional
            Array backend used to wrap the result arrays.
        **kwargs
            Forwarded to :func:`mixmnl_ate` (``data`` / ``spec`` /
            ``alternative_names``).
        """
        from pybhatlib.models.mixmnl._mixmnl_ate import mixmnl_ate

        self._require_results()
        return mixmnl_ate(
            self,
            self._theta_hat,
            scenarios=scenarios,
            draws=draws,
            xp=xp,
            **kwargs,
        )
