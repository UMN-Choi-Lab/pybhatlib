"""MORPFlex (mixed-panel MORP) model estimation on the shared MSL engine.

The mixed-panel Multivariate Ordered Response Probit facade wires the
harmonized model interface (:class:`~pybhatlib.models._base.BaseModel`) to the
rectangle-MVNCD (``pdfrectn``) kernel with an optional rc<->kernel copula
(:class:`~pybhatlib.models.morp_flex._morp_flex_kernel.RectMvncdKernel`) plus the
shared MSL engine (:mod:`pybhatlib.mixed`), ported from the GAUSS *Joint Ordered
YJ with Cross-Sectional or Panel Random Coefficients* driver.

``_fit`` assembles the panel index, the MORP *joint* mixing spec (the correlation
is over ``nrndtot = nrndcoef + nord`` random elements, ``nord`` ordinal
kernel-error dimensions), the parameter layout in GAUSS ``b`` order
``[thresh | beta | rcor | scal | lam | kernlam]`` (the threshold block leads and
the Yeo-Johnson kernel-lam block trails; MORP fixes ``wker`` to ones so there is
no kernel-scale block), the random-coefficient pipeline, the rectangle-MVNCD
kernel and :class:`~pybhatlib.mixed._engine.MixedMSLEstimator`; runs
``scipy.optimize.minimize(jac=True)``; and performs a reporting-space BHHH
standard-error pass.

``predict`` / ``ate`` are wired to the shared mixed predictor
(:mod:`pybhatlib.mixed._predict`) via
:mod:`pybhatlib.models.morp_flex._morp_flex_forecast` and
:mod:`pybhatlib.models.morp_flex._morp_flex_ate`: the shipped fixed-coefficient
MORP marginal / ATE formulation lifted over the mixing draws, returning the same
*per-outcome* list shape as the fixed MORP (one array per ordinal dimension).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.optimize as sopt
from numpy.typing import NDArray
from scipy.special import ndtr

from pybhatlib.io._data_loader import load_data, used_columns_selector
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.matgradient._radial import newcholparmscaled
from pybhatlib.mixed._draws import FixtureDrawSource, ScipyHaltonDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import (
    EstimationSpace,
    ParamLayout,
    ParamSpace,
    ReportingSpace,
    thresh_reparam,
)
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.morp_flex._morp_flex_control import MORPFlexControl
from pybhatlib.models.morp_flex._morp_flex_kernel import RectMvncdKernel
from pybhatlib.models.morp_flex._morp_flex_results import MORPFlexResults
from pybhatlib.utils._logistic import cdlogit
from pybhatlib.vecup._panel import PanelIndex
from pybhatlib.vecup._vec_ops import vecndup


class _MorpObs:
    """Per-observation ordinal-outcome bundle consumed by RectMvncdKernel.

    Exposes the ``(n_obs, nord)`` 0-based observed ordinal categories as
    ``y_ord`` (GAUSS ``yiord``).
    """

    def __init__(self, y_ord: NDArray) -> None:
        self.y_ord = y_ord


class MORPFlexModel(BaseModel):
    """Mixed-panel Multivariate Ordered Response Probit (MORPFlex).

    Parameters
    ----------
    data : str, os.PathLike, or pd.DataFrame
        Path to a data file or an in-memory DataFrame.
    dep_vars : list of str
        Column names for the ordinal outcome variables (one per ordinal
        dimension); ``nord = len(dep_vars)``.
    spec : dict
        Variable specification mapping coefficient names to per-outcome column
        names / keywords, as in :class:`~pybhatlib.models.morp.MORPModel`.
    n_categories : list of int
        Number of observed ordinal categories per dimension (aligned with
        ``dep_vars``).
    control : MORPFlexControl or None
        Estimation control structure (random-coefficient spec, copula /
        yj_kernel flags, MSL knobs, optimizer settings).

    Notes
    -----
    The random-coefficient specification lives on ``control``
    (``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``).  The joint
    correlation is sized over ``nrndtot = nrndcoef + nord`` random elements; the
    ordinal-kernel dimensions carry no free kernel-scale parameters (``wker``
    fixed to ones).
    """

    def __init__(
        self,
        data: str | os.PathLike | pd.DataFrame,
        dep_vars: list[str],
        spec: dict,
        n_categories: list[int],
        control: MORPFlexControl | None = None,
    ) -> None:
        self.control = control or MORPFlexControl()

        if isinstance(data, pd.DataFrame):
            self.data_path = "<DataFrame>"
            self.data = data
        else:
            self.data_path = str(data)
            # Load only the columns this model references. The panel-commute
            # frame is 1400+ columns while MORPFlex touches ~25; over-inclusive
            # by design, the UNO/SERO keywords are dropped and the parity test
            # guards the collector.
            usecols = used_columns_selector(
                value_cols=dep_vars,
                id_cols=[self.control.person_id, self.control.weight_var],
                specs=[spec] if spec is not None else [],
            )
            self.data = load_data(str(data), usecols=usecols)

        self.dep_vars = list(dep_vars)
        self.nord = len(self.dep_vars)
        if self.nord < 1:
            raise ValueError("MORPFlex requires at least 1 ordinal dimension")
        self.n_categories = list(int(c) for c in n_categories)
        if len(self.n_categories) != self.nord:
            raise ValueError(
                f"n_categories length ({len(self.n_categories)}) must match "
                f"dep_vars length ({self.nord})"
            )

        for dv in self.dep_vars:
            if dv not in self.data.columns:
                raise ValueError(f"Dependent variable '{dv}' not found in data")

        # --- design tensor X: (N, nord, n_beta) via the shared spec parser --
        if spec is None:
            raise ValueError("spec is required")
        self.spec_dict = spec
        self.X, self._var_names = parse_spec(spec, self.data, self.dep_vars, nseg=1)
        self.var_names = list(self._var_names)
        self.n_beta = len(self.var_names)

        # --- ordinal outcomes: (N, nord), 0-based -------------------------
        self.N = len(self.data)
        if self.control.weight_var is not None:
            self.weights = self.data[self.control.weight_var].to_numpy(
                dtype=np.float64
            )
        else:
            self.weights = np.ones(self.N, dtype=np.float64)
        self.y_ord = np.zeros((self.N, self.nord), dtype=np.int64)
        for d, dv in enumerate(self.dep_vars):
            vals = self.data[dv].to_numpy()
            self.y_ord[:, d] = (vals - vals.min()).astype(np.int64)
            if self.y_ord[:, d].max() >= self.n_categories[d]:
                raise ValueError(
                    f"Dimension '{dv}': found category "
                    f"{int(self.y_ord[:, d].max())} but "
                    f"n_categories={self.n_categories[d]}"
                )

        # --- panel person ids ---------------------------------------------
        if self.control.person_id is not None:
            self.person_ids = self.data[self.control.person_id].to_numpy()
        else:
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
            nord=self.nord,
            n_categories=self.n_categories,
            normker=not ctrl.yj_kernel,
            randdiag=ctrl.randdiag,
        )
        layout = ParamLayout(
            n_beta=spec.n_beta,
            n_rcor=spec.nrndtcor,     # joint rc + ordinal-kernel correlation
            n_scal=spec.nscale,
            n_lam=spec.numlam,
            n_kern=spec.n_kern,       # 0 for MORP (wker fixed to ones)
            kern_before_lam=True,     # GAUSS b: [thresh|beta|rcor|scal|lam|kernlam]
            n_thresh=spec.nthresh,
            n_kernlam=spec.nkernlam,
        )
        return spec, layout

    def _make_draw_source(self, n_rnd: int, n_rep: int, n_ind: int):
        """Runtime draw source (zeros when there are no random coefficients)."""
        if n_rnd == 0:
            return FixtureDrawSource(np.zeros((n_rep, n_ind * n_rnd)))
        return ScipyHaltonDrawSource(seed=self.control.draw_seed)

    def _build_estimator(
        self, spec: MixingSpec, layout: ParamLayout, panel: PanelIndex,
        *, draws=None, reporting: bool = False,
    ) -> MixedMSLEstimator:
        """Assemble the shared MSL estimator for this model.

        Parameters
        ----------
        spec, layout, panel
            Mixing spec, parameter layout and panel index for the model.
        draws : DrawSource, optional
            Override the runtime draw source; defaults to the fit-identical
            scrambled-Halton source.
        reporting : bool, default False
            If ``True`` build the reparameterization space
            (:class:`~pybhatlib.mixed._reparam.ReportingSpace`) and the kernel in
            *reporting* (``lpr1``) space, so both consume the natural
            reporting-space parameters (as stored on
            :attr:`MORPFlexResults.params`) directly -- no estimation<->reporting
            inversion.  Used by
            :func:`~pybhatlib.models.morp_flex._morp_flex_ate.morp_flex_ate_from_params`
            for gradient-free prediction / ATE at externally-supplied estimates.
            ``False`` (default) is the estimation (``lpr``) space used at fit time.
        """
        ctrl = self.control
        design = DesignData(X=self.X, obs=_MorpObs(self.y_ord))
        if reporting:
            space: ParamSpace = ReportingSpace(
                layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
            )
        else:
            space = EstimationSpace(
                layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
            )
        pipeline = RandomCoefPipeline(
            spec, layout, spher=ctrl.spher, scal=ctrl.scal, intordn1=ctrl.intordn1
        )
        kernel = RectMvncdKernel(
            self.nord, spec.nrndcoef, self.n_categories,
            copula=ctrl.copula, yj_kernel=ctrl.yj_kernel, scal=ctrl.scal,
            intordn1=ctrl.intordn1, reporting=reporting,
            iid=ctrl.iid, correst=ctrl.correst,
            method=ctrl.method,
        )
        cfg = MSLConfig(
            n_rep=ctrl.n_rep,
            spher=ctrl.spher,
            scal=ctrl.scal,
            intordn1=ctrl.intordn1,
            floor_pcomp=ctrl.floor_pcomp,
            floor_z=ctrl.floor_z,
            score_convention="mask",
        )
        if draws is None:
            draws = self._make_draw_source(spec.nrndcoef, ctrl.n_rep, panel.n_ind)
        weightind = panel.weightind(self.weights)
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

        ``thresh`` becomes the ordered threshold cut points ``tau``, ``beta`` the
        sign-reparameterized coefficients, ``rcor`` the joint correlation
        off-diagonal entries (row-based upper-triangular), ``scal`` the scale
        (std-dev) vector ``exp(xscalrand)``, ``lam`` the Yeo-Johnson powers
        ``2 cdlogit(xlam)`` in ``(0, 2)``, and ``kernlam`` the ordinal-kernel YJ
        powers ``2 cdlogit(kernlam)``.
        """
        rc = space.unpack(theta, spec, want_grad=False)
        sl = layout.slices()

        blocks: dict[str, NDArray] = {}

        if layout.n_thresh > 0:
            tau, _ = thresh_reparam(theta[sl["thresh"]], spec.numthresh)
            blocks["thresh"] = np.asarray(tau, dtype=np.float64).reshape(-1)

        blocks["beta"] = np.asarray(rc.xmu, dtype=np.float64).reshape(-1)

        if layout.n_rcor > 0:
            cholall = newcholparmscaled(theta[sl["rcor"]], space.scal)
            omega_joint = np.asarray(cholall).T @ np.asarray(cholall)
            blocks["rcor"] = vecndup(omega_joint)
        else:
            blocks["rcor"] = np.zeros(0, dtype=np.float64)

        blocks["scal"] = np.asarray(rc.wscalrand, dtype=np.float64).reshape(-1)
        blocks["kern"] = np.zeros(0, dtype=np.float64)
        blocks["lam"] = np.asarray(rc.xlamrnd, dtype=np.float64).reshape(-1)

        if layout.n_kernlam > 0:
            blocks["kernlam"] = 2.0 * cdlogit(
                np.asarray(theta[sl["kernlam"]], dtype=np.float64)
            )

        out = np.zeros(layout.n_theta, dtype=np.float64)
        for name, s in sl.items():
            out[s] = blocks[name]
        return out

    def _reporting_jacobian(
        self,
        theta: NDArray,
        spec: MixingSpec,
        layout: ParamLayout,
        space: ParamSpace,
        *,
        eps: float = 1e-6,
    ) -> NDArray:
        """Central finite-difference Jacobian ``d report / d theta``."""
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

    def _param_names(self, spec: MixingSpec, layout: ParamLayout) -> list[str]:
        """Reporting-parameter names in the physical ``theta`` block order."""
        rc_names = [self.var_names[int(p)] for p in spec.mixpos]
        # joint correlation labels: random-coef names then ordinal-kernel dims.
        joint_names = list(rc_names) + [
            f"ord{d + 1}" for d in range(spec.nord)
        ]
        thresh = [
            f"thresh[{self.dep_vars[d]},{j + 1}]"
            for d in range(self.nord)
            for j in range(spec.numthresh[d])
        ]
        beta = list(self.var_names)
        rcor = [
            f"corr[{joint_names[i]},{joint_names[j]}]"
            for i in range(spec.nrndtot)
            for j in range(i + 1, spec.nrndtot)
        ]
        scal = [f"sd[{nm}]" for nm in rc_names]
        lam = [f"lam[{nm}]" for nm in rc_names]
        kernlam = [f"kernlam[{self.dep_vars[d]}]" for d in range(spec.nkernlam)]
        blocks = {
            "thresh": thresh, "beta": beta, "rcor": rcor, "scal": scal,
            "kern": [], "lam": lam, "kernlam": kernlam,
        }
        names: list[str] = [""] * layout.n_theta
        for name, s in layout.slices().items():
            names[s] = blocks[name]
        return names

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(self, *, draws=None) -> MORPFlexResults:
        """Estimate the MORPFlex model and return :class:`MORPFlexResults`.

        Parameters
        ----------
        draws : DrawSource, optional
            Override the runtime draw source (e.g. a
            :class:`~pybhatlib.mixed._draws.FixtureDrawSource` for parity
            replication).  Defaults to the scrambled-Halton runtime source.
        """
        ctrl = self.control
        t_start = time.time()

        spec, layout = self._build_spec_layout()
        panel = PanelIndex.from_ids(self.person_ids)
        est = self._build_estimator(spec, layout, panel, draws=draws)

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
                f"  MORPFlex estimation: {panel.n_obs} obs, {panel.n_ind} "
                f"individuals, {n_theta} parameters, {self.nord} ordinal "
                f"dimension(s), {spec.nrndcoef} random coef(s), "
                f"copula={ctrl.copula}, yj_kernel={ctrl.yj_kernel}, "
                f"nrep={ctrl.n_rep}"
            )

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

        # Cache the post-fit context so ``predict`` / ``ate`` can drive the
        # shared mixed predictor without re-assembling the engine.  ``theta`` is
        # the estimation-space fitted vector consumed by ``space.unpack`` /
        # ``pipeline.realize`` / ``kernel.prepare`` (not the reporting-space
        # ``params`` stored on the results object).
        self._fit_ctx = SimpleNamespace(
            theta=theta_hat, spec=spec, layout=layout, panel=panel, est=est
        )

        ll_pi, score = est.simulated_loglik(theta_hat, want_grad=True)
        ll_total = float(ll_pi.sum())
        ll_mean = ll_total / panel.n_ind
        g_sum = np.asarray(score, dtype=np.float64).sum(0)
        grad_norm = float(np.linalg.norm(g_sum))
        converged = bool(res.success) or grad_norm < ctrl.tol_check

        params_report = self._to_reporting(theta_hat, spec, layout, est.space)
        if ctrl.want_covariance:
            if ctrl.se_method.lower() != "bhhh":
                raise ValueError(
                    "MORPFlex currently supports only se_method='bhhh'"
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

        return MORPFlexResults(
            params=params_report,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_sum,
            loglik=ll_mean,
            n_obs=panel.n_obs,
            n_ind=panel.n_ind,
            param_names=self._param_names(spec, layout),
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
    # post-estimation surface (declared for interface conformance)
    # ------------------------------------------------------------------

    #: Post-fit context (estimation-space theta + engine), set by :meth:`_fit`.
    _fit_ctx = None

    def predict(self, data=None, *, scenario=None, draws=None, xp=None):
        """Draw-integrated per-observation ordinal category probabilities.

        Delegates to
        :func:`~pybhatlib.models.morp_flex._morp_flex_forecast.morp_flex_predict`:
        the shipped fixed-coefficient MORP marginal formulation lifted over the
        mixing draws.  Signature normalized to the common mixed-model convention
        ``predict(data=None, *, scenario=None, draws=None, xp=None)``.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to predict on (the design is rebuilt from a frame; defaults
            to the fitted training frame).  Must match the fit sample size (the
            mixing draws are held per fit individual).
        scenario : dict, optional
            A single scenario's ``{column: scalar | source_column}`` overrides,
            applied before the design is rebuilt.  ``None`` predicts at the
            observed covariates.
        draws : DrawSource, optional
            Override the fit-identical draw strategy.
        xp : module, optional
            Array backend used to wrap the per-dimension arrays.

        Returns
        -------
        list of NDArray
            ``probs[d]`` has shape ``(N, n_categories[d])`` (a valid category
            distribution per row), one array per ordinal dimension.
        """
        from pybhatlib.models.morp_flex._morp_flex_forecast import (
            morp_flex_predict,
        )

        self._require_results()
        return morp_flex_predict(
            self, data, scenario=scenario, draws=draws, xp=xp
        )

    def predict_category(self, data=None, *, scenario=None, draws=None):
        """Most-likely ordinal category per observation and dimension.

        Delegates to
        :func:`~pybhatlib.models.morp_flex._morp_flex_forecast.morp_flex_predict_category`.
        Signature normalized to the common mixed-model convention (``data`` /
        ``scenario`` overrides).
        """
        from pybhatlib.models.morp_flex._morp_flex_forecast import (
            morp_flex_predict_category,
        )

        self._require_results()
        return morp_flex_predict_category(
            self, data, scenario=scenario, draws=draws
        )

    def ate(self, *, scenarios=None, draws=None, xp=None, **kwargs):
        """Draw-integrated predicted ordinal probabilities / ATE.

        Delegates to
        :func:`~pybhatlib.models.morp_flex._morp_flex_ate.morp_flex_ate`
        (per-outcome mean category probabilities; ``scenarios=`` for
        counterfactuals), lifting the fixed-coefficient MORP ATE over the mixing
        draws.  Keyword-only, matching the harmonized fixed-coefficient facade
        (:meth:`pybhatlib.models.morp.MORPModel.ate`): the design defaults to the
        training design, or is rebuilt per scenario when ``scenarios=`` is given.

        Parameters
        ----------
        scenarios : dict or pd.DataFrame, optional
            Scenario specification for counterfactual overrides; ``data`` /
            ``spec`` / ``dep_vars`` are supplied automatically from the model.
        draws : DrawSource, optional
            Override the fit-identical draw strategy (shared across scenarios).
        xp : module, optional
            Array backend used to wrap the per-dimension arrays.
        **kwargs
            Forwarded to :func:`morp_flex_ate` (e.g. an explicit single-design
            ``X`` for the no-scenario path).

        Returns
        -------
        MORPFlexATEResult
            Per-outcome ``predicted_probs`` plus, in scenario mode,
            ``shares_per_scenario`` and a ``.comparison(base, treatment)``
            helper.
        """
        from pybhatlib.models.morp_flex._morp_flex_ate import morp_flex_ate

        self._require_results()
        return morp_flex_ate(
            self, scenarios=scenarios, draws=draws, xp=xp, **kwargs
        )
