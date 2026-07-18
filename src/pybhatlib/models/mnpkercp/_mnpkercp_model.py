"""MNPKerCP (mixed-panel MNP) model estimation on the shared MSL engine.

The mixed-panel Multinomial Probit facade wires the harmonized model interface
(:class:`~pybhatlib.models._base.BaseModel`) to the MVNCD (OVUS) kernel with an
optional rc<->kernel copula (:class:`~pybhatlib.models.mnpkercp._mnpkercp_kernel.MvncdKernel`)
plus the shared MSL engine (:mod:`pybhatlib.mixed`), ported from GAUSS
``MNPKERCP.gss``.

``_fit`` assembles the panel index, the *joint* mixing spec (the correlation is
over ``nrndtot = nrndcoef + (nc - 1)`` random elements), the parameter layout in
GAUSS ``b`` order ``[beta | rcor | scal | kern | lam]`` (the kernel-scale block
interleaves **between** ``scal`` and ``lam``), the random-coefficient pipeline,
the MVNCD kernel and :class:`~pybhatlib.mixed._engine.MixedMSLEstimator`; runs
``scipy.optimize.minimize(jac=True)``; and performs a reporting-space BHHH
standard-error pass.

Copula / gradient scope
-----------------------
* The forward simulated log-likelihood reproduces GAUSS ``lpr`` value-for-value
  (the primary GAUSS-LL parity gate).
* The analytic score chains the copula ``dlogp_drc`` path through the shared
  random-coefficient reparameterization into the ``beta`` / ``scal`` / ``lam``
  blocks and the kernel-scale ``kern`` block (validated by the master
  finite-difference gate).
* The **joint correlation** (``rcor``) analytic gradient is not yet chained: the
  kernel emits no correlation-covariance derivative (plan P1.2), so the ``rcor``
  block of the analytic score is zero.  Estimation therefore holds the
  correlation parameters at their start values; the LL, and the remaining
  gradient blocks, are exact.

``predict`` / ``forecast`` / ``ate`` are wired to the shared mixed-prediction
machinery (:mod:`pybhatlib.mixed._predict`) via
:mod:`pybhatlib.models.mnpkercp._mnpkercp_forecast` and
:mod:`pybhatlib.models.mnpkercp._mnpkercp_ate`: the shipped fixed-coefficient MNP
choice-probability formulation is lifted over the mixing draws.
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
from pybhatlib.matgradient._radial import newcholparmscaled
from pybhatlib.mixed._draws import FixtureDrawSource, ScipyHaltonDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.mnpkercp._mnpkercp_control import MNPKerCPControl
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel
from pybhatlib.models.mnpkercp._mnpkercp_results import MNPKerCPResults
from pybhatlib.vecup._panel import PanelIndex
from pybhatlib.vecup._vec_ops import vecndup


class _MnpObs:
    """Per-observation availability / chosen bundle consumed by MvncdKernel.

    Mirrors the GAUSS ``availmat`` / ``depvar`` design columns as dense
    ``(n_obs, nc)`` arrays.
    """

    def __init__(self, avail: NDArray, chosen: NDArray) -> None:
        self.avail = avail
        self.chosen = chosen


class MNPKerCPModel(BaseModel):
    """Mixed-panel Multinomial Probit with a kernel-error copula (MNPKerCP).

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
        column names / keywords, as in :class:`~pybhatlib.models.mnp.MNPModel`.
    var_names : list of str or None
        Coefficient names.  Inferred from ``spec`` keys when ``None``.
    control : MNPKerCPControl or None
        Estimation control structure (random-coefficient spec, copula flag, MSL
        knobs, optimizer settings).

    Notes
    -----
    The random-coefficient specification lives on ``control``
    (``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``).  The joint
    correlation is sized over ``nrndtot = nrndcoef + (nc - 1)`` random elements,
    with ``nc - 2`` free kernel-scale parameters.
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        alternatives: list[str],
        availability: str | list[str] = "none",
        spec: dict | None = None,
        var_names: list[str] | None = None,
        control: MNPKerCPControl | None = None,
    ) -> None:
        self.control = control or MNPKerCPControl()

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
                id_cols=[self.control.person_id, self.control.weight_var],
                specs=[spec] if spec is not None else [],
            )
            self.data = load_data(data, usecols=usecols)
        else:
            self.data_path = "<DataFrame>"
            self.data = data

        self.alternatives = alternatives
        self.n_alts = len(alternatives)
        if self.n_alts < 2:
            raise ValueError("MNPKerCP requires at least 2 alternatives")

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
        if self.control.weight_var is not None:
            self.weights = self.data[self.control.weight_var].to_numpy(
                dtype=np.float64
            )
        else:
            self.weights = np.ones(self.N, dtype=np.float64)

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
            kernel_dim=self.n_alts - 1,
            randdiag=ctrl.randdiag,
        )
        layout = ParamLayout(
            n_beta=spec.n_beta,
            n_rcor=spec.nrndtcor,     # joint rc + differenced-kernel correlation
            n_scal=spec.nscale,
            n_lam=spec.numlam,
            n_kern=spec.n_kern,       # nc - 2 free kernel-scale params
            kern_before_lam=True,     # GAUSS b: [beta|rcor|scal|kern|lam]
        )
        return spec, layout

    def _make_draw_source(self, n_rnd: int, n_rep: int, n_ind: int):
        """Runtime draw source (zeros when there are no random coefficients)."""
        if n_rnd == 0:
            return FixtureDrawSource(np.zeros((n_rep, n_ind * n_rnd)))
        return ScipyHaltonDrawSource(seed=self.control.draw_seed)

    def _build_estimator(
        self, spec: MixingSpec, layout: ParamLayout, panel: PanelIndex,
        *, draws=None, space=None,
    ) -> MixedMSLEstimator:
        """Assemble the MSL estimator (design / space / pipeline / kernel).

        Parameters
        ----------
        spec, layout, panel
            Mixing spec, parameter layout and panel index from
            :meth:`_build_spec_layout` / :class:`PanelIndex`.
        draws : DrawSource, optional
            Draw source; defaults to the runtime scrambled-Halton source (zeros
            when there are no random coefficients).
        space : ParamSpace, optional
            Reparameterization strategy. Defaults to
            :class:`~pybhatlib.mixed._reparam.EstimationSpace` (the space the
            optimizer consumes). Post-estimation callers that hold *reporting*
            (natural) parameters pass a
            :class:`~pybhatlib.mixed._reparam.ReportingSpace` so the estimator
            consumes those params directly (no reporting -> estimation inversion).
        """
        ctrl = self.control
        design = DesignData(X=self.X, obs=_MnpObs(self.avail, self.chosen))
        if space is None:
            space = EstimationSpace(
                layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
            )
        pipeline = RandomCoefPipeline(
            spec, layout, spher=ctrl.spher, scal=ctrl.scal, intordn1=ctrl.intordn1
        )
        kernel = MvncdKernel(
            self.n_alts, spec.nrndcoef, copula=ctrl.copula, scal=ctrl.scal
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
        space: EstimationSpace,
    ) -> NDArray:
        """Map an estimation-space ``theta`` to natural reporting parameters.

        ``beta`` becomes the sign-reparameterized coefficients, ``rcor`` the
        joint correlation off-diagonal entries (row-based upper-triangular),
        ``scal`` the scale (std-dev) vector ``exp(xscalrand)``, ``kern`` the raw
        kernel-scale parameters (reported as-is, GAUSS ``xscalkerfinal``), and
        ``lam`` the Yeo-Johnson powers ``2 cdlogit(xlam)`` in ``(0, 2)``.
        """
        rc = space.unpack(theta, spec, want_grad=False)
        sl = layout.slices()

        beta_r = np.asarray(rc.xmu, dtype=np.float64).reshape(-1)
        if layout.n_rcor > 0:
            # joint correlation over nrndtot: rebuild from the rcor block.
            cholall = newcholparmscaled(theta[sl["rcor"]], space.scal)
            omega_joint = np.asarray(cholall).T @ np.asarray(cholall)
            rcor_r = vecndup(omega_joint)
        else:
            rcor_r = np.zeros(0, dtype=np.float64)
        scal_r = np.asarray(rc.wscalrand, dtype=np.float64).reshape(-1)
        kern_r = np.asarray(theta[sl["kern"]], dtype=np.float64).reshape(-1)
        lam_r = np.asarray(rc.xlamrnd, dtype=np.float64).reshape(-1)

        out = np.zeros(layout.n_theta, dtype=np.float64)
        blocks = {
            "beta": beta_r, "rcor": rcor_r, "scal": scal_r,
            "kern": kern_r, "lam": lam_r,
        }
        for name, s in sl.items():
            out[s] = blocks[name]
        return out

    def _reporting_jacobian(
        self,
        theta: NDArray,
        spec: MixingSpec,
        layout: ParamLayout,
        space: EstimationSpace,
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
        # joint correlation labels: random-coef names then kernel-error dims.
        joint_names = list(rc_names) + [
            f"ker{d + 1}" for d in range(spec.kernel_dim)
        ]
        beta = list(self.var_names)
        rcor = [
            f"corr[{joint_names[i]},{joint_names[j]}]"
            for i in range(spec.nrndtot)
            for j in range(i + 1, spec.nrndtot)
        ]
        scal = [f"sd[{nm}]" for nm in rc_names]
        kern = [f"kernscale{i + 1:02d}" for i in range(spec.n_kern)]
        lam = [f"lam[{nm}]" for nm in rc_names]
        blocks = {"beta": beta, "rcor": rcor, "scal": scal, "kern": kern, "lam": lam}
        names: list[str] = [""] * layout.n_theta
        for name, s in layout.slices().items():
            names[s] = blocks[name]
        return names

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(self, *, draws=None) -> MNPKerCPResults:
        """Estimate the MNPKerCP model and return :class:`MNPKerCPResults`.

        Parameters
        ----------
        draws : DrawSource, optional
            Override the runtime draw source (e.g. a
            :class:`~pybhatlib.mixed._draws.FixtureDrawSource` for GAUSS-parity
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
                f"  MNPKerCP estimation: {panel.n_obs} obs, {panel.n_ind} "
                f"individuals, {n_theta} parameters, {self.n_alts} alternatives, "
                f"{spec.nrndcoef} random coef(s), copula={ctrl.copula}, "
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
                    "MNPKerCP currently supports only se_method='bhhh'"
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

        # Persist the estimation-space fit context so the post-estimation
        # ``predict`` / ``ate`` / ``forecast`` surface can rebuild the shared
        # mixed-prediction components (the MSL engine carries the panel, mixing
        # draws, pipeline, reparameterization space, kernel, layout and config).
        self._theta_hat = theta_hat
        self._est = est
        self._mixing_spec = spec
        self._layout = layout

        return MNPKerCPResults(
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

    def predict(
        self,
        data: pd.DataFrame | None = None,
        *,
        scenario: dict | None = None,
        draws=None,
        xp=None,
    ) -> NDArray:
        """Draw-integrated, sample-averaged predicted market shares.

        Integrates the MVNCD per-alternative choice probabilities over the
        mixing distribution (random coefficients held fixed per individual,
        exactly as at fit time) and averages over the sample.  Delegates to the
        shared mixed-prediction machinery
        (:func:`~pybhatlib.mixed._predict.mixed_predict_shares`).

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to predict on; defaults to the fitted data.
        scenario : dict, optional
            A single scenario's ``{column: scalar | source_column}`` overrides
            applied before the design is rebuilt.  ``None`` predicts at the
            observed covariates.
        draws : DrawSource, optional
            Override the fit-identical draw strategy.
        xp : module, optional
            Array backend used to wrap the result. Defaults to NumPy.

        Returns
        -------
        NDArray, shape (n_alts,)
            Sample-averaged, draw-integrated market shares (non-negative,
            summing to one over the available alternatives).

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        from pybhatlib.models.mnpkercp._mnpkercp_forecast import mnpkercp_predict

        return mnpkercp_predict(
            self, data, scenario=scenario, draws=draws, xp=xp,
        )

    # ``forecast`` is the market-share forecast; an alias of ``predict`` so the
    # family exposes the harmonized ``.forecast`` surface with one code path.
    forecast = predict

    def predict_choice(
        self,
        data: pd.DataFrame | None = None,
        *,
        scenario: dict | None = None,
        draws=None,
        xp=None,
    ) -> NDArray:
        """Per-observation most-likely predicted alternative.

        Discrete-family choice helper mirroring
        :meth:`~pybhatlib.models.mnp.MNPModel.predict_choice`; delegates to
        :func:`~pybhatlib.models.mnpkercp._mnpkercp_forecast.mnpkercp_predict_choice`.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to predict on; defaults to the fitted data.
        scenario : dict, optional
            A single scenario's ``{column: scalar | source_column}`` overrides.
        draws : DrawSource, optional
            Override the fit-identical draw strategy.
        xp : module, optional
            Array backend used to wrap the result. Defaults to NumPy.

        Returns
        -------
        NDArray, shape (n_obs,)
            Predicted choice index (0-based) for each observation.

        Raises
        ------
        RuntimeError
            If the model has not been fit.
        """
        from pybhatlib.models.mnpkercp._mnpkercp_forecast import (
            mnpkercp_predict_choice,
        )

        return mnpkercp_predict_choice(
            self, data, scenario=scenario, draws=draws, xp=xp,
        )

    def ate(
        self,
        *,
        scenarios=None,
        data: pd.DataFrame | None = None,
        draws=None,
        alternative_names: list[str] | None = None,
        xp=None,
        **kwargs,
    ):
        """Draw-integrated Average Treatment Effect across scenarios.

        Delegates to :func:`~pybhatlib.models.mnpkercp._mnpkercp_ate.mnpkercp_ate`,
        which lifts the shipped fixed-coefficient MNP ATE over the mixing draws.

        Parameters
        ----------
        scenarios : dict or pd.DataFrame
            Scenario specification (see
            :func:`~pybhatlib.mixed._predict.mixed_ate`).  Required.
        data : pd.DataFrame, optional
            Dataset the scenarios override; defaults to the fitted data.
        draws : DrawSource, optional
            Override the fit-identical draw strategy (shared across scenarios).
        alternative_names : list of str, optional
            Output labels; defaults to ``self.alternatives``.
        xp : module, optional
            Array backend used to wrap the result arrays. Defaults to NumPy.

        Returns
        -------
        MixedATEResult
            Baseline ``predicted_shares`` plus per-scenario shares and the
            ``.comparison()`` / ``.summary()`` surface.

        Raises
        ------
        ValueError
            If ``scenarios`` is not supplied.
        RuntimeError
            If the model has not been fit.
        """
        from pybhatlib.models.mnpkercp._mnpkercp_ate import mnpkercp_ate

        if scenarios is None:
            raise ValueError("MNPKerCPModel.ate() requires scenarios=")
        return mnpkercp_ate(
            self,
            scenarios=scenarios,
            data=data,
            draws=draws,
            alternative_names=alternative_names,
            xp=xp,
        )
