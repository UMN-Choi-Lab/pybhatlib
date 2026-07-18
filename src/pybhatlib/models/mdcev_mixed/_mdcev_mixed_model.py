"""Mixed / panel MDCEV model on the shared mixed/panel MSL engine.

The mixed MDCEV facade wires the harmonized model interface
(:class:`~pybhatlib.models._base.BaseModel`) to the MDCEV logit-Jacobian kernel
(:class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel.LogitJacobianKernel`)
plus the shared MSL engine (:mod:`pybhatlib.mixed`). Mixing is over the
**baseline-utility** coefficients only; the translation (``gamma``) parameters
and the MDCEV kernel error scale are kernel-owned, non-mixed parameters, and
there is **no copula** (``dlogp_drc == 0``).

``_fit`` assembles the panel index, mixing spec, parameter layout,
random-coefficient pipeline, logit-Jacobian kernel and
:class:`~pybhatlib.mixed._engine.MixedMSLEstimator`; packs the estimation-space
parameter vector ``theta = [beta | gamma | rcor | kern | scal | lam]`` (the
GAUSS ``Mixed Traditional MDCEV.gss`` ``b`` order, with the kernel scale placed
between ``rcor`` and the random-coefficient scales); runs
``scipy.optimize.minimize(jac=True)`` against the engine objective with
``score_convention="divide"`` (GAUSS ``gcomp ./ Pobs``); and performs a
reporting-space BHHH standard-error pass.

The engine reproduces the GAUSS mixed MDCEV log-likelihood value-for-value at a
fixed parameter vector with fixed draws (the MDCEV logit likelihood is *exact*,
not an OVUS approximation); see ``tests/test_mixed/test_mdcev_mixed_gauss_parity.py``.

``predict`` / ``forecast`` / ``ate`` lift the shipped fixed-coefficient MDCEV
participation / allocation formulation over the mixing draws via the shared
mixed-prediction machinery (:mod:`pybhatlib.mixed._predict`); see
:mod:`pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast` and
:mod:`pybhatlib.models.mdcev_mixed._mdcev_mixed_ate`.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import scipy.optimize as sopt
from numpy.typing import NDArray
from scipy.special import ndtr

from pybhatlib.io._data_loader import load_data, used_columns_selector
from pybhatlib.mixed._draws import DrawSource, FixtureDrawSource, ScipyHaltonDrawSource
from pybhatlib.mixed._engine import DesignData, MixedMSLEstimator, MSLConfig
from pybhatlib.mixed._rc_pipeline import RandomCoefPipeline
from pybhatlib.mixed._reparam import EstimationSpace, ParamLayout
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel
from pybhatlib.models.mdcev_mixed._mdcev_mixed_results import MDCEVMixedResults
from pybhatlib.vecup._panel import PanelIndex
from pybhatlib.vecup._vec_ops import vecndup


def _ensure_special_cols(df: pd.DataFrame) -> None:
    """Ensure the ``uno`` (ones) and ``sero`` (zeros) helper columns exist.

    Mirrors the GAUSS convention that the dataset carries constant columns
    ``uno`` and ``sero`` (used as alternative-specific-constant / null design
    entries). Added in place if absent.
    """
    if "uno" not in df.columns:
        df["uno"] = 1.0
    if "sero" not in df.columns:
        df["sero"] = 0.0


def _spec_dict_to_array(
    spec_dict: dict, alternatives: list[str]
) -> tuple[NDArray, list[str]]:
    """Convert a spec dict (MNL/MNP/MDCEV format) to an ``(nc, n_par)`` name array.

    Parameters
    ----------
    spec_dict : dict
        Mapping of parameter names to per-alternative column names, e.g.
        ``{"ASC_Esc": {"Esc": "uno", ...}, ...}``. Missing alternatives default
        to the ``"sero"`` null column.
    alternatives : list of str
        Alternative names in order.

    Returns
    -------
    spec_array : NDArray of str, shape (nc, n_par)
    param_names : list of str
    """
    param_names = list(spec_dict.keys())
    nc = len(alternatives)
    n_par = len(param_names)
    arr = np.empty((nc, n_par), dtype=object)
    for j, pname in enumerate(param_names):
        pspec = spec_dict[pname]
        for i, alt in enumerate(alternatives):
            arr[i, j] = pspec.get(alt, "sero")
    return arr.astype(str), param_names


class MDCEVMixedModel(BaseModel):
    """Mixed / panel Multiple Discrete-Continuous Extreme Value (MDCEV) model.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to a data file or an in-memory DataFrame.
    alternatives : list of str
        Consumption-quantity column names (GAUSS ``dvunordname``); the first
        entry must be the essential outside / numeraire good.
    price : list of str or None
        Price column names (GAUSS ``davunordname``); ``None`` defaults to the
        ``"uno"`` column of ones for every alternative (no price variation).
    utility_spec : dict or NDArray
        Baseline-utility (``psi``) specification. Either a dict mapping parameter
        names to per-alternative column names, or an ``(nc, nvarm)`` string array
        of column names (GAUSS ``ivmt``).
    gamma_spec : dict or NDArray
        Satiation (``gamma``) specification, same format as ``utility_spec``
        (GAUSS ``ivgt``); the first parameter corresponds to the outside good.
    param_names : list of str or None
        Baseline-utility parameter names (GAUSS ``varnam``). Inferred from
        ``utility_spec`` keys when ``None``.
    gamma_names : list of str or None
        Satiation parameter names (GAUSS ``varngam``). Inferred from
        ``gamma_spec`` keys when ``None``.
    control : MDCEVMixedControl or None
        Estimation control structure (random-coefficient spec, MSL knobs,
        MDCEV utility form, optimizer settings).
    obs_id_var : str
        Column holding the observation identifier (GAUSS ``ID``); used as the
        default panel person id when ``control.person_id`` is ``None`` and the
        column is present.

    Notes
    -----
    The random-coefficient specification lives on ``control``
    (``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``); an empty
    spec yields a fixed-coefficient MDCEV through the shared engine (the
    collapse gate). The ``gamma`` block and the MDCEV kernel scale are always
    kernel-owned, non-mixed parameters.
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        alternatives: list[str],
        price: list[str] | None = None,
        utility_spec: dict | NDArray | None = None,
        gamma_spec: dict | NDArray | None = None,
        param_names: list[str] | None = None,
        gamma_names: list[str] | None = None,
        control: MDCEVMixedControl | None = None,
        obs_id_var: str = "ID",
    ) -> None:
        self.control = control or MDCEVMixedControl()
        self.obs_id_var = obs_id_var

        # populated by ._fit(); read by predict / ate / forecast to rebuild the
        # shared mixed-prediction context.
        self._fitted_theta: NDArray | None = None
        self._fitted_spec: MixingSpec | None = None
        self._fitted_layout: ParamLayout | None = None
        self._fitted_est: MixedMSLEstimator | None = None

        if isinstance(data, str):
            self.data_path = data
            # Load only the columns this model references; over-inclusive by
            # design, the uno/sero keywords are dropped (they are synthesized by
            # ``_ensure_special_cols`` below or kept if the file carries them),
            # and the parity test guards the collector.
            _prune_price = list(price) if price is not None else []
            usecols = used_columns_selector(
                value_cols=list(alternatives) + _prune_price,
                id_cols=[self.control.person_id, obs_id_var, self.control.weight_var],
                specs=[utility_spec, gamma_spec],
            )
            self.data = load_data(data, usecols=usecols)
        else:
            self.data_path = "<DataFrame>"
            self.data = data.copy()
        _ensure_special_cols(self.data)

        self.alternatives = list(alternatives)
        self.nc = len(self.alternatives)

        self.price_cols = (
            list(price) if price is not None else ["uno"] * self.nc
        )

        if utility_spec is None or gamma_spec is None:
            raise ValueError("utility_spec and gamma_spec are required")

        if isinstance(utility_spec, dict):
            self.utility_spec, u_names = _spec_dict_to_array(
                utility_spec, self.alternatives
            )
        else:
            self.utility_spec = np.asarray(utility_spec, dtype=str)
            u_names = param_names or [
                f"beta{i}" for i in range(self.utility_spec.shape[1])
            ]
        if isinstance(gamma_spec, dict):
            self.gamma_spec, g_names = _spec_dict_to_array(
                gamma_spec, self.alternatives
            )
        else:
            self.gamma_spec = np.asarray(gamma_spec, dtype=str)
            g_names = gamma_names or [
                f"gamma{i}" for i in range(self.gamma_spec.shape[1])
            ]

        self.var_names = list(param_names) if param_names is not None else u_names
        self.gamma_names = (
            list(gamma_names) if gamma_names is not None else g_names
        )
        self.nvarm = self.utility_spec.shape[1]
        self.nvargam = self.gamma_spec.shape[1]

        self._build_design()

    # ------------------------------------------------------------------
    # design tensors
    # ------------------------------------------------------------------

    def _build_design(self) -> None:
        """Build the baseline / satiation design tensors and consumption/price.

        Sets:

        * ``self.X`` -- baseline design tensor ``(n_obs, nc, nvarm)`` with
          ``X[q, k, v] = data[q, col(utility_spec[k, v])]``; the engine forms the
          drawn baseline utility ``Vsub = einsum('qcv,qv->qc', X, xmunew)``.
        * ``self.gamma_design`` -- satiation covariates ``(n_obs, nc, nvargam)``
          with ``gamma_design[q, k, j] = data[q, col(gamma_spec[k, j])]``.
        * ``self.consumption`` / ``self.price`` -- ``(n_obs, nc)`` each.
        * ``self.weights`` -- ``(n_obs,)`` observation weights.
        * ``self.person_ids`` -- ``(n_obs,)`` panel person ids.
        """
        df = self.data
        nc = self.nc

        def col(name: str) -> NDArray:
            return df[name].to_numpy(dtype=np.float64)

        n_obs = len(df)
        self.n_obs = n_obs

        # baseline design tensor (n_obs, nc, nvarm)
        X = np.zeros((n_obs, nc, self.nvarm), dtype=np.float64)
        for k in range(nc):
            for v in range(self.nvarm):
                X[:, k, v] = col(self.utility_spec[k, v])
        self.X = X

        # satiation design tensor (n_obs, nc, nvargam)
        gd = np.zeros((n_obs, nc, self.nvargam), dtype=np.float64)
        for k in range(nc):
            for j in range(self.nvargam):
                gd[:, k, j] = col(self.gamma_spec[k, j])
        self.gamma_design = gd

        self.consumption = np.column_stack([col(a) for a in self.alternatives])
        self.price = np.column_stack([col(p) for p in self.price_cols])

        ctrl = self.control
        if ctrl.weight_var is not None:
            self.weights = col(ctrl.weight_var)
        else:
            self.weights = np.ones(n_obs, dtype=np.float64)

        if ctrl.person_id is not None:
            self.person_ids = df[ctrl.person_id].to_numpy()
        elif self.obs_id_var in df.columns:
            self.person_ids = df[self.obs_id_var].to_numpy()
        else:
            self.person_ids = np.arange(n_obs)

    # ------------------------------------------------------------------
    # engine assembly
    # ------------------------------------------------------------------

    def _build_spec_layout(self) -> tuple[MixingSpec, ParamLayout]:
        """Build the mixing spec and the GAUSS-``b``-ordered parameter layout.

        The MDCEV physical ``theta`` order is
        ``[beta | gamma | rcor | kern | scal | lam]`` (``kern_before_scal``): the
        single MDCEV kernel error scale interleaves between the correlation block
        and the random-coefficient scales, matching GAUSS
        ``b = b1 | -1000 | gamma_rest | startker | xscalker | xscalrand | xlam``.
        """
        ctrl = self.control
        spec = MixingSpec.from_var_names(
            var_names=self.var_names,
            normvar=tuple(ctrl.normvar),
            logvar=tuple(ctrl.logvar),
            yjvar=tuple(ctrl.yjvar),
            varneg=tuple(ctrl.varneg),
            varpos=tuple(ctrl.varpos),
            nvargam=self.nvargam,
        )
        layout = ParamLayout(
            n_beta=spec.n_beta,
            n_gamma=self.nvargam,
            n_rcor=spec.nrndtcor,
            n_kern=1,                      # single MDCEV kernel error scale
            n_scal=spec.nscale,            # nrndcoef random-coefficient scales
            n_lam=spec.numlam,
            kern_before_scal=True,
        )
        return spec, layout

    def _make_draw_source(
        self, n_rnd: int, n_rep: int, n_ind: int
    ) -> DrawSource:
        """Runtime draw source (zero-width block when there are no random coefs)."""
        if n_rnd == 0:
            return FixtureDrawSource(np.zeros((n_rep, n_ind * n_rnd)))
        return ScipyHaltonDrawSource(seed=self.control.draw_seed)

    def build_estimator(
        self,
        spec: Optional[MixingSpec] = None,
        layout: Optional[ParamLayout] = None,
        panel: Optional[PanelIndex] = None,
        *,
        draws: Optional[DrawSource] = None,
    ) -> MixedMSLEstimator:
        """Assemble the shared MSL estimator for the mixed MDCEV model.

        Parameters
        ----------
        spec, layout, panel : optional
            Prebuilt spec / layout / panel; built from the control when omitted.
        draws : DrawSource, optional
            Draw source override (e.g. a :class:`FixtureDrawSource` for
            GAUSS-parity). Defaults to the runtime scrambled-Halton source.

        Returns
        -------
        MixedMSLEstimator
            Configured with ``score_convention="divide"`` (GAUSS ``gcomp ./ Pobs``).
        """
        ctrl = self.control
        if spec is None or layout is None:
            spec, layout = self._build_spec_layout()
        if panel is None:
            panel = PanelIndex.from_ids(self.person_ids)

        obs = SimpleNamespace(
            consumption=self.consumption,
            price=self.price,
            gamma_design=self.gamma_design,
        )
        design = DesignData(X=self.X, obs=obs)

        space = EstimationSpace(
            layout, scal=ctrl.scal, intordn1=ctrl.intordn1, spher=ctrl.spher
        )
        pipeline = RandomCoefPipeline(
            spec, layout, spher=ctrl.spher, scal=ctrl.scal, intordn1=ctrl.intordn1
        )
        kernel = LogitJacobianKernel(
            self.nc,
            self.nvargam,
            control=MDCEVControl(
                utility=ctrl.utility,
                outside_good_gamma=ctrl.outside_good_gamma,
            ),
            eqmatgam=np.eye(self.nvargam, dtype=np.float64),
        )
        cfg = MSLConfig(
            n_rep=ctrl.n_rep,
            spher=ctrl.spher,
            scal=ctrl.scal,
            intordn1=ctrl.intordn1,
            floor_pcomp=ctrl.floor_pcomp,
            floor_z=ctrl.floor_z,
            score_convention="divide",
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
    # reporting-space transform
    # ------------------------------------------------------------------

    def _to_reporting(
        self,
        theta: NDArray,
        spec: MixingSpec,
        layout: ParamLayout,
        space: EstimationSpace,
    ) -> NDArray:
        """Map an estimation-space ``theta`` to natural reporting parameters.

        ``beta`` becomes the sign-reparameterized coefficients; ``gamma`` and the
        MDCEV kernel scale pass through unchanged; ``rcor`` becomes the
        off-diagonal correlation entries (row-based upper-triangular); ``scal``
        the random-coefficient scale (std-dev) vector ``exp(xscalrand)``; and
        ``lam`` the Yeo-Johnson powers ``2 cdlogit(xlam)`` in ``(0, 2)``.
        """
        rc = space.unpack(theta, spec, want_grad=False)
        sl = layout.slices()

        beta_r = np.asarray(rc.xmu, dtype=np.float64).reshape(-1)
        gamma_r = np.asarray(theta[sl["gamma"]], dtype=np.float64).reshape(-1)
        if layout.n_rcor > 0:
            rcor_r = vecndup(np.asarray(rc.omegastar, dtype=np.float64))
        else:
            rcor_r = np.zeros(0, dtype=np.float64)
        kern_r = np.asarray(theta[sl["kern"]], dtype=np.float64).reshape(-1)
        scal_r = np.asarray(rc.wscalrand, dtype=np.float64).reshape(-1)
        lam_r = np.asarray(rc.xlamrnd, dtype=np.float64).reshape(-1)
        return np.concatenate([beta_r, gamma_r, rcor_r, kern_r, scal_r, lam_r])

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

    def _param_names(self, spec: MixingSpec) -> list[str]:
        """Reporting-parameter names in ``[beta|gamma|rcor|kern|scal|lam]`` order."""
        names = list(self.var_names)
        names.extend(self.gamma_names)
        rc_names = [self.var_names[int(p)] for p in spec.mixpos]
        for i in range(spec.nrndcoef):
            for j in range(i + 1, spec.nrndcoef):
                names.append(f"corr[{rc_names[i]},{rc_names[j]}]")
        names.append("sig_ker")
        names.extend(f"sd[{nm}]" for nm in rc_names)
        names.extend(f"lam[{nm}]" for nm in rc_names)
        return names

    def _default_startb(self, spec: MixingSpec, layout: ParamLayout) -> NDArray:
        """Zeros start with the outside-good gamma pinned (GAUSS ``b`` default)."""
        theta = np.zeros(layout.n_theta, dtype=np.float64)
        sl = layout.slices()
        gslice = sl["gamma"]
        # GAUSS: gamma = -1000 | zeros(nvargam-1); the outside-good satiation is
        # fixed. We keep it pinned at the control's outside_good_gamma.
        theta[gslice.start] = self.control.outside_good_gamma
        return theta

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def _fit(self) -> MDCEVMixedResults:
        """Estimate the mixed MDCEV model and return :class:`MDCEVMixedResults`."""
        ctrl = self.control
        t_start = time.time()

        spec, layout = self._build_spec_layout()
        panel = PanelIndex.from_ids(self.person_ids)
        est = self.build_estimator(spec, layout, panel)

        # cache the assembled engine + spec/layout so post-estimation predict /
        # ate / forecast can rebuild the shared prediction context (the fitted
        # estimation-space theta is cached after optimisation, below).
        self._fitted_spec = spec
        self._fitted_layout = layout
        self._fitted_est = est

        n_theta = layout.n_theta
        sl = layout.slices()
        if ctrl.startb is not None:
            theta0 = np.asarray(ctrl.startb, dtype=np.float64)
            if theta0.shape[0] != n_theta:
                raise ValueError(
                    f"startb has length {theta0.shape[0]}, expected {n_theta}"
                )
        else:
            theta0 = self._default_startb(spec, layout)

        # The outside-good satiation parameter is fixed (GAUSS _max_active pins
        # gamma[0]); freeze it during optimization via a projected objective.
        gamma0 = sl["gamma"].start
        fixed_val = theta0[gamma0]

        def objective(free: NDArray):
            theta = free.copy()
            theta[gamma0] = fixed_val
            neg_ll, neg_grad = est.objective(theta)
            neg_grad = np.asarray(neg_grad, dtype=np.float64).copy()
            neg_grad[gamma0] = 0.0
            return neg_ll, neg_grad

        if ctrl.verbose >= 1:
            print(
                f"  Mixed MDCEV ({ctrl.utility}) estimation: {panel.n_obs} obs, "
                f"{panel.n_ind} individuals, {n_theta} parameters, {self.nc} "
                f"alternatives, {spec.nrndcoef} random coef(s), nrep={ctrl.n_rep}"
            )

        method = "BFGS" if ctrl.optimizer == "bfgs" else "L-BFGS-B"
        options: dict = {"maxiter": ctrl.maxiter, "gtol": ctrl.tol,
                         "disp": ctrl.verbose >= 2}
        if method == "L-BFGS-B":
            options["ftol"] = ctrl.tol

        res = sopt.minimize(
            objective, theta0, jac=True, method=method, options=options,
        )
        theta_hat = np.asarray(res.x, dtype=np.float64)
        theta_hat[gamma0] = fixed_val
        # cache the fitted estimation-space vector for predict / ate / forecast.
        self._fitted_theta = theta_hat

        ll_pi, score = est.simulated_loglik(theta_hat, want_grad=True)
        ll_total = float(ll_pi.sum())
        ll_mean = ll_total / panel.n_ind
        g_sum = np.asarray(score, dtype=np.float64).sum(0)
        grad_norm = float(np.linalg.norm(np.delete(g_sum, gamma0)))
        converged = bool(res.success) or grad_norm < ctrl.tol_check

        params_report = self._to_reporting(theta_hat, spec, layout, est.space)
        if ctrl.want_covariance:
            if ctrl.se_method.lower() != "bhhh":
                raise ValueError(
                    "Mixed MDCEV currently supports only se_method='bhhh'"
                )
            S = np.asarray(score, dtype=np.float64)
            # drop the frozen outside-good gamma column from the score.
            keep = [j for j in range(n_theta) if j != gamma0]
            S_free = S[:, keep]
            B = S_free.T @ S_free
            try:
                cov_free = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                cov_free = np.linalg.pinv(B)
            cov_est = np.zeros((n_theta, n_theta), dtype=np.float64)
            for ii, pi in enumerate(keep):
                for jj, pj in enumerate(keep):
                    cov_est[pi, pj] = cov_free[ii, jj]
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

        return MDCEVMixedResults(
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
    # post-estimation surface (declared for interface conformance)
    # ------------------------------------------------------------------

    def predict(
        self,
        data: "pd.DataFrame | None" = None,
        *,
        scenario: "dict | None" = None,
        draws=None,
        xp=None,
    ) -> NDArray:
        """Draw-integrated, sample-averaged predicted participation shares.

        Delegates to :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast.mdcev_mixed_predict`,
        which lifts the shipped fixed-coefficient MDCEV participation prediction
        over the mixing draws via the shared
        :func:`pybhatlib.mixed._predict.mixed_predict_shares` machinery.  The
        signature follows the harmonized mixed-family convention
        (``data`` / ``scenario`` / ``draws`` / ``xp``): the number of mixing
        replications comes from ``control.n_rep`` (or the supplied ``draws``
        source), not a per-call draw count.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Dataset to predict on; defaults to the model's own data.
        scenario : dict, optional
            A single scenario's ``{column: scalar | source_column}`` overrides.
        draws : DrawSource, optional
            Override the fit-identical mixing-draw source.
        xp : backend, optional
            Array backend used to wrap the result. Defaults to NumPy.

        Returns
        -------
        NDArray, shape (nc,)
            Sample-averaged participation share per alternative.
        """
        from pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast import (
            mdcev_mixed_predict,
        )

        self._require_results()
        return mdcev_mixed_predict(
            self, data, scenario=scenario, draws=draws, xp=xp
        )

    def forecast(
        self,
        data: "pd.DataFrame | None" = None,
        *,
        scenario: "dict | None" = None,
        budget: "NDArray | None" = None,
        budget_col: str = "tot",
        n_replications: int = 200,
        seed: int = 1234,
        num_outside: int = 1,
        xp=None,
    ) -> NDArray:
        """Draw-integrated per-observation expenditure-allocation forecast.

        Delegates to :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast.mdcev_mixed_forecast`,
        lifting the shipped fixed-coefficient
        :func:`pybhatlib.models.mdcev._mdcev_forecast.mdcev_forecast` over the
        mixing draws.

        Returns
        -------
        NDArray, shape (n_obs, nc)
            Mean expenditure allocation per observation and good.
        """
        from pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast import (
            mdcev_mixed_forecast,
        )

        self._require_results()
        return mdcev_mixed_forecast(
            self, data, scenario=scenario, budget=budget, budget_col=budget_col,
            n_replications=n_replications, seed=seed, num_outside=num_outside,
            xp=xp,
        )

    def ate(
        self,
        *,
        scenarios=None,
        data: "pd.DataFrame | None" = None,
        draws=None,
        alternative_names: "list[str] | None" = None,
        xp=None,
        **kwargs,
    ):
        """Draw-integrated average treatment effects across scenarios.

        Delegates to :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_ate.mdcev_mixed_ate`
        (a wrapper over the shared :func:`pybhatlib.mixed._predict.mixed_ate`),
        returning a :class:`pybhatlib.mixed._predict.MixedATEResult` with the
        harmonized ``predicted_shares`` / ``shares_per_scenario`` fields and the
        ``.comparison()`` / ``.summary()`` surface.  Keyword-only (no positional
        ``data`` / design argument), matching the harmonized mixed-family facade
        (``scenarios`` / ``draws`` / ``xp``); the number of mixing replications
        comes from ``control.n_rep`` (or the supplied ``draws`` source).

        Parameters
        ----------
        scenarios : dict or pd.DataFrame
            Scenario specification (required).
        data : pd.DataFrame, optional
            Dataset the scenarios override; defaults to the model's own data.
        draws : DrawSource, optional
            Override the fit-time mixing-draw source (shared across scenarios).
        alternative_names : list of str, optional
            Output labels; defaults to the model's ``alternatives``.
        xp : backend, optional
            Array backend used to wrap the result arrays. Defaults to NumPy.
        **kwargs
            Reserved for cross-family signature compatibility; unused.

        Returns
        -------
        MixedATEResult
        """
        from pybhatlib.models.mdcev_mixed._mdcev_mixed_ate import mdcev_mixed_ate

        self._require_results()
        if scenarios is None:
            raise ValueError("MDCEVMixedModel.ate() requires scenarios=")
        names = alternative_names or list(self.alternatives)
        return mdcev_mixed_ate(
            self, data, scenarios=scenarios,
            alternative_names=names, draws=draws, xp=xp,
        )
