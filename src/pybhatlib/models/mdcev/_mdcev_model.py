"""MDCEV model estimation.

Implements the full estimation procedure from estimation_TradMDCEV.gss
and Estimation_LinearMDCEV.gss (Bhat, 2008 / 2018), including:

  - Data loading from a file path (CSV or GAUSS .dat)
  - Data preparation and index construction
  - Starting value construction matching GAUSS convention
  - Active-parameter mask replicating _max_active
  - Two-phase optimisation (log-sigma → natural-sigma reporting)
  - Cross-product covariance and delta-method standard errors
  - Results packaging into MDCEVResults

The ``utility`` attribute of ``MDCEVControl`` selects between
``"trad"`` (traditional) and ``"linear"`` (linear outside-good)
specifications without requiring separate model files.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr as _ndtr

from pybhatlib.io._data_loader import load_data
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.models.mdcev._mdcev_loglik import (
    mdcev_loglik, mdcev_gradient, mdcev_loglik_unpar, mdcev_gradient_unpar, numerical_hessian
)
from pybhatlib.models.mdcev._mdcev_results import MDCEVResults
from pybhatlib.optim._scipy_optim import minimize_scipy


class MDCEVModel:
    """Multiple Discrete-Continuous Extreme Value (MDCEV) Model.

    Supports both the traditional (Bhat 2008) and linear outside-good
    (Bhat 2018) utility specifications via ``MDCEVControl.utility``.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to data file or DataFrame.
    alternatives : list of str
        Column names for consumption quantities (first entry must be
        the outside/numeraire good).
    availability : list of str or None
        Column names for price variables. Pass None to default to ones
        for all alternatives.
    utility_spec : dict or NDArray
        Specification for baseline utility (psi). Can be a dict mapping
        parameter names to dicts (like MNL/MNP spec format) or an
        NDArray of shape (nc, nvarm) with column names.
    gamma_spec : dict or NDArray
        Specification for satiation (gamma) utility. Same format as
        utility_spec.
    param_names : list of str, optional
        Names for utility parameters. Inferred from utility_spec keys if None.
    gamma_names : list of str, optional
        Names for satiation parameters. Inferred from gamma_spec keys if None.
    control : MDCEVControl, optional
        Estimation control settings. If None, defaults are used.
    obs_id_var : str
        Column name for observation identifier. Default: "ID".

    Examples
    --------
    >>> ctrl = MDCEVControl(utility="trad", maxiter=2000)
    >>> model = MDCEVModel(
    ...     data="Workshop_SCAG_Est.csv",
    ...     alternatives=["alt_out", "Esc", "Ho", "Soc", "AR", "Eo"],
    ...     availability=None,
    ...     utility_spec={"ASC_Esc": {...}, ...},
    ...     gamma_spec={"G_Out": {...}, ...},
    ...     control=ctrl,
    ... )
    >>> results = model.fit()
    >>> results.summary()
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        alternatives: list[str],
        availability: list[str] | None = None,
        utility_spec: dict | NDArray | None = None,
        gamma_spec: dict | NDArray | None = None,
        param_names: list[str] | None = None,
        gamma_names: list[str] | None = None,
        control: MDCEVControl | None = None,
        obs_id_var: str = "ID",
    ) -> None:
        self.control = control or MDCEVControl()
        self.obs_id_var = obs_id_var

        # Load data
        if isinstance(data, str):
            self.data_path = data
            self.data = load_data(data)
        else:
            self.data_path = "<DataFrame>"
            self.data = data.copy()

        self.alternatives = alternatives
        self.n_alts = len(alternatives)

        # Store availability
        if availability is None:
            self.availability = ["uno"] * self.n_alts
        else:
            self.availability = availability

        # Convert dict specs to NDArray format if needed
        if isinstance(utility_spec, dict):
            self.utility_spec, self.param_names = self._spec_dict_to_array(
                utility_spec, alternatives
            )
        else:
            self.utility_spec = utility_spec
            self.param_names = param_names or [f"param_{i}" for i in range(utility_spec.shape[1])]

        if isinstance(gamma_spec, dict):
            self.gamma_spec, self.gamma_names = self._spec_dict_to_array(
                gamma_spec, alternatives
            )
        else:
            self.gamma_spec = gamma_spec
            self.gamma_names = gamma_names or [f"gamma_{i}" for i in range(gamma_spec.shape[1])]

        # Override with explicit names if provided
        if param_names is not None:
            self.param_names = param_names
        if gamma_names is not None:
            self.gamma_names = gamma_names

        # Ensure special columns exist
        _ensure_special_cols(self.data)

    @staticmethod
    def _spec_dict_to_array(
        spec_dict: dict,
        alternatives: list[str],
    ) -> tuple[NDArray, list[str]]:
        """Convert a spec dict (like MNL/MNP format) to NDArray format.

        Parameters
        ----------
        spec_dict : dict
            Mapping of parameter names to alternative-specific specs.
            E.g., {"ASC_Esc": {"alt_out": "sero", "Esc": "uno", ...}, ...}
        alternatives : list of str
            List of alternative names in order.

        Returns
        -------
        spec_array : NDArray, shape (nc, nvarm)
            Specification matrix.
        param_names : list of str
            Parameter names (keys from spec_dict).
        """
        param_names = list(spec_dict.keys())
        nc = len(alternatives)
        nvarm = len(param_names)

        spec_array = np.empty((nc, nvarm), dtype=object)
        for j, param_name in enumerate(param_names):
            param_spec = spec_dict[param_name]
            for i, alt_name in enumerate(alternatives):
                spec_array[i, j] = param_spec.get(alt_name, "sero")

        return spec_array.astype(str), param_names

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> MDCEVResults:
        """Fit the MDCEV model.

        Returns
        -------
        MDCEVResults
        """
        ctrl    = self.control
        t_start = time.time()

        # ---- Load and prepare data -----------------------------------
        df = self.data.copy()
        _ensure_special_cols(df)

        nc       = self.n_alts
        nvarm    = self.utility_spec.shape[1]
        nvargam  = self.gamma_spec.shape[1]
        eqmatgam = np.eye(nvargam, dtype=np.float64)

        dta, ivm, ivg, flagchm, flagprcm, wtind = _build_data_arrays(
            df, self.alternatives, self.availability, self.utility_spec, self.gamma_spec,
            ctrl.weight_var,
        )

        n_obs = dta.shape[0]

        # ---- Starting values -----------------------------------------
        # GAUSS: b = zeros(nvarm,1) | -1000 | zeros(nvargam-1,1) | 0
        # _max_active pins the outside-good gamma (index nvarm) to zero.
        if ctrl.startb is not None:
            b0 = np.asarray(ctrl.startb, dtype=np.float64)
        else:
            b0 = np.zeros(nvarm + nvargam + 1, dtype=np.float64)
            b0[nvarm] = ctrl.outside_good_gamma

        # Active-parameter mask: beta all active, outside-good gamma
        # fixed, remaining gammas active, log_sigma active.
        # GAUSS: _max_active = ones(nvarm)|zeros(1)|ones(nvargam-1)|1
        active = np.ones(nvarm + nvargam + 1, dtype=bool)
        active[nvarm] = False

        # ---- Objective and gradient ----------------------------------
        def neg_ll(x_active: NDArray) -> float:
            x = _embed(x_active, b0, active)
            return -float(mdcev_loglik(
                x, dta, ivm, ivg, flagchm, flagprcm, wtind,
                nvarm, nvargam, nc, eqmatgam, ctrl,
            ).mean())

        def neg_grad(x_active: NDArray) -> NDArray:
            x = _embed(x_active, b0, active)
            g_full = mdcev_gradient(
                x, dta, ivm, ivg, flagchm, flagprcm, wtind,
                nvarm, nvargam, nc, eqmatgam, ctrl,
            ).mean(axis=0)
            return -g_full[active]

        b0_active = b0[active]

        if ctrl.verbose >= 1:
            print(
                f"  MDCEV ({ctrl.utility}) estimation: {n_obs} obs, "
                f"{b0_active.size} free parameters, {nc} alternatives"
            )

        
        # ---- Optimise ------------------------------------------------
        method = "BFGS" if ctrl.optimizer == "bfgs" else "L-BFGS-B"

        if ctrl.analytic_grad:
            def neg_ll_with_grad(x_active: NDArray):
                return neg_ll(x_active), neg_grad(x_active)

            res = minimize_scipy(
                neg_ll_with_grad,
                b0_active,
                method=method,
                maxiter=ctrl.maxiter,
                tol=ctrl.tol,
                verbose=ctrl.verbose,
                jac=True,
            )
        else:
            res = minimize_scipy(
                neg_ll,
                b0_active,
                method=method,
                maxiter=ctrl.maxiter,
                tol=ctrl.tol,
                verbose=ctrl.verbose,
                jac=False,
            )

        x_opt_active = res.x
        x_opt        = _embed(x_opt_active, b0, active)

        # ---- Recover natural sigma -----------------------------------
        # GAUSS second phase: b = x[1:end-1] | exp(x[end])
        x_reported = x_opt.copy()
        x_reported[nvarm + nvargam] = np.exp(x_opt[nvarm + nvargam])
        sigma_hat = float(x_reported[nvarm + nvargam])

        # ---- Covariance computation -------------------------------
        se_bhhh = se_hessian = se_sandwich = None
        if ctrl.want_covariance:
            x_unpar = x_reported

            # Compute per-observation gradients using analytic gradient function
            g_obs_unpar = mdcev_gradient_unpar(
                x_unpar, dta, ivm, ivg, flagchm, flagprcm, wtind,
                nvarm, nvargam, nc, eqmatgam, ctrl,
            )

            method = ctrl.se_method
            if method == "bhhh" or method == "sandwich":
                g_active_unpar = g_obs_unpar[:, active]
                B = g_active_unpar.T @ g_active_unpar

            if method == "bhhh":
                try:
                    cov_active = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    cov_active = np.linalg.pinv(B)
            else:
                # Hessian and Sandwich methods need Hessian inverse
                def obj_unpar(x):
                    ll_obs = mdcev_loglik_unpar(
                        x, dta, ivm, ivg, flagchm, flagprcm, wtind,
                        nvarm, nvargam, nc, eqmatgam, ctrl,
                    )
                    return -ll_obs.sum()

                # Compute numerical Hessian at the optimum
                hess_unpar = numerical_hessian(obj_unpar, x_unpar)
                try:
                    hess_inv_unpar = np.linalg.inv(hess_unpar)
                except np.linalg.LinAlgError:
                    hess_inv_unpar = np.linalg.pinv(hess_unpar)
                
                hess_active = hess_inv_unpar[np.ix_(active, active)]

                if method == "hessian":
                    cov_active = hess_active
                else:
                    cov_active = hess_active @ B @ hess_active

            n_full = nvarm + nvargam + 1
            cov_full = np.zeros((n_full, n_full))
            idx = np.where(active)[0]
            for ii, pi in enumerate(idx):
                for jj, pj in enumerate(idx):
                    cov_full[pi, pj] = cov_active[ii, jj]

            se_full = np.sqrt(np.maximum(np.diag(cov_full), 0.0))
            cov_reported = cov_full.copy()  # No transformation needed

            if method == "bhhh":
                se_bhhh = se_full
            elif method == "hessian":
                se_hessian = se_full
            else:
                se_sandwich = se_full

            with np.errstate(invalid="ignore"):
                corr = cov_reported / np.outer(se_full, se_full)
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            se_full = np.zeros(nvarm + nvargam + 1)
            cov_reported = np.zeros((nvarm + nvargam + 1, nvarm + nvargam + 1))
            corr = np.zeros_like(cov_reported)

        # ---- Final LL and gradient -----------------------------------
        ll_obs_final = mdcev_loglik(
            x_opt, dta, ivm, ivg, flagchm, flagprcm, wtind,
            nvarm, nvargam, nc, eqmatgam, ctrl,
        )
        ll_total = float(ll_obs_final.sum())
        ll_mean  = ll_total / n_obs

        if ctrl.want_covariance:
            # Use the total gradient from unpar space
            g_reported = g_obs_unpar.sum(axis=0)
        else:
            # Fallback to par space gradient with delta method
            g_final_full = mdcev_gradient(
                x_opt, dta, ivm, ivg, flagchm, flagprcm, wtind,
                nvarm, nvargam, nc, eqmatgam, ctrl,
            ).sum(axis=0)
            g_reported = g_final_full.copy()
            g_reported[nvarm + nvargam] *= sigma_hat

        t_elapsed = (time.time() - t_start) / 60.0

        # ---- t-stats and p-values ------------------------------------
        with np.errstate(invalid="ignore"):
            t_stat = np.where(se_full > 0, x_reported / se_full, 0.0)
        p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

        all_param_names = list(self.param_names) + list(self.gamma_names) + ["sigma"]

        if ctrl.verbose >= 1:
            converged_msg = "converged" if res.converged else "NOT converged"
            print(
                f"  Optimisation {converged_msg} in {res.n_iter} iterations "
                f"({t_elapsed:.4f} min).  LL = {ll_total:.4f}"
            )

        return MDCEVResults(
            b=x_opt,
            b_reported=x_reported,
            se=se_full,
            se_bhhh=se_bhhh,
            se_hessian=se_hessian,
            se_sandwich=se_sandwich,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_reported,
            ll=ll_mean,
            ll_total=ll_total,
            n_obs=n_obs,
            param_names=all_param_names,
            corr_matrix=corr,
            cov_matrix=cov_reported,
            n_iterations=res.n_iter,
            convergence_time=t_elapsed,
            converged=res.converged,
            return_code=0 if res.converged else 1,
            sigma=sigma_hat,
            control=ctrl,
            data_path=self.data_path,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _load_data(data_path: str, obs_id_var: str) -> pd.DataFrame:
    """Load CSV data file and sort by observation identifier.

    Uses the centralized load_data utility and sorts by observation ID.
    """
    df = load_data(data_path)
    if obs_id_var in df.columns:
        df = df.sort_values(obs_id_var).reset_index(drop=True)
    return df


def _prepare_dataframe(data: pd.DataFrame, obs_id_var: str) -> pd.DataFrame:
    """Prepare a DataFrame for MDCEV estimation.

    Sorts by observation identifier if the column exists.
    """
    df = data.copy()
    if obs_id_var in df.columns:
        df = df.sort_values(obs_id_var).reset_index(drop=True)
    return df


def _ensure_special_cols(df: pd.DataFrame) -> None:
    """Add ``uno`` (ones) and ``sero`` (zeros) columns if not present.

    Mirrors the dataset requirement documented in both GAUSS files.
    """
    if "uno" not in df.columns:
        df["uno"] = 1.0
    if "sero" not in df.columns:
        df["sero"] = 0.0


def _build_data_arrays(
    df: pd.DataFrame,
    alternatives: Sequence[str],
    avail_cols: Sequence[str],
    utility_spec: NDArray,
    gamma_spec: NDArray,
    weight_var: str | None,
):
    """Convert DataFrame to flat NumPy array with column index vectors.

    Returns
    -------
    dta : NDArray, shape (n_obs, n_cols)
    ivm : NDArray, shape (nc * nvarm,)
        Baseline utility column indices (row-major over alts).
        Mirrors ``{ v,ivm } = indices(dataset,vecr(ivmt))`` in GAUSS.
    ivg : NDArray, shape (nc * nvargam,)
        Satiation column indices (row-major over alts).
        Mirrors ``{ v,ivg } = indices(dataset,vecr(ivgt))`` in GAUSS.
    flagchm : NDArray, shape (nc,)
        Column indices of consumption quantities.
    flagprcm : NDArray, shape (nc,)
        Column indices of price/availability variables.
    wtind : int
        Column index of observation weights.
    """
    nc       = len(alternatives)
    nvarm    = utility_spec.shape[1]
    nvargam  = gamma_spec.shape[1]
    wt_col   = weight_var if weight_var else "uno"

    # Stable ordered column list
    col_order: list[str] = []
    for v in (
        list(alternatives)
        + list(avail_cols)
        + utility_spec.flatten().tolist()
        + gamma_spec.flatten().tolist()
        + [wt_col]
    ):
        if v not in col_order:
            col_order.append(v)

    col_idx  = {name: i for i, name in enumerate(col_order)}
    dta      = df[col_order].to_numpy(dtype=np.float64)

    flagchm  = np.array([col_idx[v] for v in alternatives], dtype=int)
    flagprcm = np.array([col_idx[v] for v in avail_cols],   dtype=int)
    wtind    = col_idx[wt_col]

    ivm = np.array(
        [col_idx[utility_spec[k, j]]
         for j in range(nvarm) for k in range(nc)],
        dtype=int,
    )
    ivg = np.array(
        [col_idx[gamma_spec[k, j]]
         for j in range(nvargam) for k in range(nc)],
        dtype=int,
    )

    return dta, ivm, ivg, flagchm, flagprcm, wtind


def _embed(x_active: NDArray, b0: NDArray, active: NDArray) -> NDArray:
    """Embed active parameters back into the full parameter vector."""
    x = b0.copy()
    x[active] = x_active
    return x
