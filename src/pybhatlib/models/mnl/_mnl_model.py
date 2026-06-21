"""MNL model estimation.

Implements the full estimation procedure from MNLcasenew.gss 
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.optimize as sopt
from numpy.typing import NDArray
from scipy.special import ndtr

from pybhatlib.io._data_loader import load_data
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.mnl._mnl_control import MNLControl
from pybhatlib.models.mnl._mnl_loglik import mnl_loglik, mnl_gradient, mnl_hessian
from pybhatlib.models.mnl._mnl_results import MNLResults


class MNLModel(BaseModel):
    """Multinomial Logit (MNL) Model.

    Parameters
    ----------
    data : str or pd.DataFrame
        Path to data file or DataFrame.
    alternatives : list of str
        Column names for choice indicators (e.g., ["Alt1_ch", "Alt2_ch", "Alt3_ch"]).
    availability : str or list of str
        "none" if all alternatives always available, or list of availability
        column names matching alternatives order.
    spec : dict or None
        Variable specification mapping variable names to alternative-specific
        column names or keywords ("sero"/"uno").
    var_names : list of str or None
        Names for the coefficients (for display). Inferred from spec keys if None.
    control : MNLControl or None
        Estimation control structure.

    Examples
    --------
    >>> model = MNLModel(
    ...     data="TRAVELMODE.csv",
    ...     alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    ...     availability="none",
    ...     spec={
    ...         "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    ...         "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    ...         "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    ...         "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    ...         "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
    ...     },
    ...     control=MNLControl(maxiter=200),
    ... )
    >>> results = model.fit()
    >>> results.summary()
    """

    def __init__(
        self,
        data: str | pd.DataFrame,
        alternatives: list[str],
        availability: str | list[str] = "none",
        spec: dict | None = None,
        var_names: list[str] | None = None,
        control: MNLControl | None = None,
    ):
        self.control = control or MNLControl()

        # Load data
        if isinstance(data, str):
            self.data_path = data
            self.data = load_data(data)
        else:
            self.data_path = "<DataFrame>"
            self.data = data

        self.alternatives = alternatives
        self.n_alts = len(alternatives)

        # Parse availability
        if isinstance(availability, str) and availability.lower() == "none":
            self.avail = None
            self.avail_cols = None
        else:
            avail_cols = availability if isinstance(availability, list) else [availability]
            self.avail_cols = avail_cols
            self.avail = self.data[avail_cols].values.astype(np.float64)

        # Parse spec
        if spec is not None:
            self.spec_dict = spec
            self.X, self._var_names = parse_spec(
                spec, self.data, self.alternatives, nseg=1
            )
        else:
            raise ValueError("spec is required")

        if var_names is not None:
            self._var_names = var_names

        self.var_names = self._var_names
        self.n_beta = len(self.var_names)

        # Extract choice vector y (0-based index of chosen alternative)
        self._build_choice_vector()

    def _build_choice_vector(self) -> None:
        """Extract choice vector from data."""
        choice_data = self.data[self.alternatives].values.astype(np.float64)
        self.y = np.argmax(choice_data, axis=1).astype(np.int64)
        self.N = len(self.y)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> MNLResults:
        """Fit the MNL model.

        Returns
        -------
        MNLResults
        """
        ctrl = self.control
        t_start = time.time()

        # ---- Build flat data matrix and index vectors -----------------
        dta, indxivunord, davunord, dvunord, ddind = self._build_data_arrays()

        nc       = self.n_alts
        numunord = self.n_beta
        n_obs    = dta.shape[0]

        # ---- Starting values (GAUSS: b = zeros(ncoefunord, 1)) -------
        if ctrl.startb is not None:
            b0 = np.asarray(ctrl.startb, dtype=np.float64)
        else:
            b0 = np.zeros(numunord, dtype=np.float64)

        # ---- Objective: negative total LL ----------------------------
        def neg_ll(x: NDArray) -> float:
            ll_obs = mnl_loglik(x, dta, indxivunord, davunord, dvunord, nc, numunord)
            return -float(ll_obs.sum())

        def neg_grad(x: NDArray) -> NDArray:
            g_obs = mnl_gradient(x, dta, indxivunord, davunord, dvunord, nc, numunord)
            return -g_obs.sum(axis=0)

        def neg_hess(x: NDArray) -> NDArray:
            return -mnl_hessian(x, dta, indxivunord, davunord, dvunord, ddind, nc, numunord)

        if ctrl.verbose >= 1:
            print(
                f"  MNL estimation: {n_obs} obs, {numunord} parameters, "
                f"{nc} alternatives"
            )

        # ---- Optimise ------------------------------------------------
        # Map control.optimizer to scipy method name
        method_map = {"newton": "Newton-CG", "bfgs": "BFGS", "lbfgsb": "L-BFGS-B"}
        method = method_map.get(ctrl.optimizer, "Newton-CG")

        # Newton-CG reads the tolerance from "xtol"; BFGS/CG use "gtol".
        tol_key = "xtol" if method == "Newton-CG" else "gtol"
        opt_kwargs: dict = dict(
            fun=neg_ll,
            x0=b0,
            method=method,
            options={"maxiter": ctrl.maxiter, tol_key: ctrl.tol,
                     "disp": ctrl.verbose >= 2},
        )
        if ctrl.analytic_grad:
            opt_kwargs["jac"] = neg_grad
        if ctrl.analytic_hess and ctrl.optimizer == "newton":
            opt_kwargs["hess"] = neg_hess

        res = sopt.minimize(**opt_kwargs)
        x_opt = res.x                                           # (numunord,)

        # ---- Covariance via requested standard-error method -------------
        if ctrl.want_covariance:
            g_obs = mnl_gradient(
                x_opt, dta, indxivunord, davunord, dvunord, nc, numunord,
            )                                                   # (n_obs, numunord)
            se_method = ctrl.se_method.lower()
            if se_method not in ("bhhh", "hessian", "sandwich"):
                raise ValueError(
                    "se_method must be one of 'bhhh', 'hessian', or 'sandwich'"
                )

            B = g_obs.T @ g_obs                                # (numunord, numunord)

            if se_method == "bhhh":
                try:
                    cov = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    cov = np.linalg.pinv(B)
            else:
                hess_obs = -mnl_hessian(
                    x_opt, dta, indxivunord, davunord, dvunord, ddind, nc, numunord,
                )                                               # observed information
                try:
                    hess_inv = np.linalg.inv(hess_obs)
                except np.linalg.LinAlgError:
                    hess_inv = np.linalg.pinv(hess_obs)

                if se_method == "hessian":
                    cov = hess_inv
                else:  # sandwich
                    cov = hess_inv @ B @ hess_inv

            se = np.sqrt(np.maximum(np.diag(cov), 0.0))
            with np.errstate(invalid="ignore"):
                corr = cov / np.outer(se, se)
            corr = np.nan_to_num(corr, nan=0.0)
        else:
            cov = np.zeros((numunord, numunord))
            se = np.zeros(numunord)
            corr = np.zeros((numunord, numunord))

        # ---- Final LL and gradient -----------------------------------
        ll_obs_final = mnl_loglik(
            x_opt, dta, indxivunord, davunord, dvunord, nc, numunord,
        )
        ll_total = float(ll_obs_final.sum())
        ll_mean  = ll_total / n_obs

        g_final = mnl_gradient(
            x_opt, dta, indxivunord, davunord, dvunord, nc, numunord,
        ).mean(axis=0)
        grad_norm = float(np.linalg.norm(g_final))
        converged = grad_norm < ctrl.tol_check

        # ---- t-stats and p-values ------------------------------------
        with np.errstate(invalid="ignore", divide="ignore"):
            t_stat = np.where(se > 0, x_opt / se, 0.0)
        p_value = 2.0 * (1.0 - ndtr(np.abs(t_stat)))

        t_elapsed = (time.time() - t_start) / 60.0

        if ctrl.verbose >= 1:
            converged_msg = "converged" if converged else "NOT converged"
            print(
                f"  Optimisation {converged_msg} in {res.nit} iterations "
                f"({t_elapsed:.4f} min).  LL = {ll_total:.4f}, "
                #f"||grad|| = {grad_norm:.3e}, tol = {ctrl.tol_check:.3e}"
            )

        return MNLResults(
            b=x_opt,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_final,
            ll=ll_mean,
            ll_total=ll_total,
            n_obs=n_obs,
            param_names=self.var_names,
            corr_matrix=corr,
            cov_matrix=cov,
            n_iterations=res.nit,
            convergence_time=t_elapsed,
            converged=converged,
            return_code=0 if converged else 1,
            control=ctrl,
            data_path=self.data_path,
            message=getattr(res, "message", None),
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _build_data_arrays(
        self,
    ):
        """Convert parsed spec and data into flat NumPy arrays with column indices.

        The parsed spec (X) has shape (N, n_alts, n_vars), so we construct
        a flat data array and index vectors to match the MNL loglik interface.

        Returns
        -------
        dta : NDArray, shape (n_obs, n_cols)
        indxivunord : NDArray, shape (nc * numunord,)
            Column indices for specification matrix (row-major over alts).
        davunord : NDArray, shape (nc,)
            Column indices of availability indicators (or -1 if unavail="none").
        dvunord : NDArray, shape (nc,)
            Column indices of choice indicators.
        ddind : int
            Column index of the UNO (ones) variable, used by the Hessian.
        """
        N = self.N
        nc = self.n_alts
        numunord = self.n_beta

        # Build column list: choices + availability + variables
        col_order: list[str] = []
        col_order.extend(self.alternatives)
        
        if self.avail_cols is not None:
            col_order.extend(self.avail_cols)

        # Add variable columns: for each variable and each alternative,
        # we need to track the column (either a special value or a data column)
        var_cols: dict[tuple[int, int], str] = {}  # (var_idx, alt_idx) -> col_name
        for v_idx, var_name in enumerate(self.var_names):
            alt_spec = self.spec_dict[var_name]
            for a_idx, alt_name in enumerate(self.alternatives):
                col_or_kw = alt_spec.get(alt_name, "sero")
                var_cols[(v_idx, a_idx)] = col_or_kw

        # Collect all actual column names (not "sero"/"uno")
        for (v_idx, a_idx), col_or_kw in var_cols.items():
            col_or_kw_lower = str(col_or_kw).strip().lower()
            if col_or_kw_lower not in ("sero", "uno") and col_or_kw not in col_order:
                col_order.append(col_or_kw)

        # Ensure special columns exist
        if "uno" not in col_order:
            col_order.append("uno")
        if "sero" not in col_order:
            col_order.append("sero")

        col_idx = {name: i for i, name in enumerate(col_order)}

        # Build data frame with all needed columns
        df = self.data.copy()
        if "uno" not in df.columns:
            df["uno"] = 1.0
        if "sero" not in df.columns:
            df["sero"] = 0.0

        dta = df[col_order].to_numpy(dtype=np.float64)

        # Build index vectors
        dvunord = np.array([col_idx[v] for v in self.alternatives], dtype=int)
        
        if self.avail is None:
            # "none" availability: all alternatives available
            davunord = np.array([col_idx["uno"]] * nc, dtype=int)
        else:
            davunord = np.array([col_idx[v] for v in self.avail_cols], dtype=int)

        # Build specification indices (row-major over alternatives)
        indxivunord = []
        for k in range(nc):  # for each alternative
            for j in range(numunord):  # for each variable
                col_or_kw = var_cols.get((j, k), "sero")
                col_or_kw_lower = str(col_or_kw).strip().lower()
                if col_or_kw_lower == "sero":
                    idx = col_idx["sero"]
                elif col_or_kw_lower == "uno":
                    idx = col_idx["uno"]
                else:
                    idx = col_idx[col_or_kw]
                indxivunord.append(idx)

        indxivunord = np.array(indxivunord, dtype=int)
        ddind = col_idx.get("uno", 0)

        return dta, indxivunord, davunord, dvunord, ddind
