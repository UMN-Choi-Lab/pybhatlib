"""MNP model class: the main user-facing interface.

Provides a Pythonic API matching BHATLIB's mnpFit procedure.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import get_backend
from pybhatlib.io._data_loader import load_data
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.mnp._mnp_control import MNPControl
from pybhatlib.models.mnp._mnp_loglik import count_params, mnp_loglik
from pybhatlib.models.mnp._mnp_results import MNPResults


class MNPModel(BaseModel):
    """Multinomial Probit model.

    Supports IID, flexible covariance, random coefficients, and
    mixture-of-normals specifications.

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
    mix : bool
        Whether to include random coefficients.
    ranvars : list of str or str or None
        Names of random coefficient variables (must be in var_names/spec keys).
    control : MNPControl or None
        Estimation control structure.

    Examples
    --------
    >>> model = MNPModel(
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
    ...     control=MNPControl(iid=True),
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
        mix: bool = False,
        ranvars: list[str] | str | None = None,
        control: MNPControl | None = None,
    ):
        self.control = control or MNPControl()

        # Override mix from constructor
        if mix:
            self.control.mix = True

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
        else:
            avail_cols = availability if isinstance(availability, list) else [availability]
            self.avail = self.data[avail_cols].values.astype(np.float64)

        # Parse spec
        if spec is not None:
            self.X, self.var_names = parse_spec(
                spec, self.data, self.alternatives, nseg=1
            )
        else:
            raise ValueError("spec is required")

        if var_names is not None:
            self.var_names = var_names

        self.n_beta = len(self.var_names)

        # Parse random variables
        if isinstance(ranvars, str):
            ranvars = [ranvars]

        if ranvars is not None and len(ranvars) > 0:
            self.control.mix = True
            self.ranvar_indices = []
            for rv in ranvars:
                if rv in self.var_names:
                    self.ranvar_indices.append(self.var_names.index(rv))
                else:
                    raise ValueError(
                        f"Random variable '{rv}' not found in var_names: {self.var_names}"
                    )
        else:
            self.ranvar_indices = None

        # Extract choice vector y (0-based index of chosen alternative)
        self._build_choice_vector()

        # Compute number of parameters
        self.n_params = count_params(
            self.n_beta, self.n_alts, self.control, self.ranvar_indices
        )

    def _build_choice_vector(self) -> None:
        """Extract choice vector from data."""
        choice_data = self.data[self.alternatives].values.astype(np.float64)
        self.y = np.argmax(choice_data, axis=1).astype(np.int64)
        self.N = len(self.y)

    def fit(self) -> MNPResults:
        """Estimate the MNP model.

        Returns
        -------
        results : MNPResults
            Estimation results.
        """
        xp = get_backend("numpy")

        if self.control.seed is not None:
            from pybhatlib.utils._seeds import set_seed
            set_seed(self.control.seed)

        if self.control.verbose >= 1:
            print(f"Estimating MNP model with {self.N} observations, "
                  f"{self.n_alts} alternatives, {self.n_params} parameters")
            if self.control.iid:
                print("  Error structure: IID")
            else:
                print("  Error structure: Flexible covariance")
            if self.control.mix:
                print(f"  Random coefficients: {len(self.ranvar_indices or [])} variables")
            if self.control.nseg > 1:
                print(f"  Mixture of normals: {self.control.nseg} segments")

        # Starting values
        if self.control.startb is not None:
            theta0 = self.control.startb.copy()
        else:
            theta0 = self._default_start_values()

        # Define objective function
        def objective(theta):
            return mnp_loglik(
                theta, self.X, self.y, self.avail,
                self.n_alts, self.n_beta, self.control,
                self.ranvar_indices, return_gradient=True, xp=xp,
            )

        # Optimize
        start_time = time.time()

        from pybhatlib.optim._scipy_optim import minimize_scipy

        opt_method = "BFGS" if self.control.optimizer == "bfgs" else "L-BFGS-B"

        result = minimize_scipy(
            objective,
            theta0,
            method=opt_method,
            maxiter=self.control.maxiter,
            tol=self.control.tol,
            verbose=self.control.verbose,
            jac=True,
        )

        elapsed = (time.time() - start_time) / 60.0  # minutes

        # Extract results
        theta_hat = result.x
        hess_inv = result.hess_inv

        # BHATLIB-normalized reporting
        param_names = self._build_report_names()
        (b_report, se, t_stat, p_value,
         cov_report, corr_report, g_report) = self._normalize_for_reporting(
            theta_hat, hess_inv, result.grad,
        )

        # Extract normalized structural parameters
        lambda_hat = None
        omega_hat = None
        cholesky_L = None
        segment_probs = None

        from pybhatlib.models.mnp._mnp_loglik import (
            _build_lambda,
            _build_omega_cholesky,
            _unpack_params,
        )

        params = _unpack_params(
            theta_hat, self.n_beta, self.n_alts,
            self.control, self.ranvar_indices,
        )
        Lambda = _build_lambda(
            params.get("lambda_params"), self.n_alts, self.control,
        )
        dim = self.n_alts - 1

        # Compute sigma_1 for normalizing structural parameters
        if self.control.iid:
            sigma_1_sq = 2.0
        else:
            Sigma_diff = np.ones((dim, dim)) + Lambda + np.eye(dim)
            sigma_1_sq = Sigma_diff[0, 0]

        if not self.control.iid and "lambda_params" in params:
            Sigma_diff = np.ones((dim, dim)) + Lambda + np.eye(dim)
            lambda_hat = Sigma_diff / sigma_1_sq  # normalized

        if self.control.mix and self.ranvar_indices and "omega_params" in params:
            L_raw = _build_omega_cholesky(
                params["omega_params"], self.ranvar_indices, self.control,
            )
            cholesky_L = L_raw / np.sqrt(sigma_1_sq)
            omega_hat = cholesky_L @ cholesky_L.T

        if self.control.nseg > 1:
            seg_params = params.get("segment_params", np.array([]))
            if len(seg_params) > 0:
                raw = np.concatenate([[0.0], seg_params])
                raw -= raw.max()
                segment_probs = np.exp(raw) / np.exp(raw).sum()

        return MNPResults(
            b=theta_hat,
            b_original=b_report,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_report,
            ll=-result.fun,
            ll_total=-result.fun * self.N,
            n_obs=self.N,
            param_names=param_names,
            corr_matrix=corr_report,
            cov_matrix=cov_report,
            n_iterations=result.n_iter,
            convergence_time=elapsed,
            converged=result.converged,
            return_code=result.return_code,
            lambda_hat=lambda_hat,
            omega_hat=omega_hat,
            cholesky_L=cholesky_L,
            segment_probs=segment_probs,
            control=self.control,
            data_path=self.data_path,
        )

    def _default_start_values(self) -> np.ndarray:
        """Generate reasonable starting values for optimization."""
        theta0 = np.zeros(self.n_params, dtype=np.float64)

        # Small random perturbation for betas
        theta0[:self.n_beta] = np.random.randn(self.n_beta) * 0.1
        idx = self.n_beta

        if not self.control.iid:
            dim = self.n_alts - 1
            # Scales: start at 0 (exp(0)=1)
            n_scale = dim
            theta0[idx:idx + n_scale] = 0.0
            idx += n_scale
            # Correlations: start at moderate positive correlation
            # theta=-0.5 maps (via logistic) to ~0.4 average off-diagonal correlation,
            # similar to GAUSS 0.5*I + 0.5*ones initial correlation structure
            n_corr = dim * (dim - 1) // 2
            if not self.control.heteronly and n_corr > 0:
                theta0[idx:idx + n_corr] = -0.5
                idx += n_corr

        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                theta0[idx:idx + n_rand] = 0.0
                idx += n_rand
            else:
                n_omega = n_rand * (n_rand + 1) // 2
                theta0[idx:idx + n_omega] = 0.0
                # Set diagonal elements to small positive values
                chol_idx = 0
                for i in range(n_rand):
                    for j in range(i + 1):
                        if i == j:
                            theta0[idx + chol_idx] = 0.1
                        chol_idx += 1
                idx += n_omega

        if self.control.nseg > 1:
            # Segment params: start at 0 (equal probabilities)
            n_seg = self.control.nseg - 1
            theta0[idx:idx + n_seg] = 0.0
            idx += n_seg
            # Extra segment betas and omegas
            for _ in range(1, self.control.nseg):
                theta0[idx:idx + self.n_beta] = theta0[:self.n_beta] + np.random.randn(self.n_beta) * 0.05
                idx += self.n_beta
                if self.control.mix and self.ranvar_indices is not None:
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        theta0[idx:idx + n_rand] = 0.0
                        idx += n_rand
                    else:
                        n_omega = n_rand * (n_rand + 1) // 2
                        theta0[idx:idx + n_omega] = 0.0
                        idx += n_omega

        return theta0

    def _build_param_names(self) -> list[str]:
        """Build descriptive parameter names."""
        names = list(self.var_names)

        if not self.control.iid:
            dim = self.n_alts - 1
            for i in range(dim):
                names.append(f"scale{i + 1:02d}")
            if not self.control.heteronly:
                for i in range(dim):
                    for j in range(i + 1, dim):
                        names.append(f"parker{i + 1:02d}")

        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    rv_name = self.var_names[self.ranvar_indices[i]]
                    names.append(f"CovCOv{i + 1:02d}")
            else:
                idx = 0
                for i in range(n_rand):
                    for j in range(i + 1):
                        names.append(f"CovCOv{idx + 1:02d}")
                        idx += 1

        if self.control.nseg > 1:
            for h in range(1, self.control.nseg):
                names.append(f"segunpar")
            for h in range(1, self.control.nseg):
                for vn in self.var_names:
                    names.append(f"{vn}_s{h + 1}")
                if self.control.mix and self.ranvar_indices is not None:
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        for i in range(n_rand):
                            names.append(f"CovCOv{i + 1:02d}_s{h + 1}")
                    else:
                        idx = 0
                        for i in range(n_rand):
                            for j in range(i + 1):
                                names.append(f"CovCOv{idx + 1:02d}_s{h + 1}")
                                idx += 1

        # Pad if needed
        while len(names) < self.n_params:
            names.append(f"param{len(names) + 1}")

        return names[:self.n_params]

    def _unparametrize(self, theta: np.ndarray) -> np.ndarray:
        """Convert parametrized values to unparametrized (original scale)."""
        b_orig = theta.copy()
        idx = self.n_beta

        if not self.control.iid:
            dim = self.n_alts - 1
            # Scales: exp transform
            for i in range(dim):
                b_orig[idx + i] = np.exp(theta[idx + i])
            idx += dim
            if not self.control.heteronly:
                n_corr = dim * (dim - 1) // 2
                # Correlations: keep as-is (theta params)
                idx += n_corr

        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    b_orig[idx + i] = np.exp(theta[idx + i])
                idx += n_rand
            else:
                n_omega = n_rand * (n_rand + 1) // 2
                # Cholesky elements: keep as-is
                idx += n_omega

        return b_orig

    # ------------------------------------------------------------------
    # BHATLIB-style normalized reporting
    # ------------------------------------------------------------------

    def _theta_to_report(self, theta: np.ndarray) -> np.ndarray:
        """Map parametrized theta to BHATLIB reporting convention.

        BHATLIB normalizes the differenced error covariance so that
        Sigma_diff[0,0] = 1, absorbing one scale parameter for non-IID
        models.  All betas and random-coefficient Cholesky elements are
        divided by sigma_1 = sqrt(Sigma_diff[0,0]).
        """
        from pybhatlib.models.mnp._mnp_loglik import (
            _build_lambda,
            _build_omega_cholesky,
            _unpack_params,
        )

        params = _unpack_params(
            theta, self.n_beta, self.n_alts, self.control,
            self.ranvar_indices,
        )
        Lambda = _build_lambda(
            params.get("lambda_params"), self.n_alts, self.control,
        )
        dim = self.n_alts - 1

        # Differenced kernel error covariance
        if self.control.iid:
            Sigma_diff = np.ones((dim, dim)) + np.eye(dim)
        else:
            Sigma_diff = np.ones((dim, dim)) + Lambda + np.eye(dim)

        sigma_1 = np.sqrt(Sigma_diff[0, 0])
        report: list[float] = []

        # --- Betas ---
        report.extend(params["beta"] / sigma_1)

        # --- Covariance parameters (non-IID) ---
        if not self.control.iid:
            Sigma_norm = Sigma_diff / (sigma_1 ** 2)
            d_norm = np.sqrt(np.diag(Sigma_norm))
            Corr_norm = Sigma_norm / np.outer(d_norm, d_norm)

            # Parker (correlations, upper triangle row-by-row)
            if not self.control.heteronly:
                for i in range(dim):
                    for j in range(i + 1, dim):
                        report.append(float(Corr_norm[i, j]))

            # Relative scales (indices 1 .. dim-1; index 0 is normalized to 1)
            for i in range(1, dim):
                report.append(float(d_norm[i]))

        # --- Random-coefficient Cholesky (normalized) ---
        if (
            self.control.mix
            and self.ranvar_indices is not None
            and "omega_params" in params
        ):
            L = _build_omega_cholesky(
                params["omega_params"], self.ranvar_indices, self.control,
            )
            L_norm = L / sigma_1
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    report.append(float(L_norm[i, i]))
            else:
                for i in range(n_rand):
                    for j in range(i + 1):
                        report.append(float(L_norm[i, j]))

        # --- Segment parameters ---
        if self.control.nseg > 1:
            seg_params = params.get("segment_params", np.array([]))
            report.extend(seg_params.tolist())

            for h in range(1, self.control.nseg):
                if h - 1 < len(params.get("segment_betas", [])):
                    report.extend(
                        (params["segment_betas"][h - 1] / sigma_1).tolist()
                    )
                if (
                    self.control.mix
                    and self.ranvar_indices is not None
                    and h - 1 < len(params.get("segment_omegas", []))
                ):
                    L_h = _build_omega_cholesky(
                        params["segment_omegas"][h - 1],
                        self.ranvar_indices,
                        self.control,
                    )
                    L_h_norm = L_h / sigma_1
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        for i in range(n_rand):
                            report.append(float(L_h_norm[i, i]))
                    else:
                        for i in range(n_rand):
                            for j in range(i + 1):
                                report.append(float(L_h_norm[i, j]))

        return np.array(report, dtype=np.float64)

    def _build_report_names(self) -> list[str]:
        """Build parameter names for BHATLIB reporting convention.

        Compared to ``_build_param_names`` (theta-space), this drops one
        scale parameter (absorbed by normalization) and reorders
        covariance parameters as parker (correlations) then scale
        (relative scales).
        """
        names = list(self.var_names)
        dim = self.n_alts - 1

        if not self.control.iid:
            # Parker (correlations)
            if not self.control.heteronly:
                idx = 0
                for i in range(dim):
                    for j in range(i + 1, dim):
                        idx += 1
                        names.append(f"parker{idx:02d}")

            # Relative scales (indices 1..dim-1)
            for i in range(1, dim):
                names.append(f"scale{i:02d}")

        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    names.append(f"CovCOv{i + 1:02d}")
            else:
                idx = 0
                for i in range(n_rand):
                    for j in range(i + 1):
                        idx += 1
                        names.append(f"CovCOv{idx:02d}")

        if self.control.nseg > 1:
            for _h in range(1, self.control.nseg):
                names.append("segunpar")
            for h in range(1, self.control.nseg):
                for vn in self.var_names:
                    names.append(f"{vn}_s{h + 1}")
                if self.control.mix and self.ranvar_indices is not None:
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        for i in range(n_rand):
                            names.append(f"CovCOv{i + 1:02d}_s{h + 1}")
                    else:
                        idx = 0
                        for i in range(n_rand):
                            for j in range(i + 1):
                                idx += 1
                                names.append(f"CovCOv{idx:02d}_s{h + 1}")

        return names

    def _normalize_for_reporting(
        self,
        theta_hat: np.ndarray,
        hess_inv: np.ndarray | None,
        gradient: np.ndarray | None,
    ) -> tuple:
        """Compute BHATLIB-normalized values with delta-method standard errors.

        Returns
        -------
        b_report : ndarray  — normalized coefficient estimates
        se : ndarray         — delta-method standard errors
        t_stat : ndarray     — b_report / se
        p_value : ndarray    — two-sided p-values
        cov_report : ndarray — covariance matrix of reporting params
        corr_report : ndarray — correlation matrix of reporting params
        g_report : ndarray   — gradient projected into reporting space
        """
        b_report = self._theta_to_report(theta_hat)
        n_report = len(b_report)
        n_theta = len(theta_hat)

        # Numerical Jacobian  J[k, i] = d(report_k)/d(theta_i)
        eps = 1e-7
        J = np.zeros((n_report, n_theta), dtype=np.float64)
        for i in range(n_theta):
            theta_p = theta_hat.copy()
            theta_p[i] += eps
            theta_m = theta_hat.copy()
            theta_m[i] -= eps
            J[:, i] = (
                self._theta_to_report(theta_p) - self._theta_to_report(theta_m)
            ) / (2.0 * eps)

        # Delta method:  Cov(report) = J @ Cov(theta) @ J^T
        if hess_inv is not None:
            cov_theta = hess_inv / self.N
            cov_report = J @ cov_theta @ J.T
            se = np.sqrt(np.maximum(np.diag(cov_report), 0.0))
        else:
            cov_report = np.eye(n_report)
            se = np.full(n_report, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, b_report / se, 0.0)
            p_value = 2.0 * (1.0 - norm.cdf(np.abs(t_stat)))

        # Correlation matrix of reporting parameters
        if hess_inv is not None:
            se_outer = np.outer(se, se)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_report = np.where(
                    se_outer > 0, cov_report / se_outer, 0.0,
                )
            np.fill_diagonal(corr_report, 1.0)
        else:
            corr_report = np.eye(n_report)

        # Gradient projected into reporting space via pseudo-inverse
        if gradient is not None:
            g_report = np.linalg.pinv(J).T @ gradient
        else:
            g_report = np.zeros(n_report)

        return b_report, se, t_stat, p_value, cov_report, corr_report, g_report
