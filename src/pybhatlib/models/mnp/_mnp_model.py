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
                spec, self.data, self.alternatives, nseg=self.control.nseg
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

        # Build parameter names
        param_names = self._build_param_names()

        # Compute standard errors
        if hess_inv is not None:
            se = np.sqrt(np.abs(np.diag(hess_inv)) / self.N)
        else:
            se = np.full(self.n_params, np.nan)

        # Compute unparametrized coefficients
        b_original = self._unparametrize(theta_hat)

        # t-statistics and p-values
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, b_original / se, 0.0)
            p_value = 2.0 * (1.0 - norm.cdf(np.abs(t_stat)))

        # Gradient at convergence
        gradient = result.grad

        # Correlation matrix of parameters
        if hess_inv is not None:
            se_outer = np.outer(se, se)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_matrix = np.where(se_outer > 0, hess_inv / self.N / se_outer, 0.0)
            np.fill_diagonal(corr_matrix, 1.0)
        else:
            corr_matrix = np.eye(self.n_params)

        # Extract structural parameters
        lambda_hat = None
        omega_hat = None
        cholesky_L = None
        segment_probs = None

        from pybhatlib.models.mnp._mnp_loglik import _unpack_params, _build_lambda, _build_omega_cholesky

        params = _unpack_params(theta_hat, self.n_beta, self.n_alts, self.control, self.ranvar_indices)

        if not self.control.iid and "lambda_params" in params:
            lambda_hat = _build_lambda(params["lambda_params"], self.n_alts, self.control)

        if self.control.mix and self.ranvar_indices and "omega_params" in params:
            cholesky_L = _build_omega_cholesky(params["omega_params"], self.ranvar_indices, self.control)
            omega_hat = cholesky_L @ cholesky_L.T

        return MNPResults(
            b=theta_hat,
            b_original=b_original,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=gradient,
            ll=-result.fun,
            ll_total=-result.fun * self.N,
            n_obs=self.N,
            param_names=param_names,
            corr_matrix=corr_matrix,
            cov_matrix=hess_inv / self.N if hess_inv is not None else np.eye(self.n_params),
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
            # Correlations: start at 0
            n_corr = dim * (dim - 1) // 2
            if not self.control.heteronly and n_corr > 0:
                theta0[idx:idx + n_corr] = 0.0
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
