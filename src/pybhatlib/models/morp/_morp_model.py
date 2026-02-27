"""MORP model class: the main user-facing interface.

Provides a Pythonic API matching BHATLIB's morpFit procedure for the
Multivariate Ordered Response Probit model.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from pybhatlib.backend._array_api import get_backend
from pybhatlib.models._base import BaseModel
from pybhatlib.models.morp._morp_control import MORPControl
from pybhatlib.models.morp._morp_loglik import (
    _unpack_morp_params,
    count_morp_params,
    morp_loglik,
)
from pybhatlib.models.morp._morp_results import MORPResults


class MORPModel(BaseModel):
    """Multivariate Ordered Response Probit model.

    Models multiple ordinal outcomes simultaneously with a shared error
    covariance structure. Each dimension d has J_d ordered categories.

    Parameters
    ----------
    data : pd.DataFrame
        Dataset.
    dep_vars : list of str
        Column names for ordinal outcome variables (one per dimension).
    indep_vars : list of str
        Column names for independent variables (predictors).
    n_categories : list of int
        Number of categories per dimension.
    control : MORPControl or None
        Estimation control structure.

    Examples
    --------
    >>> model = MORPModel(
    ...     data=df,
    ...     dep_vars=["satisfaction", "likelihood"],
    ...     indep_vars=["income", "age", "education"],
    ...     n_categories=[5, 4],
    ...     control=MORPControl(method="ovus"),
    ... )
    >>> results = model.fit()
    >>> results.summary()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dep_vars: list[str],
        indep_vars: list[str],
        n_categories: list[int],
        control: MORPControl | None = None,
    ):
        self.control = control or MORPControl()
        self.data = data
        self.dep_vars = dep_vars
        self.indep_vars = indep_vars
        self.n_dims = len(dep_vars)
        self.n_categories = n_categories

        if len(n_categories) != self.n_dims:
            raise ValueError(
                f"n_categories length ({len(n_categories)}) must match "
                f"dep_vars length ({self.n_dims})"
            )

        # Build design matrix: (N, D, n_vars)
        self.N = len(data)
        self.n_beta = len(indep_vars)
        self.X = np.zeros(
            (self.N, self.n_dims, self.n_beta), dtype=np.float64
        )
        for v_idx, var_name in enumerate(indep_vars):
            if var_name not in data.columns:
                raise ValueError(f"Variable '{var_name}' not found in data")
            col_vals = data[var_name].values.astype(np.float64)
            for d in range(self.n_dims):
                self.X[:, d, v_idx] = col_vals

        # Extract ordinal outcomes: (N, D)
        self.y = np.zeros((self.N, self.n_dims), dtype=np.int64)
        for d, dv in enumerate(dep_vars):
            if dv not in data.columns:
                raise ValueError(f"Dependent variable '{dv}' not found in data")
            vals = data[dv].values
            # Ensure 0-based indexing
            min_val = vals.min()
            self.y[:, d] = (vals - min_val).astype(np.int64)
            # Validate
            max_cat = self.y[:, d].max()
            if max_cat >= n_categories[d]:
                raise ValueError(
                    f"Dimension '{dv}': found category {max_cat} but "
                    f"n_categories={n_categories[d]}"
                )

        # Count parameters
        self.n_params = count_morp_params(
            self.n_beta, self.n_dims, self.n_categories, self.control
        )

    def fit(self) -> MORPResults:
        """Estimate the MORP model.

        Returns
        -------
        results : MORPResults
        """
        xp = get_backend("numpy")

        if self.control.seed is not None:
            from pybhatlib.utils._seeds import set_seed
            set_seed(self.control.seed)

        if self.control.verbose >= 1:
            print(
                f"Estimating MORP model with {self.N} observations, "
                f"{self.n_dims} dimensions, {self.n_params} parameters"
            )
            cats_str = ", ".join(str(c) for c in self.n_categories)
            print(f"  Categories per dimension: [{cats_str}]")
            if self.control.indep:
                print("  Error structure: Independent")
            elif self.control.heteronly:
                print("  Error structure: Heteroscedastic only")
            else:
                print("  Error structure: Full covariance")

        # Starting values
        if self.control.startb is not None:
            theta0 = self.control.startb.copy()
        else:
            theta0 = self._default_start_values()

        # Define objective
        def objective(theta):
            return morp_loglik(
                theta, self.X, self.y, self.n_dims, self.n_categories,
                self.n_beta, self.control, return_gradient=True, xp=xp,
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

        elapsed = (time.time() - start_time) / 60.0

        theta_hat = result.x
        hess_inv = result.hess_inv

        # Build parameter names
        param_names = self._build_param_names()

        # Standard errors
        if hess_inv is not None:
            se = np.sqrt(np.abs(np.diag(hess_inv)) / self.N)
        else:
            se = np.full(self.n_params, np.nan)

        # t-stats and p-values
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, theta_hat / se, 0.0)
            p_value = 2.0 * (1.0 - norm.cdf(np.abs(t_stat)))

        # Extract thresholds and correlation matrix
        _, thresholds, sigma = _unpack_morp_params(
            theta_hat, self.n_beta, self.n_dims, self.n_categories, self.control
        )

        # Extract correlation matrix from sigma
        sd = np.sqrt(np.maximum(np.diag(sigma), 1e-30))
        D_inv = np.diag(1.0 / sd)
        corr_matrix = D_inv @ sigma @ D_inv
        np.fill_diagonal(corr_matrix, 1.0)

        return MORPResults(
            params=theta_hat,
            se=se,
            loglik=-result.fun,
            n_obs=self.N,
            n_params=self.n_params,
            converged=result.converged,
            n_iter=result.n_iter,
            thresholds=thresholds,
            correlation_matrix=corr_matrix if not self.control.indep else None,
            param_names=param_names,
            t_stat=t_stat,
            p_value=p_value,
            gradient=result.grad,
            cov_matrix=hess_inv / self.N if hess_inv is not None else None,
            convergence_time=elapsed,
            return_code=0 if result.converged else 2,
            control=self.control,
        )

    def _default_start_values(self) -> np.ndarray:
        """Generate reasonable starting values."""
        theta0 = np.zeros(self.n_params, dtype=np.float64)

        # Small random perturbation for betas
        theta0[: self.n_beta] = np.random.randn(self.n_beta) * 0.01
        idx = self.n_beta

        # Thresholds: equally spaced from standard normal quantiles
        for d in range(self.n_dims):
            n_thresh = self.n_categories[d] - 1
            if n_thresh <= 0:
                continue
            # First threshold: standard normal quantile
            q = norm.ppf(1.0 / self.n_categories[d])
            theta0[idx] = q
            idx += 1
            # Subsequent thresholds: log-spacing
            for j in range(1, n_thresh):
                spacing = norm.ppf((j + 1) / self.n_categories[d]) - norm.ppf(
                    j / self.n_categories[d]
                )
                theta0[idx] = np.log(max(spacing, 0.1))
                idx += 1

        # Covariance parameters: start at 0 (identity)
        # Remaining params are already 0

        return theta0

    def _build_param_names(self) -> list[str]:
        """Build descriptive parameter names."""
        names = list(self.indep_vars)

        for d in range(self.n_dims):
            n_thresh = self.n_categories[d] - 1
            for j in range(n_thresh):
                names.append(f"tau_{self.dep_vars[d]}_{j + 1}")

        if not self.control.indep:
            for d in range(1, self.n_dims):
                names.append(f"scale_{self.dep_vars[d]}")

            if not self.control.heteronly:
                for i in range(self.n_dims):
                    for j in range(i + 1, self.n_dims):
                        names.append(
                            f"corr_{self.dep_vars[i]}_{self.dep_vars[j]}"
                        )

        # Pad if needed
        while len(names) < self.n_params:
            names.append(f"param{len(names) + 1}")

        return names[: self.n_params]
