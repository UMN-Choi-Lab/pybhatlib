"""MORP model class: the main user-facing interface.

Provides a Pythonic API matching BHATLIB's morpFit procedure for the
Multivariate Ordered Response Probit model.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr as _ndtr, ndtri as _ndtri

from pybhatlib.backend._array_api import get_backend
from pybhatlib.io._data_loader import load_data
from pybhatlib.io._spec_parser import parse_spec
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
    data : pd.DataFrame, str, or os.PathLike
        Dataset as a DataFrame, or a path to a CSV file.
    dep_vars : list of str
        Column names for ordinal outcome variables (one per dimension).
    spec : dict[str, dict[str, str]]
        Variable specification mapping coefficient names to per-outcome
        column names or keywords.

        Outer keys are coefficient names (one estimated beta per key).
        Inner keys are outcome (dep_var) column names.
        Inner values are either a column name in ``data`` or the literal
        string ``"sero"`` (meaning this coefficient is zero for that outcome).

        Example::

            spec = {
                "E_rest20": {"NeatoutO": "resta20",  "Npickupo": "sero", "Ndelivo": "sero"},
                "P_rest20": {"NeatoutO": "sero",     "Npickupo": "resta20", "Ndelivo": "sero"},
                "D_urb":    {"NeatoutO": "sero",     "Npickupo": "sero",    "Ndelivo": "urb"},
            }

    n_categories : list of int
        Number of categories per dimension.
    control : MORPControl or None
        Estimation control structure.

    Examples
    --------
    >>> model = MORPModel(
    ...     data="Example_Dining.csv",
    ...     dep_vars=["NeatoutO", "Npickupo", "Ndelivo"],
    ...     spec={
    ...         "E_rest20": {"NeatoutO": "resta20",  "Npickupo": "sero",    "Ndelivo": "sero"},
    ...         "P_rest20": {"NeatoutO": "sero",     "Npickupo": "resta20", "Ndelivo": "sero"},
    ...     },
    ...     n_categories=[12, 8, 9],
    ...     control=MORPControl(method="ovus"),
    ... )
    >>> results = model.fit()
    >>> results.summary()
    """

    def __init__(
        self,
        data: pd.DataFrame | str | os.PathLike,
        dep_vars: list[str],
        spec: dict[str, dict[str, str]],
        n_categories: list[int],
        control: MORPControl | None = None,
        **kwargs,
    ):
        if "indep_vars" in kwargs:
            raise TypeError(
                "indep_vars= was removed in MORP-001; use spec={var: {outcome: column}} "
                "instead — see docs/plans/UTA_MNP_MORP_FEEDBACK_2026_04.md §MORP-001"
            )
        if kwargs:
            raise TypeError(
                f"MORPModel got unexpected keyword arguments: {sorted(kwargs)}"
            )

        self.control = control or MORPControl()

        # Load data from path if necessary
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = load_data(data)

        self.dep_vars = dep_vars
        self.n_dims = len(dep_vars)
        self.n_categories = n_categories
        self.spec = spec

        if len(n_categories) != self.n_dims:
            raise ValueError(
                f"n_categories length ({len(n_categories)}) must match "
                f"dep_vars length ({self.n_dims})"
            )

        # Validate dep_vars exist in data
        for dv in dep_vars:
            if dv not in self.data.columns:
                raise ValueError(f"Dependent variable '{dv}' not found in data")

        # Validate spec inner keys are all valid dep_var names
        for coef_name, outcome_map in spec.items():
            for outcome_key in outcome_map:
                if outcome_key not in dep_vars:
                    raise ValueError(
                        f"Spec coefficient '{coef_name}' references outcome "
                        f"'{outcome_key}' which is not in dep_vars={dep_vars!r}"
                    )

        # Build design matrix X: (N, D, n_coefs).  Routed through the shared
        # ``parse_spec`` helper (used by MNP) instead of an inline copy — this
        # picks up the ``sero`` / ``uno`` keyword handling, integer-literal
        # support, and consistent error messages.  The shared helper does not
        # care about the *meaning* of the second-dim labels (it works with
        # ``alternatives`` for MNP and with ``dep_vars`` for MORP).
        self.N = len(self.data)
        self.X, self.var_names = parse_spec(
            spec, self.data, dep_vars, nseg=1
        )
        self.n_beta = len(self.var_names)

        # Extract ordinal outcomes: (N, D)
        self.y = np.zeros((self.N, self.n_dims), dtype=np.int64)
        for d, dv in enumerate(dep_vars):
            vals = self.data[dv].values
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
            if self.control.iid:
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
            p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

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
            correlation_matrix=corr_matrix if not self.control.iid else None,
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
            q = _ndtri(1.0 / self.n_categories[d])
            theta0[idx] = q
            idx += 1
            # Subsequent thresholds: log-spacing
            for j in range(1, n_thresh):
                spacing = _ndtri((j + 1) / self.n_categories[d]) - _ndtri(
                    j / self.n_categories[d]
                )
                theta0[idx] = np.log(max(spacing, 0.1))
                idx += 1

        # Covariance parameters: start at 0 (identity)
        # Remaining params are already 0

        return theta0

    def _build_param_names(self) -> list[str]:
        """Build descriptive parameter names."""
        names = list(self.var_names)

        for d in range(self.n_dims):
            n_thresh = self.n_categories[d] - 1
            for j in range(n_thresh):
                names.append(f"tau_{self.dep_vars[d]}_{j + 1}")

        if not self.control.iid:
            # Scale params only when neither heteronly+only-scales nor
            # fix_scales (which locks them at 1).
            estimate_scales = self.control.heteronly or not getattr(
                self.control, "fix_scales", False
            )
            if estimate_scales:
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
