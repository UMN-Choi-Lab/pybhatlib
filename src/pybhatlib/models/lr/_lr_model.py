"""Single-outcome analytical ordinary least squares workflow."""

from copy import deepcopy
from os import PathLike
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from pybhatlib.io._data_loader import load_data
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._base import BaseModel
from ._lr_control import LRControl
from ._lr_loglik import lr_gradient, lr_loglik
from ._lr_results import LRResults


def _build_design(data, spec, dep_var):
    if not isinstance(spec, dict) or not spec:
        raise ValueError("spec must be a nonempty coefficient mapping")
    normalized = {}
    for name, value in spec.items():
        if isinstance(value, dict):
            if set(value) != {dep_var}:
                raise ValueError(f"spec entry '{name}' must map only '{dep_var}'")
            value = value[dep_var]
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"Invalid spec value for '{name}'")
        normalized[name] = {dep_var: value}
    X, names = parse_spec(normalized, data, [dep_var])
    X = X[:, 0, :]
    if not np.isfinite(X).all():
        raise ValueError("Design matrix contains missing or nonfinite values")
    return X, names


class LRModel(BaseModel):
    """Linear regression with one continuous ``dep_var`` column.

    ``data`` accepts a DataFrame or supported data-file path. ``spec`` maps
    coefficient names to columns, ``uno`` (constant), or numeric constants.
    Outcome-keyed entries such as ``{"B_X": {"y": "x"}}`` also work, matching
    other model specifications. No intercept is added implicitly.
    Rank-deficient designs and N <= K are rejected to keep inference identified.
    """

    def __init__(self, data, dep_var, spec=None, var_names=None, control=None):
        self.control = control or LRControl()
        if not isinstance(dep_var, str):
            raise ValueError("dep_var must name a single continuous outcome column")
        self.dep_var = dep_var
        self.data_path = str(data) if isinstance(data, (str, PathLike)) else "<DataFrame>"
        self.data = load_data(str(data)) if isinstance(data, (str, PathLike)) else data.copy()
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("data must be a DataFrame or data-file path")
        self.spec_dict = deepcopy(spec)
        self.X, names = _build_design(self.data, spec, dep_var)
        self.var_names = list(var_names) if var_names is not None else names
        self.n_beta = self.X.shape[1]
        if len(self.var_names) != self.n_beta:
            raise ValueError("var_names must match the number of coefficients")
        self.y = self.data[dep_var].to_numpy(dtype=float)
        self.N = len(self.y)
        if self.y.shape != (self.N,) or not np.isfinite(self.y).all():
            raise ValueError("Outcome must be a finite one-dimensional vector")
        if self.N <= self.n_beta:
            raise ValueError("Linear regression requires n_obs > n_beta")

    def _fit(self):
        start = perf_counter()
        ctrl = deepcopy(self.control)
        ctrl.__post_init__()
        if ctrl.verbose >= 1:
            print(f"  LR estimation: {self.N} obs, {self.n_beta} parameters")
        # SVD avoids squaring the design's condition number in the solve.
        u, s, vt = np.linalg.svd(self.X, full_matrices=False)
        if np.any(s <= np.finfo(float).eps * max(self.X.shape) * s[0]):
            raise ValueError("Design matrix is rank deficient; remove redundant columns")
        beta = vt.T @ ((u.T @ self.y) / s)
        fitted = self.X @ beta
        residuals = self.y - fitted
        sse = float(residuals @ residuals)
        df = self.N - self.n_beta
        sigma2 = sse / self.N
        exact = np.linalg.norm(residuals) <= (
            np.finfo(float).eps * max(self.X.shape) * np.linalg.norm(self.y)
        )
        if exact:
            sigma2 = 0.0
        bread = (vt.T / s**2) @ vt
        cov = np.full((self.n_beta, self.n_beta), np.nan)
        if ctrl.want_covariance:
            if ctrl.se_method == "hessian":
                cov = sigma2 * bread
            elif ctrl.se_method == "sandwich":
                xr = self.X * residuals[:, None]
                cov = bread @ (xr.T @ xr) @ bread
            elif not exact:
                scores = lr_gradient(beta, self.X, self.y, sigma2)
                if np.linalg.matrix_rank(scores) < self.n_beta:
                    raise ValueError("BHHH score matrix is rank deficient")
                cov = np.linalg.inv(scores.T @ scores)
            if ctrl.df_correction:
                cov *= self.N / df
        se = np.sqrt(np.maximum(np.diag(cov), 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            stat = beta / se
            corr = cov / np.outer(se, se)
        p = 2 * (t.sf(np.abs(stat), df) if ctrl.se_method == "hessian"
                 else norm.sf(np.abs(stat)))
        constant = np.linalg.norm(np.ones(self.N) - u @ (u.T @ np.ones(self.N))) < 1e-10 * np.sqrt(self.N)
        centered = self.y - self.y.mean() if constant else self.y
        tss = float(centered @ centered)
        r2 = 1 - sse / tss if tss > 0 else np.nan
        ssr = tss - sse
        df_model = self.n_beta - int(constant)
        if df_model <= 0:
            f_stat = np.nan
        elif exact:
            f_stat = np.inf
        else:
            f_stat = (ssr / df_model) / (sse / df)
        result = LRResults(
            params=beta, se=se, t_stat=stat, p_value=p,
            gradient=lr_gradient(beta, self.X, self.y, sigma2).mean(axis=0)
            if not exact else np.full(self.n_beta, np.nan),
            loglik=float(lr_loglik(beta, self.X, self.y, sigma2).mean())
            if not exact else np.inf,
            n_obs=self.N, param_names=self.var_names.copy(),
            corr_matrix=corr, cov_matrix=cov, control=ctrl,
            data_path=self.data_path, convergence_time=(perf_counter() - start) / 60,
            sigma2=sigma2, residual_variance=sse / df, df_resid=df,
            r_squared=r2, adj_r_squared=1 - (1 - r2) * (self.N - int(constant)) / df,
            fitted_values=fitted, residuals=residuals,
            sst=tss, ssr=ssr, sse=sse, df_model=df_model, f_stat=f_stat,
            message="Analytical least squares" + ("; exact fit (zero variance boundary)" if exact else ""),
        )
        if ctrl.verbose >= 1:
            print(
                f"  Analytical solution computed in "
                f"{result.convergence_time:.4f} min.  R-squared = {r2:.6f}"
            )
        return result

    def predict(self, X_new=None):
        """Predict from training data, an (N, K) matrix, or a new DataFrame."""
        from ._lr_forecast import lr_predict
        results = self._require_results()
        if isinstance(X_new, pd.DataFrame):
            X_new, _ = _build_design(X_new, self.spec_dict, self.dep_var)
        return lr_predict(results, self.X if X_new is None else X_new)

    def ate(self, *, scenarios=None, **kwargs):
        """Compare predicted outcome means under column-override scenarios."""
        from ._lr_ate import lr_ate
        return lr_ate(self._require_results(), data=self.data,
                      spec=self.spec_dict, dep_var=self.dep_var,
                      scenarios=scenarios, **kwargs)
