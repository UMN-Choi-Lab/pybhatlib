"""Single-outcome analytical ordinary least squares workflow."""

from __future__ import annotations

from copy import deepcopy
from os import PathLike
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.stats import norm, t

from pybhatlib.io._data_loader import load_data
from pybhatlib.io._spec_parser import parse_spec
from pybhatlib.models._ate_common import ScenarioSpec
from pybhatlib.models._base import BaseModel
from pybhatlib.models.lr._lr_control import LRControl
from pybhatlib.models.lr._lr_forecast import lr_predict
from pybhatlib.models.lr._lr_loglik import lr_gradient, lr_loglik
from pybhatlib.models.lr._lr_results import LRResults

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for annotations only
    from pybhatlib.models.lr._lr_ate import LRATEResult


def _build_design(
    data: pd.DataFrame, spec: dict, dep_var: str
) -> tuple[NDArray, list[str]]:
    """Build the ``(N, K)`` design matrix for *dep_var* from a coefficient spec.

    Reuses :func:`parse_spec` with the outcome as the single "alternative", so
    the same ``uno`` / column-name / numeric-constant grammar applies.
    """
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
        if isinstance(value, str) and value == dep_var:
            raise ValueError(
                f"spec entry '{name}' uses the outcome column '{dep_var}' as a regressor"
            )
        normalized[name] = {dep_var: value}
    X, names = parse_spec(normalized, data, [dep_var])
    X = X[:, 0, :]
    if not np.isfinite(X).all():
        raise ValueError("Design matrix contains missing or nonfinite values")
    return X, names


class LRModel(BaseModel):
    """Linear regression with one continuous outcome, solved in closed form.

    Parameters
    ----------
    data : str, PathLike, or pd.DataFrame
        Path to a data file (CSV / DAT / XLSX) or a DataFrame.
    dep_var : str
        Name of the continuous outcome column.
    spec : dict
        Maps coefficient names to a data column, ``"uno"`` (constant), or a
        numeric constant.  Outcome-keyed entries such as
        ``{"B_X": {"y": "x"}}`` also work, matching the other models'
        specifications.  No intercept is added implicitly.
    var_names : list of str or None
        Display names for the coefficients.  Defaults to the spec keys.
    control : LRControl or None
        Covariance / verbosity options.

    Notes
    -----
    Rank-deficient designs and ``N <= K`` are rejected so that inference is
    identified.  Coefficients come from the SVD of the design matrix (which
    avoids squaring its condition number).  The residual variance is the
    Gaussian MLE ``SSE / N`` (``results.sigma2``); ``SSE / (N - K)`` is
    reported as ``results.residual_variance``.

    Examples
    --------
    >>> model = LRModel(
    ...     data=df,
    ...     dep_var="ln_exp",
    ...     spec={"CON": "uno", "B_HINC": "hinc20k", "B_RURAL": "rural"},
    ...     control=LRControl(se_method="sandwich"),
    ... )
    >>> results = model.fit()
    >>> results.summary()
    """

    def __init__(
        self,
        data: str | PathLike | pd.DataFrame,
        dep_var: str,
        spec: dict | None = None,
        var_names: list[str] | None = None,
        control: LRControl | None = None,
    ) -> None:
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

    def _fit(self) -> LRResults:
        """Solve the least-squares problem and assemble :class:`LRResults`."""
        start = perf_counter()
        ctrl = deepcopy(self.control)
        ctrl.__post_init__()  # re-validate: control may have been mutated after construction
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
        ones = np.ones(self.N)
        constant = np.linalg.norm(ones - u @ (u.T @ ones)) < 1e-10 * np.sqrt(self.N)
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

    # ------------------------------------------------------------------
    # Post-estimation convenience API (delegates to the free functions;
    # shared method surface across MNP / MORP / MDCEV / MNL / LR)
    # ------------------------------------------------------------------
    def predict(self, X_new: ArrayLike | pd.DataFrame | None = None) -> NDArray:
        """Predicted conditional means (see :func:`lr_predict`).

        Parameters
        ----------
        X_new : ndarray (N, K), pd.DataFrame, or None
            ``None`` uses the training design; a DataFrame is rebuilt through
            the model's ``spec``; an array is used as the design directly.

        Returns
        -------
        y_hat : ndarray, shape (N,)
        """
        results = self._require_results()
        if isinstance(X_new, pd.DataFrame):
            X_new, _ = _build_design(X_new, self.spec_dict, self.dep_var)
        return lr_predict(results, self.X if X_new is None else X_new)

    def ate(self, *, scenarios: ScenarioSpec | None = None, **kwargs) -> LRATEResult:
        """Mean predicted outcome, optionally under counterfactual scenarios.

        ``data`` / ``spec`` / ``dep_var`` are supplied from the model; pass
        ``scenarios=`` for counterfactuals (see :func:`lr_ate`).
        """
        from pybhatlib.models.lr._lr_ate import lr_ate

        return lr_ate(self._require_results(), data=self.data,
                      spec=self.spec_dict, dep_var=self.dep_var,
                      scenarios=scenarios, **kwargs)
