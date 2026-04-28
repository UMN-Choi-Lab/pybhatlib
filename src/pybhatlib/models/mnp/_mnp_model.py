"""MNP model class: the main user-facing interface.

Provides a Pythonic API matching BHATLIB's mnpFit procedure.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr as _ndtr

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

        # Resolve device
        use_gpu = False
        gpu_device = "cpu"
        if self.control.device != "cpu":
            try:
                import torch
                if self.control.device == "auto":
                    use_gpu = (
                        torch.cuda.is_available()
                        and self.N >= self.control.gpu_threshold
                        and (self.n_alts - 1) == 2  # K=2 only for now
                    )
                else:
                    use_gpu = torch.cuda.is_available() and "cuda" in self.control.device
                if use_gpu:
                    gpu_device = "cuda" if self.control.device == "auto" else self.control.device
            except ImportError:
                pass

        if use_gpu:
            import torch
            from pybhatlib.models.mnp._mnp_grad_gpu import mnp_gradient_gpu

            X_gpu = torch.tensor(
                np.asarray(self.X, dtype=np.float64),
                dtype=torch.float64, device=gpu_device,
            )
            y_gpu = torch.tensor(
                np.asarray(self.y, dtype=np.int64),
                dtype=torch.int64, device=gpu_device,
            )

            if self.control.verbose >= 1:
                print(f"  Device: {gpu_device} (GPU accelerated)")

            def objective(theta):
                return mnp_gradient_gpu(
                    theta, X_gpu, y_gpu,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices, device=gpu_device,
                )
        else:
            # Define objective function (CPU path)
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
            ranvar_indices=self.ranvar_indices,
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
        """Map parameterized theta to BHATLIB reporting convention.

        Routes through the unparameterized form so the reporting block has
        a single source of truth: ``theta_par -> theta_unpar -> b_report``.
        This eliminates the need for a delta-method projection through the
        parameterized covariance and aligns the random-coefficient block
        with GAUSS (which reports variance/covariance entries via
        ``vecdup(L'L)``, not Cholesky entries).
        """
        from pybhatlib.models.mnp._mnp_loglik import _param_to_unpar

        theta_unpar = _param_to_unpar(
            theta, self.n_beta, self.n_alts, self.control,
            self.ranvar_indices,
        )
        return self._unpar_to_report(theta_unpar)

    def _unpar_to_report(self, theta_unpar: np.ndarray) -> np.ndarray:
        """Map unparameterized theta to BHATLIB reporting convention.

        BHATLIB normalizes the differenced error covariance so that
        Sigma_diff[0,0] = 1, absorbing one scale parameter for non-IID
        models. All betas are divided by sigma_1; the kernel block becomes
        normalized correlations (``parker``) and relative scales
        (``scale``); the random-coefficient block reports the Omega
        variance/covariance entries divided by sigma_1**2.

        Notes
        -----
        This is the only Jacobian step we keep on the SE path: a single
        scalar normalization (``sigma_1``) per block. It is NOT the
        delta-method through the spherical-Cholesky parameterization that
        the MNP-002 SE machinery used; the unparameterized theta and the
        reporting theta are identical except for sigma_1 scaling, and the
        Jacobian of that scaling is computed in closed form by finite
        differencing this function only.
        """
        from pybhatlib.models.mnp._mnp_loglik import (
            _build_lambda_direct,
            _build_omega_direct,
            _unpack_params_unpar,
        )

        params = _unpack_params_unpar(
            theta_unpar, self.n_beta, self.n_alts, self.control,
            self.ranvar_indices,
        )
        Lambda = _build_lambda_direct(
            params.get("lambda_params"), self.n_alts, self.control,
        )
        dim = self.n_alts - 1

        # Differenced kernel error covariance: Sigma_diff = ones + Lambda + I
        # (formula matches the parameterized path; identical at convergence
        # by construction, because _param_to_unpar preserves Lambda exactly).
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

            if not self.control.heteronly:
                for i in range(dim):
                    for j in range(i + 1, dim):
                        report.append(float(Corr_norm[i, j]))

            # Relative scales (indices 1 .. dim-1; index 0 is normalized to 1).
            for i in range(1, dim):
                report.append(float(d_norm[i]))

        # --- Random-coefficient Omega (variance/covariance entries) ---
        # GAUSS reference: MNP_TRAVELMODE.gss line 211 reports
        # ``vecdup(newcorker' * newcorker)`` — i.e., upper-tri (with diag)
        # of Omega. This makes ``CovCOv01`` the variance of the random
        # coefficient (not its standard deviation), matching the BHATLIB
        # paper's Table 2 fixture.
        if (
            self.control.mix
            and self.ranvar_indices is not None
            and "omega_params" in params
        ):
            Omega = _build_omega_direct(
                params["omega_params"], self.ranvar_indices, self.control,
            )
            Omega_norm = Omega / (sigma_1 ** 2)
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    report.append(float(Omega_norm[i, i]))
            else:
                for i in range(n_rand):
                    for j in range(i, n_rand):
                        report.append(float(Omega_norm[i, j]))

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
                    Omega_h = _build_omega_direct(
                        params["segment_omegas"][h - 1],
                        self.ranvar_indices,
                        self.control,
                    )
                    Omega_h_norm = Omega_h / (sigma_1 ** 2)
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        for i in range(n_rand):
                            report.append(float(Omega_h_norm[i, i]))
                    else:
                        for i in range(n_rand):
                            for j in range(i, n_rand):
                                report.append(float(Omega_h_norm[i, j]))

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
        """Compute BHATLIB-normalized values with standard errors.

        MNP-002b: SEs are computed in the unparameterized covariance/
        correlation space (the GAUSS ``lpr1``/``lgd1`` convention). The
        per-observation scores are obtained by finite differencing
        ``_per_obs_loglik_unpar`` at ``theta_unpar = _param_to_unpar(theta_hat)``,
        not at the parameterized ``theta_hat`` itself. There is no
        delta-method projection through the spherical-Cholesky
        parameterization.

        The only Jacobian that survives is the BHATLIB sigma_1 normalization
        from ``_unpar_to_report`` — a scalar division per block. This is
        unavoidable (the paper's reported parameters are normalized) and is
        explicitly NOT what the plan calls "delta method": the projection
        is a single closed-form scaling, computed by finite-differencing
        ``_unpar_to_report`` only (which is a smooth, low-dimensional
        normalization, not a re-parameterization).

        Hessian path: ``hess_inv`` from scipy is in parameterized theta
        space (the optimizer minimized in parameterized space). To project
        it into unparameterized space we need the par->unpar Jacobian,
        ``J_pu = d(theta_unpar)/d(theta_par)``, computed by finite
        differencing ``_param_to_unpar`` at ``theta_hat``. This Jacobian
        block-diagonalizes (beta and segment params pass through the
        identity; only the kernel and Omega blocks have non-trivial
        structure) so it is well-conditioned by construction.

        SE method is selected via ``self.control.se_method``:
          - "hessian": ``J_norm @ J_pu @ (hess_inv/N) @ J_pu^T @ J_norm^T``.
          - "bhhh":    ``J_norm @ (G_unpar^T G_unpar)^{-1} @ J_norm^T``.
          - "sandwich": ``J_norm @ A^{-1} B A^{-1} @ J_norm^T`` where
                       ``A = J_pu^T (hess_inv/N) J_pu`` (par->unpar) and
                       ``B = G_unpar^T G_unpar``.

        Returns
        -------
        b_report : ndarray  — normalized coefficient estimates
        se : ndarray         — standard errors
        t_stat : ndarray     — b_report / se
        p_value : ndarray    — two-sided p-values
        cov_report : ndarray — covariance matrix of reporting params
        corr_report : ndarray — correlation matrix of reporting params
        g_report : ndarray   — gradient projected into reporting space
        """
        import warnings

        from pybhatlib.models.mnp._mnp_loglik import (
            _param_to_unpar,
            _per_obs_loglik_unpar,
        )

        # 1) Translate the converged parameterized estimate to unparameterized.
        theta_unpar = _param_to_unpar(
            theta_hat, self.n_beta, self.n_alts, self.control,
            self.ranvar_indices,
        )
        n_theta = len(theta_unpar)  # equals len(theta_hat) by construction

        # 2) Build reporting estimate from unparameterized theta.
        b_report = self._unpar_to_report(theta_unpar)
        n_report = len(b_report)

        # 3) Normalization Jacobian: J_norm = d(b_report)/d(theta_unpar).
        # This is a scalar sigma_1 normalization per block, NOT a
        # delta-method through the spherical-Cholesky parameterization.
        J_norm = self._fd_jacobian(self._unpar_to_report, theta_unpar)

        se_method = self.control.se_method
        cov_unpar: np.ndarray | None = None

        if se_method in ("bhhh", "sandwich"):
            # Per-obs scores in UNPARAMETERIZED space (the lpr1/lgd1 path).
            # Scaled FD step: eps_machine^(1/3) balances truncation O(h^2)
            # and roundoff O(eps/h); the (1 + |theta_i|) scale respects
            # parameter magnitude without vanishing at theta_i ~ 0.
            base_eps = np.finfo(np.float64).eps ** (1.0 / 3.0)
            G = np.zeros((self.N, n_theta), dtype=np.float64)
            for i in range(n_theta):
                eps_i = base_eps * (1.0 + abs(theta_unpar[i]))
                theta_p = theta_unpar.copy()
                theta_p[i] += eps_i
                theta_m = theta_unpar.copy()
                theta_m[i] -= eps_i
                ll_p = _per_obs_loglik_unpar(
                    theta_p, self.X, self.y, self.avail,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices,
                )
                ll_m = _per_obs_loglik_unpar(
                    theta_m, self.X, self.y, self.avail,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices,
                )
                G[:, i] = (ll_p - ll_m) / (2.0 * eps_i)

            B = G.T @ G

            # Project out parameters whose score is numerically zero. This
            # happens routinely for IID models with unused Lambda slots, or
            # when control options pin a coefficient. We warn on
            # ill-conditioning rather than letting B be silently singular.
            score_norms = np.linalg.norm(G, axis=0)
            structural_zero = score_norms < 1e-12
            active_mask = ~structural_zero

            G_active = G[:, active_mask]
            B_active = G_active.T @ G_active
            cond_active = np.linalg.cond(B_active) if B_active.size else 0.0
            if cond_active > 1e12:
                warnings.warn(
                    f"BHHH score outer-product has condition number "
                    f"{cond_active:.2e} (>1e12) — parameters are weakly "
                    f"identified. Using pseudo-inverse; reported SEs may "
                    f"mask identification problems.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                B_inv_active = np.linalg.pinv(B_active)
            else:
                try:
                    B_inv_active = np.linalg.inv(B_active)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "BHHH B matrix is singular; falling back to pinv. "
                        "Check for redundant or unidentified parameters.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    B_inv_active = np.linalg.pinv(B_active)

            B_inv = np.zeros((n_theta, n_theta), dtype=np.float64)
            active_idx = np.where(active_mask)[0]
            B_inv[np.ix_(active_idx, active_idx)] = B_inv_active

            if se_method == "bhhh":
                cov_unpar = B_inv
            else:  # sandwich
                if hess_inv is None:
                    warnings.warn(
                        "se_method='sandwich' requested but hess_inv is None "
                        "(optimizer did not return a Hessian approximation); "
                        "falling back to BHHH covariance.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    cov_unpar = B_inv
                else:
                    # mnp_loglik minimizes mean NLL, so scipy hess_inv is in
                    # parameterized space and Var(theta_par) ≈ hess_inv / N.
                    # To translate to unparameterized space, use the par->unpar
                    # Jacobian: Var(theta_unpar) ≈ J_pu (hess_inv/N) J_pu^T.
                    J_pu = self._fd_jacobian(
                        lambda th: _param_to_unpar(
                            th, self.n_beta, self.n_alts, self.control,
                            self.ranvar_indices,
                        ),
                        theta_hat,
                    )
                    A_unpar = J_pu @ (hess_inv / self.N) @ J_pu.T
                    cov_unpar = A_unpar @ B @ A_unpar
        elif se_method == "hessian" and hess_inv is not None:
            # Project hess_inv (parameterized space) to unparameterized space
            # via the par->unpar Jacobian.
            J_pu = self._fd_jacobian(
                lambda th: _param_to_unpar(
                    th, self.n_beta, self.n_alts, self.control,
                    self.ranvar_indices,
                ),
                theta_hat,
            )
            cov_unpar = J_pu @ (hess_inv / self.N) @ J_pu.T

        if cov_unpar is not None:
            cov_report = J_norm @ cov_unpar @ J_norm.T
            se = np.sqrt(np.maximum(np.diag(cov_report), 0.0))
        else:
            cov_report = np.eye(n_report)
            se = np.full(n_report, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, b_report / se, 0.0)
            p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

        # Correlation matrix of reporting parameters
        if cov_unpar is not None:
            se_outer = np.outer(se, se)
            with np.errstate(divide="ignore", invalid="ignore"):
                corr_report = np.where(
                    se_outer > 0, cov_report / se_outer, 0.0,
                )
            np.fill_diagonal(corr_report, 1.0)
        else:
            corr_report = np.eye(n_report)

        # Gradient projected into reporting space via pseudo-inverse of the
        # combined par->report Jacobian. We compose J_norm and J_pu only when
        # we need to project the parameterized gradient; for the SE path the
        # composition is implicit in cov_unpar -> cov_report.
        if gradient is not None:
            J_pu_for_grad = self._fd_jacobian(
                lambda th: _param_to_unpar(
                    th, self.n_beta, self.n_alts, self.control,
                    self.ranvar_indices,
                ),
                theta_hat,
            )
            J_par_to_report = J_norm @ J_pu_for_grad
            g_report = np.linalg.pinv(J_par_to_report).T @ gradient
        else:
            g_report = np.zeros(n_report)

        return b_report, se, t_stat, p_value, cov_report, corr_report, g_report

    @staticmethod
    def _fd_jacobian(f, x: np.ndarray, eps: float = 1e-7) -> np.ndarray:
        """Central-difference Jacobian of a vector-valued function.

        Used for the (low-dimensional, well-conditioned) sigma_1
        normalization Jacobian and for the par->unpar translation Jacobian.
        Both are smooth scalar normalizations / smooth reparameterizations,
        so a fixed FD step is fine; we don't scale by ``|x_i|`` here.
        """
        x = np.asarray(x, dtype=np.float64)
        f0 = np.asarray(f(x), dtype=np.float64)
        n_in = x.size
        n_out = f0.size
        J = np.zeros((n_out, n_in), dtype=np.float64)
        for i in range(n_in):
            xp = x.copy()
            xp[i] += eps
            xm = x.copy()
            xm[i] -= eps
            J[:, i] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2.0 * eps)
        return J
