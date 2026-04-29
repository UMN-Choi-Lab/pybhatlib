"""MNP model class: the main user-facing interface.

Provides a Pythonic API matching BHATLIB's mnpFit procedure.
"""

from __future__ import annotations

import os
import re
import time
import warnings

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

# Regex for explicit segment-suffix forms: ``name_seg<N>`` or ``name_s<N>``.
# Matches ``OVTT_seg1``, ``OVTT_s2``, etc. but NOT ``AGE45`` or ``AGE4``.
# Reference: ``Gauss Files and Comparison/MNP/MNP Table2 d.gss:113``
# (GAUSS ``ranvars = { OVTT1 OVTT2 }`` pattern).
_EXPLICIT_SEG_RE = re.compile(r"^(.+?)_(?:seg|s)\d+$")


class MNPModel(BaseModel):
    """Multinomial Probit model.

    Supports IID, flexible covariance, random coefficients, and
    mixture-of-normals specifications.

    Parameters
    ----------
    data : pd.DataFrame, str, or os.PathLike
        DataFrame, or path to a CSV/DAT/XLSX file.
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
        data: pd.DataFrame | str | os.PathLike,
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

        # Load data: accept DataFrame, str path, or os.PathLike
        if isinstance(data, pd.DataFrame):
            self.data_path = "<DataFrame>"
            self.data = data
        elif isinstance(data, (str, os.PathLike)):
            self.data_path = os.fspath(data)
            self.data = load_data(data)
        else:
            raise TypeError(
                f"data must be a pandas DataFrame, str, or os.PathLike; "
                f"got {type(data).__name__}"
            )

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

        # Parse random variables.
        #
        # MNP-006 (M1) ergonomic auto-expansion across mixture segments:
        # when ``control.nseg > 1`` and the user supplies a base ranvar
        # name (e.g. ``"OVTT"``) that is not segment-suffixed in the
        # spec, the index is repeated ``nseg`` times so each segment
        # gets its own random-coefficient slot — mirroring the GAUSS
        # user-facing pattern ``ranvars = { OVTT1 OVTT2 }`` from
        # ``Gauss Files and Comparison/MNP/MNP Table2 d.gss:113``.
        #
        # Legacy segment-suffixed input (``ranvars=["OVTT1", "OVTT2"]``
        # with separate spec entries) continues to work unchanged.
        if isinstance(ranvars, str):
            ranvars = [ranvars]

        ranvars_expanded = False
        if ranvars is not None and len(ranvars) > 0:
            self.control.mix = True
            self.ranvar_indices, ranvars_expanded = self._resolve_ranvar_indices(
                ranvars
            )
        else:
            self.ranvar_indices = None

        # Normalize mix flag so downstream display/branching never sees
        # the meaningless ``mix=True, ranvar_indices=None`` combination
        # (e.g. when the user sets ``mix=True`` on the control but passes
        # ``ranvars=None`` or ``ranvars=[]``).
        if self.ranvar_indices is None:
            self.control.mix = False

        # Warn only when auto-expansion produced duplicate column indices,
        # which makes the random-coefficient Omega rank-deficient. User-
        # supplied duplicate names (e.g. ``ranvars=["OVTT", "OVTT"]``) are a
        # deliberate choice and not flagged here.
        if (
            ranvars_expanded
            and self.ranvar_indices
            and len(set(self.ranvar_indices)) < len(self.ranvar_indices)
        ):
            warnings.warn(
                "Mixture ranvars auto-expansion produced duplicate column indices "
                f"{self.ranvar_indices}. The resulting random-coefficient Omega is "
                "rank-deficient and the optimizer can only recover effective scalar "
                "variance per segment. To fit a model with truly per-segment random "
                "coefficients, list segment-suffixed names in spec (e.g. OVTT_seg1, "
                "OVTT_seg2) and pass them explicitly. See "
                "docs/plans/MIXTURE_SHARED_COEFFICIENTS_PLAN.md.",
                RuntimeWarning,
                stacklevel=2,
            )

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

    def _resolve_ranvar_indices(
        self, ranvars: list[str]
    ) -> tuple[list[int], bool]:
        """Resolve user-supplied ranvar names into indices in ``var_names``.

        Implements the MNP-006 (M1) auto-expansion ergonomic fix.
        Mirrors the GAUSS pattern ``ranvars = { OVTT1 OVTT2 }`` from
        ``MNP Table2 d.gss:113`` — with ``control.nseg > 1`` the user
        may write ``ranvars=["OVTT"]`` (base name) and the index is
        replicated across segments automatically.

        Resolution rules
        ----------------
        Each user-supplied name ``rv`` is classified as:

        - **base** — appears in ``var_names`` and does NOT end with a
          digit or an explicit ``_seg``/``_s`` suffix (e.g. ``"OVTT"``).
          Auto-expanded to one slot per segment when ``nseg > 1``.
        - **segment-suffixed (literal)** — appears in ``var_names`` and
          ends with a digit or ``_sN`` suffix (e.g. ``"OVTT1"``,
          ``"OVTT_s2"``). Used as-is, no expansion.
        - **segment-suffixed (inferred)** — does not appear in
          ``var_names``, but stripping a trailing digit or ``_sN``
          yields a base that does (e.g. ``"OVTT1"`` → ``"OVTT"``).
          Used as-is for that base index, no expansion.
        - Otherwise → ``ValueError``.

        is_base heuristic limitation
        ----------------------------
        A name is treated as a "base" eligible for auto-expansion only
        when it neither matches the explicit ``_seg``/``_s`` regex nor
        ends in a digit. This means a name like ``"AGE45"`` (ends in
        digit) is **never** auto-expanded — even when the user might
        want a per-segment AGE45 random coefficient. Conversely, a name
        like ``"X1Y"`` (does not end in digit) **is** auto-expanded if
        ``nseg > 1``, even when the user only wanted a single column.
        If you need the non-default behavior, list explicit segment
        names (e.g. ``"AGE45_seg1"``, ``"AGE45_seg2"``) in spec and
        pass them in ``ranvars`` directly.

        Parameters
        ----------
        ranvars : list of str
            User-supplied random-coefficient variable names.

        Returns
        -------
        indices : list of int
            Resolved indices into ``self.var_names``. When auto-expansion
            fires, length is ``n_base_names * nseg + n_non_base_names``
            (no dedup of input names); otherwise ``len(ranvars)``.
        expanded : bool
            ``True`` iff auto-expansion replicated at least one base
            index across segments. Used by ``__init__`` to attribute
            duplicate-index warnings correctly.

        Raises
        ------
        ValueError
            When a name (or its inferred base) is not in ``var_names``.
        """
        nseg = self.control.nseg
        resolved: list[int] = []
        # Classify per-name so we know which to auto-expand.
        is_base: list[bool] = []

        for rv in ranvars:
            base = self._strip_segment_suffix(rv, var_names=self.var_names)
            looks_segment_suffixed = base is not None

            if rv in self.var_names:
                resolved.append(self.var_names.index(rv))
                # A name qualifies as a "base" for per-segment auto-expansion
                # only when it carries NO segment suffix (explicit or legacy).
                # Names that end in digits but are not recognised as a legacy
                # segment form (e.g. ``"AGE45"`` when ``"AGE4"`` is absent
                # from spec) are treated as literal variable names, NOT bases
                # — even though ``_strip_segment_suffix`` returns None for them.
                # We detect the "ends-in-digits but not a recognised suffix"
                # case by checking whether the raw name ends in a digit.
                ends_in_digit = rv[-1].isdigit() if rv else False
                # Treat as base only when no suffix at all (no trailing digits,
                # no explicit _seg/_s form).
                is_base.append(not looks_segment_suffixed and not ends_in_digit)
                continue

            # Not literally in var_names — try base lookup (only
            # meaningful when nseg > 1; for nseg == 1 the segment
            # suffix has no semantic meaning).
            if nseg > 1 and looks_segment_suffixed and base in self.var_names:
                resolved.append(self.var_names.index(base))
                is_base.append(False)  # explicit segment form
                continue

            # Targeted hint when the failure is the nseg=1 + legacy-suffix
            # case (e.g. user passes ranvars=["OVTT1"] with nseg=1 and only
            # "OVTT" in var_names): with nseg=1 the segment suffix has no
            # semantic meaning, so we don't auto-strip it.
            if nseg == 1 and looks_segment_suffixed and base in self.var_names:
                raise ValueError(
                    f"Random variable '{rv}' not found in var_names "
                    f"(with nseg=1, segment suffixes are not auto-stripped). "
                    f"Use the literal var_name '{base}' instead, or set "
                    f"nseg > 1 to enable suffix resolution. "
                    f"var_names: {self.var_names}"
                )

            raise ValueError(
                f"Random variable '{rv}' not found in var_names: "
                f"{self.var_names}"
            )

        # Auto-expand base names when nseg > 1: replicate each base
        # index ``nseg`` times so that every mixture segment receives
        # its own random-coefficient slot.
        expanded = False
        if nseg > 1 and any(is_base):
            expanded = True
            replicated: list[int] = []
            for idx, base_flag in zip(resolved, is_base):
                if base_flag:
                    replicated.extend([idx] * nseg)
                else:
                    replicated.append(idx)
            resolved = replicated

        return resolved, expanded

    @staticmethod
    def _strip_segment_suffix(
        name: str,
        var_names: list[str] | None = None,
    ) -> str | None:
        """Return the base name when ``name`` carries a recognised segment suffix.

        Recognised forms
        ----------------
        1. **Explicit suffix** — ``name_seg<N>`` or ``name_s<N>``
           (e.g. ``"OVTT_seg1"``, ``"OVTT_s2"``).  Always recognised,
           regardless of ``var_names``.

        2. **Legacy GAUSS form** — plain trailing digits
           (e.g. ``"OVTT1"``, ``"OVTT2"``).  Only recognised when *both*
           the full name (``"OVTT1"``) **and** the base (``"OVTT"``)
           appear in ``var_names``.  This prevents ``"AGE45"`` from being
           silently stripped to ``"AGE4"`` when ``"AGE4"`` happens to be in
           the spec.

        Reference: ``Gauss Files and Comparison/MNP/MNP Table2 d.gss:113``.

        Parameters
        ----------
        name : str
            Candidate ranvar name.
        var_names : list of str, optional
            Spec variable names known to the model.  Required for legacy-form
            detection; if omitted the legacy form is **not** recognised.

        Returns
        -------
        str or None
            Base name if a suffix was stripped, else ``None``.
        """
        # 1. Explicit ``_seg<N>`` / ``_s<N>`` form.
        m = _EXPLICIT_SEG_RE.match(name)
        if m:
            return m.group(1)

        # 2. Legacy plain-digit form — only when BOTH name and base are in
        #    var_names.  Prevents AGE45 → AGE4 false-strip.
        if var_names is not None:
            i = len(name)
            while i > 0 and name[i - 1].isdigit():
                i -= 1
            if i < len(name) and i > 0:
                base = name[:i]
                if name in var_names and base in var_names:
                    return base

        return None

    def fit(self, bounds=None) -> MNPResults:
        """Estimate the MNP model.

        Parameters
        ----------
        bounds : list of (lower, upper) tuples or None
            Parameter bounds forwarded to scipy (L-BFGS-B only).
            Must have length ``n_params`` if provided; when ``active_mask``
            is set, the bounds are automatically filtered to the active subset.

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

        # ------------------------------------------------------------------
        # MNP-003: active_mask validation and setup
        # ------------------------------------------------------------------
        active_mask = self.control.active_mask
        frozen_theta = None  # full theta with frozen values fixed

        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool)
            if len(active_mask) != self.n_params:
                raise ValueError(
                    f"active_mask length {len(active_mask)} does not match "
                    f"n_params={self.n_params}"
                )
            n_active = int(active_mask.sum())
            if n_active == 0:
                raise ValueError(
                    "active_mask has no True entries — nothing to optimize. "
                    "Set at least one parameter to True, or pass active_mask=None "
                    "to estimate all parameters."
                )
            # frozen_theta holds the full starting vector; active entries
            # will be overwritten by the optimizer at each call.
            frozen_theta = theta0.copy()

        # ------------------------------------------------------------------
        # Fix 2: filter bounds to active subset when active_mask is set.
        # ``bounds`` has length n_params; the optimizer sees only active params.
        # ------------------------------------------------------------------
        if bounds is not None and active_mask is not None:
            bounds_active = [bounds[i] for i, a in enumerate(active_mask) if a]
        else:
            bounds_active = bounds

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

            def _full_objective(theta):
                return mnp_gradient_gpu(
                    theta, X_gpu, y_gpu,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices, device=gpu_device,
                )
        else:
            # Define objective function (CPU path)
            def _full_objective(theta):
                return mnp_loglik(
                    theta, self.X, self.y, self.avail,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices, return_gradient=True, xp=xp,
                )

        # ------------------------------------------------------------------
        # MNP-003: wrap objective for active_mask if needed
        # ------------------------------------------------------------------
        if active_mask is not None:
            # Build reduced starting vector for the optimizer
            theta0_active = theta0[active_mask]

            def objective(theta_active):
                """Reduced-space objective: reconstruct full theta before eval."""
                theta_full = frozen_theta.copy()
                theta_full[active_mask] = theta_active
                f, g_full = _full_objective(theta_full)
                g_active = g_full[active_mask]
                return f, g_active

            # Theta-space param names for verbose=3 (active params only)
            theta_names_active = [
                n for n, m in zip(self._build_param_names(), active_mask) if m
            ]
        else:
            theta0_active = theta0
            objective = _full_objective
            theta_names_active = self._build_param_names()

        # Optimize
        start_time = time.time()

        from pybhatlib.optim._scipy_optim import minimize_scipy

        opt_method = "BFGS" if self.control.optimizer == "bfgs" else "L-BFGS-B"

        result = minimize_scipy(
            objective,
            theta0_active,
            method=opt_method,
            maxiter=self.control.maxiter,
            tol=self.control.tol,
            verbose=self.control.verbose,
            jac=True,
            bounds=bounds_active,
            param_names=theta_names_active,
        )

        elapsed = (time.time() - start_time) / 60.0  # minutes

        # ------------------------------------------------------------------
        # MNP-003: reconstruct full theta_hat from active result
        # ------------------------------------------------------------------
        if active_mask is not None:
            theta_hat = frozen_theta.copy()
            theta_hat[active_mask] = result.x
            # Expand grad and hess_inv back to full space
            grad_full = np.zeros(self.n_params, dtype=np.float64)
            grad_full[active_mask] = result.grad
            # hess_inv is in active space; expand to full (frozen rows/cols = 0)
            if result.hess_inv is not None:
                hess_inv_active = result.hess_inv
                hess_inv = np.zeros(
                    (self.n_params, self.n_params), dtype=np.float64
                )
                active_idx = np.where(active_mask)[0]
                for ai, i in enumerate(active_idx):
                    for aj, j in enumerate(active_idx):
                        hess_inv[i, j] = hess_inv_active[ai, aj]
            else:
                hess_inv = None
        else:
            theta_hat = result.x
            hess_inv = result.hess_inv
            grad_full = result.grad

        # BHATLIB-normalized reporting
        param_names = self._build_report_names()
        (b_report, se, t_stat, p_value,
         cov_report, corr_report, g_report) = self._normalize_for_reporting(
            theta_hat, hess_inv, grad_full,
            active_mask=active_mask,
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
            params=theta_hat,
            b_original=b_report,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            gradient=g_report,
            loglik=-result.fun,
            n_obs=self.N,
            param_names=param_names,
            corr_matrix=corr_report,
            cov_matrix=cov_report,
            n_iter=result.n_iter,
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

    def _build_theta_to_report_map(self) -> dict[int, int | None]:
        """Map theta-space parameter indices to report-space indices.

        Returns a dict ``{theta_idx: report_idx}`` where ``report_idx`` is
        ``None`` for theta params that are absorbed by the BHATLIB
        normalization (i.e., scale01, the first scale parameter for non-IID
        models).

        Used by ``_normalize_for_reporting`` to mark SE=NaN for frozen params.
        """
        mapping: dict[int, int | None] = {}
        theta_idx = 0
        report_idx = 0

        # --- Betas: 1:1 mapping ---
        for _ in range(self.n_beta):
            mapping[theta_idx] = report_idx
            theta_idx += 1
            report_idx += 1

        if not self.control.iid:
            dim = self.n_alts - 1

            # --- Scale params in theta-space: scale01..scale0{dim} ---
            # scale01 (theta[n_beta]) is absorbed → None.
            # scale02..scale0{dim} (theta[n_beta+1..n_beta+dim-1]) map to
            # report-space *after* all parker params, so we compute their
            # report indices later.
            n_scale = dim
            scale_theta_start = theta_idx
            theta_idx += n_scale  # advance past all scales

            # --- Parker (correlations) in theta-space ---
            n_corr = dim * (dim - 1) // 2 if not self.control.heteronly else 0
            parker_theta_start = theta_idx
            theta_idx += n_corr

            # In report space: parkers come first (after betas),
            # then scale[1..dim-1] (relative scales).
            parker_report_start = report_idx
            report_idx += n_corr  # advance past parkers in report space

            scale_report_start = report_idx
            report_idx += dim - 1  # dim-1 relative scales in report space

            # Fill mapping for parkers: direct 1:1
            for k in range(n_corr):
                mapping[parker_theta_start + k] = parker_report_start + k

            # Fill mapping for scales:
            #   theta scale00 (index 0 among scales) → absorbed → None
            #   theta scale{i} for i=1..dim-1 → report_scale{i-1}
            mapping[scale_theta_start] = None  # absorbed (scale00)
            for k in range(1, n_scale):
                mapping[scale_theta_start + k] = scale_report_start + (k - 1)

        # --- Random-coefficient Cholesky params ---
        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                n_omega = n_rand
            else:
                n_omega = n_rand * (n_rand + 1) // 2
            for k in range(n_omega):
                mapping[theta_idx] = report_idx
                theta_idx += 1
                report_idx += 1

        # --- Mixture-of-normals segment params ---
        if self.control.nseg > 1:
            n_seg = self.control.nseg - 1
            for k in range(n_seg):
                mapping[theta_idx] = report_idx
                theta_idx += 1
                report_idx += 1
            # Extra segment betas and omegas
            for _ in range(1, self.control.nseg):
                for k in range(self.n_beta):
                    mapping[theta_idx] = report_idx
                    theta_idx += 1
                    report_idx += 1
                if self.control.mix and self.ranvar_indices is not None:
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        n_omega = n_rand
                    else:
                        n_omega = n_rand * (n_rand + 1) // 2
                    for k in range(n_omega):
                        mapping[theta_idx] = report_idx
                        theta_idx += 1
                        report_idx += 1

        return mapping

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
        active_mask: np.ndarray | None = None,
    ) -> tuple:
        """Compute BHATLIB-normalized values with delta-method standard errors.

        Parameters
        ----------
        theta_hat : ndarray
            Full parameter vector (theta-space), including frozen values.
        hess_inv : ndarray or None
            Full inverse Hessian in theta-space (frozen rows/cols are zero).
        gradient : ndarray or None
            Full gradient in theta-space.
        active_mask : ndarray of bool or None
            If provided, report-space params that depend *only* on frozen
            theta entries get SE/t-stat/p-value set to ``np.nan``.

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

        # MNP-003: zero columns of J corresponding to frozen theta entries
        # *before* the delta-method.  Without this, a report param that
        # depends on multiple theta entries (one frozen, one active) would
        # propagate the active column's variance even though the frozen
        # entry contributes zero (its column of cov_theta is zero, but FD
        # perturbations across frozen entries produce a non-zero J column
        # that the delta-method then mixes back in via off-diagonal cov
        # entries).  Zeroing the column makes the delta-method produce
        # se=0 naturally for the frozen-only-dependent params; the explicit
        # NaN-marking below then overrides the zero with NaN.
        if active_mask is not None:
            J[:, ~active_mask] = 0.0

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
            p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

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

        # ------------------------------------------------------------------
        # MNP-003: mark SE/t-stat/p-value as NaN for frozen report params.
        #
        # We use the explicit theta→report index map from
        # ``_build_theta_to_report_map`` to identify which report-space params
        # are *directly* determined by frozen theta-space params.  A report
        # param k is frozen if its corresponding theta param is frozen (i.e.,
        # active_mask[theta_idx] is False).  Absorbed params (mapped to None)
        # do not appear in report space, so they are skipped.
        # ------------------------------------------------------------------
        if active_mask is not None:
            theta_to_report = self._build_theta_to_report_map()
            for theta_idx, ri in theta_to_report.items():
                if ri is None:
                    continue  # absorbed param — no report slot
                if theta_idx < len(active_mask) and not active_mask[theta_idx]:
                    # This theta param is frozen → mark its report slot as NaN
                    if ri < len(se):
                        se[ri] = np.nan
                        t_stat[ri] = np.nan
                        p_value[ri] = np.nan

        return b_report, se, t_stat, p_value, cov_report, corr_report, g_report
