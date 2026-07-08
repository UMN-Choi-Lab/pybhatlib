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

        # MNP-002: compute the true observed Hessian H = ∂²(-Σℓ)/∂θ∂θᵀ via
        # central FD over the gradient when the user requested an SE method
        # that needs H (not H⁻¹).  This replaces ``result.hess_inv`` (the
        # BFGS quasi-Newton inverse) only for SE purposes; the active_mask
        # path's ``hess_inv`` is still used as the fallback covariance for
        # other reporting paths.
        #
        # PR #4 review P0 #1: when ``_observed_hessian`` raises, the
        # earlier code stored ``result.hess_inv`` (the inverse) into a
        # variable that downstream code treated as ``H`` itself, producing
        # ``inv(inv(H)) = H`` reported as the covariance.  Fix: on
        # fallback set ``hess_observed = None`` so SE goes to NaN rather
        # than silently double-inverting.
        # The observed Hessian (``2 * n_params`` extra full gradient
        # evaluations) is needed only when the primary se_method uses it, or
        # when the user opts into the full three-estimator diagnostic. With
        # the default BHHH primary and se_diagnostic=False this pass is
        # skipped — eliminating the post-convergence slowdown reported for the
        # sibling MORP model (UTA report, 2026-06). Previously (MNP-002b) it
        # was always computed so the diagnostic was available unconditionally.
        hess_observed: np.ndarray | None = None
        _need_hessian = (
            self.control.se_method in ("hessian", "sandwich")
            or getattr(self.control, "se_diagnostic", False)
        )
        if _need_hessian:
            try:
                hess_observed = self._observed_hessian(theta_hat)
            except Exception as _exc:
                if self.control.se_method in ("hessian", "sandwich"):
                    warnings.warn(
                        f"observed Hessian computation failed ({_exc}); "
                        "SE for se_method='{}' will be NaN".format(
                            self.control.se_method
                        ),
                        RuntimeWarning,
                    )
                hess_observed = None

        # BHATLIB-normalized reporting (primary se_method)
        param_names = self._build_report_names()
        (b_report, se, t_stat, p_value,
         cov_report, corr_report, g_report) = self._normalize_for_reporting(
            theta_hat, hess_inv, grad_full,
            active_mask=active_mask,
            hess_observed=hess_observed,
        )

        # Compute the other two SE estimators for the diagnostic block, but
        # only when the user opted in via se_diagnostic. Each is derived at
        # the same converged theta_hat — the only variation is the variance
        # estimator. If one fails (e.g., Hessian-based when hess_observed is
        # None), it stays None and summary() omits it from the table. When the
        # diagnostic is off, only the primary estimator is populated and the
        # side-by-side block is automatically suppressed.
        se_by_method: dict[str, np.ndarray | None] = {
            self.control.se_method: se,
        }
        if getattr(self.control, "se_diagnostic", False):
            _saved_method = self.control.se_method
            try:
                for _alt in ("bhhh", "hessian", "sandwich"):
                    if _alt == _saved_method:
                        continue
                    if _alt in ("hessian", "sandwich") and hess_observed is None:
                        se_by_method[_alt] = None
                        continue
                    try:
                        self.control.se_method = _alt
                        (_, _se_alt, _, _, _, _, _) = self._normalize_for_reporting(
                            theta_hat, hess_inv, grad_full,
                            active_mask=active_mask,
                            hess_observed=hess_observed,
                        )
                        se_by_method[_alt] = _se_alt
                    except Exception:
                        se_by_method[_alt] = None
            finally:
                self.control.se_method = _saved_method

        # Kernel correlations are reported in UNPARAMETERIZED form (the raw
        # correlation entry, e.g. 0.473), matching GAUSS BHATLIB's
        # ``_wantcovariance`` path (``MNP_TRAVELMODE.gss`` lines 199-216,
        # which re-runs ``lpr1`` on ``vecndup(cholker'*cholker)`` and labels
        # the entries ``cor``) and the ``bhatlib_table2_targets`` fixture
        # (``parker01`` documented as the correlation = 0.473, se 0.1598).
        # ``_normalize_for_reporting`` already produces these raw correlations
        # and their unparameterized SEs via the lpr1/lgd1 score path, so no
        # further transform is applied here.

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

        # GAUSS homogeneous "first-differenced-variance = 1" form: the reported
        # kernel covariance IS the differenced kernel K returned by _build_lambda
        # (K[0,0] == 1 pinned). No overall-scale renormalization is applied —
        # betas and Omega are reported on the raw (GAUSS/reported) scale.
        # For IID the kernel is the frozen GAUSS startker (K[0,0] == 1 too).
        lambda_hat = Lambda  # K, with lambda_hat[0,0] == 1

        if self.control.mix and self.ranvar_indices and "omega_params" in params:
            L_raw = _build_omega_cholesky(
                params["omega_params"], self.ranvar_indices, self.control,
            )
            cholesky_L = L_raw
            omega_hat = L_raw @ L_raw.T

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
            se_bhhh=se_by_method.get("bhhh"),
            se_hessian=se_by_method.get("hessian"),
            se_sandwich=se_by_method.get("sandwich"),
        )

    def _default_start_values(self) -> np.ndarray:
        """Generate reasonable starting values for optimization."""
        theta0 = np.zeros(self.n_params, dtype=np.float64)

        # Small random perturbation for betas
        theta0[:self.n_beta] = np.random.randn(self.n_beta) * 0.1
        idx = self.n_beta

        if not self.control.iid:
            dim = self.n_alts - 1
            # GAUSS homogeneous form: I-2 FREE scales (first diff variance pinned
            # to 1). Free scales start at 0 (exp(0)=1).
            n_scale = max(dim - 1, 0)
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
        """Map unparameterized theta to the MNP reporting convention.

        GAUSS homogeneous "first-differenced-variance = 1" form (lpr1): the
        reported kernel covariance IS the differenced kernel ``K`` with
        ``K[0,0] == 1`` PINNED. No overall-scale renormalization is applied:

          - betas are reported RAW (the GAUSS/reported scale);
          - the kernel block reports the raw correlations (``parker``) followed
            by the I-1 scales, where ``scale01`` is LITERALLY 1.0 (pinned, no
            free theta column) and ``scale0k = sqrt(K[k-1, k-1])`` for k >= 2;
          - the random-coefficient Omega entries are reported RAW.

        Notes
        -----
        Because no scalar normalization survives, the Jacobian of this map
        (``J_norm``) is an identity/selection: every reported entry except the
        pinned ``scale01`` corresponds 1:1 to an unparameterized theta entry,
        and ``scale01`` has no theta column (its SE is therefore NaN). This
        makes ``cov_report == cov_unpar`` on the free block.
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

        report: list[float] = []

        # --- Betas (raw) ---
        report.extend(params["beta"])

        # --- Kernel covariance parameters (non-IID) ---
        # K = W1 @ corr @ W1, W1 = diag(scales). Report parker (raw
        # correlations) then scales; scale01 == 1.0 by construction.
        if not self.control.iid:
            scales = np.sqrt(np.clip(np.diag(Lambda), 0.0, None))  # scale01 == 1
            if not self.control.heteronly:
                with np.errstate(divide="ignore", invalid="ignore"):
                    Corr = Lambda / np.outer(scales, scales)
                for i in range(dim):
                    for j in range(i + 1, dim):
                        report.append(float(Corr[i, j]))

            # All (I-1) kernel scales. scale01 is the pinned literal 1.0;
            # scale02.. are the free scales sqrt(K[i,i]).
            for i in range(dim):
                report.append(float(scales[i]))

        # --- Random-coefficient Omega (variance/covariance entries, raw) ---
        # GAUSS reference: MNP_TRAVELMODE.gss line 211 reports
        # ``vecdup(newcorker' * newcorker)`` — i.e., upper-tri (with diag)
        # of Omega. This makes ``CovCOv01`` the variance of the random
        # coefficient (not its standard deviation).
        if (
            self.control.mix
            and self.ranvar_indices is not None
            and "omega_params" in params
        ):
            Omega = _build_omega_direct(
                params["omega_params"], self.ranvar_indices, self.control,
            )
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    report.append(float(Omega[i, i]))
            else:
                for i in range(n_rand):
                    for j in range(i, n_rand):
                        report.append(float(Omega[i, j]))

        # --- Segment parameters ---
        if self.control.nseg > 1:
            seg_params = params.get("segment_params", np.array([]))
            report.extend(seg_params.tolist())

            for h in range(1, self.control.nseg):
                if h - 1 < len(params.get("segment_betas", [])):
                    report.extend(params["segment_betas"][h - 1].tolist())
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
                    n_rand = len(self.ranvar_indices)
                    if self.control.randdiag:
                        for i in range(n_rand):
                            report.append(float(Omega_h[i, i]))
                    else:
                        for i in range(n_rand):
                            for j in range(i, n_rand):
                                report.append(float(Omega_h[i, j]))

        return np.array(report, dtype=np.float64)

    def _build_theta_to_report_map(self) -> dict[int, int | None]:
        """Map theta-space parameter indices to report-space indices.

        Returns a dict ``{theta_idx: report_idx}``. Under the sum-of-squared-
        scales normalization every kernel scale is reported (none absorbed),
        so the only ``None`` entries would come from model-specific frozen
        params; the kernel block itself is a 1:1 theta→report mapping.

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

            # --- FREE scale params in theta-space: I-2 free scales ---
            # GAUSS homogeneous form: scale01 is PINNED (no theta column); the
            # theta block carries only I-2 free scales (scale02..scale0{dim}).
            # Theta layout: [betas, free_scales(I-2), parkers(n_corr)].
            n_free_scale = max(dim - 1, 0)
            scale_theta_start = theta_idx
            theta_idx += n_free_scale  # advance past free scales

            # --- Parker (correlations) in theta-space ---
            n_corr = dim * (dim - 1) // 2 if not self.control.heteronly else 0
            parker_theta_start = theta_idx
            theta_idx += n_corr

            # In report space: parkers come first (after betas), then all dim
            # scales (scale01 pinned, then I-2 free scales).
            parker_report_start = report_idx
            report_idx += n_corr  # advance past parkers in report space

            scale_report_start = report_idx
            report_idx += dim  # all dim scales (incl. pinned scale01) in report

            # Fill mapping for parkers: direct 1:1
            for k in range(n_corr):
                mapping[parker_theta_start + k] = parker_report_start + k

            # Fill mapping for FREE scales: theta free-scale k -> report
            # scale0{k+2} (report scale_report_start is the PINNED scale01,
            # which has NO theta column, so free scales start at offset +1).
            for k in range(n_free_scale):
                mapping[scale_theta_start + k] = scale_report_start + 1 + k

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

            # All I-1 kernel scales (sum of squares == 1); none absorbed.
            for i in range(dim):
                names.append(f"scale{i + 1:02d}")

        if self.control.mix and self.ranvar_indices is not None:
            n_rand = len(self.ranvar_indices)
            if self.control.randdiag:
                for i in range(n_rand):
                    names.append(f"CovCOv{i + 1:02d}")
            else:
                # Upper-triangle (including diagonal), row-by-row —
                # matches _unpar_to_report which loops ``for j in range(i, n_rand)``.
                idx = 0
                for i in range(n_rand):
                    for j in range(i, n_rand):
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
                        # Upper-triangle, matching _unpar_to_report segment block.
                        idx = 0
                        for i in range(n_rand):
                            for j in range(i, n_rand):
                                idx += 1
                                names.append(f"CovCOv{idx:02d}_s{h + 1}")

        return names

    def _observed_hessian(self, theta_hat: np.ndarray) -> np.ndarray:
        """Compute the observed Hessian H = ∂²(-Σ_q ℓ_q)/∂θ∂θᵀ at θ̂ via
        central finite differences over the gradient.

        Scaling note: ``mnp_loglik`` (and the optimizer objective) return the
        *mean* NLL and its gradient (i.e. sum / N). To obtain the Hessian of
        the *summed* NLL we call ``mnp_loglik(..., return_gradient=True)`` and
        multiply the returned gradient by N before differencing. This makes H
        sum-scale so that ``np.linalg.inv(H)`` equals the asymptotic variance
        of θ̂ directly — no additional 1/N correction needed (contrast with
        ``result.hess_inv / N`` which was required when using scipy's mean-scale
        BFGS approximation).

        Step size matches the adaptive eps used in the BHHH computation for
        consistency: ``eps_machine^(1/3) * (1 + |theta_i|)``.

        Returns
        -------
        H : ndarray, shape (n, n)
            Symmetric, positive-semi-definite at the MLE.
        """
        from pybhatlib.models.mnp._mnp_loglik import mnp_loglik

        n = len(theta_hat)
        # Adaptive step: classic optimum for central-FD of a smooth function
        eps_vec = np.cbrt(np.finfo(float).eps) * (1.0 + np.abs(theta_hat))
        H = np.zeros((n, n), dtype=np.float64)

        for j in range(n):
            e = np.zeros(n, dtype=np.float64)
            e[j] = eps_vec[j]
            # Evaluate mean-scale gradient, then scale up to sum-scale (* N)
            _, g_plus_mean = mnp_loglik(
                theta_hat + e, self.X, self.y, self.avail,
                self.n_alts, self.n_beta, self.control,
                self.ranvar_indices, return_gradient=True,
            )
            _, g_minus_mean = mnp_loglik(
                theta_hat - e, self.X, self.y, self.avail,
                self.n_alts, self.n_beta, self.control,
                self.ranvar_indices, return_gradient=True,
            )
            # mnp_loglik returns *negative* mean LL and its gradient.
            # We want H = ∂²(-Σℓ)/∂θ∂θᵀ = ∂²(N*nll_mean)/∂θ∂θᵀ
            # so multiply mean-scale gradient by N.
            g_plus = g_plus_mean * self.N
            g_minus = g_minus_mean * self.N
            H[:, j] = (g_plus - g_minus) / (2.0 * eps_vec[j])

        # Symmetrize to eliminate numerical asymmetry from FD noise
        return 0.5 * (H + H.T)

    def _bhhh_scores_unpar(
        self, theta_hat: np.ndarray, theta_unpar: np.ndarray,
    ) -> np.ndarray:
        """Per-observation BHHH score matrix in unparameterized space.

        ``G[q, k] = d log P_q / d theta_unpar[k]``, shape ``(N, n_theta)``.

        When the analytic gradient is available (``analytic_grad`` and
        ``method in {me, ovus}`` and single segment) this is computed in a
        single analytic pass: the parameterized-space per-observation scores
        from :func:`mnp_per_obs_scores` are projected to unparameterized space
        via the par->unpar Jacobian. BHHH scores transform *exactly* under
        reparameterisation (the score is linear in the Jacobian), so

            d log P_q / d theta_unpar = (d log P_q / d theta_par) @ inv(J_pu),
            J_pu = d theta_unpar / d theta_par,

        and ``J_pu`` is finite-differenced over the cheap parameter map
        ``_param_to_unpar`` (no data passes), preserving the single-pass speed.

        Falls back to central finite differences on the unparameterized
        per-observation log-likelihood (``2 * n_theta`` full-data passes) for
        mixture models or unsupported MVNCD methods.
        """
        from pybhatlib.models.mnp._mnp_grad_analytic import mnp_per_obs_scores
        from pybhatlib.models.mnp._mnp_loglik import (
            _param_to_unpar,
            _per_obs_loglik_unpar,
        )

        n_theta = len(theta_unpar)

        # When parameters are frozen (active_mask), the BHHH path relies on the
        # per-obs score columns being mutually independent so frozen columns can
        # be zeroed cleanly. The par->unpar Jacobian projection couples columns,
        # which would leak a frozen parameter's score into the active block, so
        # we use finite differences (native column independence) in that case.
        active_mask = getattr(self.control, "active_mask", None)
        no_frozen = active_mask is None or bool(np.all(active_mask))

        analytic_ok = (
            getattr(self.control, "analytic_grad", False)
            and self.control.method in ("me", "ovus")
            and self.control.nseg == 1
            and no_frozen
        )
        if analytic_ok:
            try:
                G_par = mnp_per_obs_scores(
                    theta_hat, self.X, self.y, self.avail,
                    self.n_alts, self.n_beta, self.control,
                    self.ranvar_indices,
                )
                J_pu = self._fd_jacobian(
                    lambda th: _param_to_unpar(
                        th, self.n_beta, self.n_alts, self.control,
                        self.ranvar_indices,
                    ),
                    theta_hat,
                )
                G = G_par @ np.linalg.inv(J_pu)
                if np.all(np.isfinite(G)):
                    return G
            except Exception:
                pass  # fall back to finite differences below

        # Central FD on the unparameterized per-obs log-likelihood.
        # Scaled step eps_machine^(1/3) * (1 + |theta_i|) balances truncation
        # and roundoff while respecting parameter magnitude.
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
        return G

    def _normalize_for_reporting(
        self,
        theta_hat: np.ndarray,
        hess_inv: np.ndarray | None,
        gradient: np.ndarray | None,
        active_mask: np.ndarray | None = None,
        hess_observed: np.ndarray | None = None,
    ) -> tuple:
        """Compute BHATLIB-normalized values with standard errors.

        MNP-002b: SEs are computed in the unparameterized covariance/
        correlation space (the GAUSS ``lpr1``/``lgd1`` convention). The
        per-observation scores are obtained by finite differencing
        ``_per_obs_loglik_unpar`` at ``theta_unpar = _param_to_unpar(theta_hat)``,
        not at the parameterized ``theta_hat`` itself. There is no
        delta-method projection through the spherical-Cholesky
        parameterization.

        The only Jacobian that survives is the overall-scale normalization
        from ``_unpar_to_report`` (``scale_norm``: trace-based for non-IID,
        sigma_1 for IID) — a scalar division per block. This is
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

        SE method is selected via ``self.control.se_method``.  The branches
        differ in whether they apply a delta-method projection through the
        par→unpar Jacobian ``J_pu`` (PR #10 review P0 #1: this is *not* a
        "no delta method" path for hessian / sandwich — only BHHH is
        unparameterized end-to-end):

          - "hessian":  ``J_norm @ V_unpar @ J_norm^T``
                        where ``V_unpar = J_pu @ (hess_inv/N) @ J_pu^T``
                        (or ``J_pu @ inv(hess_observed) @ J_pu^T`` when the
                        true observed Hessian is available — preferred).
                        Applies finite-differenced ``J_pu`` (delta-method).
          - "bhhh":     ``J_norm @ (G_unpar^T G_unpar)^{-1} @ J_norm^T``
                        — ``G_unpar`` is computed natively in unparameterized
                        space; truly no parameterization Jacobian.
          - "sandwich": ``J_norm @ V_unpar @ B @ V_unpar @ J_norm^T`` where
                        ``V_unpar`` is as in the hessian path and
                        ``B = G_unpar^T G_unpar``.  Uses ``J_pu`` for the
                        Hessian leg.

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
        se : ndarray         — standard errors
        t_stat : ndarray     — b_report / se
        p_value : ndarray    — two-sided p-values
        cov_report : ndarray — covariance matrix of reporting params
        corr_report : ndarray — correlation matrix of reporting params
        g_report : ndarray   — gradient projected into reporting space
        """
        import warnings

        from pybhatlib.models.mnp._mnp_loglik import _param_to_unpar

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
        # This is a scalar overall-scale normalization per block (scale_norm:
        # trace-based for non-IID, sigma_1 for IID), NOT a delta-method
        # through the spherical-Cholesky parameterization.
        J_norm = self._fd_jacobian(self._unpar_to_report, theta_unpar)

        # MNP-003 active_mask × unparameterized SE: the J-column zeroing
        # that lived here on the parameterized-space Jacobian J is no
        # longer applicable now that the SE projection goes through
        # J_pu (par->unpar) and J_norm (unpar->report).  Frozen theta
        # entries already contribute zero to ``cov_unpar`` via their
        # zeroed rows/cols in ``hess_inv`` (active_mask reconstruction
        # in fit()) and zeroed score columns in ``G`` (BHHH path).  The
        # explicit NaN-marking via ``_build_theta_to_report_map``
        # further down still applies — frozen-only-dependent report
        # params get SE=NaN regardless.

        # MNP-002: select Cov(theta) via control.se_method.
        se_method = self.control.se_method
        cov_unpar: np.ndarray | None = None

        if se_method in ("bhhh", "sandwich"):
            # Per-obs scores in UNPARAMETERIZED space (the lpr1/lgd1 path).
            # Single analytic pass when available (analytic_grad + me/ovus +
            # single segment), projected from parameterized space via the cheap
            # par->unpar Jacobian; otherwise 2*n_theta full-data FD passes.
            G = self._bhhh_scores_unpar(theta_hat, theta_unpar)

            # Zero out columns for frozen (inactive) parameters from PR #6
            # active_mask (MNPControl attribute).  This prevents inactive
            # directions from contaminating the score covariance with random
            # gradient noise.  The guard makes this a no-op until PR #6 lands.
            if (
                hasattr(self.control, "active_mask")
                and self.control.active_mask is not None
            ):
                G[:, ~self.control.active_mask] = 0.0

            B = G.T @ G

            # Project out parameters whose score is numerically zero. This
            # happens routinely for IID models with unused Lambda slots, or
            # when control options pin a coefficient. We warn on
            # ill-conditioning rather than letting B be silently singular.
            score_norms = np.linalg.norm(G, axis=0)
            structural_zero = score_norms < 1e-12
            scoring_active_mask = ~structural_zero

            G_active = G[:, scoring_active_mask]
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
            active_idx = np.where(scoring_active_mask)[0]
            B_inv[np.ix_(active_idx, active_idx)] = B_inv_active

            if se_method == "bhhh":
                # BHHH is the only path that's truly unparameterized end-to-end:
                # G_unpar is computed natively in unparameterized space, so
                # B_inv is already the unparameterized covariance.
                cov_unpar = B_inv
            else:  # sandwich
                # Sandwich = A^{-1} B A^{-1} with A = J_pu^T H_unpar J_pu in
                # unparameterized space.  We assemble A_unpar via the
                # par->unpar Jacobian (FD; PR #10 review P0 #1: this is a
                # delta-method projection regardless of whether we start from
                # the true observed Hessian or the BFGS approximation).
                # Prefer the true observed Hessian when available (PR #16
                # P0 #1 fix); otherwise fall back to the BFGS hess_inv / N.
                if hess_observed is not None:
                    try:
                        H_inv_par = np.linalg.inv(hess_observed)
                    except np.linalg.LinAlgError:
                        warnings.warn(
                            "Observed Hessian is singular; falling back to pinv. "
                            "Check for unidentified parameters.",
                            RuntimeWarning, stacklevel=2,
                        )
                        H_inv_par = np.linalg.pinv(hess_observed)
                    J_pu = self._fd_jacobian(
                        lambda th: _param_to_unpar(
                            th, self.n_beta, self.n_alts, self.control,
                            self.ranvar_indices,
                        ),
                        theta_hat,
                    )
                    A_unpar = J_pu @ H_inv_par @ J_pu.T
                    cov_unpar = A_unpar @ B @ A_unpar
                elif hess_inv is not None:
                    J_pu = self._fd_jacobian(
                        lambda th: _param_to_unpar(
                            th, self.n_beta, self.n_alts, self.control,
                            self.ranvar_indices,
                        ),
                        theta_hat,
                    )
                    A_unpar = J_pu @ (hess_inv / self.N) @ J_pu.T
                    cov_unpar = A_unpar @ B @ A_unpar
                else:
                    warnings.warn(
                        "se_method='sandwich' requested but neither "
                        "observed Hessian nor BFGS hess_inv is available; "
                        "falling back to BHHH covariance.",
                        RuntimeWarning, stacklevel=2,
                    )
                    cov_unpar = B_inv
        elif se_method == "hessian":
            # Hessian path: project the parameterized H to unparameterized
            # space via J_pu (delta-method).  Prefer hess_observed (true H)
            # over hess_inv / N (BFGS approximation).
            if hess_observed is not None:
                try:
                    H_inv_par = np.linalg.inv(hess_observed)
                except np.linalg.LinAlgError:
                    warnings.warn(
                        "Observed Hessian is singular; falling back to pinv. "
                        "Check for unidentified parameters.",
                        RuntimeWarning, stacklevel=2,
                    )
                    H_inv_par = np.linalg.pinv(hess_observed)
                J_pu = self._fd_jacobian(
                    lambda th: _param_to_unpar(
                        th, self.n_beta, self.n_alts, self.control,
                        self.ranvar_indices,
                    ),
                    theta_hat,
                )
                cov_unpar = J_pu @ H_inv_par @ J_pu.T
            elif hess_inv is not None:
                J_pu = self._fd_jacobian(
                    lambda th: _param_to_unpar(
                        th, self.n_beta, self.n_alts, self.control,
                        self.ranvar_indices,
                    ),
                    theta_hat,
                )
                cov_unpar = J_pu @ (hess_inv / self.N) @ J_pu.T
            else:
                warnings.warn(
                    "se_method='hessian' requested but neither observed "
                    "Hessian nor BFGS hess_inv is available; SE will be NaN.",
                    RuntimeWarning, stacklevel=2,
                )

        if cov_unpar is not None:
            cov_report = J_norm @ cov_unpar @ J_norm.T
            se = np.sqrt(np.maximum(np.diag(cov_report), 0.0))
        else:
            cov_report = np.eye(n_report)
            se = np.full(n_report, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = np.where(se > 0, b_report / se, 0.0)
            p_value = 2.0 * (1.0 - _ndtr(np.abs(t_stat)))

        # Correlation matrix of reporting parameters.  PR #4 review P0 #2:
        # gate on ``cov_unpar`` (the actual SE source), not on the observed
        # Hessian — the BHHH path produces a valid cov_unpar without ever
        # touching ``hess_observed``.
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

        # ------------------------------------------------------------------
        # Pinned kernel scale01 (GAUSS homogeneous form): the first differenced
        # scale is fixed at 1.0 by construction (no free theta column), so its
        # SE/t-stat/p-value are NaN. Its report index is n_beta + n_corr
        # (betas, then parkers, then scale01 leads the scale block).
        # ------------------------------------------------------------------
        if not self.control.iid:
            dim = self.n_alts - 1
            n_corr = dim * (dim - 1) // 2 if not self.control.heteronly else 0
            scale01_ri = self.n_beta + n_corr
            if scale01_ri < len(se):
                se[scale01_ri] = np.nan
                t_stat[scale01_ri] = np.nan
                p_value[scale01_ri] = np.nan

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
