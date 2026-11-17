"""Unmixed null benchmarks and reuse of probabilities computed during fits."""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_probability_recorder = ContextVar("fit_probability_recorder", default=None)


def record_probabilities(values, *, log=False):
    """Observe already-computed chosen-outcome probabilities when requested.

    With ``log=True`` ``values`` are log-probabilities and are exponentiated
    only if a recorder is active. On the GPU path the running sum stays a
    device tensor (no host sync); :class:`ProbabilityCache` converts it lazily.
    """
    recorder = _probability_recorder.get()
    if recorder is None:
        return
    if hasattr(values, "detach"):  # GPU: keep the sum on-device.
        v = values.detach()
        recorder[0] = recorder[0] + (v.exp() if log else v).sum()
        recorder[1] += values.numel()
    else:
        values = np.asarray(values)
        if log:
            values = np.exp(values)
        recorder[0] = recorder[0] + values.sum()
        recorder[1] += values.size


@contextmanager
def pause_probability_recording():
    """Exclude perturbed likelihoods used by numerical differentiation."""
    token = _probability_recorder.set(None)
    try:
        yield
    finally:
        _probability_recorder.reset(token)


class ProbabilityCache:
    """Cache an arithmetic mean per evaluated parameter vector, without refits.

    Only the ``max_entries`` most recently evaluated vectors are kept (the
    optimizer's final accepted point may be followed by line-search trials).
    Values may be device tensors until read through :meth:`get`.
    """

    def __init__(self, objective, n_obs, max_entries=8):
        self.objective = objective
        self.n_obs = n_obs
        self.max_entries = max_entries
        self.values = {}

    def __call__(self, theta):
        return self.evaluate(theta, self.objective)

    def evaluate(self, theta, function):
        recorder = [0.0, 0]
        token = _probability_recorder.set(recorder)
        try:
            result = function(theta)
        finally:
            _probability_recorder.reset(token)
        if recorder[1] == self.n_obs:
            key = np.asarray(theta, dtype=np.float64).tobytes()
            self.values.pop(key, None)  # re-evaluation refreshes recency
            self.values[key] = recorder[0] / self.n_obs
            while len(self.values) > self.max_entries:
                self.values.pop(next(iter(self.values)))
        return result

    def get(self, theta):
        value = self.values.get(np.asarray(theta, dtype=np.float64).tobytes(), np.nan)
        return float(value)


def rho_squared(loglik, loglik_constants, loglik_zero, n_free, n_norm,
                n_constants=-1):
    """Return rho2 against zero/constants, then both adjusted versions.

    ``loglik``/``loglik_*`` are normalized by ``n_norm`` (cases, or individuals
    for panel models). Adjusted rho2 is ``1 - (LL - K) / LL(0)`` on *total*
    log likelihoods. The constants-based adjustment is
    ``1 - (LL - (K - Kc)) / LL(C)``. Penalties are divided by ``n_norm``
    here to match mean likelihoods. Unknown Kc or zero denominators give NaN.
    """
    def ratio(num, den):
        with np.errstate(all="ignore"):
            r = 1.0 - np.divide(num, den)
        return float(r) if np.isfinite(r) else np.nan

    k = float(n_free) / n_norm if n_norm and n_free >= 0 else np.nan
    kc = float(n_constants) / n_norm if n_norm and n_constants >= 0 else np.nan
    return (
        ratio(loglik, loglik_zero),
        ratio(loglik, loglik_constants),
        ratio(loglik - k, loglik_zero),
        ratio(loglik - (k - kc), loglik_constants),
    )


class FitMetricsMixin:
    """rho-squared properties/summary lines for results with LL(C)/LL(0)."""

    @property
    def rho_squared(self):
        """``(rho2_zero, rho2_constants, adjusted_zero, adjusted_constants)``."""
        k = self.n_free_params if self.n_free_params >= 0 else len(self.params)
        n = getattr(self, "n_ind", 0) or self.n_obs
        return rho_squared(
            self.loglik, self.loglik_constants, getattr(self, "loglik_zero", np.nan),
            k, n, getattr(self, "n_constants_params", -1),
        )

    @property
    def rho2_zero(self):
        return self.rho_squared[0]

    @property
    def rho2_constants(self):
        return self.rho_squared[1]

    @property
    def rho2_adjusted(self):
        return self.rho_squared[2]

    @property
    def rho2_adjusted_constants(self):
        """``1 - (LL - (K - Kc)) / LL(C)`` on total likelihoods."""
        return self.rho_squared[3]

    def rho2_summary_lines(self, *, include_zero=True):
        out = []
        labels = ("Rho2 vs LL(0)", "Rho2 vs LL(C)",
                  "Adj. rho2 vs LL(0)", "Adj. rho2 vs LL(C)")
        for index, (label, value) in enumerate(zip(labels, self.rho_squared)):
            if not include_zero and index in (0, 2):
                continue
            formatted = f"{value:>14.6f}" if not np.isnan(value) else f"{'N/A':>14s}"
            out.append(f"  {label:<23s}{formatted}")
        return out


def baseline_loglikelihoods(y, category_counts, *, availability=None,
                           row_to_person=None, person_weights=None, family="mnl"):
    """Mean unmixed constants-only and uniform null log likelihoods.

    For ordinal vectors, category dimensions are independent in both nulls.
    With full availability, empirical shares give the analytic baseline. With
    unavailable alternatives, fit one shared set of alternative constants
    using the basic MNL or IID MNP likelihood (``family``), not separate shares
    per availability pattern. No random mixing or covariance estimation is
    performed. Panel nulls assume independent occasions; their weights and
    per-individual normalization match the fitted model.

    Ordinal nulls retain independent empirical category shares, equivalent to
    unrestricted thresholds for each dimension with independent normal errors.
    This is not a correlated MORP null fit. LL(0) remains uniform over each
    observation's available categories.
    """
    y = np.asarray(y, dtype=np.int64)
    if y.ndim == 1:
        y = y[:, None]
    n = len(y)
    persons = np.arange(n) if row_to_person is None else np.asarray(row_to_person)
    n_ind = int(persons.max()) + 1
    weights = np.ones(n_ind) if person_weights is None else np.asarray(person_weights)
    case_weights = weights[persons]
    log_c = np.zeros(n)
    log_0 = np.zeros(n)
    for d, k in enumerate(category_counts):
        if availability is None:
            log_0 -= np.log(k)
        else:
            avail = np.asarray(availability) > 0
            if y.shape[1] != 1:
                raise ValueError("Availability is only supported for single-choice outcomes")
            counts_available = avail.sum(axis=1)
            if np.any(counts_available == 0) or not np.all(avail[np.arange(n), y[:, d]]):
                raise ValueError("Observed choices must be available")
            log_0 -= np.log(counts_available)
            if not np.all(avail):
                ll_c = _choice_constants_loglik(y[:, d], avail, case_weights, family)
                active = case_weights > 0
                return ll_c / n_ind, float(case_weights[active] @ log_0[active] / n_ind)
        counts = np.bincount(y[:, d], weights=case_weights, minlength=k)
        if counts.sum() > 0:
            chosen_shares = counts[y[:, d]] / counts.sum()
            np.log(chosen_shares, out=chosen_shares, where=chosen_shares > 0)
            log_c += chosen_shares
    active = case_weights > 0
    return (
        float(case_weights[active] @ log_c[active] / n_ind),
        float(case_weights[active] @ log_0[active] / n_ind),
    )


def _choice_constants_loglik(
    y: np.ndarray, availability: np.ndarray, weights: np.ndarray, family: str,
) -> float:
    """Fit weighted ASCs through the basic unmixed model's likelihood routines.

    Aggregate identical (choice set, chosen alternative) rows: constants-only
    probabilities are identical within each cell, including for panel data.
    The first constant is fixed to zero for identification. A failed auxiliary
    fit yields NaN with a warning rather than a misleading fitted benchmark.
    """
    import warnings
    import pandas as pd
    from scipy.optimize import minimize

    if family not in ("mnl", "mnp"):
        raise ValueError("Constants-only family must be 'mnl' or 'mnp'")
    active = weights > 0
    if not active.any():
        return 0.0
    cells, inverse = np.unique(
        np.column_stack([availability[active], y[active]]), axis=0,
        return_inverse=True,
    )
    counts = np.bincount(inverse, weights=weights[active])
    k = availability.shape[1]
    choices = cells[:, -1].astype(int)
    avail = cells[:, :-1]
    alts = [f"a{j}" for j in range(k)]
    frame = pd.DataFrame({
        alt: (choices == j).astype(int) for j, alt in enumerate(alts)
    })
    for j in range(k):
        frame[f"av{j}"] = avail[:, j]
    frame["uno"] = 1.0
    spec = {f"ASC{j}": {alts[j]: "uno"} for j in range(1, k)}
    if k == 1:
        return 0.0
    if family == "mnl":
        from pybhatlib.models.mnl import MNLControl, MNLModel
        from pybhatlib.models.mnl._mnl_loglik import mnl_loglik, mnl_gradient
        model = MNLModel(frame, alts, spec=spec,
                         availability=[f"av{j}" for j in range(k)],
                         control=MNLControl(verbose=0, want_covariance=False))
        dta, iv, av, dv, _ = model._build_data_arrays()

        def objective(theta):
            ll = mnl_loglik(theta, dta, iv, av, dv, k, k - 1)
            score = mnl_gradient(theta, dta, iv, av, dv, k, k - 1)
            return (
                -float(counts @ ll) / counts.sum(),
                -(counts @ score) / counts.sum(),
            )
    else:
        from pybhatlib.models.mnp import MNPControl, MNPModel
        from pybhatlib.models.mnp._mnp_loglik import _per_obs_loglik
        from pybhatlib.models.mnp._mnp_grad_analytic import mnp_per_obs_scores
        ctrl = MNPControl(iid=True, mix=False, nseg=1, method="ovus",
                          verbose=0, want_covariance=False)
        model = MNPModel(frame, alts, spec=spec,
                         availability=[f"av{j}" for j in range(k)], control=ctrl)

        def objective(theta):
            args = (theta, model.X, model.y, model.avail, k, k - 1, ctrl)
            ll = _per_obs_loglik(*args)
            score = mnp_per_obs_scores(*args)
            return (
                -float(counts @ ll) / counts.sum(),
                -(counts @ score) / counts.sum(),
            )

    with pause_probability_recording():
        result = minimize(objective, np.zeros(k - 1), jac=True, method="BFGS",
                          options={"gtol": 1e-8, "maxiter": 1000})
    if not np.isfinite(result.fun) or (
        not result.success and np.max(np.abs(result.jac)) > 1e-6
    ):
        warnings.warn(f"Constants-only {family.upper()} fit failed: {result.message}",
                      RuntimeWarning, stacklevel=2)
        return np.nan
    return -float(result.fun) * counts.sum()


def mdcev_constants_loglik(
    consumption: np.ndarray, prices: np.ndarray, *, utility: str,
    outside_good_gamma: float,
    observation_weights: np.ndarray | None = None,
    row_to_person: np.ndarray | None = None,
    person_weights: np.ndarray | None = None,
) -> float:
    """Fit basic MDCEV constants in both utility and inside-good gamma specs.

    The outside utility is the reference, outside gamma remains fixed, and
    sigma is estimated. Preserve consumption, prices, utility form and weights.
    Mixed-model baselines use this basic engine without simulation/mixing;
    panel contributions are weighted as in the mixed engine and divided by
    the number of individuals. LL(0) is intentionally not defined for MDCEV.
    """
    import warnings
    import pandas as pd
    from pybhatlib.models.mdcev import MDCEVControl, MDCEVModel

    n, k = consumption.shape
    persons = np.arange(n) if row_to_person is None else np.asarray(row_to_person)
    n_ind = int(persons.max()) + 1
    weights = np.ones(n) if observation_weights is None else np.asarray(observation_weights)
    if person_weights is not None:
        weights = np.asarray(person_weights)[persons]
    alts = [f"c{j}" for j in range(k)]
    pcols = [f"p{j}" for j in range(k)]
    frame = pd.DataFrame(consumption, columns=alts)
    frame[pcols] = prices
    frame["baseline_weight"] = weights
    utility_spec = {f"ASC{j}": {alts[j]: "uno"} for j in range(1, k)}
    gamma_spec = {f"G{j}": {alts[j]: "uno"} for j in range(1, k)}
    ctrl = MDCEVControl(utility=utility, outside_good_gamma=outside_good_gamma,
                        weight_var="baseline_weight", verbose=0,
                        want_covariance=False, maxiter=1000, tol=1e-6)
    model = MDCEVModel(frame, alts, availability=pcols,
                       utility_spec=utility_spec, gamma_spec=gamma_spec, control=ctrl)
    model._skip_fit_diagnostics = True  # Do not recursively fit the null model.
    with pause_probability_recording():
        result = model.fit()
    if not result.converged or not np.isfinite(result.loglik):
        warnings.warn("Constants-only MDCEV fit did not converge; LL(C) is unavailable",
                      RuntimeWarning, stacklevel=2)
        return np.nan
    return float(result.loglik * n / n_ind)
