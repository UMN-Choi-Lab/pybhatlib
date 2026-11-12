"""Share-based null benchmarks and reuse of probabilities computed during fits."""

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


def rho_squared(loglik, loglik_constants, loglik_zero, n_free, n_norm):
    """Return ``(rho2_zero, rho2_constants, rho2_adjusted)``.

    ``loglik``/``loglik_*`` are normalized by ``n_norm`` (cases, or individuals
    for panel models). Adjusted rho2 is ``1 - (LL - K) / LL(0)`` on *total*
    log likelihoods, i.e. ``K`` is divided by ``n_norm`` here. NaN-safe.
    """
    def ratio(num, den):
        with np.errstate(all="ignore"):
            r = 1.0 - num / den
        return float(r) if np.isfinite(r) else np.nan

    k = float(n_free) / n_norm if n_norm and n_free >= 0 else np.nan
    return (
        ratio(loglik, loglik_zero),
        ratio(loglik, loglik_constants),
        ratio(loglik - k, loglik_zero),
    )


class FitMetricsMixin:
    """rho-squared properties/summary lines for results with LL(C)/LL(0)."""

    @property
    def rho_squared(self):
        """``(rho2_zero, rho2_constants, adjusted_rho2_zero)``."""
        k = self.n_free_params if self.n_free_params >= 0 else len(self.params)
        n = getattr(self, "n_ind", 0) or self.n_obs
        return rho_squared(self.loglik, self.loglik_constants, self.loglik_zero, k, n)

    @property
    def rho2_zero(self):
        return self.rho_squared[0]

    @property
    def rho2_constants(self):
        return self.rho_squared[1]

    @property
    def rho2_adjusted(self):
        return self.rho_squared[2]

    def rho2_summary_lines(self):
        out = []
        for label, value in zip(
            ("Rho2 vs LL(0)", "Rho2 vs LL(C)", "Adj. rho2 vs LL(0)"), self.rho_squared
        ):
            formatted = f"{value:>14.6f}" if not np.isnan(value) else f"{'N/A':>14s}"
            out.append(f"  {label:<23s}{formatted}")
        return out


def baseline_loglikelihoods(y, category_counts, *, availability=None,
                           row_to_person=None, person_weights=None):
    """Mean independent empirical-share and uniform null log likelihoods.

    For ordinal vectors, category dimensions are independent in both nulls.
    For varying choice sets, shares are conditional on the availability set:
    this is a descriptive share benchmark, not a restricted fit of the full
    kernel/covariance model. Panel nulls additionally assume independent
    occasions. Their normalization/weights match the fitted individual LL.
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
            groups = np.zeros(n, dtype=np.int64)
        else:
            avail = np.asarray(availability) > 0
            if y.shape[1] != 1:
                raise ValueError("Availability is only supported for single-choice outcomes")
            counts_available = avail.sum(axis=1)
            if np.any(counts_available == 0) or not np.all(avail[np.arange(n), y[:, d]]):
                raise ValueError("Observed choices must be available")
            log_0 -= np.log(counts_available)
            if np.all(avail == avail[0]):
                groups = np.zeros(n, dtype=np.int64)
            else:
                _, groups = np.unique(avail, axis=0, return_inverse=True)
        for group in np.unique(groups):
            rows = groups == group
            counts = np.bincount(y[rows, d], weights=case_weights[rows], minlength=k)
            if counts.sum() == 0:
                counts = np.bincount(y[rows, d], minlength=k)
            chosen_shares = counts[y[rows, d]] / counts.sum()
            # Zero-weight outcomes may have zero weighted shares; their LL is
            # omitted below, rather than evaluating 0 * log(0).
            np.log(chosen_shares, out=chosen_shares, where=chosen_shares > 0)
            log_c[rows] += chosen_shares
    active = case_weights > 0
    return (
        float(case_weights[active] @ log_c[active] / n_ind),
        float(case_weights[active] @ log_0[active] / n_ind),
    )
