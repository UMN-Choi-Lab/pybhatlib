"""Share-based null benchmarks and reuse of probabilities computed during fits."""

from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np


_probability_recorder = ContextVar("fit_probability_recorder", default=None)


def record_probabilities(values):
    """Observe already-computed chosen-outcome probabilities when requested."""
    recorder = _probability_recorder.get()
    if recorder is None:
        return
    if hasattr(values, "detach"):  # GPU: transfer just a sum, not the tensor.
        recorder[0] += float(values.detach().sum().item())
        recorder[1] += values.numel()
    else:
        values = np.asarray(values)
        recorder[0] += float(values.sum())
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
    """Cache an arithmetic mean per evaluated parameter vector, without refits."""

    def __init__(self, objective, n_obs):
        self.objective = objective
        self.n_obs = n_obs
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
            self.values[np.asarray(theta, dtype=np.float64).tobytes()] = recorder[0] / self.n_obs
        return result

    def get(self, theta):
        return self.values.get(np.asarray(theta, dtype=np.float64).tobytes(), np.nan)


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
