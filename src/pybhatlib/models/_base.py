"""Base model abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all pybhatlib models.

    Subclasses implement the estimation routine as :meth:`_fit` (returning the
    model-specific results object).  The public :meth:`fit` wrapper delegates to
    it and caches the result on ``self.results_`` so post-estimation helpers
    (``predict`` / ``ate`` / ...) can be called without threading the results
    object back in by hand.
    """

    #: Most recent results object, populated by :meth:`fit` (``None`` until then).
    results_ = None

    @abstractmethod
    def _fit(self, *args, **kwargs):
        """Estimate parameters and return the model-specific results object."""
        ...

    def fit(self, *args, **kwargs):
        """Estimate model parameters.

        Delegates to the model-specific :meth:`_fit`, stores the returned
        results object on ``self.results_``, and returns it.
        """
        self.results_ = self._fit(*args, **kwargs)
        return self.results_

    def _require_results(self):
        """Return ``self.results_`` or raise if the model has not been fit."""
        if self.results_ is None:
            raise RuntimeError(
                f"{type(self).__name__} has not been fit yet; call .fit() first."
            )
        return self.results_
