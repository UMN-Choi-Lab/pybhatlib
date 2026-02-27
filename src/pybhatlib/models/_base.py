"""Base model abstract class."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for all pybhatlib models."""

    @abstractmethod
    def fit(self):
        """Estimate model parameters."""
        ...
