"""Multinomial Probit (MNP) model."""

from pybhatlib.models.mnp._mnp_ate import (
    ATEResult,
    MNPATEResult,
    mnp_ate,
    mnp_ate_from_params,
)
from pybhatlib.models.mnp._mnp_control import MNPControl
from pybhatlib.models.mnp._mnp_forecast import mnp_predict, mnp_predict_choice
from pybhatlib.models.mnp._mnp_model import MNPModel
from pybhatlib.models.mnp._mnp_results import MNPResults

__all__ = [
    "MNPControl",
    "MNPResults",
    "MNPModel",
    "MNPATEResult",
    "ATEResult",  # deprecated alias for MNPATEResult
    "mnp_ate",
    "mnp_ate_from_params",
    "mnp_predict",
    "mnp_predict_choice",
]
