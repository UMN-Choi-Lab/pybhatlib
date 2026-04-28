"""pybhatlib: Python implementation of BHATLIB for matrix-based econometric inference."""

from pybhatlib._version import __version__
from pybhatlib.models.mnp import (
    ATEResult,
    MNPControl,
    MNPModel,
    MNPResults,
    mnp_ate,
    mnp_predict,
    mnp_predict_choice,
)
from pybhatlib.models.morp import (
    MORPATEResult,
    MORPControl,
    MORPModel,
    MORPResults,
    morp_ate,
    morp_predict,
    morp_predict_category,
)

__all__ = [
    "__version__",
    # MNP
    "MNPControl",
    "MNPModel",
    "MNPResults",
    "ATEResult",
    "mnp_ate",
    "mnp_predict",
    "mnp_predict_choice",
    # MORP
    "MORPControl",
    "MORPModel",
    "MORPResults",
    "MORPATEResult",
    "morp_ate",
    "morp_predict",
    "morp_predict_category",
]
