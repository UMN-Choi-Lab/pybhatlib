"""pybhatlib: Python implementation of BHATLIB for matrix-based econometric inference."""

from pybhatlib._version import __version__
from pybhatlib.models.mdcev import (
    MDCEVATEResult,
    MDCEVControl,
    MDCEVModel,
    MDCEVResults,
    mdcev_ate,
)
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
from pybhatlib.models.mixmnl import MixMNLControl, MixMNLModel, MixMNLResults
from pybhatlib.models.mnpkercp import (
    MNPKerCPControl,
    MNPKerCPModel,
    MNPKerCPResults,
)

__all__ = [
    "__version__",
    # MDCEV
    "MDCEVControl",
    "MDCEVModel",
    "MDCEVResults",
    "MDCEVATEResult",
    "mdcev_ate",
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
    # Mixed / panel families (random coefficients, MSL over Halton draws)
    "MixMNLControl",
    "MixMNLModel",
    "MixMNLResults",
    "MNPKerCPControl",
    "MNPKerCPModel",
    "MNPKerCPResults",
]
