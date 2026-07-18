"""Mixed (random-coefficient / panel) MNL model — MSL over Halton draws.

Ports Bhat's ``MIXMNL.gss`` (Bhat 2001 mixed logit with normal / log-normal /
Yeo-Johnson mixing, cross-sectional or panel). Built on the shared MSL engine in
:mod:`pybhatlib.mixed`; see ``docs/plans/MIXED_PANEL_MODELS_PLAN.md``.
"""

from pybhatlib.models.mixmnl._mixmnl_control import MixMNLControl
from pybhatlib.models.mixmnl._mixmnl_results import MixMNLResults
from pybhatlib.models.mixmnl._mixmnl_model import MixMNLModel
from pybhatlib.models.mixmnl._mixmnl_kernel import SoftmaxKernel
from pybhatlib.models.mixmnl._mixmnl_ate import (
    MixedATEResult,
    mixmnl_ate,
    mixmnl_ate_from_params,
)
from pybhatlib.models.mixmnl._mixmnl_forecast import (
    mixmnl_predict,
    mixmnl_predict_choice,
)

__all__ = [
    "MixMNLControl",
    "MixMNLResults",
    "MixMNLModel",
    "SoftmaxKernel",
    "MixedATEResult",
    "mixmnl_ate",
    "mixmnl_ate_from_params",
    "mixmnl_predict",
    "mixmnl_predict_choice",
]
