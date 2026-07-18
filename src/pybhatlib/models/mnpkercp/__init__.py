"""Mixed (random-coefficient / panel) MNP with rc<->kernel correlation — MSL over Halton draws.

Ports Bhat's ``MNPKERCP.gss`` (cross-sectional or panel MNP with Yeo-Johnson random
coefficients and, cross-sectionally, correlation between the random coefficients and the
kernel errors). Built on the shared MSL engine in :mod:`pybhatlib.mixed`; the MVNCD kernel
uses :func:`pybhatlib.gradmvn.mvncd` (OVUS) and the copula uses
:mod:`pybhatlib.mixed._copula`. See ``docs/plans/MIXED_PANEL_MODELS_PLAN.md``.

STATUS (2026-07-16): log-likelihood matches GAUSS value-for-value (rel 2.9e-9); the
mean/scale/kernel/lambda gradients are exact. Two follow-ups remain before the joint
correlation is fully estimable and the copula-active path is GAUSS-validated — see the
package NOTES and the plan.
"""

from pybhatlib.models.mnpkercp._mnpkercp_ate import (
    MixedATEResult,
    mnpkercp_ate,
    mnpkercp_ate_from_params,
)
from pybhatlib.models.mnpkercp._mnpkercp_control import MNPKerCPControl
from pybhatlib.models.mnpkercp._mnpkercp_forecast import (
    mnpkercp_forecast,
    mnpkercp_predict,
    mnpkercp_predict_choice,
)
from pybhatlib.models.mnpkercp._mnpkercp_kernel import MvncdKernel
from pybhatlib.models.mnpkercp._mnpkercp_model import MNPKerCPModel
from pybhatlib.models.mnpkercp._mnpkercp_results import MNPKerCPResults

__all__ = [
    "MNPKerCPControl",
    "MNPKerCPResults",
    "MNPKerCPModel",
    "MvncdKernel",
    "MixedATEResult",
    "mnpkercp_ate",
    "mnpkercp_ate_from_params",
    "mnpkercp_predict",
    "mnpkercp_predict_choice",
    "mnpkercp_forecast",
]
