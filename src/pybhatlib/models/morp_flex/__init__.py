"""Mixed (random-coefficient / panel) MORP with flexible Yeo-Johnson kernel errors.

Ports Bhat's "Joint Ordered YJ with Cross-Sectional or Panel Random Coefficients"
driver (multivariate ordered-response probit with YJ/normal/log-normal random
coefficients, optional YJ kernel errors, and cross-sectional rc<->kernel correlation).
Built on the shared MSL engine in :mod:`pybhatlib.mixed`; the rectangle-MVNCD kernel
uses :func:`pybhatlib.gradmvn._pdfrectn.pdfrectn` and reuses the copula seam from
:mod:`pybhatlib.mixed._copula`. See ``docs/plans/MIXED_PANEL_MODELS_PLAN.md``.
"""

from pybhatlib.models.morp_flex._morp_flex_control import MORPFlexControl
from pybhatlib.models.morp_flex._morp_flex_results import MORPFlexResults
from pybhatlib.models.morp_flex._morp_flex_model import MORPFlexModel
from pybhatlib.models.morp_flex._morp_flex_kernel import RectMvncdKernel
from pybhatlib.models.morp_flex._morp_flex_ate import (
    MORPFlexATEResult,
    morp_flex_ate,
    morp_flex_ate_from_params,
)
from pybhatlib.models.morp_flex._morp_flex_forecast import (
    morp_flex_predict,
    morp_flex_predict_category,
)

__all__ = [
    "MORPFlexControl",
    "MORPFlexResults",
    "MORPFlexModel",
    "RectMvncdKernel",
    "MORPFlexATEResult",
    "morp_flex_ate",
    "morp_flex_ate_from_params",
    "morp_flex_predict",
    "morp_flex_predict_category",
]
