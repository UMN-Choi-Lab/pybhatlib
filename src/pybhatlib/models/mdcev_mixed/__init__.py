"""MDCEV mixed / panel family (Phase 3): logit-Jacobian kernel over the shared
MSL engine.

This package hosts the mixed / panel Multiple Discrete-Continuous Extreme Value
(MDCEV) model facade. Mixing is over the **baseline-utility** ``beta`` only; the
translation (``gamma``) and the MDCEV kernel scale are kernel-owned, non-mixed
parameters. There is **no copula** (``kernel_dim == 0``, ``nord == 0``).

Public API:

* :class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_control.MDCEVMixedControl`
* :class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_results.MDCEVMixedResults`
* :class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_model.MDCEVMixedModel`
* :class:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel.LogitJacobianKernel`
* :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_ate.mdcev_mixed_ate`
* :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_ate.mdcev_mixed_ate_from_params`
* :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast.mdcev_mixed_predict`
* :func:`~pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast.mdcev_mixed_predict_choice`
* :class:`~pybhatlib.mixed._predict.MixedATEResult` (harmonized ATE result type)
"""

from pybhatlib.models.mdcev_mixed._mdcev_mixed_control import MDCEVMixedControl
from pybhatlib.models.mdcev_mixed._mdcev_mixed_kernel import LogitJacobianKernel
from pybhatlib.models.mdcev_mixed._mdcev_mixed_model import MDCEVMixedModel
from pybhatlib.models.mdcev_mixed._mdcev_mixed_results import MDCEVMixedResults
from pybhatlib.models.mdcev_mixed._mdcev_mixed_ate import (
    MixedATEResult,
    mdcev_mixed_ate,
    mdcev_mixed_ate_from_params,
)
from pybhatlib.models.mdcev_mixed._mdcev_mixed_forecast import (
    mdcev_mixed_forecast,
    mdcev_mixed_predict,
    mdcev_mixed_predict_choice,
)

__all__ = [
    "MDCEVMixedControl",
    "MDCEVMixedResults",
    "MDCEVMixedModel",
    "LogitJacobianKernel",
    "MixedATEResult",
    "mdcev_mixed_ate",
    "mdcev_mixed_ate_from_params",
    "mdcev_mixed_predict",
    "mdcev_mixed_predict_choice",
    "mdcev_mixed_forecast",
]
