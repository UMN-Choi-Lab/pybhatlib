"""Shared Maximum-Simulated-Likelihood engine for mixed / panel choice models.

One kernel-agnostic MSL skeleton (Halton draws -> Dmask panel broadcast ->
correlation Cholesky -> Yeo-Johnson / log-normal / normal transform -> scale ->
inject -> per-draw kernel probability -> per-person product -> average over draws
-> analytic score through the reparameterisation Jacobians). Per-model families
(``models/mixmnl`` etc.) supply only a :class:`MixedKernel`.

See ``docs/plans/MIXED_PANEL_MODELS_PLAN.md``.
"""

from pybhatlib.mixed._draws import (
    DrawSource,
    FixtureDrawSource,
    ScipyHaltonDrawSource,
    panel_reshape_gauss,
)
from pybhatlib.mixed._spec import MixingSpec
from pybhatlib.mixed._reparam import (
    EstimationSpace,
    ParamLayout,
    ParamSpace,
    RCState,
    ReportingSpace,
)
from pybhatlib.mixed._rc_pipeline import (
    RandomCoefPipeline,
    RCJacobian,
    RealizationCache,
)
from pybhatlib.mixed._kernel import KernelObsResult, MixedKernel
from pybhatlib.mixed._copula import (
    condition,
    gcondnewcov,
    gcondnewmean,
    gcondspecialnewmean,
)
from pybhatlib.mixed._engine import (
    DesignData,
    MixedMSLEstimator,
    MSLConfig,
    Tracer,
)
from pybhatlib.mixed._predict import (
    MixedATEResult,
    MixedPredictComponents,
    default_kernel_predict,
    mixed_ate,
    mixed_predict_shares,
)

__all__ = [
    "DrawSource",
    "FixtureDrawSource",
    "ScipyHaltonDrawSource",
    "panel_reshape_gauss",
    "MixingSpec",
    "ParamLayout",
    "ParamSpace",
    "EstimationSpace",
    "ReportingSpace",
    "RCState",
    "RandomCoefPipeline",
    "RCJacobian",
    "RealizationCache",
    "KernelObsResult",
    "MixedKernel",
    "condition",
    "gcondnewcov",
    "gcondnewmean",
    "gcondspecialnewmean",
    "DesignData",
    "MixedMSLEstimator",
    "MSLConfig",
    "Tracer",
    "MixedPredictComponents",
    "MixedATEResult",
    "mixed_predict_shares",
    "mixed_ate",
    "default_kernel_predict",
]
