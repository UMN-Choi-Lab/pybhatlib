"""Matrix gradient operations (reimplements GAUSS Matgradient.src)."""

from pybhatlib.matgradient._gradcovcor import GradCovCorResult, gradcovcor
from pybhatlib.matgradient._gomegxomegax import gomegxomegax
from pybhatlib.matgradient._spherical import grad_corr_theta, theta_to_corr
from pybhatlib.matgradient._chain_rules import chain_grad

__all__ = [
    "GradCovCorResult",
    "gradcovcor",
    "gomegxomegax",
    "theta_to_corr",
    "grad_corr_theta",
    "chain_grad",
]
