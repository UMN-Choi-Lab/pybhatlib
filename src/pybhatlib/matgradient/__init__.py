"""Matrix gradient operations (reimplements GAUSS Matgradient.src)."""

from pybhatlib.matgradient._gradcovcor import GradCovCorResult, gradcovcor
from pybhatlib.matgradient._gomegxomegax import gomegxomegax
from pybhatlib.matgradient._spherical import (
    corr_to_theta,
    grad_corr_theta,
    theta_to_corr,
)
from pybhatlib.matgradient._radial import (
    gnewcholparmcorscaled,
    grad_radial_theta,
    newcholparmscaled,
    radial_to_corr,
    revnewcholparmscaled,
)
from pybhatlib.matgradient._corr_chol import gcholeskycor, ggradchol
from pybhatlib.matgradient._chain_rules import chain_grad

__all__ = [
    "GradCovCorResult",
    "gradcovcor",
    "gomegxomegax",
    "theta_to_corr",
    "corr_to_theta",
    "grad_corr_theta",
    "radial_to_corr",
    "grad_radial_theta",
    "newcholparmscaled",
    "revnewcholparmscaled",
    "gnewcholparmcorscaled",
    "gcholeskycor",
    "ggradchol",
    "chain_grad",
]
