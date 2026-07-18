"""Low-level matrix operations (reimplements GAUSS Vecup.src)."""

from pybhatlib.vecup._vec_ops import (
    matdupdiagonefull,
    matdupfull,
    vecdup,
    vecndup,
    vecsymmetry,
)
from pybhatlib.vecup._nondiag import nondiag
from pybhatlib.vecup._ldlt import ldlt_decompose, ldlt_rank1_update
from pybhatlib.vecup._truncnorm import truncated_mvn_moments
from pybhatlib.vecup._panel import PanelIndex
from pybhatlib.vecup._yj import (
    gradmeanyj,
    gradstandyjinvnonpgiven,
    gyjinvnonp,
    gyjnonp,
    meanyj,
    standyjinvnonpgiven,
    yjinvnonp,
    yjnonp,
)

__all__ = [
    "vecdup",
    "vecndup",
    "matdupfull",
    "matdupdiagonefull",
    "vecsymmetry",
    "nondiag",
    "ldlt_decompose",
    "ldlt_rank1_update",
    "truncated_mvn_moments",
    "PanelIndex",
    "yjnonp",
    "gyjnonp",
    "yjinvnonp",
    "gyjinvnonp",
    "meanyj",
    "gradmeanyj",
    "standyjinvnonpgiven",
    "gradstandyjinvnonpgiven",
]
