"""MixMNL model control structure.

Configures estimation of the Mixed Multinomial Logit (MixMNL) model — an MNL
softmax kernel mixed over random coefficients via the shared MSL engine
(``pybhatlib.mixed``).  The random-coefficient specification is given as lists
of variable names (mirroring the GAUSS ``MIXMNL.gss`` globals ``normvar`` /
``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``); the MSL / reparameterization
knobs (``n_rep`` / ``intordn1`` / ``spher`` / ``scal``) and the panel person-id
column mirror the corresponding ``MNPControl`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from numpy.typing import NDArray


@dataclass
class MixMNLControl:
    """Control structure for MixMNL model estimation.

    Attributes
    ----------
    normvar : sequence of str
        Variables carrying a normal random coefficient (GAUSS ``normvar``).
    logvar : sequence of str
        Variables carrying a log-normal random coefficient (GAUSS ``logvar``).
        Each must also appear in ``varneg`` or ``varpos``.
    yjvar : sequence of str
        Variables carrying a Yeo-Johnson-transformed random coefficient
        (GAUSS ``yjvar``).
    varneg, varpos : sequence of str
        Fixed components constrained to a strictly negative / positive sign
        (GAUSS ``varneg`` / ``varpos``).
    person_id : str or None
        Column holding the panel person identifier.  ``None`` (default) treats
        every observation as its own person (cross-sectional; ``Dmask = I``).
    randdiag : bool
        If True, fix the random-coefficient correlation matrix to the identity
        (GAUSS ``_randdiag``) instead of estimating a full correlation matrix.
    n_rep : int
        Number of MSL replications (GAUSS ``nrep``, default 125).
    intordn1 : int
        Gauss-Hermite node count for ``meanyj`` (GAUSS ``intordn1``).
    spher : bool
        Spherical (True) vs radial (False) correlation-Cholesky
        parameterization.  Only the radial path is implemented downstream.
    scal : float
        Scaling constant in ``newcholparmscaled`` (GAUSS ``scal``).
    floor_pcomp : float
        Per-observation kernel-probability floor applied before the log
        (GAUSS ``w1``).
    floor_z : float
        Per-person simulated-probability floor after averaging over draws
        (GAUSS ``w2``).
    draw_seed : int or None
        Seed for the runtime scrambled-Halton draw source.  ``None`` yields
        non-deterministic draws.
    maxiter : int
        Maximum number of optimizer iterations.
    optimizer : str
        Optimization method: "bfgs" (default), "lbfgsb", or "newton" (Newton-CG,
        gradient only).
    tol : float
        Optimizer tolerance passed to scipy (``gtol`` for BFGS/Newton-CG,
        ``ftol``/``gtol`` for L-BFGS-B).
    tol_check : float
        Gradient-norm threshold used to flag convergence after the fit.
    want_covariance : bool
        If True, compute the parameter covariance matrix at convergence via the
        reporting-space BHHH pass.
    se_method : str
        Standard-error method.  Only "bhhh" (cross-product of the
        per-individual score, matching GAUSS ``_max_CovPar = 2``) is currently
        implemented.
    verbose : int
        Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    seed : int or None
        Random seed for reproducibility (alias kept for cross-model symmetry;
        the draw source uses ``draw_seed``).
    startb : NDArray or None
        User-supplied starting values (full ``theta`` in estimation space).
        If None, defaults to zeros.
    """

    # --- random-coefficient specification (GAUSS name lists) ----------------
    normvar: Sequence[str] = field(default_factory=tuple)
    logvar: Sequence[str] = field(default_factory=tuple)
    yjvar: Sequence[str] = field(default_factory=tuple)
    varneg: Sequence[str] = field(default_factory=tuple)
    varpos: Sequence[str] = field(default_factory=tuple)

    # --- panel ---------------------------------------------------------------
    person_id: str | None = None

    # --- MSL / reparameterization -------------------------------------------
    randdiag: bool = False
    n_rep: int = 125
    intordn1: int = 20
    spher: bool = False
    scal: float = 1.0
    floor_pcomp: float = 1e-4
    floor_z: float = 1e-4
    draw_seed: int | None = None

    # --- optimizer -----------------------------------------------------------
    maxiter: int = 2000
    optimizer: Literal["bfgs", "lbfgsb", "newton"] = "bfgs"
    tol: float = 1e-8
    tol_check: float = 1e-5
    want_covariance: bool = True
    se_method: Literal["bhhh"] = "bhhh"
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
