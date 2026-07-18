"""MORPFlex model control structure.

Configures estimation of the mixed-panel Multivariate Ordered Response Probit
(GAUSS *Joint Ordered YJ with Cross-Sectional or Panel Random Coefficients*
driver) — a rectangle-MVNCD (``pdfrectn``) kernel mixed over random
coefficients via the shared MSL engine (:mod:`pybhatlib.mixed`).  The
random-coefficient specification is given as lists of variable names (mirroring
:class:`~pybhatlib.models.mnpkercp._mnpkercp_control.MNPKerCPControl` and the
GAUSS globals ``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``);
the extra ordered-response knobs are the ``copula`` flag (the inverse of GAUSS
``_nocorrrcker``), the ``yj_kernel`` flag (the inverse of GAUSS ``_normker``)
selecting a Yeo-Johnson kernel-error transform, and the ordinal-outcome sizing
(``nord`` / ``n_categories``) that determines the leading threshold block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from numpy.typing import NDArray


@dataclass
class MORPFlexControl:
    """Control structure for MORPFlex (mixed-panel MORP) model estimation.

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
    copula : bool
        If ``True`` (GAUSS ``_nocorrrcker = 0``) condition the ordinal kernel
        errors on the drawn random coefficients (the rc<->kernel copula,
        cross-sectional) and estimate the joint correlation gradient.  If
        ``False`` (GAUSS ``_nocorrrcker = 1``; **required for panel data**) the
        ordinal kernel covariance is the unconditional block and the copula
        gradient path vanishes.  ``copula = True`` requires at least one random
        coefficient.  Default ``False``.
    yj_kernel : bool
        If ``True`` (GAUSS ``_normker = 0``) apply a Yeo-Johnson transform to the
        ordinal kernel errors, adding an ``nord``-parameter ``kernlam`` block
        that trails the vector.  If ``False`` (GAUSS ``_normker = 1``; default)
        the kernel errors are standard normal and there is no ``kernlam`` block.
    scaling : {"none"}
        Kernel-scale reparameterization.  MORP fixes ``wker = ones`` (GAUSS
        line 617), so there is no free kernel-scale block (``"none"`` only).
    n_rep : int
        Number of MSL replications (GAUSS ``nrep``).
    intordn1 : int
        Gauss-Hermite node count for ``meanyj`` (GAUSS ``_intordn1``).
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
        Optimization method: "bfgs" (default), "lbfgsb", or "newton".
    tol : float
        Optimizer tolerance passed to scipy.
    tol_check : float
        Gradient-norm threshold used to flag convergence after the fit.
    want_covariance : bool
        If True, compute the parameter covariance at convergence via the
        reporting-space BHHH pass.
    se_method : str
        Standard-error method.  Only "bhhh" is implemented.
    verbose : int
        Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    seed : int or None
        Random seed kept for cross-model symmetry (the draw source uses
        ``draw_seed``).
    startb : NDArray or None
        User-supplied starting values (full ``theta`` in estimation space).
        If None, defaults to zeros.

    Notes
    -----
    Panel parity note (GAUSS): for genuine panel data GAUSS requires
    ``_nocorrrcker = 1`` (``copula = False``); the copula path is
    cross-sectional only.
    """

    # --- random-coefficient specification (GAUSS name lists) ----------------
    normvar: Sequence[str] = field(default_factory=tuple)
    logvar: Sequence[str] = field(default_factory=tuple)
    yjvar: Sequence[str] = field(default_factory=tuple)
    varneg: Sequence[str] = field(default_factory=tuple)
    varpos: Sequence[str] = field(default_factory=tuple)

    # --- panel / weights -----------------------------------------------------
    person_id: str | None = None
    weight_var: str | None = None

    # --- kernel / copula / ordered-response knobs ---------------------------
    copula: bool = False
    iid: bool = False
    correst: NDArray | None = None
    yj_kernel: bool = False
    scaling: Literal["none"] = "none"

    # --- MSL / reparameterization -------------------------------------------
    randdiag: bool = False
    n_rep: int = 5
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
