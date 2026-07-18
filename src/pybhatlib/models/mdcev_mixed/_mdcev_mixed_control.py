"""Mixed MDCEV model control structure.

Configures estimation of the mixed / panel Multiple Discrete-Continuous Extreme
Value (MDCEV) model — an MDCEV logit-Jacobian kernel mixed over random
*baseline-utility* coefficients via the shared MSL engine (:mod:`pybhatlib.mixed`).

The random-coefficient specification is given as lists of baseline-utility
variable names (mirroring the GAUSS ``Mixed Traditional MDCEV.gss`` globals
``normvar`` / ``logvar`` / ``yjvar`` / ``varneg`` / ``varpos``); the MSL /
reparameterization knobs (``n_rep`` / ``intordn1`` / ``spher`` / ``scal``) and the
panel person-id column mirror :class:`~pybhatlib.models.mixmnl._mixmnl_control.MixMNLControl`.
The MDCEV-specific fields (``utility`` / ``outside_good_gamma`` / ``weight_var``)
mirror :class:`~pybhatlib.models.mdcev._mdcev_control.MDCEVControl`.

The translation (``gamma``) parameters and the MDCEV kernel error scale are
kernel-owned, **non-mixed** parameters; only the baseline-utility coefficients
are mixed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from numpy.typing import NDArray


@dataclass
class MDCEVMixedControl:
    """Control structure for mixed MDCEV model estimation.

    Attributes
    ----------
    utility : {"trad", "linear"}
        Outside-good utility specification passed through to the shipped MDCEV
        likelihood (:class:`~pybhatlib.models.mdcev._mdcev_control.MDCEVControl`):
        ``"trad"`` (Bhat 2008) or ``"linear"`` (Bhat 2018).
    normvar : sequence of str
        Baseline-utility variables carrying a normal random coefficient (GAUSS
        ``normvar``).
    logvar : sequence of str
        Baseline-utility variables carrying a log-normal random coefficient
        (GAUSS ``logvar``). Each must also appear in ``varneg`` or ``varpos``.
    yjvar : sequence of str
        Baseline-utility variables carrying a Yeo-Johnson-transformed random
        coefficient (GAUSS ``yjvar``).
    varneg, varpos : sequence of str
        Fixed components constrained to a strictly negative / positive sign
        (GAUSS ``varneg`` / ``varpos``).
    person_id : str or None
        Column holding the panel person identifier. ``None`` (default) treats
        every observation as its own person (cross-sectional; ``Dmask = I``).
        When set to a column name the model reads that column for panel grouping.
    weight_var : str or None
        Column name for observation weights. ``None`` defaults to the ``"uno"``
        column of ones (GAUSS ``{ weight, wtind } = indices(dataset, "uno")``).
    outside_good_gamma : float
        Fixed satiation value forced on the outside good (GAUSS
        ``u[.,1] = -1000``); passed through to the kernel's MDCEV control.
    n_rep : int
        Number of MSL replications (GAUSS ``nrep``, default 10 for the SCAG
        driver).
    intordn1 : int
        Gauss-Hermite node count for ``meanyj`` (GAUSS ``_intordn1``).
    spher : bool
        Spherical (True) vs radial (False) correlation-Cholesky parameterization.
        Only the radial path is implemented downstream.
    scal : float
        Scaling constant in ``newcholparmscaled`` (GAUSS ``scal``).
    floor_pcomp : float
        Per-observation kernel-probability floor applied before the log (GAUSS
        ``w1``). Defaults to ``0.0`` to match the GAUSS mixed MDCEV driver, which
        sets ``w1 = 0`` (the MDCEV logit likelihood is strictly positive).
    floor_z : float
        Per-person simulated-probability floor after averaging over draws (GAUSS
        ``w2``). Defaults to ``0.0`` (GAUSS ``w2 = 0``).
    draw_seed : int or None
        Seed for the runtime scrambled-Halton draw source. ``None`` yields
        non-deterministic draws.
    maxiter : int
        Maximum number of optimizer iterations.
    optimizer : {"bfgs", "lbfgsb"}
        Optimization method.
    tol : float
        Optimizer tolerance passed to scipy.
    tol_check : float
        Gradient-norm threshold used to flag convergence after the fit.
    want_covariance : bool
        If True, compute the parameter covariance matrix at convergence via the
        reporting-space BHHH pass.
    se_method : str
        Standard-error method. Only "bhhh" (cross-product of the per-individual
        score, matching GAUSS ``_max_CovPar = 2``) is currently implemented.
    verbose : int
        Verbosity: 0 = silent, 1 = summary, 2 = per-iteration.
    seed : int or None
        Random seed for reproducibility (kept for cross-model symmetry; the draw
        source uses ``draw_seed``).
    startb : NDArray or None
        User-supplied starting values (full ``theta`` in estimation space,
        ``[beta | gamma | rcor | kern | scal | lam]`` order). If None, defaults
        to zeros with the outside-good gamma pinned to ``outside_good_gamma``.
    """

    utility: Literal["trad", "linear"] = "trad"

    # --- random-coefficient specification (GAUSS name lists) ----------------
    normvar: Sequence[str] = field(default_factory=tuple)
    logvar: Sequence[str] = field(default_factory=tuple)
    yjvar: Sequence[str] = field(default_factory=tuple)
    varneg: Sequence[str] = field(default_factory=tuple)
    varpos: Sequence[str] = field(default_factory=tuple)

    # --- panel / weights -----------------------------------------------------
    person_id: str | None = None
    weight_var: str | None = None
    outside_good_gamma: float = -1000.0

    # --- MSL / reparameterization -------------------------------------------
    randdiag: bool = False
    n_rep: int = 10
    intordn1: int = 20
    spher: bool = False
    scal: float = 1.0
    floor_pcomp: float = 0.0
    floor_z: float = 0.0
    draw_seed: int | None = None

    # --- optimizer -----------------------------------------------------------
    maxiter: int = 2000
    optimizer: Literal["bfgs", "lbfgsb"] = "bfgs"
    tol: float = 1e-8
    tol_check: float = 1e-5
    want_covariance: bool = True
    se_method: Literal["bhhh"] = "bhhh"
    verbose: int = 1
    seed: int | None = None
    startb: NDArray | None = None
