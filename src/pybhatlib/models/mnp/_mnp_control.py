"""MNP model control structure.

Equivalent to BHATLIB's mnpControl struct, this dataclass configures
the estimation procedure for the Multinomial Probit model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class MNPControl:
    """Control structure for MNP model estimation.

    Attributes
    ----------
    iid : bool
        If True, assume IID error terms (homoscedastic, uncorrelated).
        If False, estimate flexible covariance structure.
    mix : bool
        If True, include random coefficients (mixed MNP).
    correst : NDArray or None
        Correlation restriction matrix. Upper-diagonal matrix with 1s on
        diagonal, 1s in off-diagonal positions indicate active correlations.
        None means no restrictions (full correlation).
    heteronly : bool
        If True, estimate only heteroscedastic errors (no correlations).
    randdiag : bool
        If True, random coefficients covariance is diagonal (uncorrelated).
    nseg : int
        Number of mixture-of-normals segments. 1 = single normal,
        >1 = discrete mixture of normals.
    method : str
        MVNCD approximation method: "me", "ovus", "tvbs", "bme", "ovbs",
        "ssj", "scipy".
    spherical : bool
        If True, use spherical parameterization for correlation matrix.
    scal : float
        Scaling factor for starting values.
    IID_first : bool
        If True, run IID model first to obtain starting values.
    want_covariance : bool
        If True, compute parameter covariance matrix at convergence.
    seed10 : int
        Secondary seed for QMC sequences.
    perms : int
        Number of permutations for MVNCD variable reordering.
    maxiter : int
        Maximum number of optimizer iterations.
    tol : float
        Convergence tolerance (gradient norm).
    optimizer : str
        Optimization method: "bfgs", "lbfgsb", "torch_adam", "torch_lbfgs".
    se_method : str
        Standard-error method: "bhhh" (default, matches GAUSS _max_CovPar=2),
        "hessian" (inverse observed information), or "sandwich" (robust).
        Per-observation scores are computed by finite differencing the
        per-observation log-likelihood in UNPARAMETERIZED space (the GAUSS
        lpr1/lgd1 convention) at the converged estimate. The only
        Jacobian step that survives is the BHATLIB sigma_1 normalization
        (one scalar per block). There is no delta-method projection through
        the spherical-Cholesky parameterization. See MNP-002b for the
        plan-fidelity rationale.

        BHHH is the default to match GAUSS BHATLIB output (which uses
        ``_max_CovPar = 2`` = cross-product of first derivatives). On
        BHATLIB Table 1 Model (a)(i), BHHH matches the published GAUSS
        s.e. to ~0.05 % on every parameter. Set ``se_method="hessian"``
        to use the inverse observed Hessian instead, or
        ``se_method="sandwich"`` for the robust estimator that does not
        assume the information-matrix equality.

        Regardless of which method is chosen as the primary, all three
        estimators are computed at fit time and exposed on the results
        object as ``se_bhhh``, ``se_hessian``, ``se_sandwich`` so that
        ``summary()`` can print a side-by-side diagnostic comparison
        (large divergence is a misspecification signal).
    verbose : int
        Verbosity: 0=silent, 1=summary, 2=per-iteration NLL,
        3=per-iteration NLL + parameter/gradient/rel-gradient table.
    seed : int or None
        Random seed for reproducibility.
    startb : NDArray or None
        User-supplied starting values for parameters.
    active_mask : NDArray of bool or None
        Boolean mask of shape ``(n_params,)``.  ``True`` = parameter is
        estimated; ``False`` = parameter is frozen at its ``startb`` value
        (or default starting value when ``startb`` is None).  ``None``
        (default) estimates all parameters with no overhead.  A
        ``ValueError`` is raised if the length does not match ``n_params``
        or if all entries are ``False``.  SE/t-stat/p-value are set to
        ``np.nan`` for frozen parameters in the results.
    """

    iid: bool = False
    mix: bool = False
    correst: NDArray | None = None
    heteronly: bool = False
    randdiag: bool = False
    nseg: int = 1
    method: str = "ovus"
    spherical: bool = True
    scal: float = 1.0
    IID_first: bool = True
    want_covariance: bool = True
    seed10: int = 1234
    perms: int = 0
    maxiter: int = 200
    tol: float = 1e-5
    optimizer: Literal["bfgs", "lbfgsb", "torch_adam", "torch_lbfgs"] = "bfgs"
    se_method: Literal["bhhh", "hessian", "sandwich"] = "bhhh"
    verbose: int = 1
    seed: int | None = None
    analytic_grad: bool = True
    startb: NDArray | None = None
    active_mask: NDArray | None = None
    """Boolean mask of shape (n_params,).  True = parameter is estimated,
    False = parameter is frozen at its ``startb`` value (or at the default
    starting value when ``startb`` is None).  ``None`` (default) means all
    parameters are estimated — the fast path with no overhead.

    If provided, ``len(active_mask)`` must equal ``n_params``; a
    ``ValueError`` is raised at fit-time otherwise.  If all entries are
    False, a ``ValueError`` is raised (nothing to optimize).
    SE/t-stat/p-value are set to ``np.nan`` for frozen parameters.
    """
    device: str = "cpu"
    """Device for computation: "cpu", "cuda", "cuda:0", or "auto".
    "auto" selects GPU when N >= gpu_threshold and CUDA is available."""
    gpu_threshold: int = 5000
    """Minimum N for automatic GPU dispatch when device="auto"."""
    torch_compile: bool = False
    """If True, use torch.compile on GPU MVNCD kernels (requires device != "cpu").
    Adds ~5s one-time compilation cost, then 2-20x faster per iteration."""
