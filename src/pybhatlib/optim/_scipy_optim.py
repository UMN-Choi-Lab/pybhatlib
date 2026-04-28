"""SciPy optimization wrapper."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize


@dataclass
class OptimResult:
    """Optimization result container.

    Attributes
    ----------
    x : NDArray
        Optimal parameters.
    fun : float
        Objective value at optimum.
    grad : NDArray
        Gradient at optimum.
    hess_inv : NDArray or None
        Inverse Hessian approximation (BFGS only).
    n_iter : int
        Number of iterations.
    converged : bool
        Whether optimization converged.
    return_code : int
        Return code (0 = converged).
    message : str
        Status message.
    """

    x: NDArray
    fun: float
    grad: NDArray
    hess_inv: NDArray | None
    n_iter: int
    converged: bool
    return_code: int
    message: str


def minimize_scipy(
    func,
    x0: NDArray,
    *,
    method: str = "BFGS",
    maxiter: int = 200,
    tol: float = 1e-5,
    verbose: int = 1,
    jac: bool = True,
    bounds=None,
    param_names: list[str] | None = None,
) -> OptimResult:
    """Minimize a scalar function using scipy.optimize.minimize.

    Parameters
    ----------
    func : callable
        If jac=True: func(x) -> (f, grad)
        If jac=False: func(x) -> f
    x0 : ndarray
        Initial parameter vector.
    method : str
        "BFGS", "L-BFGS-B", or "Nelder-Mead".
    maxiter : int
        Maximum iterations.
    tol : float
        Gradient tolerance.
    verbose : int
        0=silent, 1=summary, 2=per-iteration NLL,
        3=per-iteration NLL + parameter/gradient/rel-gradient table.
    jac : bool
        Whether func returns (f, grad) tuple.
    bounds : list of tuples or None
        Parameter bounds for L-BFGS-B.
    param_names : list of str or None
        Parameter names for verbose=3 display.  When None, labels fall back
        to ``θ[k]`` placeholders.  These are the *theta-space* names, so
        they may differ from the reporting-space names in MNPResults.

    Returns
    -------
    result : OptimResult
    """
    x0 = np.asarray(x0, dtype=np.float64)
    n_params = len(x0)

    iteration_count = [0]
    start_time = time.time()

    def _get_fval_grad(xk):
        """Return (fval, grad) regardless of jac mode."""
        if jac:
            fval, g = func(xk)
            return float(fval), np.asarray(g, dtype=np.float64)
        else:
            return float(func(xk)), np.zeros_like(xk)

    def _print_param_table(xk, fval, grad):
        """Print per-iteration parameter/gradient/rel-gradient table (verbose=3)."""
        # Build labels: use param_names if available, else θ[k]
        if param_names is not None and len(param_names) == len(xk):
            labels = param_names
        else:
            labels = [f"θ[{k}]" for k in range(len(xk))]

        # Relative gradient: |g_k| / max(|x_k|, 1)
        rel_grad = np.abs(grad) / np.maximum(np.abs(xk), 1.0)

        col_w_name = max(14, max(len(lb) for lb in labels) + 2)
        header = (
            f"  {'param':<{col_w_name}s}  {'val':>14s}  {'grad':>14s}  "
            f"{'rel_grad':>14s}"
        )
        print(f"  iter={iteration_count[0]}  f={fval:+.6f}")
        print(header)
        print("  " + "-" * (col_w_name + 3 * 16 + 4))
        for k, (lb, v, g, rg) in enumerate(zip(labels, xk, grad, rel_grad)):
            print(
                f"  {lb:<{col_w_name}s}  {v:>+14.6f}  {g:>+14.6f}  {rg:>14.6f}"
            )

    def callback(xk):
        iteration_count[0] += 1
        if verbose >= 3:
            elapsed = time.time() - start_time
            fval, grad = _get_fval_grad(xk)
            print(f"  Iter {iteration_count[0]:4d}: f = {fval:.6f}  ({elapsed:.1f}s)")
            _print_param_table(xk, fval, grad)
        elif verbose >= 2:
            elapsed = time.time() - start_time
            if jac:
                fval, _ = func(xk)
            else:
                fval = func(xk)
            print(f"  Iter {iteration_count[0]:4d}: f = {fval:.6f}  ({elapsed:.1f}s)")

    if jac:
        # func returns (f, grad) — need to wrap for scipy
        def scipy_func(x):
            f, g = func(x)
            return float(f), np.asarray(g, dtype=np.float64)

        result = minimize(
            scipy_func,
            x0,
            method=method,
            jac=True,
            options={"maxiter": maxiter, "gtol": tol, "disp": False},
            bounds=bounds,
            callback=callback,
        )
    else:
        result = minimize(
            func,
            x0,
            method=method,
            options={"maxiter": maxiter, "gtol": tol, "disp": False},
            bounds=bounds,
            callback=callback,
        )

    # Extract gradient
    if hasattr(result, "jac") and result.jac is not None:
        grad = np.asarray(result.jac)
    else:
        grad = np.zeros_like(x0)

    # Extract inverse Hessian
    hess_inv = None
    if hasattr(result, "hess_inv"):
        if hasattr(result.hess_inv, "todense"):
            hess_inv = np.asarray(result.hess_inv.todense())
        else:
            hess_inv = np.asarray(result.hess_inv)

    converged = result.success
    return_code = 0 if converged else 2

    if verbose >= 1:
        elapsed = time.time() - start_time
        status = "converged" if converged else "did not converge"
        print(f"  Optimization {status} in {result.nit} iterations ({elapsed:.2f}s)")
        print(f"  Final objective: {result.fun:.6f}")

    return OptimResult(
        x=result.x,
        fun=float(result.fun),
        grad=grad,
        hess_inv=hess_inv,
        n_iter=result.nit,
        converged=converged,
        return_code=return_code,
        message=result.message,
    )
