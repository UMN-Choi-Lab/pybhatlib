"""PyTorch optimization wrapper (optional dependency)."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from pybhatlib.optim._scipy_optim import OptimResult


def minimize_torch(
    func,
    x0: NDArray,
    *,
    method: str = "lbfgs",
    maxiter: int = 200,
    tol: float = 1e-5,
    verbose: int = 1,
    device: str = "cpu",
) -> OptimResult:
    """Minimize using PyTorch optimizers.

    Parameters
    ----------
    func : callable
        func(x) -> (f, grad) where x is numpy array.
    x0 : ndarray
        Initial parameters.
    method : str
        "lbfgs" or "adam".
    maxiter : int
        Maximum iterations.
    tol : float
        Convergence tolerance.
    verbose : int
        Verbosity level.
    device : str
        PyTorch device.

    Returns
    -------
    result : OptimResult
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            "PyTorch required for torch optimizer. Install: pip install pybhatlib[torch]"
        ) from e

    x0_np = np.asarray(x0, dtype=np.float64)
    x_param = torch.tensor(x0_np, dtype=torch.float64, device=device, requires_grad=True)

    start_time = time.time()
    best_f = float("inf")
    best_x = x0_np.copy()
    n_iter = 0

    if method.lower() == "lbfgs":
        optimizer = torch.optim.LBFGS(
            [x_param],
            max_iter=1,
            tolerance_grad=tol,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        for i in range(maxiter):
            def closure():
                optimizer.zero_grad()
                x_np = x_param.detach().cpu().numpy()
                f_val, g_val = func(x_np)
                x_param.grad = torch.tensor(g_val, dtype=torch.float64, device=device)
                return torch.tensor(f_val, dtype=torch.float64)

            loss = optimizer.step(closure)
            n_iter = i + 1
            f_val = float(loss)

            if f_val < best_f:
                best_f = f_val
                best_x = x_param.detach().cpu().numpy().copy()

            if verbose >= 2:
                print(f"  Iter {n_iter:4d}: f = {f_val:.6f}")

            # Check convergence
            if x_param.grad is not None:
                grad_norm = float(x_param.grad.norm())
                if grad_norm < tol:
                    break

    elif method.lower() == "adam":
        optimizer = torch.optim.Adam([x_param], lr=0.01)

        for i in range(maxiter):
            optimizer.zero_grad()
            x_np = x_param.detach().cpu().numpy()
            f_val, g_val = func(x_np)
            x_param.grad = torch.tensor(g_val, dtype=torch.float64, device=device)
            optimizer.step()
            n_iter = i + 1

            if f_val < best_f:
                best_f = f_val
                best_x = x_param.detach().cpu().numpy().copy()

            if verbose >= 2 and (i + 1) % 10 == 0:
                print(f"  Iter {n_iter:4d}: f = {f_val:.6f}")

            grad_norm = float(np.linalg.norm(g_val))
            if grad_norm < tol:
                break
    else:
        raise ValueError(f"Unknown method: {method}. Use 'lbfgs' or 'adam'.")

    elapsed = time.time() - start_time

    # Final gradient
    _, final_grad = func(best_x)
    grad_norm = float(np.linalg.norm(final_grad))
    converged = grad_norm < tol

    if verbose >= 1:
        status = "converged" if converged else "did not converge"
        print(f"  Optimization {status} in {n_iter} iterations ({elapsed:.2f}s)")
        print(f"  Final objective: {best_f:.6f}")

    return OptimResult(
        x=best_x,
        fun=best_f,
        grad=final_grad,
        hess_inv=None,
        n_iter=n_iter,
        converged=converged,
        return_code=0 if converged else 2,
        message="converged" if converged else "max iterations reached",
    )
