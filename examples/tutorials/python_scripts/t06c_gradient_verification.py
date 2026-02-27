"""Tutorial T06c: Gradient Verification via Finite Differences.

Correct gradients are critical for optimization convergence.
This capstone tutorial shows the systematic pattern for verifying
any analytic gradient against numerical finite differences.

What you will learn:
  - Central finite differences: formula, eps selection
  - Verifying vecup gradients (det of matdupfull)
  - Verifying matgradient (gradcovcor)
  - Verifying gradmvn (mvncd_grad)
  - A reusable verify_gradient() template function
  - Best practices: eps, relative vs absolute tolerance

Prerequisites: t01a, t02a, t03b.
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.vecup import vecdup, matdupfull
from pybhatlib.matgradient import gradcovcor, theta_to_corr, grad_corr_theta
from pybhatlib.gradmvn import mvncd, mvncd_grad

# ============================================================
#  Step 1: Central finite differences
# ============================================================
print("=" * 60)
print("  Step 1: Central Finite Differences")
print("=" * 60)

print("""
  Forward difference:  df/dx ~ [f(x+eps) - f(x)] / eps          O(eps)
  Central difference:  df/dx ~ [f(x+eps) - f(x-eps)] / (2*eps)  O(eps^2)

  Central differences are more accurate and should always be preferred.

  Choosing eps:
  - Too large: truncation error dominates
  - Too small: floating-point cancellation dominates
  - Rule of thumb: eps ~ sqrt(machine_epsilon) ~ 1e-7 for float64
""")

# Demonstrate with f(x) = x^3, f'(x) = 3x^2
def f_cube(x):
    return x ** 3

def grad_cube(x):
    return 3 * x ** 2

x0 = 2.0
true_grad = grad_cube(x0)

print(f"  f(x) = x^3, f'(x) = 3x^2, at x={x0}")
print(f"  True gradient: {true_grad:.6f}\n")
print(f"  {'eps':>12s} {'forward':>12s} {'central':>12s} {'fwd_err':>12s} {'ctr_err':>12s}")
print(f"  {'-'*62}")

for exp in range(-2, -12, -1):
    eps = 10.0 ** exp
    fwd = (f_cube(x0 + eps) - f_cube(x0)) / eps
    ctr = (f_cube(x0 + eps) - f_cube(x0 - eps)) / (2 * eps)
    print(f"  {eps:>12.0e} {fwd:>12.6f} {ctr:>12.6f} "
          f"{abs(fwd - true_grad):>12.2e} {abs(ctr - true_grad):>12.2e}")

# ============================================================
#  Step 2: Reusable verification function
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Reusable verify_gradient() Function")
print("=" * 60)

def verify_gradient(f, grad_analytic, x0, eps=1e-7, rtol=1e-4, atol=1e-6):
    """Verify analytic gradient against central finite differences.

    Parameters
    ----------
    f : callable
        Scalar function f(x) -> float.
    grad_analytic : ndarray
        Analytic gradient at x0.
    x0 : ndarray
        Point at which to verify.
    eps : float
        Finite difference step size.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.

    Returns
    -------
    passed : bool
    max_err : float
    """
    n = len(x0)
    grad_fd = np.zeros(n)

    for i in range(n):
        x_plus = x0.copy(); x_plus[i] += eps
        x_minus = x0.copy(); x_minus[i] -= eps
        grad_fd[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    abs_err = np.abs(grad_analytic - grad_fd)
    max_err = np.max(abs_err)

    # Use relative tolerance where gradient is large, absolute where small
    scale = np.maximum(np.abs(grad_analytic), np.abs(grad_fd))
    scale = np.maximum(scale, 1.0)
    rel_err = abs_err / scale

    passed = np.all(rel_err < rtol) or max_err < atol

    return passed, max_err, grad_fd

print("""
  verify_gradient(f, grad_analytic, x0) -> (passed, max_err, grad_fd)

  Uses combined relative/absolute tolerance:
  - Relative for large gradient values
  - Absolute for near-zero gradient values
""")

# ============================================================
#  Step 3: Vecup example — gradient of det(matdupfull(v))
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Vecup — Gradient of det(matdupfull(v))")
print("=" * 60)

def det_from_vec(v):
    """Determinant of symmetric matrix from its vecdup."""
    return np.linalg.det(matdupfull(v))

# Analytic gradient: d(det A)/d(vec A) is cofactor matrix
v0 = np.array([4.0, 1.0, 0.5, 3.0, 0.3, 2.0])  # vecdup of 3x3 PD matrix
A0 = matdupfull(v0)
det_A0 = np.linalg.det(A0)

# For vecdup gradient, we need the chain through matdupfull
# d(det)/d(A_ij) = cofactor(i,j) = det(A) * (A^{-1})_ji
cofactor = det_A0 * np.linalg.inv(A0).T
# Account for symmetry: off-diagonal elements appear twice
grad_analytic = vecdup(cofactor)
# Off-diagonal: multiply by 2 (since A_ij = A_ji both contribute)
idx = 0
K = 3
for i in range(K):
    for j in range(i, K):
        if i != j:
            grad_analytic[idx] *= 2
        idx += 1

passed, max_err, grad_fd = verify_gradient(det_from_vec, grad_analytic, v0)
print(f"\n  f(v) = det(matdupfull(v))")
print(f"  v0 = {v0}")
print(f"  det = {det_A0:.4f}")
print(f"\n  Analytic gradient: {grad_analytic}")
print(f"  Numerical gradient: {grad_fd}")
print(f"  Max error: {max_err:.2e}")
print(f"  Passed: {passed}")

# ============================================================
#  Step 4: Matgradient example — verify gradcovcor.glitomega
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Matgradient — Verify gradcovcor")
print("=" * 60)

omega = np.array([2.0, 1.5, 1.0])
Omega_star = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])
Omega = np.diag(omega) @ Omega_star @ np.diag(omega)

gc = gradcovcor(Omega)

# Verify: perturb omega_k, see how vecdup(Omega) changes
# glitomega has shape (K, n_cov) = (3, 6) in BHATLIB convention
eps = 1e-7
glitomega_fd = np.zeros_like(gc.glitomega)
for k in range(3):
    om_p = omega.copy(); om_p[k] += eps
    om_m = omega.copy(); om_m[k] -= eps
    Om_p = np.diag(om_p) @ Omega_star @ np.diag(om_p)
    Om_m = np.diag(om_m) @ Omega_star @ np.diag(om_m)
    glitomega_fd[k, :] = (vecdup(Om_p) - vecdup(Om_m)) / (2 * eps)

max_err_gc = np.max(np.abs(gc.glitomega - glitomega_fd))
print(f"\n  gradcovcor.glitomega verification:")
print(f"  Max error: {max_err_gc:.2e}")
print(f"  Passed: {max_err_gc < 1e-4}")

# ============================================================
#  Step 5: Gradmvn example — verify mvncd_grad
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Gradmvn — Verify mvncd_grad")
print("=" * 60)

sigma = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.4], [0.1, 0.4, 1.0]])
a = np.array([1.0, 0.5, 0.0])

result = mvncd_grad(a, sigma)

# Verify grad_a
def mvncd_scalar(a_vec):
    return mvncd(a_vec, sigma, method="scipy")

passed_a, max_err_a, grad_a_fd = verify_gradient(mvncd_scalar, result.grad_a, a, eps=1e-5)

print(f"\n  mvncd_grad.grad_a verification:")
print(f"  Analytic: {result.grad_a}")
print(f"  Numerical: {grad_a_fd}")
print(f"  Max error: {max_err_a:.2e}")
print(f"  Passed: {passed_a}")

# ============================================================
#  Step 6: Best practices
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Best Practices for Gradient Verification")
print("=" * 60)

print("""
  1. ALWAYS verify analytic gradients before using in optimization.
     A small bug in the gradient can cause silent convergence failure.

  2. eps selection:
     - Default: 1e-7 for float64 (sqrt of machine epsilon)
     - For MVNCD gradients: 1e-5 (MVNCD has inherent approximation noise)
     - If errors are large, try both smaller and larger eps

  3. Tolerance selection:
     - Relative tolerance (rtol): 1e-4 to 1e-3
     - Absolute tolerance (atol): 1e-6
     - Use combined: pass if rel_err < rtol OR abs_err < atol

  4. Where to check:
     - At multiple random points (not just zeros or special cases)
     - At the initial parameter values used for optimization
     - At the converged solution (gradients should be ~0)

  5. Common failure modes:
     - Forgot factor of 2 for symmetric matrix off-diagonals
     - Sign errors (maximizing vs minimizing)
     - Missing chain rule terms
     - Numerical noise in MVNCD approximations (use larger eps)

  Template usage:
    passed, err, grad_fd = verify_gradient(f, grad_analytic, x0)
    assert passed, f"Gradient check failed: max error = {err}"
""")

print("=" * 60)
print("  Tutorial series complete!")
print("=" * 60)
print("""
  You've covered the full pybhatlib stack:

  Level 1: vecup     — Matrix vectorization and LDLT decomposition
  Level 2: matgradient — Covariance gradients and chain rules
  Level 3: gradmvn   — MVNCD methods, gradients, and building blocks
  Level 4: models/mnp — MNP estimation, control, and forecasting
  Level 5: models/morp — MORP ordered probit
  Level 6: advanced  — Backends, specifications, and verification

  For more examples, see the examples/ directory.
  For API reference, see the docstrings in src/pybhatlib/.
""")
