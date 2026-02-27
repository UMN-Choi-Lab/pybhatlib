"""Tutorial T02a: Gradients of Covariance Decomposition (gradcovcor).

The covariance matrix Omega is parameterized as:
    Omega = diag(omega) @ Omega_star @ diag(omega)
where omega are standard deviations and Omega_star is the correlation matrix.

gradcovcor computes the Jacobians in BHATLIB's row-based arrangement:
  - glitomega: shape (K, n_cov) — d(vecdup Omega) / d(omega)
  - gomegastar: shape (n_corr, n_cov) — d(vecdup Omega) / d(off-diag Omega*)

What you will learn:
  - Building Omega from omega and Omega_star
  - Computing and interpreting gradcovcor output
  - Verifying gradients via finite differences
  - Why this decomposition matters for unconstrained optimization

Prerequisites: t01a (vectorization).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.matgradient import gradcovcor, GradCovCorResult
from pybhatlib.vecup import vecdup, vecndup, matdupfull

# ============================================================
#  Step 1: Build Omega from omega and Omega_star
# ============================================================
print("=" * 60)
print("  Step 1: Build Covariance from Std Devs and Correlations")
print("=" * 60)

K = 3
omega = np.array([2.0, 1.5, 1.0])   # standard deviations
Omega_star = np.array([              # correlation matrix
    [1.0, 0.6, 0.3],
    [0.6, 1.0, 0.5],
    [0.3, 0.5, 1.0],
])

# Omega = diag(omega) @ Omega_star @ diag(omega)
Omega = np.diag(omega) @ Omega_star @ np.diag(omega)

print(f"\n  omega (std devs) = {omega}")
print(f"  Omega_star (correlations) =\n{Omega_star}")
print(f"  Omega (covariance) =\n{Omega}")
print(f"\n  Note: Omega[0,0] = omega[0]^2 = {omega[0]**2:.1f}")
print(f"  Omega[0,1] = omega[0]*rho_01*omega[1] = {omega[0]*0.6*omega[1]:.1f}")

# ============================================================
#  Step 2: Compute gradcovcor
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Compute gradcovcor Jacobians")
print("=" * 60)

result = gradcovcor(Omega)

n_cov = K * (K + 1) // 2   # = 6 upper-tri elements of Omega
n_corr = K * (K - 1) // 2  # = 3 off-diagonal correlation elements

print(f"\n  glitomega shape: {result.glitomega.shape}")
print(f"    Rows = K = {K} (one per std dev omega_k)")
print(f"    Cols = K*(K+1)/2 = {n_cov} (vecdup of Omega)")
print(f"    glitomega[k, c] = d(Omega_c) / d(omega_k)")

print(f"\n  gomegastar shape: {result.gomegastar.shape}")
print(f"    Rows = K*(K-1)/2 = {n_corr} (off-diagonal correlations)")
print(f"    Cols = K*(K+1)/2 = {n_cov} (vecdup of Omega)")

print(f"\n  glitomega =\n{result.glitomega}")
print(f"\n  gomegastar =\n{result.gomegastar}")

# ============================================================
#  Step 3: Verify glitomega via finite differences
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Verify glitomega (d vecdup(Omega) / d omega)")
print("=" * 60)

eps = 1e-7
glitomega_fd = np.zeros_like(result.glitomega)

for k in range(K):
    omega_plus = omega.copy(); omega_plus[k] += eps
    omega_minus = omega.copy(); omega_minus[k] -= eps
    Omega_plus = np.diag(omega_plus) @ Omega_star @ np.diag(omega_plus)
    Omega_minus = np.diag(omega_minus) @ Omega_star @ np.diag(omega_minus)
    # Row k of glitomega: how each vecdup(Omega) element changes with omega_k
    glitomega_fd[k, :] = (vecdup(Omega_plus) - vecdup(Omega_minus)) / (2 * eps)

max_err = np.max(np.abs(result.glitomega - glitomega_fd))
print(f"\n  Finite difference step: eps = {eps}")
print(f"  Max error |analytic - numerical|: {max_err:.2e}")
print(f"  Verification passed: {max_err < 1e-4}")

# ============================================================
#  Step 4: Verify gomegastar via finite differences
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Verify gomegastar (d vecdup(Omega) / d off-diag Omega*)")
print("=" * 60)

# Off-diagonal elements of Omega_star (upper-triangular, no diagonal)
offdiag_elements = vecndup(Omega_star)
gomegastar_fd = np.zeros_like(result.gomegastar)

# Enumerate off-diagonal upper-tri positions
offdiag_idx = 0
for i in range(K):
    for j in range(i + 1, K):
        Ostar_plus = Omega_star.copy()
        Ostar_plus[i, j] += eps; Ostar_plus[j, i] += eps
        Omega_plus = np.diag(omega) @ Ostar_plus @ np.diag(omega)

        Ostar_minus = Omega_star.copy()
        Ostar_minus[i, j] -= eps; Ostar_minus[j, i] -= eps
        Omega_minus = np.diag(omega) @ Ostar_minus @ np.diag(omega)

        gomegastar_fd[offdiag_idx, :] = (
            vecdup(Omega_plus) - vecdup(Omega_minus)
        ) / (2 * eps)
        offdiag_idx += 1

max_err2 = np.max(np.abs(result.gomegastar - gomegastar_fd))
print(f"\n  Max error |analytic - numerical|: {max_err2:.2e}")
print(f"  Verification passed: {max_err2 < 1e-4}")

# ============================================================
#  Step 5: Interpretation
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Why Decompose Covariance This Way?")
print("=" * 60)

print("""
  Optimization requires unconstrained parameters, but:
  - Standard deviations omega must be positive -> log(omega) is unconstrained
  - Correlations must satisfy |rho| < 1 and Omega* must be PD
    -> spherical parameterization (see t02b) maps R^n to valid correlations

  gradcovcor provides the chain rule pieces:
    d(loss)/d(omega) = glitomega @ d(loss)/d(vecdup Omega)
    d(loss)/d(Omega*) = gomegastar @ d(loss)/d(vecdup Omega)

  This makes gradient-based optimization of the covariance matrix possible
  while maintaining all required constraints.
""")

print(f"  Next: t02b_spherical.py — Spherical parameterization of correlations")
