"""Tutorial T02b: Spherical Parameterization of Correlations.

For unconstrained optimization, we need to map free parameters theta in R^n
to a valid positive definite correlation matrix Omega*. The spherical
parameterization achieves this via hyperspherical coordinates.

What you will learn:
  - theta_to_corr: convert angles to a valid PD correlation matrix
  - grad_corr_theta: Jacobian d(vecdup Omega*)/d(theta)
  - Verification via finite differences
  - Why this is better than direct correlation parameterization

Prerequisites: t02a (gradcovcor).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.matgradient import theta_to_corr, grad_corr_theta
from pybhatlib.vecup import vecdup

# ============================================================
#  Step 1: Basic theta_to_corr for K=3
# ============================================================
print("=" * 60)
print("  Step 1: Angles -> Correlation Matrix (K=3)")
print("=" * 60)

K = 3
n_theta = K * (K - 1) // 2  # = 3 free parameters
n_upper = K * (K + 1) // 2  # = 6 upper-tri elements

theta = np.array([0.5, 0.3, -0.2])
R = theta_to_corr(theta, K)

print(f"\n  K = {K}, n_theta = {n_theta}")
print(f"  theta = {theta}")
print(f"\n  Omega* = theta_to_corr(theta, K) =\n{R}")
print(f"\n  Diagonal = {np.diag(R)}  (should be all 1.0)")
print(f"  Symmetric: {np.allclose(R, R.T)}")
print(f"  Eigenvalues: {np.linalg.eigvalsh(R)}  (should be all positive)")

# ============================================================
#  Step 2: Verify properties for random thetas
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Always Valid for Any Theta")
print("=" * 60)

rng = np.random.default_rng(42)
print(f"\n  Testing 5 random theta vectors:")
print(f"  {'theta':>30s} {'min_eig':>10s} {'diag=1':>8s} {'symm':>6s}")
print(f"  {'-'*56}")

for _ in range(5):
    theta_rand = rng.uniform(-3, 3, size=n_theta)
    R_rand = theta_to_corr(theta_rand, K)
    min_eig = np.linalg.eigvalsh(R_rand).min()
    diag_ok = np.allclose(np.diag(R_rand), 1.0)
    symm_ok = np.allclose(R_rand, R_rand.T)
    theta_str = "[" + ", ".join(f"{t:+.2f}" for t in theta_rand) + "]"
    print(f"  {theta_str:>30s} {min_eig:>10.4f} {str(diag_ok):>8s} {str(symm_ok):>6s}")

print(f"\n  Key property: ANY theta in R^{n_theta} maps to a valid PD correlation matrix.")
print(f"  This makes unconstrained optimization possible.")

# ============================================================
#  Step 3: Gradient d(vecdup Omega*)/d(theta)
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Jacobian grad_corr_theta")
print("=" * 60)

J = grad_corr_theta(theta, K)

print(f"\n  Jacobian shape: {J.shape}")
print(f"    Rows = n_theta = {n_theta} (free angle parameters)")
print(f"    Cols = K*(K+1)/2 = {n_upper} (vecdup of Omega*)")
print(f"    J[p, q] = d(Omega*_q) / d(theta_p)")
print(f"\n  J =\n{J}")

# Verify via finite differences
eps = 1e-7
J_fd = np.zeros_like(J)
for p in range(n_theta):
    theta_plus = theta.copy(); theta_plus[p] += eps
    theta_minus = theta.copy(); theta_minus[p] -= eps
    R_plus = theta_to_corr(theta_plus, K)
    R_minus = theta_to_corr(theta_minus, K)
    J_fd[p, :] = (vecdup(R_plus) - vecdup(R_minus)) / (2 * eps)

max_err = np.max(np.abs(J - J_fd))
print(f"\n  Finite difference verification:")
print(f"  Max error: {max_err:.2e}")
print(f"  Passed: {max_err < 1e-4}")

# ============================================================
#  Step 4: K=4 example
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Scalability — K=4")
print("=" * 60)

K4 = 4
n_theta4 = K4 * (K4 - 1) // 2  # = 6
theta4 = rng.standard_normal(n_theta4) * 0.5
R4 = theta_to_corr(theta4, K4)
J4 = grad_corr_theta(theta4, K4)

print(f"\n  K=4: n_theta = {n_theta4}")
print(f"  theta = {theta4}")
print(f"  Omega* =\n{R4}")
print(f"  Eigenvalues: {np.linalg.eigvalsh(R4)}")
print(f"  Jacobian shape: {J4.shape}")

# ============================================================
#  Step 5: Why spherical?
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Why Spherical Parameterization?")
print("=" * 60)

print("""
  Direct parameterization of correlations is problematic:
  1. Individual correlations must satisfy |rho_ij| < 1
  2. The correlation matrix must be positive definite
  3. These constraints are coupled — not a simple box constraint

  Spherical parameterization solves all three:
  - Maps any vector theta in R^(K(K-1)/2) to a valid PD correlation
  - The mapping is smooth and differentiable
  - Gradient computation is exact via grad_corr_theta

  In the MNP estimation pipeline:
    theta (unconstrained) -> Omega* (correlation) -> Omega (covariance)
  with gradients flowing backward through:
    grad_corr_theta -> gomegastar -> glitomega
""")

print(f"  Next: t02c_chain_rules.py — Composing matrix gradients via chain rules")
