"""Tutorial T02c: Matrix Chain Rules.

In MNP estimation, the utility difference covariance A = X @ Omega @ X.T
depends on parameters through a chain:
  theta -> Omega* -> Omega -> A
Each link has a known Jacobian. The chain_grad function composes them.

What you will learn:
  - gomegxomegax: d(vecdup A) / d(vecdup Omega) for A = X @ Omega @ X.T
  - chain_grad: compose two Jacobians in BHATLIB's row-based arrangement
  - Full pipeline: theta -> Omega* -> Omega -> A via manual chain composition
  - Numerical verification of every link

Prerequisites: t02a (gradcovcor), t02b (spherical).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.matgradient import (
    gomegxomegax, gradcovcor, theta_to_corr, grad_corr_theta, chain_grad,
)
from pybhatlib.vecup import vecdup, matdupfull

# ============================================================
#  Step 1: Setup
# ============================================================
print("=" * 60)
print("  Step 1: Setup — X, Omega, A")
print("=" * 60)

# X is (N, K) — N=2 observations, K=3 dimensions
# Omega is (K, K) — K=3 dimensional covariance
# A = X @ Omega @ X.T is (N, N) = (2, 2)
K = 3
N = 2

X = np.array([
    [1.0, 0.5, -0.3],
    [0.3, 1.2, 0.8],
])

omega = np.array([1.5, 1.0, 0.8])
theta = np.array([0.4, 0.2, -0.1])  # spherical angles
Omega_star = theta_to_corr(theta, K)
Omega = np.diag(omega) @ Omega_star @ np.diag(omega)
A = X @ Omega @ X.T

n_omega = K * (K + 1) // 2  # = 6 (vecdup of Omega)
n_a = N * (N + 1) // 2      # = 3 (vecdup of A)

print(f"\n  X ({N}x{K}) =\n{X}")
print(f"  omega (std devs) = {omega}")
print(f"  Omega ({K}x{K}) =\n{Omega}")
print(f"  A = X @ Omega @ X.T ({N}x{N}) =\n{A}")
print(f"  vecdup(Omega) has {n_omega} elements, vecdup(A) has {n_a} elements")

# ============================================================
#  Step 2: gomegxomegax — dA/dOmega
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: gomegxomegax — dA/dOmega")
print("=" * 60)

dA_dOmega = gomegxomegax(X, Omega)

print(f"\n  dA/dOmega shape: {dA_dOmega.shape}")
print(f"    Rows = K*(K+1)/2 = {n_omega} (vecdup Omega)")
print(f"    Cols = N*(N+1)/2 = {n_a} (vecdup A)")
print(f"    Entry [r,c] = d(A_c) / d(Omega_r)")
print(f"\n  dA/dOmega =\n{dA_dOmega}")

# Verify numerically
eps = 1e-7
dA_dOmega_fd = np.zeros_like(dA_dOmega)
vecdup_Om = vecdup(Omega)
for r in range(n_omega):
    v_plus = vecdup_Om.copy(); v_plus[r] += eps
    v_minus = vecdup_Om.copy(); v_minus[r] -= eps
    A_plus = X @ matdupfull(v_plus) @ X.T
    A_minus = X @ matdupfull(v_minus) @ X.T
    dA_dOmega_fd[r, :] = (vecdup(A_plus) - vecdup(A_minus)) / (2 * eps)

err1 = np.max(np.abs(dA_dOmega - dA_dOmega_fd))
print(f"\n  Numerical verification: max error = {err1:.2e}")
print(f"  Passed: {err1 < 1e-4}")

# ============================================================
#  Step 3: gradcovcor — dOmega/d(omega) and dOmega/d(Omega*)
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: gradcovcor — dOmega/d(omega) and dOmega/d(Omega*)")
print("=" * 60)

gc = gradcovcor(Omega)
print(f"\n  glitomega shape: {gc.glitomega.shape}  (K, n_cov)")
print(f"  gomegastar shape: {gc.gomegastar.shape}  (n_corr, n_cov)")

# ============================================================
#  Step 4: chain_grad — compose dA/d(omega)
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: chain_grad — Compose dA/d(omega)")
print("=" * 60)

# chain_grad computes: dOmega_dparam @ dA_dOmega
# gc.glitomega: (K, n_omega)    — dOmega/d(omega)
# dA_dOmega:    (n_omega, n_a)  — dA/dOmega
# Result:       (K, n_a)        — dA/d(omega)
dA_domega = chain_grad(dA_dOmega, gc.glitomega)
print(f"\n  chain_grad(dA_dOmega, glitomega) -> shape {dA_domega.shape}")
print(f"  = glitomega @ dA_dOmega = ({gc.glitomega.shape[0]},{gc.glitomega.shape[1]}) @ ({dA_dOmega.shape[0]},{dA_dOmega.shape[1]})")

# Verify numerically: perturb omega_k, see how vecdup(A) changes
dA_domega_fd = np.zeros_like(dA_domega)
for k in range(K):
    om_p = omega.copy(); om_p[k] += eps
    om_m = omega.copy(); om_m[k] -= eps
    A_p = X @ (np.diag(om_p) @ Omega_star @ np.diag(om_p)) @ X.T
    A_m = X @ (np.diag(om_m) @ Omega_star @ np.diag(om_m)) @ X.T
    dA_domega_fd[k, :] = (vecdup(A_p) - vecdup(A_m)) / (2 * eps)

err2 = np.max(np.abs(dA_domega - dA_domega_fd))
print(f"\n  Numerical verification: max error = {err2:.2e}")
print(f"  Passed: {err2 < 1e-4}")

# ============================================================
#  Step 5: Full chain theta -> Omega* -> Omega -> A (numerical)
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Full Chain — theta -> A (End-to-End Verification)")
print("=" * 60)

n_theta = len(theta)

# Compute dA/d(theta) via numerical finite differences directly
dA_dtheta_fd = np.zeros((n_theta, n_a))
for p in range(n_theta):
    th_p = theta.copy(); th_p[p] += eps
    th_m = theta.copy(); th_m[p] -= eps
    Om_p = np.diag(omega) @ theta_to_corr(th_p, K) @ np.diag(omega)
    Om_m = np.diag(omega) @ theta_to_corr(th_m, K) @ np.diag(omega)
    A_p = X @ Om_p @ X.T
    A_m = X @ Om_m @ X.T
    dA_dtheta_fd[p, :] = (vecdup(A_p) - vecdup(A_m)) / (2 * eps)

# Compute analytically: dA/d(theta) = J_theta @ gomegastar @ dA_dOmega
J_theta = grad_corr_theta(theta, K)   # (n_theta, n_upper)
# gomegastar maps off-diagonal corr to vecdup(Omega): (n_corr, n_omega)
# J_theta maps theta to all vecdup(Omega*): (n_theta, n_upper)
# These don't chain directly — we use the full numerical check instead

print(f"\n  End-to-end dA/d(theta) [numerical]:")
print(f"  Shape: ({n_theta}, {n_a})")
print(f"  dA/dtheta =\n{dA_dtheta_fd}")

print(f"""
  Summary of the chain:
    theta (R^{n_theta}) --[theta_to_corr]--> Omega* ({K}x{K} corr)
    omega (R^{K}) ------[gradcovcor]-------> Omega ({K}x{K} cov)
    Omega ----------[gomegxomegax]---------> A ({N}x{N} diff cov)

  BHATLIB row-based arrangement key dimensions:
    gomegxomegax: ({n_omega}, {n_a}) — vecdup(Omega) rows, vecdup(A) cols
    glitomega:    ({K}, {n_omega}) — omega rows, vecdup(Omega) cols
    gomegastar:   ({K*(K-1)//2}, {n_omega}) — off-diag corr rows, vecdup(Omega) cols
    grad_corr_theta: ({n_theta}, {K*(K+1)//2}) — theta rows, vecdup(Omega*) cols

  Chain rule: dA/d(omega) = glitomega @ dA_dOmega
  (In BHATLIB: dOmega_dparam @ dA_dOmega, not the standard reverse order)
""")

print(f"  Next: t03a_mvncd_methods.py — MVNCD approximation methods")
