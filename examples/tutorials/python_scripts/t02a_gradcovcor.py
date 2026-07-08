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

Expected runtime: <5 sec
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

print("""
  GAUSS cross-reference (matgradient.src, proc gradcovcor):
  The GAUSS routine returns { glitomega, gomegastar } in exactly this
  row-based arrangement: each ROW of glitomega is one std-dev omega_k and
  each COLUMN is one vecdup(Omega) element, and each row of gomegastar is
  one off-diagonal correlation. The GAUSS header documents the intended use
  as the LEFT-multiplied chain rule  dF/dw = glitomega * dF/da  (and
  dF/drho = gomegastar * dF/da), which is what we demonstrate in Step 5.
  Because gradcovcor is a deterministic Jacobian (no estimation), the
  finite-difference checks below ARE the numerical cross-check against the
  GAUSS algorithm.
""")

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
#  Step 5: Apply the chain rule to a scalar function
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Chain Rule in Action (dF/domega, dF/dOmega*)")
print("=" * 60)

print("""
  The whole point of gradcovcor is the chain rule. Suppose an estimator
  produces dF/d(vecdup Omega) -- the gradient of a scalar loss F with
  respect to the covariance elements. To optimize over the std devs omega
  and correlations Omega* directly we left-multiply by the Jacobians:

      dF/domega = glitomega   @ dF/d(vecdup Omega)
      dF/dOmega* = gomegastar @ dF/d(vecdup Omega)

  Here we pick a concrete scalar F(Omega) = sum(vecdup(Omega) ** 2), so
  dF/d(vecdup Omega) = 2 * vecdup(Omega), then verify the propagated
  gradients against finite differences of F taken in (omega, Omega*) space.
""")

vec_Omega = vecdup(Omega)
dF_dvecOmega = 2.0 * vec_Omega          # analytic dF/d(vecdup Omega)

# Propagate via gradcovcor Jacobians (LEFT multiply, GAUSS convention)
dF_domega = result.glitomega @ dF_dvecOmega
dF_dOmegastar = result.gomegastar @ dF_dvecOmega

# Finite-difference dF directly w.r.t. omega
dF_domega_fd = np.zeros(K)
for k in range(K):
    op = omega.copy(); op[k] += eps
    om = omega.copy(); om[k] -= eps
    Fp = np.sum(vecdup(np.diag(op) @ Omega_star @ np.diag(op)) ** 2)
    Fm = np.sum(vecdup(np.diag(om) @ Omega_star @ np.diag(om)) ** 2)
    dF_domega_fd[k] = (Fp - Fm) / (2 * eps)

# Finite-difference dF directly w.r.t. off-diagonal correlations
dF_dOmegastar_fd = np.zeros(n_corr)
idx = 0
for i in range(K):
    for j in range(i + 1, K):
        Sp = Omega_star.copy(); Sp[i, j] += eps; Sp[j, i] += eps
        Sm = Omega_star.copy(); Sm[i, j] -= eps; Sm[j, i] -= eps
        Fp = np.sum(vecdup(np.diag(omega) @ Sp @ np.diag(omega)) ** 2)
        Fm = np.sum(vecdup(np.diag(omega) @ Sm @ np.diag(omega)) ** 2)
        dF_dOmegastar_fd[idx] = (Fp - Fm) / (2 * eps)
        idx += 1

err_omega = np.max(np.abs(dF_domega - dF_domega_fd))
err_star = np.max(np.abs(dF_dOmegastar - dF_dOmegastar_fd))

print(f"  dF/domega   (chain rule) = {dF_domega}")
print(f"  dF/domega   (finite diff)= {dF_domega_fd}")
print(f"  max error                = {err_omega:.2e}")
print(f"\n  dF/dOmega*  (chain rule) = {dF_dOmegastar}")
print(f"  dF/dOmega*  (finite diff)= {dF_dOmegastar_fd}")
print(f"  max error                = {err_star:.2e}")
print(f"\n  Chain-rule verification passed: "
      f"{err_omega < 1e-4 and err_star < 1e-4}")

# ============================================================
#  Step 6: Interpretation
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Why Decompose Covariance This Way?")
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
