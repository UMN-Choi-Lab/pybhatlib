"""Tutorial T03b: MVNCD Gradients.

For gradient-based MNP estimation, we need not just P(X <= b) but also
its partial derivatives with respect to the integration limits (b) and
the covariance matrix (sigma).

What you will learn:
  - mvncd_grad: simultaneous computation of MVNCD and its gradients
  - Interpreting grad_a (sensitivity to integration limits)
  - Interpreting grad_sigma (sensitivity to covariance)
  - Numerical verification via finite differences

Prerequisites: t03a (MVNCD methods).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.gradmvn import mvncd, mvncd_grad

# ============================================================
#  Step 1: K=3 gradient computation
# ============================================================
print("=" * 60)
print("  Step 1: MVNCD Gradients for K=3")
print("=" * 60)

sigma = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.4],
    [0.1, 0.4, 1.0],
])
a = np.array([1.0, 0.5, 0.0])

result = mvncd_grad(a, sigma)

print(f"\n  a (integration limits) = {a}")
print(f"  sigma =\n{sigma}")
print(f"\n  P(X <= a) = {result.prob:.6f}")
print(f"  grad_a = {result.grad_a}")
print(f"  grad_sigma shape = {result.grad_sigma.shape}")
print(f"  grad_sigma =\n{result.grad_sigma}")

# ============================================================
#  Step 2: Interpret grad_a
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Interpreting grad_a")
print("=" * 60)

print(f"\n  grad_a[k] = dP/da_k = how P changes when limit k increases")
print(f"\n  grad_a = {result.grad_a}")
print(f"\n  All positive (as expected): increasing any limit increases P(X <= a)")

for k in range(3):
    print(f"    dP/da_{k+1} = {result.grad_a[k]:.6f}"
          f" — raising a_{k+1} from {a[k]:.1f} increases probability")

# ============================================================
#  Step 3: Verify grad_a via finite differences
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Verify grad_a (Finite Differences)")
print("=" * 60)

eps = 1e-5
grad_a_fd = np.zeros(3)

# IMPORTANT: use the same method as mvncd_grad (default: "me").
# scipy's CDF is stochastic for K>=3 (Genz algorithm with randomization),
# so finite-differencing scipy gives noisy gradients.  ME is deterministic.
for k in range(3):
    a_plus = a.copy(); a_plus[k] += eps
    a_minus = a.copy(); a_minus[k] -= eps
    p_plus = mvncd(a_plus, sigma, method="me")
    p_minus = mvncd(a_minus, sigma, method="me")
    grad_a_fd[k] = (p_plus - p_minus) / (2 * eps)

print(f"\n  {'k':>4s} {'mvncd_grad':>12s} {'FD (me)':>12s} {'error':>12s}")
print(f"  {'-'*42}")
for k in range(3):
    err = abs(result.grad_a[k] - grad_a_fd[k])
    print(f"  {k+1:>4d} {result.grad_a[k]:>12.6f} {grad_a_fd[k]:>12.6f} {err:>12.2e}")

max_err_a = np.max(np.abs(result.grad_a - grad_a_fd))
print(f"\n  Max error: {max_err_a:.2e}  (should be < 1e-8)")

# ============================================================
#  Step 4: Verify grad_sigma via finite differences
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Verify grad_sigma (Finite Differences)")
print("=" * 60)

# grad_sigma is a vecdup vector (upper-tri elements, row-by-row)
K_dim = 3
n_upper = K_dim * (K_dim + 1) // 2
grad_sigma_fd = np.zeros(n_upper)

# Again use method="me" to match mvncd_grad's default.
idx = 0
for i in range(K_dim):
    for j in range(i, K_dim):
        sigma_plus = sigma.copy()
        sigma_plus[i, j] += eps
        sigma_plus[j, i] += eps  # keep symmetric
        p_plus = mvncd(a, sigma_plus, method="me")

        sigma_minus = sigma.copy()
        sigma_minus[i, j] -= eps
        sigma_minus[j, i] -= eps
        p_minus = mvncd(a, sigma_minus, method="me")

        grad_sigma_fd[idx] = (p_plus - p_minus) / (2 * eps)
        idx += 1

print(f"\n  mvncd_grad grad_sigma: {result.grad_sigma}")
print(f"  Numerical grad_sigma:  {grad_sigma_fd}")

max_err_s = np.max(np.abs(result.grad_sigma - grad_sigma_fd))
print(f"\n  Max error: {max_err_s:.2e}  (should be < 1e-8)")

# ============================================================
#  Step 5: Connection to MNP
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Connection to MNP Log-Likelihood")
print("=" * 60)

print("""
  In MNP estimation, the log-likelihood for observation q choosing
  alternative i is:

    log P_qi = log P(V_qi - V_qj > eps_qj - eps_qi, for all j != i)
             = log MVNCD(a_qi, Lambda_qi)

  where a_qi depends on beta (utility parameters) and Lambda_qi depends
  on the error covariance structure.

  The gradient of the log-likelihood is:
    d(log P_qi)/d(beta) uses grad_a (through da/dbeta = X differences)
    d(log P_qi)/d(Lambda) uses grad_sigma (through chain rules from t02c)

  mvncd_grad computes these via finite differences of a deterministic
  MVNCD method (ME by default), providing stable gradients for optimization.
  Note: do NOT use scipy for FD — its CDF is stochastic for K>=3.
""")

print(f"  Next: t03c_mvncd_rect.py — Rectangular MVNCD for ordered probit")
