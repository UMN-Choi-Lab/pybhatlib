"""Tutorial T03a: MVNCD Approximation Methods Compared.

Computing the multivariate normal CDF P(X <= b) is the computational
core of MNP estimation. pybhatlib implements 7 methods with different
accuracy-speed tradeoffs.

What you will learn:
  - All 7 MVNCD methods and how to select them
  - Accuracy comparison against scipy reference
  - Speed comparison across methods
  - Recommended method for different problem sizes

Prerequisites: t03d (univariate CDFs) recommended.
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.gradmvn import mvncd

# ============================================================
#  Step 1: K=2 — All methods should be exact
# ============================================================
print("=" * 60)
print("  Step 1: K=2 — Bivariate (All Methods Exact)")
print("=" * 60)

sigma2 = np.array([[1.0, 0.5], [0.5, 1.0]])
a2 = np.array([1.0, 0.5])

methods = ["me", "ovus", "bme", "tvbs", "ovbs", "ssj", "scipy"]
ref2 = mvncd(a2, sigma2, method="scipy")

print(f"\n  sigma = [[1.0, 0.5], [0.5, 1.0]]")
print(f"  a = [1.0, 0.5]")
print(f"  scipy reference = {ref2:.6f}")
print(f"\n  {'Method':>8s} {'P(X<=b)':>10s} {'RelErr':>10s}")
print(f"  {'-'*30}")

for m in methods:
    p = mvncd(a2, sigma2, method=m, seed=42)
    rel_err = abs(p - ref2) / max(ref2, 1e-15)
    print(f"  {m:>8s} {p:>10.6f} {rel_err:>10.2e}")

# ============================================================
#  Step 2: K=3 — Accuracy starts to diverge
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: K=3 — Trivariate")
print("=" * 60)

sigma3 = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.4],
    [0.1, 0.4, 1.0],
])
a3 = np.array([1.0, 0.5, 0.0])
ref3 = mvncd(a3, sigma3, method="scipy")

print(f"\n  scipy reference = {ref3:.6f}")
print(f"\n  {'Method':>8s} {'P(X<=b)':>10s} {'RelErr%':>10s} {'Time(ms)':>10s}")
print(f"  {'-'*42}")

for m in methods:
    t0 = time.perf_counter()
    n_trials = 50
    for _ in range(n_trials):
        p = mvncd(a3, sigma3, method=m, seed=42)
    elapsed = (time.perf_counter() - t0) / n_trials * 1000
    rel_pct = abs(p - ref3) / max(ref3, 1e-15) * 100
    print(f"  {m:>8s} {p:>10.6f} {rel_pct:>9.1f}% {elapsed:>9.2f}")

# ============================================================
#  Step 3: K=5 — Accuracy degrades for analytic methods
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: K=5 — Higher Dimensional")
print("=" * 60)

# Build a valid 5x5 correlation matrix
rng = np.random.default_rng(42)
L5 = np.eye(5) + 0.3 * np.tril(rng.standard_normal((5, 5)), -1)
sigma5 = L5 @ L5.T
# Normalize to correlation matrix
d5 = np.sqrt(np.diag(sigma5))
sigma5 = sigma5 / np.outer(d5, d5)

a5 = np.array([0.5, 0.3, 0.1, -0.2, 0.4])
ref5 = mvncd(a5, sigma5, method="scipy")

print(f"\n  scipy reference = {ref5:.6f}")
print(f"\n  {'Method':>8s} {'P(X<=b)':>10s} {'RelErr%':>10s} {'Time(ms)':>10s}")
print(f"  {'-'*42}")

for m in methods:
    t0 = time.perf_counter()
    n_trials = 20
    for _ in range(n_trials):
        p = mvncd(a5, sigma5, method=m, seed=42)
    elapsed = (time.perf_counter() - t0) / n_trials * 1000
    rel_pct = abs(p - ref5) / max(ref5, 1e-15) * 100
    print(f"  {m:>8s} {p:>10.6f} {rel_pct:>9.1f}% {elapsed:>9.2f}")

# ============================================================
#  Step 4: Summary table
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Method Summary")
print("=" * 60)

print("""
  Method  | Basis CDFs      | Complexity  | Best For
  --------|-----------------|-------------|------------------
  ME      | univariate      | O(K^2)      | K=2-3, speed
  OVUS    | uni + bivariate | O(K^2)      | K=2-5 (default)
  BME     | bivariate       | O(K^2)      | K=3-5
  OVBS    | BME + trivariate| O(K^3)      | K=4-7, accuracy
  TVBS    | BME + quad      | O(K^4)      | K=3-5, best accuracy
  SSJ     | QMC simulation  | O(K*N_draws)| K>5, large problems
  scipy   | numerical integ | varies      | reference only

  Notes:
  - 'ovus' (default) is the best general-purpose choice
  - For K <= 2, all methods give identical results
  - Analytic methods have known tolerances: ~20% for K=3, ~35% for K=5
  - SSJ accuracy improves with n_draws but is stochastic
  - 'scipy' is the gold standard but can be slow for K > 5
""")

print(f"  Next: t03b_mvncd_gradients.py — Gradients of the MVNCD")
