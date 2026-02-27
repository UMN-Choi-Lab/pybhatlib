"""Tutorial T01b: LDLT Decomposition.

The LDLT decomposition factorizes a symmetric positive definite matrix
as A = L @ D @ L.T where L is unit lower triangular and D is diagonal.
This is numerically stable and avoids square roots (unlike Cholesky).

What you will learn:
  - ldlt_decompose: compute L and D from a PD matrix
  - ldlt_rank1_update: efficient O(n^2) update when A changes by alpha*v*v.T
  - Why this matters for MVNCD sequential conditioning

Prerequisites: t01a (vectorization basics).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.vecup import ldlt_decompose, ldlt_rank1_update

# ============================================================
#  Step 1: Basic LDLT decomposition
# ============================================================
print("=" * 60)
print("  Step 1: Basic LDLT Decomposition")
print("=" * 60)

# A 4x4 positive definite matrix
A = np.array([
    [4.0, 2.0, 1.0, 0.5],
    [2.0, 5.0, 2.0, 1.0],
    [1.0, 2.0, 6.0, 3.0],
    [0.5, 1.0, 3.0, 7.0],
])

L, D = ldlt_decompose(A)

print(f"\n  A =\n{A}")
print(f"\n  L (unit lower triangular) =\n{L}")
print(f"\n  D (diagonal) = {np.diag(D)}")

# ============================================================
#  Step 2: Verify properties
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Verify Properties")
print("=" * 60)

# L is unit lower triangular
print(f"\n  L diagonal (should be all 1s): {np.diag(L)}")
print(f"  L upper triangle (should be all 0s): {L[np.triu_indices(4, k=1)]}")

# D is positive (since A is PD)
print(f"  D diagonal (should be all positive): {np.diag(D)}")

# Reconstruction: A = L @ D @ L.T
A_recon = L @ D @ L.T
print(f"\n  Reconstruction error: {np.max(np.abs(A - A_recon)):.2e}")
print(f"  L @ D @ L.T == A: {np.allclose(A, A_recon)}")

# ============================================================
#  Step 3: Rank-1 update
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Rank-1 Update (O(n^2) instead of O(n^3))")
print("=" * 60)

v = np.array([1.0, 0.5, -0.3, 0.8])
alpha = 2.0

# Direct computation: A_new = A + alpha * v @ v.T
A_new = A + alpha * np.outer(v, v)

# Efficient update using existing LDLT
L_new, D_new = ldlt_rank1_update(L, D, v, alpha=alpha)
A_new_recon = L_new @ D_new @ L_new.T

print(f"\n  v = {v}")
print(f"  alpha = {alpha}")
print(f"\n  A + alpha * v @ v.T =\n{A_new}")
print(f"\n  L_new @ D_new @ L_new.T =\n{A_new_recon}")
print(f"\n  Update error: {np.max(np.abs(A_new - A_new_recon)):.2e}")
print(f"  Match: {np.allclose(A_new, A_new_recon)}")

# ============================================================
#  Step 4: Complexity comparison
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Why This Matters")
print("=" * 60)

print("""
  Full LDLT decomposition: O(n^3)
  Rank-1 update:           O(n^2)

  In MVNCD (multivariate normal CDF computation), we sequentially
  condition on each variable. Each conditioning step modifies the
  covariance matrix by a rank-1 update. Using ldlt_rank1_update
  avoids recomputing the full decomposition at each step.

  For a K-dimensional problem:
    Naive:   K x O(K^3) = O(K^4)
    With update: K x O(K^2) = O(K^3)

  This speedup is critical for MNP estimation where MVNCD is called
  thousands of times during optimization.
""")

# Demonstrate: single rank-1 update is much cheaper than full decomposition
import time
n = 50
A_big = np.eye(n) + 0.1 * np.random.default_rng(0).standard_normal((n, n))
A_big = A_big @ A_big.T  # make PD
L_big, D_big = ldlt_decompose(A_big)
v_big = np.random.default_rng(1).standard_normal(n)

t0 = time.perf_counter()
for _ in range(100):
    ldlt_rank1_update(L_big, D_big, v_big, alpha=0.1)
t_update = (time.perf_counter() - t0) / 100

t0 = time.perf_counter()
for _ in range(100):
    ldlt_decompose(A_big + 0.1 * np.outer(v_big, v_big))
t_full = (time.perf_counter() - t0) / 100

print(f"  n={n}: rank-1 update = {t_update*1000:.3f} ms, full decomp = {t_full*1000:.3f} ms")
print(f"  Speedup: {t_full/t_update:.1f}x")

print(f"\n  Next: t01c_truncated_mvn.py — Truncated multivariate normal moments")
