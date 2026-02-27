"""Tutorial T01a: Matrix Vectorization Operations.

The vecup module provides low-level operations for converting between
matrices and vectors. These operations follow BHATLIB's row-based
arrangement convention, which is critical for all higher-level computations.

What you will learn:
  - vecdup: extract upper-triangular elements from a symmetric matrix
  - vecndup: extract off-diagonal upper-triangular elements
  - matdupfull: reconstruct a symmetric matrix from its vector form
  - matdupdiagonefull: build a correlation matrix from off-diagonal elements
  - vecsymmetry: the pattern matrix that relates full and vectorized forms
  - nondiag: extract all non-diagonal elements

Prerequisites: None.
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.vecup import (
    vecdup, vecndup, matdupfull, matdupdiagonefull, vecsymmetry, nondiag,
)

# ============================================================
#  Step 1: vecdup — Vectorize a symmetric matrix
# ============================================================
print("=" * 60)
print("  Step 1: vecdup — Vectorize a Symmetric Matrix")
print("=" * 60)

# Example from BHATLIB paper p.6
A = np.array([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6],
], dtype=float)

v = vecdup(A)
print(f"\n  A =\n{A}")
print(f"\n  vecdup(A) = {v}")
print(f"  Expected:    [1. 2. 3. 4. 5. 6.]")
print(f"\n  Convention: row-by-row, upper triangular including diagonal")
print(f"  Row 1: [1, 2, 3]  ->  1, 2, 3")
print(f"  Row 2:    [4, 5]  ->  4, 5")
print(f"  Row 3:       [6]  ->  6")

# ============================================================
#  Step 2: vecndup — Off-diagonal elements only
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: vecndup — Off-Diagonal Elements Only")
print("=" * 60)

v_offdiag = vecndup(A)
print(f"\n  vecndup(A) = {v_offdiag}")
print(f"  Expected:    [2. 3. 5.]")
print(f"\n  This extracts just the unique off-diagonal elements.")
print(f"  For K=3: K*(K-1)/2 = {3*2//2} elements")

# ============================================================
#  Step 3: matdupfull — Reconstruct symmetric matrix
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: matdupfull — Reconstruct from Vector")
print("=" * 60)

A_reconstructed = matdupfull(v)
print(f"\n  matdupfull(vecdup(A)) =\n{A_reconstructed}")
print(f"\n  Roundtrip check: {np.allclose(A, A_reconstructed)}")

# Also works from a length-6 vector directly
v2 = np.array([10., 20., 30., 40., 50., 60.])
M = matdupfull(v2)
print(f"\n  matdupfull([10,20,30,40,50,60]) =\n{M}")
print(f"  Always symmetric: M == M.T is {np.allclose(M, M.T)}")

# ============================================================
#  Step 4: matdupdiagonefull — Correlation matrix from off-diagonals
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: matdupdiagonefull — Build Correlation Matrix")
print("=" * 60)

# Off-diagonal correlations for 3x3 matrix
rho = np.array([0.6, 0.5, 0.5])
R = matdupdiagonefull(rho)
print(f"\n  Off-diagonal correlations: {rho}")
print(f"  matdupdiagonefull(rho) =\n{R}")
print(f"\n  Notice: diagonal is always 1 (unit diagonal)")
print(f"  This is ideal for building correlation matrices.")

# ============================================================
#  Step 5: vecsymmetry — Pattern matrix
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: vecsymmetry — Position Pattern Matrix")
print("=" * 60)

S = vecsymmetry(np.zeros((3, 3)))
print(f"\n  vecsymmetry for K=3: shape = {S.shape}")
print(f"  S is the elimination matrix: vecdup(A) = S @ vec(A)")
print(f"  (maps K^2 = 9 full elements to K*(K+1)/2 = 6 unique elements)")
print(f"\n  S =\n{S}")

# Verify
vec_A = A.T.ravel()   # column-major vectorization
vecdup_A = vecdup(A)
reconstructed = S @ vec_A
print(f"\n  vec(A)          = {vec_A}")
print(f"  S @ vec(A)      = {reconstructed}")
print(f"  vecdup(A)       = {vecdup_A}")
print(f"  Match: {np.allclose(vecdup_A, reconstructed)}")

# ============================================================
#  Step 6: nondiag — All non-diagonal elements
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: nondiag — All Non-Diagonal Elements")
print("=" * 60)

nd = nondiag(A)
print(f"\n  A =\n{A}")
print(f"\n  nondiag(A) = {nd}")
print(f"  (row-by-row, both upper and lower triangular)")
print(f"\n  For K=3: K^2 - K = {9-3} non-diagonal elements")

# ============================================================
#  Summary
# ============================================================
print("\n" + "=" * 60)
print("  Summary: Vectorization Dimensions")
print("=" * 60)

for K in [2, 3, 4, 5]:
    n_dup = K * (K + 1) // 2
    n_ndup = K * (K - 1) // 2
    n_nondiag = K * K - K
    print(f"  K={K}: vecdup={n_dup}, vecndup={n_ndup}, nondiag={n_nondiag}")

print(f"\n  Next: t01b_ldlt.py — LDLT decomposition for positive definite matrices")
