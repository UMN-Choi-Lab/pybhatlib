"""Tutorial T03c: Rectangular MVNCD.

Standard MVNCD computes P(X <= b). Rectangular MVNCD computes
P(a <= X <= b), the probability that X falls within a box.
This is needed for ordered response models (MORP).

What you will learn:
  - mvncd_rect: P(lower <= X <= upper)
  - Relation to standard MVNCD via inclusion-exclusion
  - K=2 detailed verification with 4 terms
  - Connection to ordered probit threshold bounds

Prerequisites: t03a (MVNCD methods).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.gradmvn import mvncd, mvncd_rect

# ============================================================
#  Step 1: Standard vs Rectangular MVNCD
# ============================================================
print("=" * 60)
print("  Step 1: Standard vs Rectangular MVNCD")
print("=" * 60)

sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

# Standard: P(X <= b)
b = np.array([1.0, 0.5])
p_standard = mvncd(b, sigma, method="scipy")

# Rectangular: P(a <= X <= b)
a = np.array([-1.0, -0.5])
p_rect = mvncd_rect(a, b, sigma, method="scipy")

print(f"\n  sigma = {sigma.tolist()}")
print(f"\n  Standard MVNCD:   P(X <= [{b[0]}, {b[1]}]) = {p_standard:.6f}")
print(f"  Rectangular MVNCD: P([{a[0]}, {a[1]}] <= X <= [{b[0]}, {b[1]}]) = {p_rect:.6f}")
print(f"\n  Rectangular probability < standard (tighter region)")

# ============================================================
#  Step 2: Inclusion-exclusion verification (K=2)
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: K=2 Inclusion-Exclusion Verification")
print("=" * 60)

# P(a <= X <= b) = P(X <= b) - P(X1 <= a1, X2 <= b2)
#                              - P(X1 <= b1, X2 <= a2)
#                              + P(X1 <= a1, X2 <= a2)
# = P(X<=b) - P(X<=[a1,b2]) - P(X<=[b1,a2]) + P(X<=a)

p_bb = mvncd(np.array([b[0], b[1]]), sigma, method="scipy")   # P(X <= [b1, b2])
p_ab = mvncd(np.array([a[0], b[1]]), sigma, method="scipy")   # P(X <= [a1, b2])
p_ba = mvncd(np.array([b[0], a[1]]), sigma, method="scipy")   # P(X <= [b1, a2])
p_aa = mvncd(np.array([a[0], a[1]]), sigma, method="scipy")   # P(X <= [a1, a2])

p_incl_excl = p_bb - p_ab - p_ba + p_aa

print(f"\n  Inclusion-exclusion for K=2:")
print(f"    P(X <= [b1,b2])  = {p_bb:>10.6f}  (all within upper)")
print(f"  - P(X <= [a1,b2])  = {p_ab:>10.6f}  (X1 too low)")
print(f"  - P(X <= [b1,a2])  = {p_ba:>10.6f}  (X2 too low)")
print(f"  + P(X <= [a1,a2])  = {p_aa:>10.6f}  (double subtraction)")
print(f"  = P(a <= X <= b)   = {p_incl_excl:>10.6f}")
print(f"\n  mvncd_rect result  = {p_rect:>10.6f}")
print(f"  Match: {abs(p_rect - p_incl_excl) < 1e-4}")

# ============================================================
#  Step 3: K=3 with -inf lower bound
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: K=3 with Partial Lower Bounds")
print("=" * 60)

sigma3 = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.4],
    [0.1, 0.4, 1.0],
])

lower3 = np.array([-np.inf, -1.0, -0.5])
upper3 = np.array([1.0, 0.5, np.inf])

p_rect3 = mvncd_rect(lower3, upper3, sigma3, method="scipy")

print(f"\n  lower = [{lower3[0]}, {lower3[1]}, {lower3[2]}]")
print(f"  upper = [{upper3[0]}, {upper3[1]}, {upper3[2]}]")
print(f"\n  P(lower <= X <= upper) = {p_rect3:.6f}")
print(f"\n  -inf lower bound on X1 means: no lower constraint on X1")
print(f"  +inf upper bound on X3 means: no upper constraint on X3")

# When all lowers are -inf, rectangular = standard
lower_all_inf = np.array([-np.inf, -np.inf, -np.inf])
upper_std = np.array([1.0, 0.5, 0.0])
p_rect_std = mvncd_rect(lower_all_inf, upper_std, sigma3, method="scipy")
p_std = mvncd(upper_std, sigma3, method="scipy")

print(f"\n  Special case: all lower = -inf")
print(f"    mvncd_rect([-inf,-inf,-inf], [1,0.5,0])  = {p_rect_std:.6f}")
print(f"    mvncd([1, 0.5, 0])                       = {p_std:.6f}")
print(f"    Match: {abs(p_rect_std - p_std) < 1e-4}")

# ============================================================
#  Step 4: Connection to ordered probit
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Connection to Ordered Probit (MORP)")
print("=" * 60)

print("""
  In the Multivariate Ordered Response Probit (MORP), each dimension d
  has ordinal categories 0, 1, ..., J_d mapped by thresholds:

    Y_d = j  iff  tau_{j-1} < Y*_d <= tau_j

  where tau_0 = -inf, tau_{J_d} = +inf, and Y*_d is the latent utility.

  For a D-dimensional MORP, the probability of observing category
  vector (j_1, j_2, ..., j_D) is:

    P(Y = j) = P(tau_{j1-1} < Y*_1 <= tau_{j1}, ..., tau_{jD-1} < Y*_D <= tau_{jD})

  After standardizing: P(Y = j) = mvncd_rect(lower, upper, sigma)

  where lower_d = (tau_{jd-1} - mu_d) / sigma_d
        upper_d = (tau_{jd}   - mu_d) / sigma_d

  This is why rectangular MVNCD is essential for ordered probit models.
""")

# Quick demonstration
print("  Example: 2D ordered probit with 3 categories each")
thresholds = [np.array([-0.5, 0.5]), np.array([-0.3, 0.8])]
mu = np.array([0.2, -0.1])
sigma_err = np.array([[1.0, 0.4], [0.4, 1.0]])

# P(Y1=1, Y2=2) = P(tau10 < Y1* <= tau11, tau21 < Y2* <= tau22)
j1, j2 = 1, 2  # middle category dim1, high category dim2
lo = np.array([
    (thresholds[0][j1-1] - mu[0]),   # tau_0 for dim1 = -0.5
    (thresholds[1][j2-1] - mu[1]),   # tau_1 for dim2 = 0.8
])
hi = np.array([
    (thresholds[0][j1] - mu[0]),     # tau_1 for dim1 = 0.5
    np.inf,                           # tau_2 for dim2 = +inf
])

p_joint = mvncd_rect(lo, hi, sigma_err, method="scipy")
print(f"  P(Y1={j1}, Y2={j2}) = {p_joint:.6f}")
print(f"  (lower={lo}, upper=[{hi[0]:.1f}, inf])")

print(f"\n  Next: t04c_mnp_heteronly.py — MNP heteroscedastic-only model")
