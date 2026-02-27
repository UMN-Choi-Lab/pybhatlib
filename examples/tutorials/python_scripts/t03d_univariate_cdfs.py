"""Tutorial T03d: Univariate and Low-Dimensional CDF Building Blocks.

The gradmvn module provides analytic CDF functions for standard normal
distributions in 1, 2, 3, and 4 dimensions. These are the building blocks
used by the MVNCD approximation methods.

What you will learn:
  - normal_pdf and normal_cdf for the univariate standard normal
  - bivariate_normal_cdf and the role of correlation rho
  - trivariate_normal_cdf and quadrivariate_normal_cdf
  - How these building blocks compose into MVNCD methods

Prerequisites: None (standalone reference).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.gradmvn import (
    normal_pdf, normal_cdf,
    bivariate_normal_cdf,
    trivariate_normal_cdf,
    quadrivariate_normal_cdf,
)

# ============================================================
#  Step 1: Univariate normal PDF and CDF
# ============================================================
print("=" * 60)
print("  Step 1: Univariate Standard Normal")
print("=" * 60)

from scipy.stats import norm as scipy_norm

x_vals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print(f"\n  {'x':>6s} {'pdf(x)':>10s} {'scipy':>10s} {'cdf(x)':>10s} {'scipy':>10s}")
print(f"  {'-'*48}")
for x in x_vals:
    pdf_val = float(normal_pdf(np.array([x])))
    cdf_val = float(normal_cdf(np.array([x])))
    sp_pdf = scipy_norm.pdf(x)
    sp_cdf = scipy_norm.cdf(x)
    print(f"  {x:>6.1f} {pdf_val:>10.6f} {sp_pdf:>10.6f} {cdf_val:>10.6f} {sp_cdf:>10.6f}")

print(f"\n  Max PDF difference: {max(abs(float(normal_pdf(np.array([x]))) - scipy_norm.pdf(x)) for x in x_vals):.2e}")
print(f"  Max CDF difference: {max(abs(float(normal_cdf(np.array([x]))) - scipy_norm.cdf(x)) for x in x_vals):.2e}")

# ============================================================
#  Step 2: Bivariate normal CDF
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Bivariate Normal CDF")
print("=" * 60)

from scipy.stats import multivariate_normal

print(f"\n  P(X1 <= x1, X2 <= x2) for the standard bivariate normal")
print(f"\n  {'rho':>6s} {'x1':>6s} {'x2':>6s} {'pybhat':>10s} {'scipy':>10s} {'diff':>10s}")
print(f"  {'-'*50}")

test_cases = [
    (0.0, 0.0, 0.0),    # Independent, at origin
    (0.0, 1.0, 1.0),    # Independent, upper quadrant
    (0.8, 0.0, 0.0),    # Strong positive correlation
    (-0.8, 0.0, 0.0),   # Strong negative correlation
    (0.5, 1.5, -0.5),   # Mixed
]

for rho, x1, x2 in test_cases:
    pb_val = bivariate_normal_cdf(x1, x2, rho)
    sigma_bvn = np.array([[1.0, rho], [rho, 1.0]])
    sp_val = multivariate_normal.cdf([x1, x2], mean=[0, 0], cov=sigma_bvn)
    diff = abs(pb_val - sp_val)
    print(f"  {rho:>6.1f} {x1:>6.1f} {x2:>6.1f} {pb_val:>10.6f} {sp_val:>10.6f} {diff:>10.2e}")

# Special case: rho=0 means P(X1<=x1, X2<=x2) = P(X1<=x1) * P(X2<=x2)
bvn_indep = bivariate_normal_cdf(1.0, 1.0, 0.0)
product = float(normal_cdf(np.array([1.0]))) * float(normal_cdf(np.array([1.0])))
print(f"\n  When rho=0: bivariate CDF = product of marginals")
print(f"    bivariate_normal_cdf(1, 1, 0) = {bvn_indep:.6f}")
print(f"    Phi(1) * Phi(1)               = {product:.6f}")
print(f"    Match: {abs(bvn_indep - product) < 1e-6}")

# ============================================================
#  Step 3: Trivariate normal CDF
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Trivariate Normal CDF")
print("=" * 60)

# Correlation matrix for 3 dimensions
sigma3 = np.array([
    [1.0, 0.3, 0.1],
    [0.3, 1.0, 0.2],
    [0.1, 0.2, 1.0],
])

x1, x2, x3 = 1.0, 0.5, 0.0
pb_tri = trivariate_normal_cdf(x1, x2, x3, sigma3)
sp_tri = multivariate_normal.cdf([x1, x2, x3], mean=[0, 0, 0], cov=sigma3)

print(f"\n  sigma =\n{sigma3}")
print(f"  x = [{x1}, {x2}, {x3}]")
print(f"\n  trivariate_normal_cdf = {pb_tri:.6f}")
print(f"  scipy reference       = {sp_tri:.6f}")
print(f"  difference            = {abs(pb_tri - sp_tri):.2e}")

# ============================================================
#  Step 4: Quadrivariate normal CDF
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Quadrivariate Normal CDF")
print("=" * 60)

sigma4 = np.array([
    [1.0, 0.3, 0.1, 0.2],
    [0.3, 1.0, 0.2, 0.1],
    [0.1, 0.2, 1.0, 0.3],
    [0.2, 0.1, 0.3, 1.0],
])

x1, x2, x3, x4 = 1.0, 0.5, 0.0, -0.5
pb_quad = quadrivariate_normal_cdf(x1, x2, x3, x4, sigma4)
sp_quad = multivariate_normal.cdf([x1, x2, x3, x4], mean=np.zeros(4), cov=sigma4)

print(f"\n  sigma =\n{sigma4}")
print(f"  x = [{x1}, {x2}, {x3}, {x4}]")
print(f"\n  quadrivariate_normal_cdf = {pb_quad:.6f}")
print(f"  scipy reference          = {sp_quad:.6f}")
print(f"  difference               = {abs(pb_quad - sp_quad):.2e}")

# ============================================================
#  Step 5: Role in MVNCD methods
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: How These Build Up to MVNCD")
print("=" * 60)

print("""
  MVNCD methods approximate P(X <= b) for K-dimensional MVN by
  combining low-dimensional CDFs:

  Method   Basis CDFs          Accuracy   Complexity
  ------   -----------------   --------   ----------
  ME       univariate only     low        O(K^2)
  OVUS     ME + bivariate      medium     O(K^2)
  BME      bivariate only      medium     O(K^2)
  OVBS     BME + trivariate    good       O(K^3)
  TVBS     BME + quadrivariate best       O(K^4)
  SSJ      QMC simulation      good       O(K * N_draws)
  scipy    numerical integration exact    varies

  As we use higher-dimensional building blocks, accuracy improves
  but computational cost increases. The default 'ovus' balances
  these tradeoffs well for most applications.

  See t03a_mvncd_methods.py for a full comparison.
""")
