"""Tutorial T01c: Truncated Multivariate Normal Moments.

Computing the mean and covariance of a multivariate normal distribution
truncated to a rectangular region. This is essential for Gibbs sampling
in MNP and for MVNCD computations.

What you will learn:
  - truncated_mvn_moments: compute E[X | a <= X <= b] and Cov[X | a <= X <= b]
  - Monte Carlo validation of the analytic moments
  - Different truncation scenarios and their effects
  - The nature (and limits) of the sequential-conditioning approximation
    that underlies the MVNCD algorithm

Note on accuracy
----------------
``truncated_mvn_moments`` uses the *sequential-conditioning* scheme that the
MVNCD CDF algorithm relies on (Bhat 2018). The truncated **mean** and the
**diagonal** of the truncated covariance track the true moments closely;
the **off-diagonal** covariance is an approximation (the routine's own
docstring flags the covariance as approximate). This tutorial makes that
explicit by cross-checking every quantity against brute-force Monte Carlo.

Prerequisites: t01a (vectorization), t01b (LDLT).

Expected runtime: <5 sec
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.vecup import truncated_mvn_moments

# ============================================================
#  Step 1: Bivariate truncation — basic example
# ============================================================
print("=" * 60)
print("  Step 1: Bivariate Truncated Normal")
print("=" * 60)

mu = np.array([0.0, 0.0])
sigma = np.array([[1.0, 0.5], [0.5, 1.0]])
lower = np.array([-1.0, -1.0])
upper = np.array([1.0, 1.0])

trunc_mean, trunc_cov = truncated_mvn_moments(mu, sigma, lower, upper)

print(f"\n  mu = {mu}")
print(f"  sigma =\n{sigma}")
print(f"  Truncation: [{lower}] to [{upper}]")
print(f"\n  Truncated mean: {trunc_mean}")
print(f"  Truncated cov:\n{trunc_cov}")
print(f"\n  Note: truncated mean should be ~0 (symmetric truncation)")
print(f"  Truncated variance < 1 (truncation reduces spread)")

# ============================================================
#  Step 2: Monte Carlo validation
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Monte Carlo Validation")
print("=" * 60)

rng = np.random.default_rng(42)
n_samples = 200_000
samples = rng.multivariate_normal(mu, sigma, size=n_samples)

# Keep only samples within the truncation region
mask = np.all((samples >= lower) & (samples <= upper), axis=1)
truncated_samples = samples[mask]

mc_mean = truncated_samples.mean(axis=0)
mc_cov = np.cov(truncated_samples.T)

print(f"\n  Generated {n_samples:,} samples, {mask.sum():,} within bounds "
      f"({100*mask.mean():.1f}%)")
print(f"\n  Analytic truncated mean: {trunc_mean}")
print(f"  Monte Carlo mean:        {mc_mean}")
print(f"  Max difference:          {np.max(np.abs(trunc_mean - mc_mean)):.4f}")
print(f"\n  Analytic truncated cov:\n{trunc_cov}")
print(f"  Monte Carlo cov:\n{mc_cov}")
diag_diff = np.max(np.abs(np.diag(trunc_cov) - np.diag(mc_cov)))
offdiag_diff = abs(trunc_cov[0, 1] - mc_cov[0, 1])
print(f"\n  Diagonal (variance) max diff : {diag_diff:.4f}  (close -> good)")
print(f"  Off-diagonal (cov) diff      : {offdiag_diff:.4f}  (larger -> approx)")

print("""
  Reading the comparison
  ----------------------
  - The truncated MEAN matches Monte Carlo to ~1e-3 (sampling noise).
  - The truncated VARIANCES (diagonal) match closely.
  - The off-diagonal COVARIANCE is only approximate: the sequential
    conditioning that powers MVNCD propagates the truncation effect one
    variable at a time and does not fully recover cross-truncation
    dependence. This is by design and acceptable for CDF evaluation.
""")

# ============================================================
#  Step 3: Different truncation scenarios
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Truncation Scenarios")
print("=" * 60)

# Each scenario is cross-checked against Monte Carlo so the analytic mean /
# variance can be trusted at a glance. NOTE: an *infinite* upper bound on a
# lower-truncated tail can make the sequential routine return NaN, so we use a
# large finite cap (+8 sigma) to represent a one-sided lower truncation.
BIG = 8.0
scenarios = [
    ("Symmetric tight", [-0.5, -0.5], [0.5, 0.5]),
    ("One-sided upper", [-np.inf, -np.inf], [0.0, 0.0]),
    ("One-sided lower", [0.0, 0.0], [BIG, BIG]),
    ("Asymmetric", [-2.0, -0.5], [0.5, 2.0]),
]

# A fresh, large sample reused for every scenario's Monte Carlo check.
big_samples = rng.multivariate_normal(mu, sigma, size=400_000)

for name, lo, hi in scenarios:
    lo_arr = np.array(lo)
    hi_arr = np.array(hi)
    t_mean, t_cov = truncated_mvn_moments(mu, sigma, lo_arr, hi_arr)

    sc_mask = np.all((big_samples >= lo_arr) & (big_samples <= hi_arr), axis=1)
    sc = big_samples[sc_mask]
    mc_m = sc.mean(axis=0)
    mc_v = sc.var(axis=0)

    print(f"\n  {name}: [{lo}] to [{hi}]")
    print(f"    Truncated mean (analytic): {t_mean}")
    print(f"    Truncated mean (MonteCarlo): {mc_m}")
    print(f"    Truncated var  (analytic): [{t_cov[0,0]:.4f}, {t_cov[1,1]:.4f}]")
    print(f"    Truncated var  (MonteCarlo): [{mc_v[0]:.4f}, {mc_v[1]:.4f}]")

print("""
  Interpreting the scenarios
  --------------------------
  Symmetric/centered truncations reproduce Monte Carlo almost exactly.
  For strongly one-sided or asymmetric truncations the sequential routine
  shows a modest bias in the first variable's mean and in the variances,
  because it conditions one coordinate at a time rather than jointly. The
  approximation is intentional: MVNCD only needs the conditional ordering
  to evaluate orthant probabilities, not exact joint truncated moments.
""")

# ============================================================
#  Step 4: Connection to MVNCD
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Connection to MVNCD")
print("=" * 60)

print("""
  In the MVNCD algorithm (multivariate normal CDF computation):

  1. We compute P(X <= b) by sequentially conditioning:
     P(X1 <= b1) * P(X2 <= b2 | X1 <= b1) * ...

  2. At each step, we need the conditional distribution of the
     remaining variables given that earlier ones are truncated.

  3. truncated_mvn_moments provides exactly these conditional
     moments: the mean and covariance of the truncated distribution.

  4. The LDLT decomposition (t01b) is used to efficiently update
     the covariance after each conditioning step.
""")

print(f"  Next: t02a_gradcovcor.py — Gradients of covariance decomposition")
