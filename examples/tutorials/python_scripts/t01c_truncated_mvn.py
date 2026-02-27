"""Tutorial T01c: Truncated Multivariate Normal Moments.

Computing the mean and covariance of a multivariate normal distribution
truncated to a rectangular region. This is essential for Gibbs sampling
in MNP and for MVNCD computations.

What you will learn:
  - truncated_mvn_moments: compute E[X | a <= X <= b] and Cov[X | a <= X <= b]
  - Monte Carlo validation of the analytic moments
  - Different truncation scenarios and their effects

Prerequisites: t01a (vectorization), t01b (LDLT).
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
print(f"  Max cov difference:      {np.max(np.abs(trunc_cov - mc_cov)):.4f}")

# ============================================================
#  Step 3: Different truncation scenarios
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Truncation Scenarios")
print("=" * 60)

scenarios = [
    ("Symmetric tight", [-0.5, -0.5], [0.5, 0.5]),
    ("One-sided upper", [-np.inf, -np.inf], [0.0, 0.0]),
    ("One-sided lower", [0.0, 0.0], [np.inf, np.inf]),
    ("Asymmetric", [-2.0, -0.5], [0.5, 2.0]),
]

for name, lo, hi in scenarios:
    lo_arr = np.array(lo)
    hi_arr = np.array(hi)
    t_mean, t_cov = truncated_mvn_moments(mu, sigma, lo_arr, hi_arr)
    print(f"\n  {name}: [{lo}] to [{hi}]")
    print(f"    Truncated mean: {t_mean}")
    print(f"    Truncated var:  [{t_cov[0,0]:.4f}, {t_cov[1,1]:.4f}]")

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
