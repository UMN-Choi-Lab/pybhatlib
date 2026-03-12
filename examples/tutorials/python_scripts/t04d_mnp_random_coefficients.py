"""Tutorial T04d: MNP Random Coefficients.

Standard MNP models assume that every individual shares the same
preference coefficients (beta). The random coefficients extension
relaxes this by treating the coefficient on one or more variables as
a random draw from a normal distribution, capturing unobserved taste
heterogeneity across the population.

What you will learn:
  - How random coefficients extend fixed-coefficient MNP
  - What beta_i ~ N(mu, Omega) means for behavioral interpretation
  - How to enable random coefficients with mix=True and ranvars=[...]
  - How to interpret the estimated mean mu and covariance Omega
  - Computational tradeoffs relative to the fixed-coefficient model

Prerequisites: t04b (flexible covariance).

Expected runtime: ~6 min
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# Model (c) spec: 5-variable base + AGE45 demographics (Bhat 2018 Table 1)
spec = {
    "CON_SR":   {"Alt1_ch": "sero",    "Alt2_ch": "uno",    "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",    "Alt2_ch": "sero",   "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45",   "Alt2_ch": "sero",   "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",    "Alt2_ch": "AGE45",  "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# ============================================================
#  Step 1: Random Coefficients Concept
# ============================================================
print("=" * 60)
print("  Step 1: Random Coefficients Concept")
print("=" * 60)

print("""
  Fixed-coefficient MNP assumes every individual in the sample
  shares the same preference vector beta.  Utility for person n
  choosing alternative j is:

      U_nj = beta' * x_nj + epsilon_nj

  When preferences genuinely vary across the population — e.g.,
  some travellers are far more sensitive to travel time than others
  — a single mean vector cannot capture this.

  Random coefficients relax the assumption by treating one or more
  elements of beta as individual-specific draws:

      beta_i ~ N(mu, Omega)

  where mu is the population mean and Omega is the covariance of
  taste heterogeneity.  The log-likelihood integrates out the
  latent beta_i using simulation (QMC / Halton draws):

      L_n = E_{beta_i}[ P(y_n | beta_i, x_n) ]

  In pybhatlib, enabling random coefficients requires two arguments
  on MNPModel:

      mix=True          — activates the mixed (random-coefficient) mode
      ranvars=["OVTT"]  — lists which variables get random coefficients

  The model then estimates:
    - The mean utility coefficients mu (one per variable)
    - The lower-Cholesky factor L of Omega, where Omega = L L'
      (Omega captures the variance and covariance of taste
       heterogeneity for the random variables)

  For a single random variable (OVTT), Omega is a 1x1 scalar:
  the variance of the OVTT coefficient across individuals.
  The square root of this variance is the standard deviation of
  taste heterogeneity — a larger value means greater diversity
  in how travellers respond to out-of-vehicle travel time.
""")

# ============================================================
#  Step 2: Model Setup and Estimation
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Model Setup and Estimation")
print("=" * 60)

print("""
  We use the 7-variable Model (c) specification from BHATLIB
  Table 1, which extends Model (b) by adding a random coefficient
  on OVTT.  The spec includes:

    CON_SR, CON_TR  — alternative-specific constants
    AGE45_DA        — AGE45 interaction for drive-alone
    AGE45_SR        — AGE45 interaction for shared-ride
    IVTT            — in-vehicle travel time (fixed coefficient)
    OVTT            — out-of-vehicle travel time (RANDOM coefficient)
    COST            — monetary cost (fixed coefficient)

  MNPControl settings:
    iid=False   — use flexible differenced covariance
    maxiter=200 — allow sufficient iterations for convergence
    verbose=1   — print progress every iteration
    seed=42     — reproducible QMC draws
""")

ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

t0 = time.perf_counter()
model = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec,
    mix=True,
    ranvars=["OVTT"],
    control=ctrl,
)
results = model.fit()
t_elapsed = time.perf_counter() - t0

print(f"\n  Log-likelihood: {results.ll_total:.3f}")
print(f"  Parameters: {len(results.b)}")
print(f"  Estimation time: {t_elapsed:.1f}s")

# ============================================================
#  Step 3: Interpreting Results
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Interpreting Results")
print("=" * 60)

print("\n  Estimated coefficients:")
if hasattr(results, "param_names") and results.param_names is not None:
    for name, val in zip(results.param_names, results.b):
        print(f"    {name:<25s}  {val:>10.4f}")
else:
    param_labels = list(spec.keys())
    for i, val in enumerate(results.b):
        label = param_labels[i] if i < len(param_labels) else f"param_{i}"
        print(f"    {label:<25s}  {val:>10.4f}")

print(f"\n  Target log-likelihood : -635.871  (BHATLIB paper Table 1, Model c)")
print(f"  Achieved log-likelihood: {results.ll_total:.3f}")

diff = abs(results.ll_total - (-635.871))
print(f"  Absolute difference   : {diff:.4f}")
if diff < 0.005:
    print("  Result matches the paper target to within 0.005 LL units.")

if results.omega_hat is not None:
    print("\n  Random coefficient covariance (Omega):")
    print("  " + str(results.omega_hat).replace("\n", "\n  "))
    sigma_ovtt = float(np.sqrt(np.atleast_2d(results.omega_hat)[0, 0]))
    print(f"\n  Std dev of OVTT taste heterogeneity: {sigma_ovtt:.4f}")
    print("  Interpretation: OVTT coefficients across individuals are")
    print(f"  distributed as N(mu_OVTT, {sigma_ovtt:.4f}^2).")

if results.cholesky_L is not None:
    print("\n  Cholesky factor L  (Omega = L L'):")
    print("  " + str(results.cholesky_L).replace("\n", "\n  "))

print("""
  Reading the output:
    - Coefficients for CON_SR, CON_TR, AGE45_DA, AGE45_SR, IVTT, COST
      are fixed — they are the same for every individual.
    - The OVTT coefficient has a mean (mu_OVTT) and a variance (Omega).
    - A statistically meaningful Omega > 0 confirms genuine taste
      heterogeneity: different travellers weight out-of-vehicle time
      very differently.
    - AGE45 interactions capture whether travellers over 45 have
      systematically different alternative preferences.
""")

# ============================================================
#  Step 4: Comparison and Guidance
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Comparison and Guidance")
print("=" * 60)

print("""
  Random coefficients vs. fixed coefficients:

    Fixed (Model b):
      - All individuals share the same OVTT sensitivity
      - Faster estimation: no simulation integral
      - May underestimate welfare effects if heterogeneity is large

    Random (Model c, this tutorial):
      - OVTT sensitivity varies across individuals ~ N(mu, Omega)
      - Requires simulation-based integration at each likelihood call
      - More parameters: adds Cholesky elements of Omega
      - Better captures real-world preference diversity

  Computational considerations:
    - Each likelihood evaluation integrates over the random draws
      using QMC (quasi-Monte Carlo) Halton sequences
    - Number of draws is set by MNPControl(ndraws=...) — more draws
      give more accurate approximation but slower estimation
    - Adding more random variables increases the dimension of the
      integration and computation grows roughly exponentially
    - For a single random variable (OVTT), computation is manageable;
      for 3+ random variables, consider pruning or parallelisation

  Practical tips:
    - Always check Omega: if it is near zero, the random coefficient
      adds no explanatory power and you can revert to the fixed spec
    - The sign of mu_OVTT should still be negative (more time = less
      utility); a positive mean would be a red flag
    - Use seed= in MNPControl for reproducibility across runs

  Summary of Table 1 log-likelihoods (BHATLIB paper):
    Model (a)(i)  IID                   LL = -670.956
    Model (a)(ii) Flexible covariance   LL = -661.111
    Model (b)     + AGE45 demographics  LL = -659.285
    Model (c)     + Random OVTT (this)  LL = -635.871
    Model (d)     2-segment mixture     LL = -634.975

  The jump from Model (b) to (c) (-659.3 -> -635.9) is the largest
  single improvement in the sequence, driven entirely by allowing
  the OVTT coefficient to vary across individuals.
""")

print("  Next: t04e_mnp_mixture.py — Mixture-of-normals model")
