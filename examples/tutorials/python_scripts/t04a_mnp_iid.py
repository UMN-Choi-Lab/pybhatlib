"""Tutorial T04a: MNP IID Model.

The IID (independently and identically distributed) error structure is the
simplest MNP specification. This tutorial introduces the IID model, explains
the differenced covariance structure it implies, and shows how to estimate and
interpret results.

What you will learn:
  - What the IID assumption means in the MNP context
  - How the differenced covariance matrix Sigma_diff is constructed
  - How to set up and estimate an IID MNP model with MNPControl(iid=True)
  - How to read the estimated coefficients from results.b_original
    (the readable reported view, with the kernel scales spelled out)
  - How to compute an Average Treatment Effect (ATE) on predicted mode
    shares with the scenarios= API of mnp_ate
  - When to use (and not use) the IID specification

Prerequisites: t00 (quickstart).

Expected runtime: ~2 sec (includes JIT compilation on first run)
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl, mnp_ate

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno",      "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero",     "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# ============================================================
#  Step 1: IID Error Structure
# ============================================================
print("=" * 60)
print("  Step 1: IID Error Structure")
print("=" * 60)

print("""
  In the MNP model each alternative j has an unobserved error
  epsilon_j.  Under the IID assumption:

      epsilon_j ~ N(0, 1)  independently for all j

  When we difference out the chosen alternative (say alt 0) to
  form the utility differences used in the likelihood, the
  differenced errors are:

      d_j = epsilon_j - epsilon_0,   j = 1, ..., K

  where K = n_alternatives - 1.

  The covariance of these differences is:

      Sigma_diff[j, k] = Cov(d_j, d_k)
                       = Cov(eps_j - eps_0, eps_k - eps_0)

  Because the errors are independent:
      Cov(eps_j, eps_k) = 0  for j != k
      Var(eps_j)        = 1  for all j

  Therefore:
      diagonal:     Sigma_diff[j,j] = Var(eps_j) + Var(eps_0) = 1 + 1 = 2
      off-diagonal: Sigma_diff[j,k] = Var(eps_0)              = 1

  For K = 2 (three alternatives, two differences):

      Sigma_diff = [[2, 1],
                    [1, 2]]
""")

K = 2
Sigma_diff = np.ones((K, K)) + np.eye(K)
print("  Sigma_diff (K=2):")
print("  " + str(Sigma_diff).replace("\n", "\n  "))
print()
print("  Note: ones(K,K) + I(K,K) = off-diagonals 1, diagonals 2.")
print("  This matrix is FIXED — IID has zero free covariance parameters.")

# ============================================================
#  Step 2: Model Setup and Estimation
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Model Setup and Estimation")
print("=" * 60)

print("""
  We use the 5-variable TRAVELMODE specification from the BHATLIB
  paper (Table 1, Model a-i):

    CON_SR  — alternative-specific constant for SR (shared ride)
    CON_TR  — alternative-specific constant for TR (transit)
    IVTT    — in-vehicle travel time (generic across alternatives)
    OVTT    — out-of-vehicle travel time (generic)
    COST    — monetary cost (generic)

  Setting MNPControl(iid=True) fixes Sigma_diff = ones+I and
  estimates only the 5 mean utility parameters (beta vector).
""")

t0 = time.perf_counter()
model = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec,
    control=MNPControl(iid=True, maxiter=100, verbose=1, seed=42),
)
results = model.fit()
t_elapsed = time.perf_counter() - t0

print(f"\n  Log-likelihood: {results.loglik * results.n_obs:.3f}")
print(f"  Parameters: {len(results.b_original)}")
print(f"  Estimation time: {t_elapsed:.1f}s")

# ============================================================
#  Step 3: Interpreting Results
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Interpreting Results")
print("=" * 60)

# pybhatlib reports on the GAUSS first-differenced-variance=1 scale, so
# results.b_original (the readable reported view) and results.params share
# one scale; for the mean coefficients they are equal. Use b_original for
# reporting: it carries the readable kernel block and is aligned with the
# standard errors / t / p.
print("\n  Estimated coefficients (match GAUSS output):")
for name, val, se, t, p in zip(
    results.param_names, results.b_original, results.se, results.t_stat, results.p_value
):
    print(f"    {name:<10s}  {val:>10.4f}   s.e.={se:>8.4f}   t={t:>8.3f}   p={p:>6.4f}")

print(f"\n  Target log-likelihood : -670.956  (BHATLIB paper Table 1)")
print(f"  Achieved log-likelihood: {results.loglik * results.n_obs:.3f}")

diff = abs(results.loglik * results.n_obs - (-670.956))
print(f"  Absolute difference   : {diff:.4f}")
if diff < 0.005:
    print("  Result matches the paper target to within 0.005 LL units.")

print("""
  Interpretation of signs:
    CON_SR, CON_TR < 0  =>  DA (drive alone) is the preferred base
    IVTT, OVTT, COST < 0 =>  higher time/cost reduces utility (expected)
""")

# ============================================================
#  Step 4: Average Treatment Effect (ATE) via scenarios=
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Average Treatment Effect (ATE) via scenarios=")
print("=" * 60)

print("""
  A common policy question is: how would predicted mode shares
  change if a covariate were set to a counterfactual value?  The
  scenarios= API of mnp_ate answers this WITHOUT re-estimation:

    1. rebuild the design matrix for each scenario's overrides
    2. recompute every observation's choice probabilities
    3. average to obtain predicted shares per scenario

  To demonstrate, we add the binary AGE45 indicator (1 if the
  traveler is 45+) to the utilities of the DA and SR alternatives,
  with transit (TR) as the base — the same AGE45 arrangement used
  in BHATLIB Table 1 Model (b).  We keep the IID kernel here (the
  published Model (b) LL of -659.285 uses the FLEXIBLE covariance;
  the IID+AGE45 variant below is its IID counterpart).  We then
  compare the AGE45=0 ("younger") and AGE45=1 ("older") scenarios.

  NOTE: we use scenarios=, not the legacy changevar=/changeval=
  path.  The scenario API returns shares_per_scenario for each
  named scenario, so the ATE is the difference of two well-defined
  counterfactual predictions (the legacy .predicted_shares is the
  UNCONDITIONAL prediction and would yield a spurious 0% ATE).
""")

# Table 1 Model (b): IID + AGE45 on DA and SR (TR base), matching the
# GAUSS reference spec (AGE45_DA, AGE45_SR; TR omitted as base).
spec_age = dict(spec)
spec_age["AGE45_DA"] = {"Alt1_ch": "AGE45", "Alt2_ch": "sero",  "Alt3_ch": "sero"}
spec_age["AGE45_SR"] = {"Alt1_ch": "sero",  "Alt2_ch": "AGE45", "Alt3_ch": "sero"}

model_age = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_age,
    control=MNPControl(iid=True, maxiter=100, verbose=0, seed=42),
)
results_age = model_age.fit()
ll_age = results_age.loglik * results_age.n_obs
ll_base = results.loglik * results.n_obs
print(f"  IID baseline LL (no AGE45, Table 1 a-i) : {ll_base:.3f}")
print(f"  IID + AGE45 LL                          : {ll_age:.3f}")
print(f"  LR gain from 2 AGE45 betas              : {ll_age - ll_base:+.3f}")
print(f"  (Reference: FLEXIBLE-cov + AGE45 is Table 1 Model b, LL = -659.285)")
for name, val in zip(results_age.param_names, results_age.b_original):
    if name.startswith("AGE45"):
        print(f"    estimated {name:<10s} = {val:>8.4f}")

# Counterfactual shares: everyone "younger" (AGE45=0) vs "older" (AGE45=1).
import pandas as pd  # noqa: E402 — local import keeps Step 1-3 dependency-free

ate = mnp_ate(
    results_age,
    data=pd.read_csv(data_path),
    spec=spec_age,
    alternatives=alternatives,
    scenarios={"younger": {"AGE45": 0}, "older": {"AGE45": 1}},
)

# shares_per_scenario / comparison are already averaged over observations
# (shape = (n_alternatives,)).
alt_labels = ["DA", "SR", "TR"]
shares_young = np.asarray(ate.shares_per_scenario["younger"])
shares_old = np.asarray(ate.shares_per_scenario["older"])
pct_change = np.asarray(ate.comparison("younger", "older"))

print("\n  Predicted aggregate mode shares under each AGE45 scenario:")
print(f"    {'Alt':<6s} {'AGE45=0':>10s} {'AGE45=1':>10s} {'%change':>10s}")
for lab, s0, s1, pc in zip(alt_labels, shares_young, shares_old, pct_change):
    print(f"    {lab:<6s} {s0:>10.4f} {s1:>10.4f} {pc:>9.2f}%")

print("""
  Interpretation: a positive AGE45 coefficient on an alternative
  raises that alternative's predicted share when AGE45 flips 0->1.
  The %change column is the share-weighted ATE on each mode.  Shares
  in each column sum to 1 by construction.
""")

# ============================================================
#  Step 5: IID Limitations
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: IID Limitations and When to Use")
print("=" * 60)

print("""
  Strengths of the IID specification:
    - Fastest estimation: zero free covariance parameters
    - Fewest parameters: lower risk of overfit on small datasets
    - Good starting point for exploring the mean utility structure
    - Closed-form differenced covariance — no MVNCD tuning needed

  Limitations:
    - Assumes ALL alternatives have equal unobserved variance (sigma^2 = 1)
    - Assumes ZERO correlation between error terms across alternatives
    - Imposes IIA (Independence of Irrelevant Alternatives) implicitly
    - Can misfit data where alternatives share unobserved factors
      (e.g., SR and TR both involve waiting, unlike DA)

  When IID is appropriate:
    - Exploratory analysis or quick sanity checks
    - Very small samples where covariance parameters cannot be identified
    - When theoretical considerations support equal unobserved variance
    - As a baseline model for likelihood-ratio tests against richer specs

  Rule of thumb:
    - Always estimate IID first, then compare to flexible covariance
    - A large LL improvement (>3 per added cov param) justifies complexity
""")

print("  Next: t04b_mnp_flexible_cov.py — Flexible covariance structure")
