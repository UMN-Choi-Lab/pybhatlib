"""Tutorial T04b: MNP Flexible Covariance.

The flexible covariance MNP model relaxes the IID assumption by allowing
both heteroscedastic error variances and correlations across alternatives.
This tutorial walks through estimation, interpretation, and model comparison.

What you will learn:
  - Flexible covariance parameterization: Omega = omega * Omega_star * omega
  - Spherical coordinates for positive-definite correlation matrices
  - Reading the reported coefficient table (b_original, se, t, p)
  - The UNPARAMETERIZED kernel correlation (no 2*atanh transform)
  - How to read the differenced covariance Lambda (lambda_hat)
  - Likelihood ratio test: IID vs flexible covariance
  - Counterfactual / ATE share analysis via the scenarios= API
  - When flexible covariance is worth the extra parameters

Prerequisites: t04a (IID model).

Expected runtime: ~3 sec (includes JIT compilation on first run)
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl, mnp_ate

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno",     "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero",    "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# ============================================================
#  Step 1: IID Baseline
# ============================================================
print("=" * 60)
print("  Step 1: IID Baseline")
print("=" * 60)

print("\n  Estimating IID model for comparison.")
print("  Error structure: Lambda = ones(K,K) + I  (differenced identity)")
print("  Free covariance parameters: 0 (fixed)")

t0 = time.perf_counter()
model_iid = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(iid=True, maxiter=100, verbose=0, seed=42),
)
res_iid = model_iid.fit()
t_iid = time.perf_counter() - t0

print(f"\n  Log-likelihood: {res_iid.loglik * res_iid.n_obs:.3f}")
print(f"  Parameters:     {len(res_iid.params)}")
print(f"  Time:           {t_iid:.1f}s")

# ============================================================
#  Step 2: Flexible Covariance
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Flexible Covariance")
print("=" * 60)

print("""
  The full covariance structure decomposes as:

      Omega = omega * Omega_star * omega

  where:
    omega      = diagonal matrix of error standard deviations
                 (captures heteroscedasticity across alternatives)
    Omega_star = correlation matrix, parameterized via spherical
                 coordinates to guarantee positive definiteness

  After differencing w.r.t. the chosen alternative, the model
  works with Lambda = differenced covariance matrix.

  Estimating flexible covariance model (iid=False, heteronly=False)...
""")

t0 = time.perf_counter()
model_flex = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(
        iid=False,
        heteronly=False,
        maxiter=100,
        verbose=1,
        seed=42,
    ),
)
res_flex = model_flex.fit()
t_flex = time.perf_counter() - t0

print(f"\n  PyBhatLib LL                : {res_flex.loglik * res_flex.n_obs:.3f}")
print(f"  GAUSS / paper reference LL   : -661.111")
print(f"  Reporting parameters         : {len(res_flex.b_original)}")
print(f"  Time                         : {t_flex:.1f}s")

# ============================================================
#  Step 3: Coefficient Table & Kernel Correlation
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Coefficient Table & Kernel Correlation")
print("=" * 60)

print("""
  Report the fitted model using results.b_original, the readable
  reported view of the estimates.  pybhatlib reports on the GAUSS
  first-differenced-variance=1 scale: the kernel is identified by
  pinning the first differenced error variance to 1, so the reported
  scale01 is fixed at 1.0 (a pinned row, SE=NaN).  results.b_original
  and results.params share one scale and are equal for the mean
  coefficients; b_original additionally spells out the readable kernel
  block, is aligned with param_names, and carries the standard errors,
  t-stats, and p-values from the delta method.

  The covariance block of b_original contains:
    parker01  = the UNPARAMETERIZED kernel correlation.  This is
                reported directly (no 2*atanh re-parameterization),
                so it reads as an ordinary correlation in [-1, 1].
    scale01,  = error standard-deviation scales.  scale01 = 1.0 is
    scale02     the pinned first differenced variance (SE=NaN);
                scale02 = sqrt(Lambda[1,1]) is freely estimated.
""")

print(f"  {'Parameter':<12s} {'Estimate':>10s} {'Std.err':>10s} "
      f"{'t-stat':>9s} {'p-value':>9s}")
print("  " + "-" * 52)
for name, est, se, t, p in zip(
    res_flex.param_names, res_flex.b_original,
    res_flex.se, res_flex.t_stat, res_flex.p_value,
):
    print(f"  {name:<12s} {est:>10.4f} {se:>10.4f} {t:>9.3f} {p:>9.4f}")

# The unparameterized kernel correlation is recoverable two ways:
#   (1) directly from b_original ('parker01'), and
#   (2) from the differenced kernel covariance lambda_hat via cov->corr.
# They must agree (this is the "no 2*atanh transform" check).
if "parker01" in res_flex.param_names:
    parker = res_flex.b_original[res_flex.param_names.index("parker01")]
else:
    parker = float("nan")

if res_flex.lambda_hat is not None:
    L = res_flex.lambda_hat
    d = np.sqrt(np.diag(L))
    kernel_corr = L / np.outer(d, d)
    print("\n  Kernel correlation cross-check (unparameterized):")
    print(f"    parker01 (from b_original)        : {parker:.4f}")
    print(f"    corr from lambda_hat (cov->corr)  : {kernel_corr[0, 1]:.4f}")
    print(f"    GAUSS / paper reference           :  0.4731")

# ============================================================
#  Step 4: Interpreting Lambda
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Interpreting Lambda")
print("=" * 60)

print("\n  lambda_hat = estimated differenced covariance matrix:")
if hasattr(res_flex, "lambda_hat") and res_flex.lambda_hat is not None:
    print(f"\n  {res_flex.lambda_hat}\n")
else:
    print("\n  (lambda_hat not available on this result object)\n")

print("""  How to read Lambda:

    Diagonals Lambda[i,i]:
      Variance of the i-th differenced error term.
      Larger values indicate more unobserved variation in that
      error difference, relative to the chosen alternative.

    Off-diagonals Lambda[i,j]:
      Covariance between differenced error pairs.
      Positive values: alternatives i+1 and j+1 share common
      unobserved factors (their errors move together).
      Negative values: errors move in opposite directions.

  IID reference:
    Under IID, Lambda = ones(K,K) + I  (for K=2 differenced alts):
      [[2, 1],
       [1, 2]]
    Every diagonal = 2, every off-diagonal = 1.
    Deviations from this pattern signal heteroscedasticity
    or cross-alternative correlations.
""")

# ============================================================
#  Step 5: Model Comparison
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Model Comparison")
print("=" * 60)

n_iid  = len(res_iid.params)
n_flex = len(res_flex.params)
extra_params = n_flex - n_iid
lr_stat = -2.0 * (res_iid.loglik * res_iid.n_obs - res_flex.loglik * res_flex.n_obs)

print(f"\n  {'Model':<15s} {'n_params':>10s} {'LL':>12s} {'Time(s)':>10s}")
print(f"  {'-'*49}")
print(f"  {'IID':<15s} {n_iid:>10d} {res_iid.loglik * res_iid.n_obs:>12.3f} {t_iid:>10.1f}")
print(f"  {'Flexible':<15s} {n_flex:>10d} {res_flex.loglik * res_flex.n_obs:>12.3f} {t_flex:>10.1f}")

print(f"""
  Likelihood Ratio Test:
    H0: covariance parameters jointly zero (IID is true)
    LR = -2 * (LL_iid - LL_flex) = {lr_stat:.3f}
    Extra parameters (df): {extra_params}
    chi-squared critical value at 0.05, df={extra_params}: see scipy.stats.chi2

  Interpretation:
    A large LR statistic (above the chi-squared critical value)
    rejects the IID null, indicating that the flexible covariance
    significantly improves fit.
""")

# ============================================================
#  Step 6: Counterfactual Share Analysis (ATE scenarios)
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Counterfactual Share Analysis (ATE scenarios)")
print("=" * 60)

print("""
  With the fitted flexible-covariance model we can run a policy
  counterfactual using the scenarios= API.  Each scenario is a
  dict of {column: override_value}; mnp_ate re-predicts choice
  probabilities under each scenario and aggregates to market
  shares.  Comparing scenarios gives the average treatment
  effect (ATE) on predicted shares.

  Here we raise the COST of every alternative by 1 unit (a uniform
  price increase) and contrast it with the base case.  Because the
  flexible covariance lets the motorized alternatives share
  unobserved factors, the substitution pattern differs from what
  an IID model would predict.

  Note: use scenarios= (NOT the legacy changevar=/changeval= path,
  whose .predicted_shares returns the unmodified prediction and can
  silently report a 0% ATE).
""")

ate = mnp_ate(
    res_flex,
    data=model_flex.data,
    spec=spec,
    alternatives=alternatives,
    scenarios={
        "base":       {},
        "cost_plus1": {"COST_DA": 1.0, "COST_SR": 1.0, "COST_TR": 1.0},
    },
)

base_sh = ate.shares_per_scenario["base"]
trt_sh = ate.shares_per_scenario["cost_plus1"]
pct = ate.comparison("base", "cost_plus1")

alt_labels = ["Drive-alone", "Shared-ride", "Transit"]
print(f"  {'Alternative':<14s} {'Base share':>11s} "
      f"{'+1 cost':>11s} {'ATE (%)':>10s}")
print("  " + "-" * 48)
for lbl, b, t, p in zip(alt_labels, base_sh, trt_sh, pct):
    print(f"  {lbl:<14s} {b:>11.4f} {t:>11.4f} {p:>10.2f}")

print("""
  Reading the result:
    A uniform cost increase shifts probability mass toward the
    cheapest / least cost-sensitive alternative.  The percentage
    ATE column quantifies the relative share change per alternative.
""")

# ============================================================
#  Step 7: When to Use Flexible Covariance
# ============================================================
print("\n" + "=" * 60)
print("  Step 7: When to Use Flexible Covariance")
print("=" * 60)

print("""
  Use flexible covariance when:
    - Alternatives share unobserved factors (e.g., motorized modes
      share traffic-related unobservables)
    - Error variances plausibly differ across alternatives
    - Sample size is large enough to identify extra parameters
    - Theory or prior research suggests non-IID errors

  Be cautious when:
    - Sample is small — more covariance parameters reduce degrees
      of freedom and can cause near-singular Hessians
    - Estimation time is a constraint — MVNCD evaluation is O(K^2)
      and each extra correlation adds an OVUS/TVBS integration
    - All alternatives are structurally similar — IID may suffice

  Parameter count guide (I = number of alternatives):
    IID:       0 covariance params
    Heteronly: I - 2 variance params  (one alt normalized)
    Flexible:  I*(I-1)/2 correlations + (I-2) variances

  Next: t04c_mnp_heteronly.py — Heteroscedastic-only model
""")
