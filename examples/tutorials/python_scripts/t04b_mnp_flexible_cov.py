"""Tutorial T04b: MNP Flexible Covariance.

The flexible covariance MNP model relaxes the IID assumption by allowing
both heteroscedastic error variances and correlations across alternatives.
This tutorial walks through estimation, interpretation, and model comparison.

What you will learn:
  - Flexible covariance parameterization: Omega = omega * Omega_star * omega
  - Spherical coordinates for positive-definite correlation matrices
  - How to read the differenced covariance Lambda (lambda_hat)
  - Likelihood ratio test: IID vs flexible covariance
  - When flexible covariance is worth the extra parameters

Prerequisites: t04a (IID model).

Expected runtime: ~4 min
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

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

print(f"\n  Log-likelihood: {res_iid.ll_total:.3f}")
print(f"  Parameters:     {len(res_iid.b)}")
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

print(f"\n  Log-likelihood: {res_flex.ll_total:.3f}  (target: -661.111)")
print(f"  Parameters:     {len(res_flex.b)}")
print(f"  Time:           {t_flex:.1f}s")

# ============================================================
#  Step 3: Interpreting Lambda
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Interpreting Lambda")
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
#  Step 4: Model Comparison
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Model Comparison")
print("=" * 60)

n_iid  = len(res_iid.b)
n_flex = len(res_flex.b)
extra_params = n_flex - n_iid
lr_stat = -2.0 * (res_iid.ll_total - res_flex.ll_total)

print(f"\n  {'Model':<15s} {'n_params':>10s} {'LL':>12s} {'Time(s)':>10s}")
print(f"  {'-'*49}")
print(f"  {'IID':<15s} {n_iid:>10d} {res_iid.ll_total:>12.3f} {t_iid:>10.1f}")
print(f"  {'Flexible':<15s} {n_flex:>10d} {res_flex.ll_total:>12.3f} {t_flex:>10.1f}")

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
#  Step 5: When to Use Flexible Covariance
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: When to Use Flexible Covariance")
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
