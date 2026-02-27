"""Tutorial T04c: MNP Heteroscedastic-Only Model.

Between IID (homoscedastic, no correlation) and full covariance lies
the heteronly model: different variances per alternative, but no
correlations. This tutorial compares all three.

What you will learn:
  - heteronly=True: different error variances, zero correlations
  - How to compare IID vs heteronly vs full covariance models
  - Model selection: when each specification is appropriate
  - Interpreting the covariance structure

Prerequisites: t00 (quickstart).
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# ============================================================
#  Step 1: IID Model (homoscedastic, no correlations)
# ============================================================
print("=" * 60)
print("  Step 1: IID Model (Homoscedastic)")
print("=" * 60)

print("\n  Error structure: Lambda = sigma^2 * I")
print("  Free covariance parameters: 0 (fixed at identity)")

t0 = time.perf_counter()
model_iid = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(iid=True, maxiter=100, verbose=0, seed=42),
)
res_iid = model_iid.fit()
t_iid = time.perf_counter() - t0

print(f"\n  Log-likelihood: {res_iid.ll_total:.3f}")
print(f"  Parameters: {len(res_iid.b)}")
print(f"  Time: {t_iid:.1f}s")

# ============================================================
#  Step 2: Heteronly Model (different variances, no correlations)
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Heteroscedastic-Only Model")
print("=" * 60)

print("\n  Error structure: Lambda = diag(sigma_1^2, sigma_2^2, ...)")
print("  Free covariance parameters: I-2 (one normalized, one for scale)")

t0 = time.perf_counter()
model_het = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(
        iid=False,
        heteronly=True,
        maxiter=100,
        verbose=1,
        seed=42,
    ),
)
res_het = model_het.fit()
t_het = time.perf_counter() - t0

print(f"\n  Log-likelihood: {res_het.ll_total:.3f}")
print(f"  Parameters: {len(res_het.b)}")
print(f"  Time: {t_het:.1f}s")

# ============================================================
#  Step 3: Full Covariance Model
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Full Covariance Model")
print("=" * 60)

print("\n  Error structure: Lambda = full PD matrix")
print("  Free covariance parameters: I*(I-1)/2 correlations + I-2 variances")

t0 = time.perf_counter()
model_full = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(
        iid=False,
        heteronly=False,
        maxiter=100,
        verbose=0,
        seed=42,
    ),
)
res_full = model_full.fit()
t_full = time.perf_counter() - t0

print(f"\n  Log-likelihood: {res_full.ll_total:.3f}")
print(f"  Parameters: {len(res_full.b)}")
print(f"  Time: {t_full:.1f}s")

# ============================================================
#  Step 4: Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Model Comparison")
print("=" * 60)

print(f"\n  {'Model':<15s} {'n_params':>10s} {'LL':>12s} {'Time(s)':>10s}")
print(f"  {'-'*49}")
print(f"  {'IID':<15s} {len(res_iid.b):>10d} {res_iid.ll_total:>12.3f} {t_iid:>10.1f}")
print(f"  {'Heteronly':<15s} {len(res_het.b):>10d} {res_het.ll_total:>12.3f} {t_het:>10.1f}")
print(f"  {'Full Cov':<15s} {len(res_full.b):>10d} {res_full.ll_total:>12.3f} {t_full:>10.1f}")

# ============================================================
#  Step 5: When to use each
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: When to Use Each Model")
print("=" * 60)

print("""
  IID:
  - Simplest, fastest, fewest parameters
  - Assumes all alternatives have equal error variance
  - Good starting point; use when covariance is not of interest

  Heteroscedastic-only:
  - Allows different error variances per alternative
  - Still assumes zero correlation between error terms
  - Good when alternatives differ in unobserved variation

  Full covariance:
  - Most flexible, most parameters
  - Captures both variance heterogeneity and error correlations
  - Needed when alternatives share unobserved factors
  - Slower due to more parameters and MVNCD computation

  Rule of thumb:
  - Start with IID for quick exploration
  - Try heteronly if IID fits poorly
  - Use full covariance for final specification
""")

print(f"  Next: t04f_mnp_control_options.py — All MNPControl configuration options")
