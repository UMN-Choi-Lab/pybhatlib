"""Tutorial T04c: MNP Heteroscedastic-Only Model.

Between IID (homoscedastic, no correlation) and full covariance lies
the heteronly model: different variances per alternative, but no
correlations. This tutorial compares all three.

What you will learn:
  - heteronly=True: different error variances, zero correlations
  - How to compare IID vs heteronly vs full covariance models
  - Model selection: when each specification is appropriate
  - Reporting the fitted coefficients (results.b_original)
    with std errors / t / p
  - The first-differenced-variance=1 scale identification (scale01 = 1.0)

GAUSS / paper cross-check (BHATLIB Table 1):
  - Model (a)(i)  IID      : LL = -670.956
  - Model (a)(ii) Flexible : LL = -661.111
The heteroscedastic-only kernel has no separately published Table-1 row;
its LL must lie between the IID and full-covariance bounds.

Prerequisites: t00 (quickstart).

Expected runtime: ~3 sec
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


def print_coefs(res):
    """Print the fitted coefficients with std errors / t / p.

    pybhatlib reports on the GAUSS first-differenced-variance=1 scale, so
    results.b_original (the readable reported view) and results.params share
    one scale and are equal for the mean coefficients. Report b_original: it
    spells out the readable kernel block (parker / scale rows) and is aligned
    with the standard errors / t / p.
    """
    print(f"\n  {'Parameter':<12s} {'Estimate':>10s} {'Std.Err':>10s}"
          f" {'t-stat':>9s} {'p-value':>9s}")
    print(f"  {'-' * 52}")
    for name, b, se, t, p in zip(
        res.param_names, res.b_original, res.se, res.t_stat, res.p_value
    ):
        print(f"  {name:<12s} {b:>10.4f} {se:>10.4f} {t:>9.3f} {p:>9.3f}")
    print(f"  Parameters reported: {len(res.b_original)}")


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

print(f"\n  GAUSS / paper reference LL : -670.956")
print(f"  PyBhatLib LL               : {res_iid.loglik * res_iid.n_obs:.3f}")
print(f"  Time: {t_iid:.1f}s")
print_coefs(res_iid)

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

ll_het = res_het.loglik * res_het.n_obs
print(f"\n  IID lower bound  LL : -670.956")
print(f"  Heteronly        LL : {ll_het:.3f}")
print(f"  Full-cov upper   LL : -661.111")
print(f"  In-bounds check     : {-670.956 <= ll_het <= -661.111}")
print(f"  Time: {t_het:.1f}s")
print_coefs(res_het)

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

print(f"\n  GAUSS / paper reference LL : -661.111")
print(f"  PyBhatLib LL               : {res_full.loglik * res_full.n_obs:.3f}")
print(f"  Time: {t_full:.1f}s")
print_coefs(res_full)

# ============================================================
#  Step 4: Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Model Comparison")
print("=" * 60)

print(f"\n  {'Model':<15s} {'n_params':>10s} {'LL':>12s} {'Time(s)':>10s}")
print(f"  {'-'*49}")
print(f"  {'IID':<15s} {len(res_iid.b_original):>10d} {res_iid.loglik * res_iid.n_obs:>12.3f} {t_iid:>10.1f}")
print(f"  {'Heteronly':<15s} {len(res_het.b_original):>10d} {res_het.loglik * res_het.n_obs:>12.3f} {t_het:>10.1f}")
print(f"  {'Full Cov':<15s} {len(res_full.b_original):>10d} {res_full.loglik * res_full.n_obs:>12.3f} {t_full:>10.1f}")

# Likelihood-ratio tests (nested models): heteronly vs IID, full vs heteronly.
from scipy.stats import chi2  # noqa: E402

ll_iid = res_iid.loglik * res_iid.n_obs
ll_het2 = res_het.loglik * res_het.n_obs
ll_full = res_full.loglik * res_full.n_obs

lr_het = 2.0 * (ll_het2 - ll_iid)
df_het = len(res_het.b_original) - len(res_iid.b_original)
lr_full = 2.0 * (ll_full - ll_het2)
df_full = len(res_full.b_original) - len(res_het.b_original)

print(f"\n  LR test heteronly vs IID : LR={lr_het:.3f}, df={df_het},"
      f" p={chi2.sf(lr_het, df_het):.4f}")
print(f"  LR test full vs heteronly: LR={lr_full:.3f}, df={df_full},"
      f" p={chi2.sf(lr_full, df_full):.4f}")

# ============================================================
#  Step 5: Scale identification (first differenced variance = 1)
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Heteroscedastic Scale Identification")
print("=" * 60)

print("""
The MNP likelihood depends only on UTILITY DIFFERENCES, so the overall
scale and one variance of the error vector are not identified. PyBhatLib
fixes this with the GAUSS first-differenced-variance=1 normalization:
the first differenced error variance is pinned to 1, i.e.

      lambda_hat[0, 0] = 1   <=>   scale01 = 1.0

scale01 is therefore a FIXED row in the reported coefficients (SE=NaN);
the remaining scale parameters (scale02, ...) are freely estimated on
that homogeneous kernel scale, with scale_i = sqrt(lambda_hat[i, i]).
This is the modern default and is the same identification PyBhatLib uses
for the flexible kernel, so the reported scales are directly comparable
across specifications.
""")

scale_names = [n for n in res_het.param_names if n.startswith("scale")]
scale_idx = [res_het.param_names.index(n) for n in scale_names]
scales = res_het.b_original[scale_idx]

print(f"  Reported scale parameters : {scale_names}")
print(f"  Scale estimates           : {scales}")
print(f"  scale01 (pinned to 1.0)   : {scales[0]:.6f}")
print(f"  lambda_hat[0,0] (==1)     : {res_het.lambda_hat[0, 0]:.6f}")
print(f"  Normalization holds       : {np.isclose(res_het.lambda_hat[0, 0], 1.0)}")

print("\n  Differenced kernel covariance (lambda_hat):")
print(res_het.lambda_hat)
print("""
  Note: lambda_hat is the (I-1)x(I-1) DIFFERENCED covariance. Its
  off-diagonal terms are non-zero even though the ORIGINAL utility errors
  are uncorrelated -- differencing against the base alternative induces
  correlation. This is the 'heteroscedasticity in the differenced
  utilities' that the GAUSS BHATLIB driver (_heteronly=1) documents.
""")

# ============================================================
#  Step 6: When to use each
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: When to Use Each Model")
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
