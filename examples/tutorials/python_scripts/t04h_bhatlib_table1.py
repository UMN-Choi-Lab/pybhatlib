"""Tutorial T04h: BHATLIB Manual Table 1 — All Five MNP Specifications.

This tutorial consolidates the five MNP model specifications from Table 1 of
the BHATLIB manual (Bhat 2018) and estimates them sequentially on the
TRAVELMODE dataset.  The five models form a progression of increasing
behavioral richness:

    (a)(i)   IID error structure — simplest baseline
    (a)(ii)  Flexible covariance — heteroscedastic, correlated errors
    (b)      +AGE45 demographics — alternative-specific demographics
    (c)      +Random OVTT       — random coefficient on out-of-vehicle time
    (d)      2-segment mixture   — discrete mixture of normals (nseg=2)

Each model adds complexity (more parameters) and should yield a higher
log-likelihood.  At the end, a comparison table summarises all five results
side-by-side, reproducing Table 1 from the BHATLIB paper.

What you will learn:
  - How to set up all five MNP specifications in pybhatlib
  - How model complexity grows from IID to mixture-of-normals
  - How to compare log-likelihoods and parameter counts across specifications
  - The practical tradeoff between model fit and estimation time

Prerequisites: t04a through t04e (individual model tutorials).

Target log-likelihoods (BHATLIB paper Table 1):
  (a)(i)   -670.956
  (a)(ii)  -661.111
  (b)      -659.285
  (c)      -635.871
  (d)      -634.975
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# ------------------------------------------------------------------
#  Specification dictionaries
# ------------------------------------------------------------------

# 5-variable spec used in Models (a)(i) and (a)(ii)
spec_5var = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno",      "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero",     "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# 7-variable spec used in Models (b), (c), and (d)
# Adds AGE45 interactions for drive-alone and shared-ride
spec_7var = {
    "CON_SR":   {"Alt1_ch": "sero",    "Alt2_ch": "uno",      "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",    "Alt2_ch": "sero",     "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45",   "Alt2_ch": "sero",     "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",    "Alt2_ch": "AGE45",    "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR",  "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR",  "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR",  "Alt3_ch": "COST_TR"},
}

# Accumulator for results across all models
all_results = []

# ============================================================
#  Step 1: Overview of Table 1
# ============================================================
print("=" * 60)
print("  Step 1: Overview of Table 1")
print("=" * 60)

print("""
  Table 1 in the BHATLIB manual (Bhat 2018) presents five MNP model
  specifications of increasing complexity on a mode choice dataset
  with three alternatives: Drive Alone (DA), Shared Ride (SR), and
  Transit (TR).

  The five specifications are:

    (a)(i)   IID
             5 beta parameters, 0 covariance parameters.
             Errors are iid N(0,1), so the differenced covariance
             Sigma_diff = ones(K,K) + I is fixed.

    (a)(ii)  Flexible covariance
             5 betas + free covariance parameters.
             Relaxes the IID assumption: error variances and
             correlations are estimated from the data.

    (b)      +AGE45 demographics
             7 betas (adds AGE45_DA, AGE45_SR) + free cov params.
             Tests whether age > 45 shifts alternative-specific
             preferences.

    (c)      +Random OVTT coefficient
             7 betas + 1 random-coefficient variance + free cov params.
             The OVTT coefficient varies across individuals as
             N(mu, sigma^2), capturing taste heterogeneity.

    (d)      2-segment mixture of normals
             2 x 7 betas + 1 segment prob + free cov params.
             Two latent population segments, each with its own
             coefficient vector. Segment membership is estimated
             via softmax probabilities.

  Each model nests the one above it, so the log-likelihood must
  weakly improve as we move down the list.  We will estimate all
  five models and confirm this progression.
""")


# ============================================================
#  Step 2: Model (a)(i) — IID
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Model (a)(i) — IID")
print("=" * 60)

print("""
  The IID model is the simplest MNP specification.  All error terms
  are independently and identically distributed as N(0,1), which
  implies the differenced covariance is fixed at Sigma_diff = I + 11'.
  Only the 5 mean utility parameters are estimated.
""")

t0 = time.perf_counter()
model_ai = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_5var,
    control=MNPControl(iid=True, maxiter=100, verbose=0, seed=42),
)
res_ai = model_ai.fit()
t_ai = time.perf_counter() - t0

print(f"  Parameters : {model_ai.n_params}")
print(f"  LL achieved: {res_ai.ll_total:.3f}")
print(f"  LL target  : -670.956")
print(f"  Time       : {t_ai:.1f}s")

all_results.append({
    "label": "(a)(i)  IID",
    "n_params": model_ai.n_params,
    "ll": res_ai.ll_total,
    "target_ll": -670.956,
    "time_s": t_ai,
})


# ============================================================
#  Step 3: Model (a)(ii) — Flexible Covariance
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Model (a)(ii) — Flexible Covariance")
print("=" * 60)

print("""
  The flexible covariance model estimates error standard deviations
  (omega) and correlations (Omega_star) in addition to the 5 betas.
  The covariance decomposition is Omega = omega * Omega_star * omega,
  parameterized via spherical coordinates for positive definiteness.

  With K=2 differenced alternatives, the free covariance parameters
  include scale and correlation terms.  Setting iid=False activates
  the full covariance structure.
""")

t0 = time.perf_counter()
model_aii = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_5var,
    control=MNPControl(iid=False, maxiter=100, verbose=0, seed=42),
)
res_aii = model_aii.fit()
t_aii = time.perf_counter() - t0

print(f"  Parameters : {model_aii.n_params}")
print(f"  LL achieved: {res_aii.ll_total:.3f}")
print(f"  LL target  : -661.111")
print(f"  Time       : {t_aii:.1f}s")

all_results.append({
    "label": "(a)(ii) Flexible",
    "n_params": model_aii.n_params,
    "ll": res_aii.ll_total,
    "target_ll": -661.111,
    "time_s": t_aii,
})


# ============================================================
#  Step 4: Model (b) — +AGE45 Demographics
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Model (b) — +AGE45 Demographics")
print("=" * 60)

print("""
  Model (b) extends the flexible covariance specification by adding
  two demographic interaction variables:

    AGE45_DA — effect of being over 45 on drive-alone utility
    AGE45_SR — effect of being over 45 on shared-ride utility

  These capture whether older travellers have systematically
  different mode preferences, conditional on the level-of-service
  variables (IVTT, OVTT, COST).  Transit is the reference category,
  so its AGE45 coefficient is implicitly zero.
""")

t0 = time.perf_counter()
model_b = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_7var,
    control=MNPControl(iid=False, maxiter=200, verbose=0, seed=42),
)
res_b = model_b.fit()
t_b = time.perf_counter() - t0

print(f"  Parameters : {model_b.n_params}")
print(f"  LL achieved: {res_b.ll_total:.3f}")
print(f"  LL target  : -659.285")
print(f"  Time       : {t_b:.1f}s")

all_results.append({
    "label": "(b)     +AGE45",
    "n_params": model_b.n_params,
    "ll": res_b.ll_total,
    "target_ll": -659.285,
    "time_s": t_b,
})


# ============================================================
#  Step 5: Model (c) — +Random OVTT
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Model (c) — +Random OVTT")
print("=" * 60)

print("""
  Model (c) builds on Model (b) by making the OVTT coefficient
  random across individuals:

      beta_OVTT_i ~ N(mu_OVTT, sigma_OVTT^2)

  This captures unobserved taste heterogeneity — some travellers
  are far more sensitive to out-of-vehicle travel time than others.
  The mean mu_OVTT and variance sigma_OVTT^2 are both estimated.

  In pybhatlib, this is activated by passing mix=True and
  ranvars=["OVTT"] to the MNPModel constructor.
""")

t0 = time.perf_counter()
model_c = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_7var,
    mix=True,
    ranvars=["OVTT"],
    control=MNPControl(iid=False, maxiter=200, verbose=0, seed=42),
)
res_c = model_c.fit()
t_c = time.perf_counter() - t0

print(f"  Parameters : {model_c.n_params}")
print(f"  LL achieved: {res_c.ll_total:.3f}")
print(f"  LL target  : -635.871")
print(f"  Time       : {t_c:.1f}s")

all_results.append({
    "label": "(c)     +Rand OVTT",
    "n_params": model_c.n_params,
    "ll": res_c.ll_total,
    "target_ll": -635.871,
    "time_s": t_c,
})


# ============================================================
#  Step 6: Model (d) — 2-Segment Mixture
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Model (d) — 2-Segment Mixture")
print("=" * 60)

print("""
  Model (d) is the most complex specification in Table 1.  Instead
  of a single coefficient vector, the population is divided into
  two latent segments, each with its own set of beta coefficients:

      Segment 1: beta_1   with probability pi_1
      Segment 2: beta_2   with probability pi_2 = 1 - pi_1

  Segment probabilities are parameterized via softmax (one free
  parameter for S=2).  The total parameter count roughly doubles
  the betas and adds the segment probability parameter.

  Note: Mixture models are computationally expensive with numerical
  gradients and have multiple local optima.  The achieved LL may
  differ slightly from the BHATLIB target, especially on synthetic
  data.
""")

t0 = time.perf_counter()
model_d = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec_7var,
    control=MNPControl(iid=False, nseg=2, maxiter=200, verbose=0, seed=42),
)
res_d = model_d.fit()
t_d = time.perf_counter() - t0

print(f"  Parameters : {model_d.n_params}")
print(f"  LL achieved: {res_d.ll_total:.3f}")
print(f"  LL target  : -634.975")
print(f"  Time       : {t_d:.1f}s")

all_results.append({
    "label": "(d)     2-seg mix",
    "n_params": model_d.n_params,
    "ll": res_d.ll_total,
    "target_ll": -634.975,
    "time_s": t_d,
})


# ============================================================
#  Step 7: Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("  Step 7: Comparison Table — BHATLIB Table 1")
print("=" * 60)

# Determine match status for each model
def match_status(achieved, target, tol_exact=0.005, tol_close=5.0):
    """Return a descriptive match string."""
    diff = abs(achieved - target)
    if diff < tol_exact:
        return "EXACT"
    elif diff < tol_close:
        return f"~{diff:.1f} LL"
    else:
        return f"DIFF ({diff:.1f})"

# Header
print()
hdr = (f"  {'Model':<22s} {'n_params':>8s} {'LL':>12s} "
       f"{'Target LL':>12s} {'Match?':>12s} {'Time(s)':>10s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))

total_time = 0.0
for r in all_results:
    status = match_status(r["ll"], r["target_ll"])
    row = (f"  {r['label']:<22s} {r['n_params']:>8d} {r['ll']:>12.3f} "
           f"{r['target_ll']:>12.3f} {status:>12s} {r['time_s']:>10.1f}")
    print(row)
    total_time += r["time_s"]

print("  " + "-" * (len(hdr) - 2))
print(f"  {'Total time':>68s} {total_time:>10.1f}")
print()


# ============================================================
#  Step 8: Interpretation
# ============================================================
print("\n" + "=" * 60)
print("  Step 8: Interpretation")
print("=" * 60)

print("""
  Log-likelihood progression:

  The five models show a monotonic improvement in log-likelihood as
  model complexity increases.  This is expected because each model
  nests the one above it — additional parameters can only improve
  (or maintain) the fit.

  Key transitions:

    (a)(i) -> (a)(ii):  IID -> Flexible covariance
      Adding free covariance parameters allows heteroscedastic
      error variances and cross-alternative correlations.  The LL
      improvement indicates that the IID assumption is restrictive
      for this dataset.

    (a)(ii) -> (b):     +AGE45 demographics
      Adding age interactions provides a modest improvement,
      suggesting that travellers over 45 have somewhat different
      mode preferences.

    (b) -> (c):         +Random OVTT coefficient
      This is typically the largest single LL improvement in the
      sequence.  Allowing the OVTT coefficient to vary across
      individuals captures substantial unobserved taste
      heterogeneity — different people weigh out-of-vehicle time
      very differently.

    (c) -> (d):         2-segment mixture
      The mixture model allows the entire coefficient vector to
      differ across two latent population segments.  The LL
      improvement over (c) is modest, but the parameter count
      roughly doubles.  Whether this is worthwhile depends on the
      research question: if segment-level policy analysis is
      needed, the mixture model provides it; otherwise, (c) may
      offer a better complexity-fit tradeoff.

  Model selection guidance:

    - For quick exploration or when covariance is not of interest,
      start with IID.
    - For final specifications, use flexible covariance at minimum.
    - Add demographics when theory suggests heterogeneous segments
      (e.g., age, income, trip purpose).
    - Random coefficients are strongly recommended when taste
      heterogeneity is expected — the LL gain is often dramatic.
    - Mixture models are powerful but expensive; use when you need
      discrete population segments for policy analysis and have
      sufficient data to identify them.

  Computational note:

    Estimation time grows substantially from IID (fastest) to
    mixture (slowest).  The IID model has a closed-form differenced
    covariance, requiring no MVNCD integration.  Models (a)(ii)
    through (c) require MVNCD evaluations per observation.  The
    mixture model (d) doubles the MVNCD calls (one per segment per
    observation) and uses numerical gradients, making it the most
    computationally demanding specification.
""")

print("  Tutorial complete.")
print("  See t04a-t04e for detailed explanations of each specification.")
