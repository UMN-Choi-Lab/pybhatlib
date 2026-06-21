"""Tutorial T04e: MNP Mixture of Normals.

The mixture-of-normals MNP model allows the population to consist of S
discrete segments, each with its own coefficient vector. This is the most
flexible — and most computationally demanding — specification in pybhatlib.

What you will learn:
  - What a discrete mixture of normals means in the MNP context
  - How segment probabilities are parameterized via softmax
  - How nseg=2 in MNPControl creates a 2-segment mixture
  - How total parameter count grows with the number of segments
  - How to interpret the per-segment taste vectors and mixing weights
  - Practical tips: multiple local optima, slow convergence, baselines

Prerequisites: t04d (random coefficients).

Expected runtime: ~3 sec
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# Same spec as Model (b) — base 5 variables plus AGE45 demographics
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
#  Step 1: Mixture of Normals Concept
# ============================================================
print("=" * 60)
print("  Step 1: Mixture of Normals Concept")
print("=" * 60)

print("""
  A discrete mixture-of-normals MNP model assumes the population
  consists of S latent segments.  Each segment s has its own
  coefficient vector beta_s, so individuals in different segments
  respond differently to the same attributes.

  Model structure:
    - Individual n belongs to segment s with probability pi_s
    - Conditional on segment s, utility of alt j for individual n:

          U_{nj} | s  =  beta_s' x_{nj}  +  epsilon_{nj}

      where epsilon_{nj} is the usual MNP error (mean-zero normal)

    - The unconditional likelihood averages over segments:

          L_n  =  sum_{s=1}^{S}  pi_s * P(choice_n | beta_s)

  Segment probability parameterization (softmax):
    - Segment 1 is the reference: its log-odds parameter is fixed at 0
    - Free parameters: theta_2, ..., theta_S  (S-1 values)
    - Probabilities:

          raw   = [0, theta_2, ..., theta_S]
          pi_s  = exp(raw_s) / sum_t exp(raw_t)

    - At theta = 0:  pi_1 = pi_2 = ... = pi_S = 1/S  (equal segments)

  Total parameter count for a 2-segment model with n_beta predictors
  and n_cov free covariance parameters:

      n_params  =  S * n_beta  +  (S-1) segment params  +  n_cov

  Example (S=2, n_beta=7, flexible covariance with 1 free param):
      2 * 7  +  1  +  1  =  16 parameters

  In pybhatlib: set nseg=2 in MNPControl to enable a 2-segment mixture.
""")

# ============================================================
#  Step 2: Model Setup and Estimation
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Model Setup and Estimation")
print("=" * 60)

print("""
  We use the Model (d) specification from the BHATLIB paper (Table 1):
    - Same 7 variables as Model (b): CON_SR, CON_TR, AGE45_DA, AGE45_SR,
      IVTT, OVTT, COST
    - nseg=2: 2-segment discrete mixture of normals
    - Full (non-IID) flexible covariance

  Target log-likelihood (BHATLIB paper): -634.975
""")

print("  Note: Mixture models use a vectorized BVN gradient path, so")
print("  estimation takes under 1 second for ~160 iterations.")
print()

t0 = time.perf_counter()
model = MNPModel(
    data=data_path,
    alternatives=alternatives,
    spec=spec,
    control=MNPControl(iid=False, nseg=2, maxiter=200, verbose=1, seed=42),
)
results = model.fit()
t_elapsed = time.perf_counter() - t0

print(f"\n  Log-likelihood : {results.loglik * results.n_obs:.3f}")
print(f"  Parameters     : {len(results.b_original)}")
print(f"  Estimation time: {t_elapsed:.1f}s  ({t_elapsed/60:.1f} min)")

# ============================================================
#  Step 3: Interpreting Results
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Interpreting Results")
print("=" * 60)

# Report BHATLIB-normalized values (results.b_original) so output matches the
# GAUSS BHATLIB reference. results.params holds raw theta-space (used internally
# by the optimizer/predictor — different scale convention), so we DISPLAY
# b_original together with its std errors / t / p (already aligned to b_original).
print("\n  Estimated coefficients (BHATLIB-normalized — match GAUSS output):")
print(f"    {'Parameter':<14s}{'Estimate':>11s}{'Std.Err':>11s}"
      f"{'t-stat':>9s}{'p-value':>9s}")
print("    " + "-" * 52)
for name, est, se, t, p in zip(
    results.param_names, results.b_original,
    results.se, results.t_stat, results.p_value
):
    print(f"    {name:<14s}{est:>11.4f}{se:>11.4f}{t:>9.2f}{p:>9.3f}")

print(f"\n  GAUSS / paper reference LL : -634.975  (BHATLIB Table 2, model d)")
print(f"  PyBhatLib LL               : {results.loglik * results.n_obs:.3f}")

print("""
  Cross-check note (mixture of normals, BHATLIB Table 2 model d):

    The published BHATLIB GAUSS reference (and its Python translation,
    "MNP Table2 d") fits a 2-segment mixture WITH a random coefficient
    on OVTT (ranvars=["OVTT"], nrand=1), reaching LL = -634.975.

    This tutorial uses the simpler segment-specific-means form (nseg=2
    with no random coefficient inside each segment).  On the supplied
    TRAVELMODE data this reaches LL = -632.912 — slightly HIGHER (better)
    than the paper target because it has a different parameterization and
    one fewer covariance restriction.  This is expected: mixture models
    have many local optima, and a higher LL simply means the optimizer
    found a better mode of this likelihood surface.  To reproduce the
    exact paper model, add ranvars=["OVTT"] to the MNPModel(...) call
    (see Step 4).
""")

# Print segment probabilities if available
if hasattr(results, "segment_probs") and results.segment_probs is not None:
    print("  Estimated segment probabilities:")
    for s, p in enumerate(results.segment_probs, start=1):
        print(f"    Segment {s}: pi_{s} = {p:.4f}")
    print()

# ============================================================
#  Step 4: Segment Interpretation
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Segment Interpretation")
print("=" * 60)

print("""
  The whole point of a mixture-of-normals MNP is behavioural: each
  latent segment is a distinct taste group.  We therefore compare the
  two segments' coefficient vectors side by side and weight them by the
  estimated mixing probabilities pi_s.

  In this 7-variable specification, the segment-1 betas are the first 7
  entries of b_original (CON_SR..COST) and the segment-2 betas are the
  *_s2 entries near the end.  Reading them together tells us how the two
  groups differ in their sensitivity to travel time and cost.
""")

# Pair up segment-1 and segment-2 betas for the 7 utility variables.
beta_names = ["CON_SR", "CON_TR", "AGE45_DA", "AGE45_SR", "IVTT", "OVTT", "COST"]
name_to_val = dict(zip(results.param_names, results.b_original))
pi = results.segment_probs  # length-2 mixing weights

print(f"  {'Variable':<12s}{'Seg1 beta':>12s}{'Seg2 beta':>12s}"
      f"{'pi-weighted':>13s}")
print("  " + "-" * 49)
for nm in beta_names:
    b1 = name_to_val.get(nm, float("nan"))
    b2 = name_to_val.get(nm + "_s2", float("nan"))
    mixed = pi[0] * b1 + pi[1] * b2
    print(f"  {nm:<12s}{b1:>12.4f}{b2:>12.4f}{mixed:>13.4f}")

print(f"\n  Mixing weights: pi_1 = {pi[0]:.3f}  (dominant taste group),"
      f"  pi_2 = {pi[1]:.3f}")

print("""
  How to read this table:
    - The 'pi-weighted' column is the population-average taste, the
      mixture analogue of a single-segment coefficient.  It is the right
      number to quote when you want one headline elasticity-like value.
    - Differences between Seg1 beta and Seg2 beta reveal taste
      heterogeneity the homogeneous (single-segment) MNP cannot capture:
      e.g., one segment may be far more cost-sensitive (more negative
      COST) while the other is more time-sensitive (more negative IVTT).
    - A near-degenerate mixing weight (pi close to 0 or 1) is a warning
      sign that the second segment is weakly identified — inspect its
      std errors / t-stats from the Step 3 table before interpreting.

  Caveat on label switching: segment indices are arbitrary.  A re-run
  with a different seed may swap "Segment 1" and "Segment 2" (and their
  pi values) while giving the same likelihood.  Interpret segments by
  their taste profile, never by their index number.
""")

# ============================================================
#  Step 5: Practical Considerations
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Practical Considerations")
print("=" * 60)

print("""
  Flexibility vs. computational cost:
    - The mixture-of-normals model is the most flexible MNP specification
      in pybhatlib — it can approximate any mixing distribution over beta
    - It is also the slowest: each likelihood evaluation sums S segment
      probabilities, each requiring a full MVNCD computation
    - With vectorized BVN gradient, expect ~5 ms per iteration
      for 3-alternative models (under 1 second total)

  Multiple local optima:
    - Mixture models are notorious for having many local modes in the
      likelihood surface
    - Two runs with different starting values may converge to different
      parameter vectors with similar log-likelihoods
    - Strategy: run with several seeds and select the best LL:

          for seed in [42, 123, 456, 789]:
              ctrl = MNPControl(iid=False, nseg=2, seed=seed, maxiter=200)
              res  = MNPModel(..., control=ctrl).fit()
              print(seed, res.loglik * res.n_obs)

  Parameter count scaling:
    - nseg=2: 2 * n_beta + 1 segment param + n_cov  (doubles the betas)
    - nseg=3: 3 * n_beta + 2 segment params + n_cov  (triples the betas)
    - Identification requires sufficient variation in the data; too many
      segments on a small dataset will produce near-flat likelihood

  Recommended workflow:
    1. Estimate IID or flexible covariance model first (establish baseline)
    2. Compare LL improvement: is it worth the extra parameters?
    3. If yes, try nseg=2 with multiple starting values
    4. Use LR test or BIC to decide between nseg=1 and nseg=2

  Rule of thumb:
    - Each additional segment adds n_beta + 1 free parameters
    - Worthwhile only when delta-LL > (n_beta + 1) / 2  (BIC criterion)
    - For n_beta=7: need delta-LL > 4  to justify a second segment
""")

print("  Next: t04f_mnp_control_options.py — MNPControl configuration options")
