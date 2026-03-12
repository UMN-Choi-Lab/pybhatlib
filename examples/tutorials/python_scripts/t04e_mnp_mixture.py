"""Tutorial T04e: MNP Mixture of Normals.

The mixture-of-normals MNP model allows the population to consist of S
discrete segments, each with its own coefficient vector. This is the most
flexible — and most computationally demanding — specification in pybhatlib.

What you will learn:
  - What a discrete mixture of normals means in the MNP context
  - How segment probabilities are parameterized via softmax
  - How nseg=2 in MNPControl creates a 2-segment mixture
  - How total parameter count grows with the number of segments
  - Practical tips: multiple local optima, slow convergence, baselines

Prerequisites: t04d (random coefficients).
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

print("  Note: Mixture models are computationally expensive. With numerical")
print("  gradients, this may take ~26 minutes for 163 iterations.")
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

print(f"\n  Log-likelihood : {results.ll_total:.3f}")
print(f"  Parameters     : {len(results.b)}")
print(f"  Estimation time: {t_elapsed:.1f}s  ({t_elapsed/60:.1f} min)")

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
    # Fallback: label by segment + variable name
    n_beta = len(spec)
    seg_labels = []
    for s in range(1, 3):
        for vname in spec.keys():
            seg_labels.append(f"seg{s}_{vname}")
    seg_labels.append("theta_seg2")   # 1 segment probability param
    # remaining are covariance params
    for i, val in enumerate(results.b):
        label = seg_labels[i] if i < len(seg_labels) else f"cov_param_{i - len(seg_labels)}"
        print(f"    {label:<25s}  {val:>10.4f}")

print(f"\n  Target log-likelihood  : -634.975  (BHATLIB paper Table 1)")
print(f"  Achieved log-likelihood: {results.ll_total:.3f}")

print("""
  Note: With synthetic TRAVELMODE data, the optimizer may find a
  different local optimum (e.g., LL = -632.912) than the original
  BHATLIB result.  This is expected — mixture models have multiple
  local optima, and the synthetic data does not perfectly replicate
  the original dataset.  A higher LL than the target simply means
  the optimizer found a better local mode on this dataset.
""")

# Print segment probabilities if available
if hasattr(results, "segment_probs") and results.segment_probs is not None:
    print("  Estimated segment probabilities:")
    for s, p in enumerate(results.segment_probs, start=1):
        print(f"    Segment {s}: pi_{s} = {p:.4f}")
    print()

# ============================================================
#  Step 4: Practical Considerations
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Practical Considerations")
print("=" * 60)

print("""
  Flexibility vs. computational cost:
    - The mixture-of-normals model is the most flexible MNP specification
      in pybhatlib — it can approximate any mixing distribution over beta
    - It is also the slowest: each likelihood evaluation sums S segment
      probabilities, each requiring a full MVNCD computation
    - With numerical gradients (current default), expect 5-30 seconds
      per iteration depending on sample size and number of alternatives

  Multiple local optima:
    - Mixture models are notorious for having many local modes in the
      likelihood surface
    - Two runs with different starting values may converge to different
      parameter vectors with similar log-likelihoods
    - Strategy: run with several seeds and select the best LL:

          for seed in [42, 123, 456, 789]:
              ctrl = MNPControl(iid=False, nseg=2, seed=seed, maxiter=200)
              res  = MNPModel(..., control=ctrl).fit()
              print(seed, res.ll_total)

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
