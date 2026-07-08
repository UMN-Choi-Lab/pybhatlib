"""Tutorial T04g: MNP Forecasting and Scenario Analysis.

After estimating an MNP model, we can predict choice probabilities for
new observations and conduct scenario analysis by varying key variables.

What you will learn:
  - mnp_predict: predicted choice probabilities
  - mnp_predict_choice: most likely alternative
  - Scenario analysis: how cost changes affect mode shares
  - mnp_ate with the scenarios= API: counterfactual treatment effects
    (cross-checked against the GAUSS MNP_TRAVELMODE_ATE driver)

Prerequisites: t00 (quickstart).

Expected runtime: ~10 sec
"""
import os, sys
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import (
    MNPControl,
    MNPModel,
    mnp_ate,
    mnp_predict,
    mnp_predict_choice,
)
from pybhatlib.io import parse_spec

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
#  Step 1: Fit IID model
# ============================================================
print("=" * 60)
print("  Step 1: Fit IID MNP Model")
print("=" * 60)

model = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(iid=True, maxiter=100, verbose=1, seed=42),
)
results = model.fit()
print(f"\n  GAUSS / paper reference LL : -670.956")
print(f"  PyBhatLib LL               : {results.loglik * results.n_obs:.3f}")

# results.b_original is the readable reported view (matches GAUSS and the
# published tables). pybhatlib reports on the GAUSS first-differenced-
# variance=1 scale, so b_original and results.params share one scale and are
# equal for the mean coefficients; report b_original because it additionally
# spells out the readable kernel block (parker / scale rows).
print(f"\n  {'Parameter':<12s} {'Estimate':>10s} {'Std.Err':>10s} "
      f"{'t-stat':>8s} {'p-value':>8s}")
print(f"  {'-' * 50}")
for name, b, se, t, p in zip(results.param_names, results.b_original,
                             results.se, results.t_stat, results.p_value):
    print(f"  {name:<12s} {b:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.4f}")
print(f"\n  Number of reported parameters: {len(results.b_original)}")

# ============================================================
#  Step 2: Build X_new manually
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Build Design Matrix for New Observations")
print("=" * 60)

# 5 hypothetical observations
# X_new shape: (N, n_alts=3, n_vars=5)
# Variables: CON_SR, CON_TR, IVTT, OVTT, COST
N_new = 5
n_alts = 3
n_vars = 5

X_new = np.zeros((N_new, n_alts, n_vars))

# Constants: CON_SR=1 for Alt2, CON_TR=1 for Alt3
for q in range(N_new):
    X_new[q, 1, 0] = 1.0  # CON_SR for Alt2
    X_new[q, 2, 1] = 1.0  # CON_TR for Alt3

# IVTT (in-vehicle travel time in minutes)
ivtt_values = [
    [20, 25, 35],  # Obs 1: DA fast
    [30, 30, 30],  # Obs 2: equal IVTT
    [40, 35, 25],  # Obs 3: transit fast
    [15, 20, 40],  # Obs 4: DA very fast
    [35, 35, 20],  # Obs 5: transit very fast
]

# OVTT (out-of-vehicle travel time)
ovtt_values = [
    [5, 10, 15],
    [5, 10, 15],
    [5, 10, 15],
    [5, 10, 15],
    [5, 10, 15],
]

# COST (in dollars)
cost_values = [
    [10, 5, 3],
    [10, 5, 3],
    [10, 5, 3],
    [10, 5, 3],
    [10, 5, 3],
]

for q in range(N_new):
    for a in range(n_alts):
        X_new[q, a, 2] = ivtt_values[q][a]  # IVTT
        X_new[q, a, 3] = ovtt_values[q][a]  # OVTT
        X_new[q, a, 4] = cost_values[q][a]  # COST

print(f"\n  X_new shape: {X_new.shape}")
print(f"  (5 observations, 3 alternatives, 5 variables)")

# ============================================================
#  Step 3: mnp_predict — Probability table
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Predicted Choice Probabilities")
print("=" * 60)

probs = mnp_predict(results, X_new)

print(f"\n  {'Obs':>5s} {'P(DA)':>10s} {'P(SR)':>10s} {'P(TR)':>10s} {'Scenario':<20s}")
print(f"  {'-'*55}")
scenarios = ["DA fast", "Equal IVTT", "TR fast", "DA very fast", "TR very fast"]
for q in range(N_new):
    print(f"  {q+1:>5d} {probs[q,0]:>10.4f} {probs[q,1]:>10.4f} {probs[q,2]:>10.4f} {scenarios[q]:<20s}")

# ============================================================
#  Step 4: mnp_predict_choice — Best alternative
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Predicted Best Alternative")
print("=" * 60)

choices = mnp_predict_choice(results, X_new)
alt_names = ["DA", "SR", "TR"]

print(f"\n  {'Obs':>5s} {'Chosen':>10s} {'Probability':>12s}")
print(f"  {'-'*30}")
for q in range(N_new):
    print(f"  {q+1:>5d} {alt_names[choices[q]]:>10s} {probs[q, choices[q]]:>12.4f}")

# ============================================================
#  Step 5: Scenario analysis — Vary COST
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Scenario Analysis — Varying DA Cost")
print("=" * 60)

print(f"\n  Holding all else fixed, what happens as DA cost changes?")
print(f"\n  {'DA_COST':>10s} {'P(DA)':>10s} {'P(SR)':>10s} {'P(TR)':>10s}")
print(f"  {'-'*42}")

# Use observation 1 as baseline
X_scenario = X_new[0:1].copy()  # shape (1, 3, 5)
cost_range = [2, 5, 10, 15, 20, 30]

for cost_da in cost_range:
    X_scenario[0, 0, 4] = cost_da  # Change DA cost
    p = mnp_predict(results, X_scenario)
    print(f"  {cost_da:>10d} {p[0,0]:>10.4f} {p[0,1]:>10.4f} {p[0,2]:>10.4f}")

print(f"""
  As DA cost increases:
  - DA share decreases (people switch away from expensive option)
  - SR and TR shares increase (substitutes absorb the demand)
  - This is the key insight from MNP: substitution patterns depend
    on the covariance structure of the error terms.
""")

# ============================================================
#  Step 6: Average Treatment Effects (scenarios= API)
# ============================================================
print("=" * 60)
print("  Step 6: Average Treatment Effects via mnp_ate")
print("=" * 60)

print("""
The forecasting above used a hand-built design matrix. For policy
analysis on the ESTIMATION sample we instead use `mnp_ate`, which
re-applies the fitted model to the real data under counterfactual
scenarios and averages the resulting individual probabilities into
predicted aggregate shares.

We re-estimate the GAUSS MNP_TRAVELMODE_ATE specification (the same
flexible-covariance model plus AGE45 effects) and ask: what is the
treatment effect of being under 45 (AGE45 = 1) versus 45+ (AGE45 = 0)
on aggregate mode shares?  Use the scenarios= dict API — NOT the legacy
scalar changevar=/changeval= path, whose .predicted_shares is the
unconditional unmodified-data prediction.
""")

# Flexible-covariance spec with AGE45 effects (matches the GAUSS driver
# MNP_TRAVELMODE_ATE.gss: IVTT, OVTT, COST, AGE45_DA, AGE45_SR, ASCs).
ate_spec = {
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
    "AGE45_DA": {"Alt1_ch": "AGE45",   "Alt2_ch": "sero",    "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",    "Alt2_ch": "AGE45",   "Alt3_ch": "sero"},
    "CON_SR":   {"Alt1_ch": "sero",    "Alt2_ch": "uno",     "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",    "Alt2_ch": "sero",    "Alt3_ch": "uno"},
}

ate_model = MNPModel(
    data=data_path, alternatives=alternatives, availability="none",
    spec=ate_spec,
    control=MNPControl(iid=False, maxiter=200, verbose=0, seed=42),
)
ate_results = ate_model.fit()

print(f"  GAUSS / paper reference LL (Model b + AGE45) : -659.285")
print(f"  PyBhatLib LL                                 : "
      f"{ate_results.loglik * ate_results.n_obs:.3f}")

# Counterfactual shares via the scenarios= API.
data = pd.read_csv(data_path)
ate = mnp_ate(
    ate_results, data=data, spec=ate_spec, alternatives=alternatives,
    scenarios={"base": {"AGE45": 0}, "treatment": {"AGE45": 1}},
)

base_shares = ate.shares_per_scenario["base"]
treat_shares = ate.shares_per_scenario["treatment"]
pct = ate.comparison("base", "treatment")

print(f"\n  {'Alt':>5s} {'Base (45+)':>12s} {'Treat (<45)':>12s} {'%ATE':>10s}")
print(f"  {'-' * 41}")
for i, name in enumerate(alt_names):
    print(f"  {name:>5s} {base_shares[i]:>12.6f} {treat_shares[i]:>12.6f} "
          f"{pct[i]:>+9.2f}%")

# GAUSS cross-check: shares produced by MNP_TRAVELMODE_ATE.gss driver.
print(f"\n  GAUSS reference shares (AGE45=0 base): "
      f"DA=0.6924  SR=0.1411  TR=0.1665")
print(f"  PyBhatLib  shares (AGE45=0 base)     : "
      f"DA={base_shares[0]:.4f}  SR={base_shares[1]:.4f}  TR={base_shares[2]:.4f}")

print("""
  Interpretation: being under 45 raises the drive-alone (DA) share and
  pulls travelers away from shared-ride (SR) and transit (TR). Because
  mnp_ate averages individual choice probabilities under the fitted
  covariance structure, the cross-substitution is governed by the
  estimated error correlations, not by an IIA logit assumption.
""")

print(f"  Next: t05b_morp_ate_predict.py — MORP prediction and ATE analysis")
