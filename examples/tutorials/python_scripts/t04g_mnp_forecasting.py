"""Tutorial T04g: MNP Forecasting and Scenario Analysis.

After estimating an MNP model, we can predict choice probabilities for
new observations and conduct scenario analysis by varying key variables.

What you will learn:
  - mnp_predict: predicted choice probabilities
  - mnp_predict_choice: most likely alternative
  - Scenario analysis: how cost changes affect mode shares

Prerequisites: t00 (quickstart).
"""
import os, sys
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl
from pybhatlib.models.mnp._mnp_forecast import mnp_predict, mnp_predict_choice
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
print(f"\n  Log-likelihood: {results.ll_total:.3f}")
print(f"  Estimated beta: {results.b}")

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

print(f"  Next: t05b_morp_ate_predict.py — MORP prediction and ATE analysis")
