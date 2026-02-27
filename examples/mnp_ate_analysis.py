"""Example: MNP Post-Estimation ATE Analysis.

Replicates the ATE analysis from BHATLIB paper Figure 12.
Expected base-level predicted shares: [0.692, 0.141, 0.167]

Uses Model (b) results to compute Average Treatment Effects of the
AGE45 variable on predicted mode shares.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from pybhatlib.models.mnp import MNPModel, MNPControl, mnp_ate

data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

# First, estimate Model (b) with AGE45
spec = {
    "CON_SR":   {"Alt1_ch": "sero",  "Alt2_ch": "uno",   "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",  "Alt2_ch": "sero",  "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero",  "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",  "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    control=ctrl,
)

print("=" * 50)
print("Step 1: Estimating Model (b)")
print("=" * 50)
results = model.fit()
results.summary()

# ATE Analysis (Figure 11-12)
print("\n" + "=" * 50)
print("Step 2: ATE Analysis — Base Level (AGE45 = 0)")
print("=" * 50)

data = pd.read_csv(data_path)
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

ate_base = mnp_ate(
    results,
    changevar="AGE45",
    changeval=0,
    data=data,
    alternatives=alternatives,
    spec=spec,
)

print(f"\nNumber of observations: {ate_base.n_obs}")
print(f"Predicted shares at base level (AGE45=0):")
for i, alt in enumerate(["DA", "SR", "TR"]):
    print(f"  {alt}: {ate_base.predicted_shares[i]:.6f}")

print(f"\nTarget: [0.692, 0.141, 0.167]")

print("\n" + "=" * 50)
print("Step 3: ATE Analysis — Treatment Level (AGE45 = 1)")
print("=" * 50)

ate_treat = mnp_ate(
    results,
    changevar="AGE45",
    changeval=1,
    data=data,
    alternatives=alternatives,
    spec=spec,
)

print(f"\nPredicted shares at treatment level (AGE45=1):")
for i, alt in enumerate(["DA", "SR", "TR"]):
    print(f"  {alt}: {ate_treat.predicted_shares[i]:.6f}")

# Compute percentage ATE
print("\nPercentage ATE:")
for i, alt in enumerate(["DA", "SR", "TR"]):
    base = ate_base.predicted_shares[i]
    treat = ate_treat.predicted_shares[i]
    pct = (treat - base) / base * 100 if base > 0 else 0
    print(f"  {alt}: {pct:+.1f}%")
