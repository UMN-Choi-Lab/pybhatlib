"""Example: MNP Model (b) — Flexible covariance with AGE45 demographic.

Replicates Model (b) from the BHATLIB paper Table 1.
Expected log-likelihood at convergence: -659.285

Extends Model (a)(ii) by adding alternative-specific AGE45 variables
for DA and SR modes, capturing age-related heterogeneity in mode choice.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

# Model (b) spec: adds AGE45_DA and AGE45_SR to the base specification
spec = {
    "CON_SR":   {"Alt1_ch": "sero",  "Alt2_ch": "uno",   "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",  "Alt2_ch": "sero",  "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero",  "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",  "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# Flexible covariance (same as Model (a)(ii))
ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()

print(f"\nTarget log-likelihood: -659.285")
print(f"Achieved log-likelihood: {results.ll_total:.3f}")
