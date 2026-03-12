"""Example: MNP Model (d) — 2-segment mixture of normals.

Replicates Model (d) from the BHATLIB paper Table 1.
Target log-likelihood at convergence: -634.975

Note: With synthetic TRAVELMODE data, the optimizer may find a different
local optimum (e.g., LL = -632.912) than the original BHATLIB result.
This is expected — mixture models have multiple local optima, and
synthetic data does not perfectly replicate the original dataset.

Uses a 2-segment discrete mixture of normal distributions for the
error terms, allowing for unobserved heterogeneity in the population.
Same specification as Model (b) with AGE45 demographics.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

# Same spec as Model (b) with AGE45
spec = {
    "CON_SR":   {"Alt1_ch": "sero",  "Alt2_ch": "uno",   "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",  "Alt2_ch": "sero",  "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero",  "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",  "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# 2-segment mixture of normals
ctrl = MNPControl(iid=False, nseg=2, maxiter=200, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()

print(f"\nTarget log-likelihood: -634.975")
print(f"Achieved log-likelihood: {results.ll_total:.3f}")
