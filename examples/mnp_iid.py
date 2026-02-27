"""Example: MNP Model (a)(i) — IID error structure.

Replicates Model (a)(i) from the BHATLIB paper Table 1.
Expected log-likelihood at convergence: -670.956

Uses TRAVELMODE.csv with 3 modes (DA, SR, TR) and 5 generic variables.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

# Data path
data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

# Model specification (Figure 4 from BHATLIB paper)
spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# IID control structure (Figure 5)
ctrl = MNPControl(iid=True, maxiter=200, verbose=1, seed=42)

# Create and estimate model (Figure 6)
model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()

print(f"\nTarget log-likelihood: -670.956")
print(f"Achieved log-likelihood: {results.ll_total:.3f}")
