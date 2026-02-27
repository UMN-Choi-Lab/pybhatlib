"""Example: MNP Model (c) — Random coefficients on OVTT.

Replicates Model (c) from the BHATLIB paper Table 1.
Expected log-likelihood at convergence: -635.871

Extends Model (b) by adding a random coefficient on the OVTT variable,
allowing the sensitivity to out-of-vehicle travel time to vary across
individuals following a normal distribution.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

# Model (b) spec with AGE45 (Figure 7)
spec = {
    "CON_SR":   {"Alt1_ch": "sero",  "Alt2_ch": "uno",   "Alt3_ch": "sero"},
    "CON_TR":   {"Alt1_ch": "sero",  "Alt2_ch": "sero",  "Alt3_ch": "uno"},
    "AGE45_DA": {"Alt1_ch": "AGE45", "Alt2_ch": "sero",  "Alt3_ch": "sero"},
    "AGE45_SR": {"Alt1_ch": "sero",  "Alt2_ch": "AGE45", "Alt3_ch": "sero"},
    "IVTT":     {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":     {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":     {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# Mixed MNP with random coefficient on OVTT (Figure 8)
ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    mix=True,
    ranvars=["OVTT"],
    control=ctrl,
)

results = model.fit()
results.summary()

print(f"\nTarget log-likelihood: -635.871")
print(f"Achieved log-likelihood: {results.ll_total:.3f}")

if results.omega_hat is not None:
    print(f"\nRandom coefficient covariance (Omega):")
    print(results.omega_hat)
    print(f"Cholesky L:")
    print(results.cholesky_L)
