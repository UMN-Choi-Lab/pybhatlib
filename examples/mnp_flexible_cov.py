"""Example: MNP Model (a)(ii) — Flexible covariance structure.

Replicates Model (a)(ii) from the BHATLIB paper Table 1.
Expected log-likelihood at convergence: -661.111

Same specification as Model (a)(i) but with flexible (heteroscedastic,
correlated) error terms instead of IID.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl

data_path = os.path.join(os.path.dirname(__file__), "data", "TRAVELMODE.csv")

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# Flexible covariance: IID=False, heteronly=False (Figure 5, mCtl.IID=0)
ctrl = MNPControl(iid=False, heteronly=False, maxiter=200, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()

print(f"\nTarget log-likelihood: -661.111")
print(f"Achieved log-likelihood: {results.ll_total:.3f}")

if results.lambda_hat is not None:
    print(f"\nEstimated kernel error covariance (differenced):")
    print(results.lambda_hat)
