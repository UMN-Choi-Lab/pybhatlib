import os
import sys

from pybhatlib.models.mnp import MNPModel, MNPControl

# Data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "TRAVELMODE.csv")

# Model specification (Figure 4 from BHATLIB paper)
spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# IID control structure (Figure 5)
ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

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
