import os
import sys

from pybhatlib.models.mnp import MNPModel, MNPControl

# Data path
data_path = os.path.join(os.path.dirname(__file__), "VehTypePurchase.csv")

# Model specification (Figure 4 from BHATLIB paper)
spec = {
    "AV_TECH":   {"REGVEH": "SERO", "AV": "TECH", "SAV": "SERO"},
    "AV_SAFE":   {"REGVEH": "SERO", "AV": "SAFE", "SAV": "SERO"},
    "AV_DRIVE":   {"REGVEH": "SERO", "AV": "DRIVE", "SAV": "SERO"},
    "AV_FEMALE": {"REGVEH": "SERO", "AV": "FEMALE", "SAV": "SERO"},
    "AV_AGE65": {"REGVEH": "SERO", "AV": "AGE65", "SAV": "SERO"},
    "SAV_MOB": {"REGVEH": "SERO", "AV": "SERO", "SAV": "MOB"},
    "SAV_VEH": {"REGVEH": "SERO", "AV": "SERO", "SAV": "IFVEH"},
    "SAV_AGE65": {"REGVEH": "SERO", "AV": "SERO", "SAV": "AGE65"},
    "AV_CON": {"REGVEH": "SERO", "AV": "UNO", "SAV": "SERO"},
    "SAV_CON": {"REGVEH": "SERO", "AV": "SERO", "SAV": "UNO"},
    }

# IID control structure (Figure 5)
ctrl = MNPControl(iid=False, maxiter=200, verbose=1, seed=42)

# Create and estimate model (Figure 6)
model = MNPModel(
    data=data_path,
    alternatives=["REGVEH", "AV", "SAV"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()
