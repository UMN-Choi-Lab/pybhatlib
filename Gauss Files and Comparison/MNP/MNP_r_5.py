import os
import sys

from pybhatlib.models.mnp import MNPModel, MNPControl

# Data path
data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Joint Model","r_dat_t_1000.csv")

# Model specification (Figure 4 from BHATLIB paper)
spec = {
    "ASC2": {"FCH_1": "SERO", "FCH_2": "UNO", "FCH_3": "SERO", "FCH_4": "SERO", "FCH_5": "SERO"},
    "ASC3": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "UNO", "FCH_4": "SERO", "FCH_5": "SERO"},
    "ASC4": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "SERO", "FCH_4": "UNO", "FCH_5": "UNO"},
    "ASC5": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "SERO", "FCH_4": "SERO", "FCH_5": "SERO"},
    "ATT1": {"FCH_1": "Fatt1_1", "FCH_2": "Fatt1_2", "FCH_3": "Fatt1_3", "FCH_4": "Fatt1_4", "FCH_5": "Fatt1_5"},
    "ATT2": {"FCH_1": "Fatt2_1", "FCH_2": "Fatt2_2", "FCH_3": "Fatt2_3", "FCH_4": "Fatt2_4", "FCH_5": "Fatt2_5"},
    "IND1_2": {"FCH_1": "SERO", "FCH_2": "ind1", "FCH_3": "SERO", "FCH_4": "SERO", "FCH_5": "SERO"},
    "IND1_3": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "ind1", "FCH_4": "SERO", "FCH_5": "SERO"},
    "IND1_4": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "SERO", "FCH_4": "ind1", "FCH_5": "SERO"},
    "IND1_5": {"FCH_1": "SERO", "FCH_2": "SERO", "FCH_3": "SERO", "FCH_4": "SERO", "FCH_5": "ind1"},
    }

# IID control structure (Figure 5)
ctrl = MNPControl(iid=False, maxiter=200, verbose=2, seed=42, tol = 5e-4)

# Create and estimate model (Figure 6)
model = MNPModel(
    data=data_path,
    alternatives=["FCH_1", "FCH_2", "FCH_3", "FCH_4", "FCH_5"],
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
results.summary()
