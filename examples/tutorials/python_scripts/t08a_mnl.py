#MNL Example
import pandas as pd
import os

from pybhatlib.models.mnl import MNLControl, MNLModel

# Load data
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "modeData.csv")
data = pd.read_csv(data_path)

# Add neessary columns for MNL
data["MODE1"] = (data["chosen"] == 1).astype(int)
data["MODE2"] = (data["chosen"] == 2).astype(int)
data["MODE3"] = (data["chosen"] == 3).astype(int)

# Define model specification
alternatives = ["MODE1", "MODE2", "MODE3"]

spec= {
    "ASC_AIR": {"MODE1": "SERO", "MODE2": "UNO", "MODE3": "SERO"},
    "ASC_RAIL": {"MODE1": "SERO", "MODE2": "SERO", "MODE3": "UNO"},
    "IVTT": {"MODE1": "IVTT_CAR", "MODE2": "IVTT_AIR", "MODE3": "IVTT_RAIL"},
    "OVTT": {"MODE1": "SERO", "MODE2": "OVTT_AIR", "MODE3": "OVTT_RAIL"},
    "FREQ": {"MODE1": "SERO", "MODE2": "FREQ_AIR", "MODE3": "FREQ_RAIL"},
    "COST": {"MODE1": "TCCAR", "MODE2": "TCAIR", "MODE3": "TCRAIL"},
    "FEM_AIR": {"MODE1": "SERO", "MODE2": "FEMSEX", "MODE3": "SERO"},
    "FEM_RAIL": {"MODE1": "SERO", "MODE2": "SERO", "MODE3": "FEMSEX"}}

control = MNLControl(se_method="hessian")

model= MNLModel(data=data, 
                       alternatives=alternatives, 
                       spec=spec, 
                       control=control,
)

results = model.fit()
results.summary()
