"""Example: Traditional MDCEV Model
"""

import pandas as pd
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mdcev import MDCEVModel, MDCEVControl

# Data path
data_path = os.path.join(os.path.dirname(__file__), "data", "WorkshopData_ToursimExp.csv")

# Model specification 
utility_spec = {
    "ASCAcc":{"Transp": "sero", "Accomod": "uno", "FandB": "sero", "Shp": "sero", "Recr": "sero"},
    "ASCFnB": {"Transp": "sero", "Accomod": "sero", "FandB": "uno", "Shp": "sero", "Recr": "sero"},
    "ASCShp":{"Transp": "sero", "Accomod": "sero", "FandB": "sero", "Shp": "uno", "AR": "sero"},
    "ASCRec": {"Transp": "sero", "Accomod": "sero", "FandB": "sero", "Shp": "sero", "Recr": "uno"},
    }
gamma_spec = {
    "G_Out":{"Transp": "uno", "Accomod": "sero", "FandB": "sero", "Shp": "sero", "Recr": "sero"},
    "GAcc":{"Transp": "sero", "Accomod": "uno", "FandB": "sero", "Shp": "sero", "Recr": "sero"},
    "GFnB": {"Transp": "sero", "Accomod": "sero", "FandB": "uno", "Shp": "sero", "Recr": "sero"},
    "GShp":{"Transp": "sero", "Accomod": "sero", "FandB": "sero", "Shp": "uno", "Recr": "sero"},
    "GRec": {"Transp": "sero", 'Accomod': "sero", 'FandB': "sero", 'Shp': "sero", 'Recr': "uno"},
}
# control structure 
ctrl = MDCEVControl(maxiter=200, verbose=2, seed=42, tol = 1e-5, optimizer="bfgs", utility="linear")

# Create and estimate model
model = MDCEVModel(
    data=data_path,
    alternatives=["Transp", "Accomod", "FandB", "Shp", "Recr"],
    availability=None,
    utility_spec=utility_spec,
    gamma_spec=gamma_spec,
    control=ctrl,
)

results = model.fit()
results.summary()

