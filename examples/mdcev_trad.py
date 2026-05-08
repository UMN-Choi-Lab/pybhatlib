"""Example: Traditional MDCEV Model
"""

import pandas as pd
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pybhatlib.models.mdcev import MDCEVModel, MDCEVControl

# Data path
data_path = os.path.join(os.path.dirname(__file__), "data", "Workshop_SCAG_Est.csv")

# Model specification 
utility_spec = {
    "ASC_Esc":{"alt_out": "sero", "Esc": "uno", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "ASC_ho": {"alt_out": "sero", "Esc": "sero", "Ho": "uno", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "ASC_Soc":{"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "uno", "AR": "sero", "Eo": "sero"},
    "ASC_AR": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "uno", "Eo": "sero"},
    "ASC_Eo": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "uno"},
    "male_Esc": {"alt_out": "sero", "Esc": "gend1", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "male_ho": {"alt_out": "sero", "Esc": "sero", "Ho": "gend1", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "male_Soc": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "gend1", "AR": "sero", "Eo": "sero"},
    "Lin_Esc": {"alt_out": "sero", "Esc": "Lin", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "Lin_ho": {"alt_out": "sero", "Esc": "sero", "Ho": "Lin", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "Lin_Eo": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "Lin"},
    "Min_Eo": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "Min"},
}
gamma_spec = {
    "G_Out":{"alt_out": "uno", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "G_Esc":{"alt_out": "sero", "Esc": "uno", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "G_ho": {"alt_out": "sero", "Esc": "sero", "Ho": "uno", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "G_Soc":{"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "uno", "AR": "sero", "Eo": "sero"},
    "G_AR": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "uno", "Eo": "sero"},
    "G_Eo": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "uno"},
    "male_ho": {"alt_out": "sero", "Esc": "sero", "Ho": "gend1", "Soc": "sero", "AR": "sero", "Eo": "sero"},
    "male_Soc": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "gend1", "AR": "sero", "Eo": "sero"},
    "male_Eo": {"alt_out": "sero", "Esc": "sero", "Ho": "sero", "Soc": "sero", "AR": "sero", "Eo": "gend1"},
}
# control structure 
ctrl = MDCEVControl(maxiter=200, verbose=2, seed=42, tol = 1e-5, optimizer="bfgs", utility="trad", se_method="bhhh")

# Create and estimate model
model = MDCEVModel(
    data=data_path,
    alternatives=["alt_out", "Esc", "Ho", "Soc", "AR", "Eo"],
    availability=None,
    utility_spec=utility_spec,
    gamma_spec=gamma_spec,
    control=ctrl,
)

results = model.fit()
results.summary()

