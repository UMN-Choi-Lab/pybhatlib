import os
import sys
import pandas as pd

from pybhatlib.models.morp import MORPControl, MORPModel

data_path = os.path.join(os.path.dirname(__file__), "Example_Dining.csv")

data = pd.read_csv(data_path)

model = MORPModel(
    data=data,
    dep_vars=["NeatoutO", "Npickupo", "Ndelivo"],
    indep_vars=["resta20", "in150", "urb", "wrk_H"],
    n_categories=[11, 7, 7],
    control=MORPControl(indep=True, verbose=2, seed=42),
)
results = model.fit()
results.summary()