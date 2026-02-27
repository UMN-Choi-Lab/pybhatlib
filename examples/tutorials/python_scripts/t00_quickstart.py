"""Tutorial T00: Quick Start.

Your first pybhatlib model in under 20 lines. This tutorial verifies
your installation and runs a simple IID Multinomial Probit model on
the TRAVELMODE dataset.

What you will learn:
  - How to verify pybhatlib is installed correctly
  - How to define a model specification
  - How to estimate an IID MNP model
  - How to interpret the output

Prerequisites: None (this is the first tutorial).
"""
import os, sys
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

# ============================================================
#  Step 1: Verify installation
# ============================================================
print("=" * 60)
print("  Step 1: Verify Installation")
print("=" * 60)

import pybhatlib
print(f"  pybhatlib version : {pybhatlib.__version__}")
print(f"  NumPy version     : {np.__version__}")
import scipy; print(f"  SciPy version     : {scipy.__version__}")
print("  All dependencies OK!\n")

# ============================================================
#  Step 2: Load data
# ============================================================
print("=" * 60)
print("  Step 2: Load TRAVELMODE Data")
print("=" * 60)

import pandas as pd
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
data = pd.read_csv(data_path)
print(f"  Shape: {data.shape}")
print(f"  Columns: {list(data.columns)}")
print(f"\n  First 3 rows:")
print(data.head(3).to_string(index=False))
print()

# ============================================================
#  Step 3: Define model specification
# ============================================================
print("=" * 60)
print("  Step 3: Define Model Specification")
print("=" * 60)

from pybhatlib.models.mnp import MNPModel, MNPControl

# 3 alternatives: DA (drive alone), SR (shared ride), TR (transit)
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# 5 variables: 2 alternative-specific constants + 3 generic variables
spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

print('  "sero" = zero coefficient (variable not in that alternative)')
print('  "uno"  = constant (=1 for all observations)')
print(f"\n  Number of variables: {len(spec)}")
print(f"  Number of alternatives: {len(alternatives)}")
print()

# ============================================================
#  Step 4: Estimate IID MNP
# ============================================================
print("=" * 60)
print("  Step 4: Estimate IID MNP Model")
print("=" * 60)

ctrl = MNPControl(iid=True, maxiter=100, verbose=1, seed=42)

model = MNPModel(
    data=data_path,
    alternatives=alternatives,
    availability="none",
    spec=spec,
    control=ctrl,
)

results = model.fit()
print()
results.summary()

# ============================================================
#  Step 5: Interpret results
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Interpret Results")
print("=" * 60)

print(f"\n  Log-likelihood: {results.ll_total:.3f}")
print(f"  Number of observations: {results.n_obs}")
print(f"  Number of parameters: {len(results.b)}")
print(f"  Converged: {results.converged}")

print("\n  Key findings:")
print("  - Negative IVTT coefficient: higher in-vehicle travel time reduces utility")
print("  - Negative COST coefficient: higher cost reduces utility")
print("  - Constants capture alternative-specific baseline preferences")

# ============================================================
#  Next steps
# ============================================================
print("\n" + "=" * 60)
print("  Next Steps")
print("=" * 60)
print("""
  Tutorials are organized by the pybhatlib build chain:

  Level 1 — Low-level ops:   t01a (vectorization), t01b (LDLT), t01c (truncated MVN)
  Level 2 — Matrix gradients: t02a (gradcovcor), t02b (spherical), t02c (chain rules)
  Level 3 — Distributions:    t03a (MVNCD methods), t03b (gradients), t03c (rectangular)
  Level 4 — MNP models:       t04c (heteronly), t04f (control options), t04g (forecasting)
  Level 5 — MORP model:       t05b (ATE & prediction)
  Level 6 — Advanced:         t06a (backends), t06b (specs), t06c (gradient verification)
""")
