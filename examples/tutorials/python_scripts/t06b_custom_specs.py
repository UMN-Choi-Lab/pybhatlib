"""Tutorial T06b: Custom Model Specifications.

The specification (spec) dictionary controls how data columns map to
the utility function for each alternative. This tutorial covers all
the patterns you'll need for real MNP models.

What you will learn:
  - Basic spec dict structure (sero, uno, column names)
  - Alternative-specific constants
  - Generic vs alternative-specific variables
  - parse_ivunord: GAUSS-style row-per-alt matrix
  - Availability masks
  - Inspecting the design matrix X

Prerequisites: t00 (quickstart).
"""
import os, sys
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.io import parse_spec, load_data
from pybhatlib.io._spec_parser import parse_ivunord

# Load TRAVELMODE data
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
data = load_data(data_path)
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

# ============================================================
#  Step 1: Basic spec dict
# ============================================================
print("=" * 60)
print("  Step 1: Basic Spec Dictionary")
print("=" * 60)

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

X, var_names = parse_spec(spec, data, alternatives)

print(f"\n  Spec defines {len(var_names)} variables: {var_names}")
print(f"  X shape: {X.shape}  (N observations, {X.shape[1]} alternatives, {X.shape[2]} variables)")

print(f"\n  Keyword reference:")
print(f'    "sero" = 0.0 for all observations (variable not in utility)')
print(f'    "uno"  = 1.0 for all observations (intercept/constant)')
print(f'    "COL"  = use column COL from the data')

# ============================================================
#  Step 2: Alternative-specific constants
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Alternative-Specific Constants")
print("=" * 60)

print(f"""
  With 3 alternatives, we can have at most 2 alternative-specific
  constants (one is the reference). In our spec:

  CON_SR: [sero, uno, sero]  -> constant for SR (Alt2) only
  CON_TR: [sero, sero, uno]  -> constant for TR (Alt3) only
  DA (Alt1) is the reference alternative (no constant).

  This is equivalent to having intercepts in the utility function:
    V_DA = beta_IVTT * IVTT_DA + beta_OVTT * OVTT_DA + beta_COST * COST_DA
    V_SR = alpha_SR + beta_IVTT * IVTT_SR + ...
    V_TR = alpha_TR + beta_IVTT * IVTT_TR + ...
""")

# Show the design matrix for observation 0
print(f"  X[0, :, :] (first observation):")
print(f"  {'':>10s}", end="")
for vn in var_names:
    print(f" {vn:>10s}", end="")
print()
for a in range(3):
    alt_label = ["DA", "SR", "TR"][a]
    print(f"  {alt_label:>10s}", end="")
    for v in range(len(var_names)):
        print(f" {X[0, a, v]:>10.2f}", end="")
    print()

# ============================================================
#  Step 3: Generic vs alternative-specific variables
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Generic vs Alternative-Specific Variables")
print("=" * 60)

print(f"""
  Generic variable: same coefficient across alternatives, but different
  data column per alternative.

  Example: IVTT has one coefficient but uses IVTT_DA, IVTT_SR, IVTT_TR.
  The coefficient captures the marginal utility of in-vehicle time,
  which is assumed the same for all modes.

  Alternative-specific variable: different coefficient per alternative,
  often from the same data column.

  Example: If income affects mode choice differently:
""")

spec_altspec = {
    "INC_DA": {"Alt1_ch": "INCOME", "Alt2_ch": "sero",   "Alt3_ch": "sero"},
    "INC_SR": {"Alt1_ch": "sero",   "Alt2_ch": "INCOME", "Alt3_ch": "sero"},
}

# Only show this if INCOME column exists
if "INCOME" in data.columns:
    X_alt, vn_alt = parse_spec(spec_altspec, data, alternatives)
    print(f"  Alternative-specific income spec gives:")
    print(f"  Variables: {vn_alt}")
    print(f"  X shape: {X_alt.shape}")
else:
    print(f"  (INCOME column not in TRAVELMODE data — shown as example pattern)")
    print(f"  Alternative-specific specs use the same column in different rows.")

# ============================================================
#  Step 4: parse_ivunord — GAUSS-style specification
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: parse_ivunord — GAUSS-Style Specification")
print("=" * 60)

# Equivalent to the spec above, but in matrix format
ivunord = [
    ["sero", "sero", "IVTT_DA", "OVTT_DA", "COST_DA"],  # Alt1 (DA)
    ["uno",  "sero", "IVTT_SR", "OVTT_SR", "COST_SR"],  # Alt2 (SR)
    ["sero", "uno",  "IVTT_TR", "OVTT_TR", "COST_TR"],  # Alt3 (TR)
]

var_names_iv = ["CON_SR", "CON_TR", "IVTT", "OVTT", "COST"]
X_iv, vn_iv = parse_ivunord(ivunord, data, alternatives, var_names=var_names_iv)

print(f"\n  ivunord (row = alternative, column = variable):")
for i, row in enumerate(ivunord):
    alt_label = ["DA", "SR", "TR"][i]
    print(f"    {alt_label}: {row}")

print(f"\n  Result shape: {X_iv.shape}")
print(f"  Variable names: {vn_iv}")
print(f"  Match parse_spec? {np.allclose(X, X_iv)}")

print(f"""
  When to use parse_ivunord:
  - Translating GAUSS BHATLIB code to pybhatlib
  - When you have specifications in matrix form
  - It internally converts to dict spec and calls parse_spec
""")

# ============================================================
#  Step 5: Availability masks
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Availability Masks")
print("=" * 60)

print(f"""
  Not all alternatives may be available to all individuals.
  The availability parameter controls this:

    availability="none"     -> all alternatives available to everyone
    availability="AV_COL"   -> single column with availability codes
    availability=["AV1", "AV2", "AV3"]  -> one column per alternative

  When an alternative is unavailable (avail=0), its probability is
  set to zero and the remaining alternatives share 100%.
""")

# Demonstrate availability with synthetic mask
avail = np.ones((len(data), 3))
avail[:50, 2] = 0  # First 50 people can't use transit

print(f"  Example: transit unavailable for first 50 observations")
print(f"  avail shape: {avail.shape}")
print(f"  Rows with transit available: {int(avail[:, 2].sum())}/{len(data)}")

# ============================================================
#  Step 6: Inspect design matrix structure
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Design Matrix Structure")
print("=" * 60)

print(f"\n  X[0, :, :] — First observation, all alternatives:")
header = "  " + f"{'':>6s}"
for vn in var_names:
    header += f" {vn:>10s}"
print(header)
print(f"  {'-'*len(header)}")

for a in range(3):
    alt_label = ["DA", "SR", "TR"][a]
    row = f"  {alt_label:>6s}"
    for v in range(len(var_names)):
        row += f" {X[0, a, v]:>10.2f}"
    print(row)

print(f"\n  X[1, :, :] — Second observation:")
for a in range(3):
    alt_label = ["DA", "SR", "TR"][a]
    row = f"  {alt_label:>6s}"
    for v in range(len(var_names)):
        row += f" {X[1, a, v]:>10.2f}"
    print(row)

print(f"""
  Notice:
  - Constants (CON_SR, CON_TR) are the same across observations
  - Data columns (IVTT, OVTT, COST) vary by observation
  - Each variable has one coefficient shared across all observations
""")

print(f"  Next: t06c_gradient_verification.py — Finite-difference verification")
