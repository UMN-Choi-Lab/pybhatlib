"""Tutorial T04f: MNPControl Configuration Options.

MNPControl has many knobs for controlling the MNP estimation procedure.
This tutorial demonstrates the most important ones.

What you will learn:
  - method: MVNCD approximation method selection
  - spherical: spherical vs direct correlation parameterization
  - IID_first: warm-starting from IID estimates
  - correst: restricting specific correlations to zero
  - optimizer: BFGS vs L-BFGS-B

Prerequisites: t00 (quickstart), t04c (heteronly).
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl
from dataclasses import fields

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")
alternatives = ["Alt1_ch", "Alt2_ch", "Alt3_ch"]

spec = {
    "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
    "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
    "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
}

# ============================================================
#  Step 1: Print all MNPControl defaults
# ============================================================
print("=" * 60)
print("  Step 1: All MNPControl Options")
print("=" * 60)

ctrl_default = MNPControl()
print(f"\n  {'Field':<20s} {'Default':>15s}  Description")
print(f"  {'-'*75}")

descriptions = {
    "iid": "IID error structure",
    "mix": "Random coefficients",
    "indep": "Independent ordinal dimensions",
    "correst": "Correlation restrictions",
    "heteronly": "Heteroscedastic only (no correlations)",
    "randdiag": "Diagonal random coef covariance",
    "nseg": "Mixture-of-normals segments",
    "method": "MVNCD method",
    "spherical": "Spherical parameterization",
    "scal": "Starting value scale factor",
    "IID_first": "Warm-start from IID model",
    "want_covariance": "Compute parameter covariance",
    "seed10": "QMC secondary seed",
    "perms": "MVNCD variable reorderings",
    "maxiter": "Max optimizer iterations",
    "tol": "Convergence tolerance",
    "optimizer": "Optimization algorithm",
    "verbose": "Verbosity level",
    "seed": "Random seed",
    "startb": "User starting values",
}

for f in fields(MNPControl):
    val = getattr(ctrl_default, f.name)
    val_str = str(val) if val is not None else "None"
    desc = descriptions.get(f.name, "")
    print(f"  {f.name:<20s} {val_str:>15s}  {desc}")

# ============================================================
#  Step 2: Method switching
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: MVNCD Method Comparison on IID Model")
print("=" * 60)

print(f"\n  {'Method':>8s} {'LL':>12s} {'Time(s)':>10s}")
print(f"  {'-'*32}")

for method in ["me", "ovus", "tvbs"]:
    t0 = time.perf_counter()
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(iid=True, method=method, maxiter=100, verbose=0, seed=42),
    )
    res = model.fit()
    elapsed = time.perf_counter() - t0
    print(f"  {method:>8s} {res.ll_total:>12.3f} {elapsed:>10.1f}")

print(f"\n  Note: For IID models, method choice has minimal impact since")
print(f"  the utility differences are uncorrelated (K-1 = 2 dimensions).")

# ============================================================
#  Step 3: spherical=True vs False
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Spherical Parameterization")
print("=" * 60)

print(f"\n  spherical=True (default): angles -> correlation (always PD)")
print(f"  spherical=False: direct correlation parameters (may need bounds)")

for sph in [True, False]:
    t0 = time.perf_counter()
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(
            iid=False, spherical=sph, maxiter=100, verbose=0, seed=42,
        ),
    )
    res = model.fit()
    elapsed = time.perf_counter() - t0
    print(f"\n  spherical={str(sph):<5s}: LL={res.ll_total:.3f}, "
          f"n_params={len(res.b)}, time={elapsed:.1f}s")

# ============================================================
#  Step 4: IID_first — warm starting
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: IID_first — Warm Starting")
print("=" * 60)

print(f"\n  IID_first=True: estimate IID first, use as starting values")
print(f"  IID_first=False: use default/user starting values")

for iid_first in [True, False]:
    t0 = time.perf_counter()
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(
            iid=False, IID_first=iid_first, maxiter=100, verbose=0, seed=42,
        ),
    )
    res = model.fit()
    elapsed = time.perf_counter() - t0
    print(f"  IID_first={str(iid_first):<5s}: LL={res.ll_total:.3f}, time={elapsed:.1f}s")

# ============================================================
#  Step 5: correst — Correlation restrictions
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: correst — Restricting Correlations")
print("=" * 60)

# For 3 alternatives, the kernel has dimension I-1=2.
# correst is a 2x2 upper triangular matrix with 1s on diagonal.
# Off-diagonal 0 means restrict that correlation to 0.
correst_full = np.array([[1.0, 1.0], [0.0, 1.0]])  # estimate correlation
correst_zero = np.array([[1.0, 0.0], [0.0, 1.0]])   # restrict to zero

for label, cr in [("Full correlation", correst_full), ("Zero correlation", correst_zero)]:
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(
            iid=False, correst=cr, maxiter=100, verbose=0, seed=42,
        ),
    )
    res = model.fit()
    print(f"\n  {label}: LL={res.ll_total:.3f}, n_params={len(res.b)}")

# ============================================================
#  Step 6: Optimizer choice
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: Optimizer Choice")
print("=" * 60)

for opt in ["bfgs", "lbfgsb"]:
    t0 = time.perf_counter()
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(
            iid=True, optimizer=opt, maxiter=100, verbose=0, seed=42,
        ),
    )
    res = model.fit()
    elapsed = time.perf_counter() - t0
    print(f"  optimizer='{opt}': LL={res.ll_total:.3f}, time={elapsed:.1f}s")

print(f"""
  BFGS: Full Hessian approximation, good for small-medium problems.
  L-BFGS-B: Limited-memory BFGS with bounds, good for many parameters.
""")

print(f"  Next: t04g_mnp_forecasting.py — Prediction and scenario analysis")
