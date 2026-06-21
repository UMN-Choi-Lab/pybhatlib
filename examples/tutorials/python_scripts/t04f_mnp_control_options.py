"""Tutorial T04f: MNPControl Configuration Options.

MNPControl has many knobs for controlling the MNP estimation procedure.
This tutorial demonstrates the most important ones.

What you will learn:
  - method: MVNCD approximation method selection
  - spherical: spherical vs direct correlation parameterization
  - IID_first: warm-starting from IID estimates
  - correst: restricting specific correlations to zero
  - optimizer: BFGS vs L-BFGS-B
  - analytic_grad / se_method: analytic BHHH scores and SE source
  - reporting a fitted model with b_original (the readable reported view)
  - ATE scenarios on a control-tuned model

Prerequisites: t00 (quickstart), t04c (heteronly).

Expected runtime: ~30 sec
"""
import os, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.mnp import MNPModel, MNPControl, mnp_ate
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
    "se_method": "Std-error source (bhhh/hessian/sandwich)",
    "se_diagnostic": "Report all SE variants",
    "analytic_grad": "Use analytic BHHH scores (fast)",
    "verbose": "Verbosity level",
    "seed": "Random seed",
    "startb": "User starting values",
    "active_mask": "Fix/free individual parameters",
    "device": "Backend device (cpu/cuda)",
    "gpu_threshold": "N obs before GPU is used",
    "torch_compile": "JIT-compile the torch kernel",
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
    print(f"  {method:>8s} {res.loglik * res.n_obs:>12.3f} {elapsed:>10.1f}")

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
    print(f"\n  spherical={str(sph):<5s}: LL={res.loglik * res.n_obs:.3f}, "
          f"n_params={len(res.params)}, time={elapsed:.1f}s")

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
    print(f"  IID_first={str(iid_first):<5s}: LL={res.loglik * res.n_obs:.3f}, time={elapsed:.1f}s")

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
    print(f"\n  {label}: LL={res.loglik * res.n_obs:.3f}, n_params={len(res.params)}")

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
    print(f"  optimizer='{opt}': LL={res.loglik * res.n_obs:.3f}, time={elapsed:.1f}s")

print("""
  BFGS: Full Hessian approximation, good for small-medium problems.
  L-BFGS-B: Limited-memory BFGS with bounds, good for many parameters.
""")

# ============================================================
#  Step 7: analytic_grad and se_method
# ============================================================
print("\n" + "=" * 60)
print("  Step 7: Analytic Scores and Standard-Error Source")
print("=" * 60)

print("""
  analytic_grad=True (default) uses single-pass analytic per-observation
  scores (the MORP A3 -> MNP port). These same per-obs scores form the
  BHHH outer-product, so se_method='bhhh' is essentially free once the
  gradient is computed. analytic_grad=False falls back to finite
  differences (slower, only for debugging).

  se_method controls which standard errors are reported:
    'bhhh'     - outer product of analytic scores (default, robust+fast)
    'hessian'  - inverse numerical Hessian
    'sandwich' - robust (Huber/White) sandwich estimator
""")

print(f"  {'analytic_grad':>14s} {'se_method':>10s} {'LL':>12s} {'time(s)':>9s}")
print(f"  {'-'*48}")
for ag, sem in [(True, "bhhh"), (True, "hessian"), (False, "bhhh")]:
    t0 = time.perf_counter()
    model = MNPModel(
        data=data_path, alternatives=alternatives, spec=spec,
        control=MNPControl(iid=True, analytic_grad=ag, se_method=sem,
                           maxiter=100, verbose=0, seed=42),
    )
    res = model.fit()
    elapsed = time.perf_counter() - t0
    print(f"  {str(ag):>14s} {sem:>10s} {res.loglik * res.n_obs:>12.3f} "
          f"{elapsed:>9.1f}")

print(f"\n  GAUSS / BHATLIB Table 1 (a)(i) IID reference LL : -670.956")
print(f"  (LL is invariant to gradient/SE choice; only speed and the")
print(f"   reported standard errors differ.)")

# ============================================================
#  Step 8: Reporting a fitted model (b_original)
# ============================================================
print("\n" + "=" * 60)
print("  Step 8: Reporting Estimates with b_original")
print("=" * 60)

print("""
  For REPORTING, use results.b_original -- the readable reported view of
  the estimates, which match the published GAUSS tables. pybhatlib reports
  on the GAUSS first-differenced-variance=1 scale, so b_original and
  results.params share one scale and are equal for the mean coefficients;
  b_original is preferred because it additionally spells out the readable
  kernel block (parker / scale rows) aligned with the std errors / t / p.
""")

model = MNPModel(
    data=data_path, alternatives=alternatives, spec=spec,
    control=MNPControl(iid=True, se_method="bhhh", maxiter=100,
                       verbose=0, seed=42),
)
results = model.fit()

print(f"  Log-likelihood : {results.loglik * results.n_obs:.3f}")
print(f"  GAUSS reference : -670.956")
print(f"  N parameters   : {len(results.b_original)}\n")
print(f"  {'Parameter':<12s} {'Estimate':>10s} {'Std.Err':>10s} "
      f"{'t-stat':>9s} {'p-value':>9s}")
print(f"  {'-'*54}")
for name, b, se, t, p in zip(results.param_names, results.b_original,
                             results.se, results.t_stat, results.p_value):
    print(f"  {name:<12s} {b:>10.4f} {se:>10.4f} {t:>9.2f} {p:>9.3f}")

print(f"\n  params     : {results.params}")
print(f"  b_original : {results.b_original}")
print(f"  (same scale; equal for the mean coefficients under this IID model)")

# ============================================================
#  Step 9: ATE via scenarios on a control-tuned model
# ============================================================
print("\n" + "=" * 60)
print("  Step 9: ATE with the scenarios= API")
print("=" * 60)

print("""
  The current ATE API takes a scenarios= dict and computes choice
  shares under each counterfactual on the SAME fitted model. Scenario
  override keys are DATA COLUMN names (not spec variable labels). An
  empty override dict is the untouched baseline. Use
  comparison(base, treatment) to read the average treatment effect.
  (The legacy changevar=/changeval= scalar path is avoided: its
  predicted_shares is the unconditional unmodified-data prediction and
  silently returns a 0% ATE.)

  Here we shock the drive-alone cost (COST_DA) to zero as a simple
  illustrative scenario on the IID model fit reported above; lowering
  drive-alone cost should raise the drive-alone (Alt1) share.
""")

ate = mnp_ate(
    results, data=model.data, spec=spec, alternatives=alternatives,
    scenarios={"base": {}, "free_da": {"COST_DA": 0.0}},
)
base = ate.shares_per_scenario["base"]
treat = ate.shares_per_scenario["free_da"]
print(f"  {'Alt':<10s} {'base share':>12s} {'free-DA share':>14s} "
      f"{'delta':>9s}")
print(f"  {'-'*47}")
for alt, b, t in zip(alternatives, base, treat):
    print(f"  {alt:<10s} {b:>12.4f} {t:>14.4f} {t - b:>9.4f}")

cmp = ate.comparison("base", "free_da")
print(f"\n  comparison('base','free_da') pct change per alt: {cmp}")
print(f"\n  Shares sum to 1 (base={base.sum():.4f}, "
      f"free_da={treat.sum():.4f}); deltas net to zero as expected.")

print(f"\n  Next: t04g_mnp_forecasting.py — Prediction and scenario analysis")
