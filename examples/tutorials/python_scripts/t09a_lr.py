"""Tutorial T09a: Linear Regression (Single Continuous Outcome).

This tutorial fits an ordinary least-squares linear regression with
pybhatlib's ``LRModel`` on the tourism-expenditure data used in t07a/t07b,
explaining log total trip expenditure with traveller and trip attributes.

What you will learn:
  - How the ``spec`` convention shared by all pybhatlib models maps
    coefficient names to data columns (``"uno"`` = constant)
  - That estimation is analytical (SVD least squares): no optimizer,
    starting values, or convergence settings
  - How to read ``results.summary()``: coefficient table, R-squared,
    F-test, residual variance
  - How ``LRControl(se_method=...)`` switches between classical,
    heteroscedasticity-robust (White), and BHHH standard errors
  - Prediction on the training data or on a new DataFrame
  - Average treatment effects via the ``scenarios=`` API shared with
    MNP / MNL / MORP / MDCEV, and ATEs from externally supplied coefficients

Prerequisites: t00 (quickstart).

Expected runtime: < 1 sec
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.lr import LRControl, LRModel, lr_ate_from_params

data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "WorkshopData_ToursimExp.csv")

# ============================================================
#  Step 1: Data and Specification
# ============================================================
print("=" * 60)
print("  Step 1: Data and Specification")
print("=" * 60)

data = pd.read_csv(data_path)

# Outcome: log of total trip expenditure across the five spending categories
# (accommodation, food & beverage, transport, shopping, recreation).
data["total_exp"] = data[["Accomod", "FandB", "Transp", "Shp", "Recr"]].sum(axis=1)
data["ln_exp"] = np.log(data["total_exp"])

print(f"  n = {len(data)} trips")
print(f"  total expenditure: mean = {data['total_exp'].mean():,.0f}, "
      f"median = {data['total_exp'].median():,.0f}")
print(f"  ln_exp: mean = {data['ln_exp'].mean():.3f}, sd = {data['ln_exp'].std():.3f}")

print("""
  The spec grammar is the one every pybhatlib model uses: a coefficient
  name maps to a data column, to "uno" (a column of ones, i.e. the
  constant), or to a numeric constant. Nothing is added implicitly, so
  the intercept is declared explicitly. The outcome-keyed form used by
  the multi-outcome models, e.g. {"B_RURAL": {"ln_exp": "rural"}}, is
  also accepted.
""")

spec = {
    "CON":      "uno",      # constant
    "B_RURAL":  "rural",    # rural residence (urban = base)
    "B_FEM":    "fr_fem",   # fraction of the travel party that is female
    "B_HINC":   "hinc20k",  # household income above $20k
    "B_ST410":  "st410",    # stay of 4-10 days (< 3 days = base)
    "B_STGT10": "stgt10",   # stay longer than 10 days
}

for name, col in spec.items():
    print(f"  {name:<10s} <- {col}")

# ============================================================
#  Step 2: Estimate and Summarize
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Estimate and Summarize")
print("=" * 60)

print("""
  LRModel follows the same object surface as the other models
  (fit / predict / ate / results.summary()), but the least-squares
  solution is computed in closed form from the SVD of the design
  matrix. There is no optimizer, so LRControl carries only covariance
  and verbosity options; results.n_iter is 0 and results.converged is
  True by construction.
""")

model = LRModel(
    data=data,
    dep_var="ln_exp",
    spec=spec,
    control=LRControl(verbose=1),
)
results = model.fit()
print()
results.summary()

print("""
  Reading the summary:
    - Mean log-likelihood is the Gaussian log-likelihood per observation
      at the MLE residual variance sigma2 = SSE / N (results.sigma2).
    - Residual variance is the unbiased SSE / (N - K)
      (results.residual_variance); the default se_method="hessian" with
      df_correction=True gives the textbook OLS covariance
      SSE / (N - K) * (X'X)^-1. p-values use Student's t with N - K
      degrees of freedom for every se_method.
    - F(df_model, df_resid) jointly tests all non-constant slopes.
    - R-squared is centered because the design spans a constant.
""")

print("  Coefficient table as a DataFrame:")
print(results.to_dataframe().round(4).to_string())

# ============================================================
#  Step 3: Standard-Error Options
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: Standard-Error Options")
print("=" * 60)

print("""
  Expenditure data are typically heteroscedastic, so compare the
  classical covariance with the White sandwich. With df_correction=True
  (the default) the sandwich is the HC1 form; "bhhh" inverts the score
  cross-product of the Gaussian likelihood. Coefficients are identical
  across methods; only the standard errors change.
""")

se_table = {}
for method in ["hessian", "sandwich", "bhhh"]:
    ctrl = LRControl(se_method=method, verbose=0)
    se_table[method] = LRModel(data, "ln_exp", spec, control=ctrl).fit().se

print(f"  {'coef':<10s} {'estimate':>10s} {'se hessian':>12s} {'se sandwich':>12s} {'se bhhh':>10s}")
print("  " + "-" * 58)
for k, name in enumerate(results.param_names):
    print(f"  {name:<10s} {results.params[k]:>10.4f} {se_table['hessian'][k]:>12.4f} "
          f"{se_table['sandwich'][k]:>12.4f} {se_table['bhhh'][k]:>10.4f}")

# ============================================================
#  Step 4: Prediction
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: Prediction")
print("=" * 60)

print("""
  model.predict() with no argument returns the in-sample fitted values.
  Passing a DataFrame rebuilds the design through the model's spec, so
  new observations only need the columns the spec refers to. (Passing a
  ready-made (N, K) array is also accepted.)
""")

y_hat = model.predict()
rmse = np.sqrt(np.mean((data["ln_exp"] - y_hat) ** 2))
print(f"  In-sample RMSE of ln_exp: {rmse:.4f}")
print(f"  R-squared recomputed from fitted values: "
      f"{1 - np.sum((data['ln_exp'] - y_hat) ** 2) / np.sum((data['ln_exp'] - data['ln_exp'].mean()) ** 2):.6f}")

new_trips = pd.DataFrame({
    "rural":   [0, 1],
    "fr_fem":  [0.5, 0.5],
    "hinc20k": [1, 0],
    "st410":   [1, 0],
    "stgt10":  [0, 1],
})
pred_ln = model.predict(new_trips)
print("\n  Two hypothetical trips:")
print(f"  {'trip':<6s} {'rural':>6s} {'hinc':>5s} {'stay':>8s} {'ln_exp':>8s} {'exp(ln_exp)':>12s}")
for i in range(len(new_trips)):
    stay = "4-10 d" if new_trips.st410[i] else (">10 d" if new_trips.stgt10[i] else "<3 d")
    print(f"  {i + 1:<6d} {new_trips.rural[i]:>6d} {new_trips.hinc20k[i]:>5d} {stay:>8s} "
          f"{pred_ln[i]:>8.3f} {np.exp(pred_ln[i]):>12,.0f}")

print("""
  Note: exp(predicted ln_exp) is the conditional median-type prediction
  of expenditure, not its conditional mean (Jensen's inequality).
""")

# ============================================================
#  Step 5: Average Treatment Effects (scenarios= API)
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: Average Treatment Effects (scenarios= API)")
print("=" * 60)

print("""
  model.ate(scenarios=...) uses the scenario grammar shared with the other
  models: each scenario overrides data columns with a scalar or with
  another column, the design is rebuilt through the spec, and the mean
  prediction is reported per scenario. For a linear model the effect in
  outcome units of switching a dummy from 0 to 1 is exactly its
  coefficient, which makes this a good check of the machinery.
""")

ate = model.ate(scenarios={
    "urban": {"rural": 0},
    "rural": {"rural": 1},
    "long_stay_hinc": {"st410": 0, "stgt10": 1, "hinc20k": 1},
})
ate.summary()

effect_units = ate.comparison("urban", "rural")
print(f"\n  ATE rural vs urban, in ln_exp units:  {effect_units:+.4f}")
print(f"  B_RURAL coefficient:                  {results.params[1]:+.4f}")
print(f"  Implied expenditure ratio exp(B):     {np.exp(results.params[1]):.3f}")

print("""
  comparison() reports the effect in outcome units (here ln_exp).
  percent=True gives the percentage change of the *predicted mean of
  ln_exp*, the share-model convention; for a log outcome that is not the
  percentage change in expenditure. Use exp(effect) - 1 for that, as
  shown above.
""")

print("  Scenario table:")
print(ate.to_dataframe().round(4).to_string())

# ATEs from externally supplied coefficients (e.g. GAUSS output) without
# re-fitting: same scenario API, same numbers.
external = lr_ate_from_params(
    results.params,
    param_names=results.param_names,
    data=data, spec=spec, dep_var="ln_exp",
    scenarios={"urban": {"rural": 0}, "rural": {"rural": 1}},
)
print(f"\n  lr_ate_from_params reproduces the fitted ATE: "
      f"{external.comparison('urban', 'rural'):+.4f}")
