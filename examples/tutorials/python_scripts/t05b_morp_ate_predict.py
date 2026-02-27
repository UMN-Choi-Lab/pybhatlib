"""Tutorial T05b: MORP Prediction and ATE Analysis.

The Multivariate Ordered Response Probit (MORP) models multiple ordinal
outcomes simultaneously, capturing correlations between dimensions.

What you will learn:
  - Fitting a MORP model with synthetic data
  - morp_ate: predicted ordinal probabilities
  - morp_predict: probabilities for new observations
  - morp_predict_category: most likely category per dimension

Prerequisites: t03c (rectangular MVNCD).
"""
import os, sys
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.morp import (
    MORPModel, MORPControl, morp_ate, morp_predict, morp_predict_category,
)

# ============================================================
#  Step 1: Generate synthetic data
# ============================================================
print("=" * 60)
print("  Step 1: Generate Synthetic MORP Data")
print("=" * 60)

rng = np.random.default_rng(42)
n = 200
n_dims = 2
n_categories = [3, 3]
n_beta = 3

# True parameters
beta_true = np.array([0.5, -0.3, 0.2])
tau_true = [np.array([-0.5, 0.8]), np.array([-0.3, 1.0])]
rho_true = 0.4
sigma_true = np.array([[1.0, rho_true], [rho_true, 1.0]])

# Covariates
income = rng.standard_normal(n)
age = rng.standard_normal(n)
education = rng.standard_normal(n)
X_vars = np.column_stack([income, age, education])

# Latent utilities and observed ordinal outcomes
eps = rng.multivariate_normal(np.zeros(2), sigma_true, size=n)
Y_star = np.column_stack([
    X_vars @ beta_true + eps[:, 0],
    X_vars @ beta_true + eps[:, 1],
])

y1 = np.digitize(Y_star[:, 0], tau_true[0])
y2 = np.digitize(Y_star[:, 1], tau_true[1])

df = pd.DataFrame({
    "income": income, "age": age, "education": education,
    "satisfaction": y1, "recommendation": y2,
})

print(f"\n  n = {n} observations")
print(f"  Dimensions: satisfaction (3 categories), recommendation (3 categories)")
print(f"  True beta: {beta_true}")
print(f"  True rho: {rho_true}")

for col in ["satisfaction", "recommendation"]:
    counts = df[col].value_counts().sort_index()
    print(f"  {col}: {dict(counts)}")

# ============================================================
#  Step 2: Fit MORP model (independent errors for speed)
# ============================================================
print("\n" + "=" * 60)
print("  Step 2: Fit MORP Model (Independent Errors)")
print("=" * 60)

model = MORPModel(
    data=df,
    dep_vars=["satisfaction", "recommendation"],
    indep_vars=["income", "age", "education"],
    n_categories=n_categories,
    control=MORPControl(indep=True, maxiter=200, verbose=1, seed=42),
)
results = model.fit()
print()
results.summary()

print(f"\n  True beta:      {beta_true}")
print(f"  Estimated beta: {results.params[:n_beta]}")

# ============================================================
#  Step 3: morp_ate — Average predicted probabilities
# ============================================================
print("\n" + "=" * 60)
print("  Step 3: morp_ate — Average Predicted Probabilities")
print("=" * 60)

# Build X for ATE: shape (N, D, n_vars)
X_ate = np.zeros((n, n_dims, n_beta))
for d in range(n_dims):
    X_ate[:, d, :] = X_vars

ate_result = morp_ate(results, X_ate, n_dims, n_categories, n_beta)

print(f"\n  Number of observations: {ate_result.n_obs}")
for d in range(n_dims):
    dim_name = ["satisfaction", "recommendation"][d]
    print(f"\n  {dim_name} — average predicted probabilities:")
    for j in range(n_categories[d]):
        print(f"    Category {j}: {ate_result.predicted_probs[d][j]:.4f}")

# ============================================================
#  Step 4: morp_predict — Probabilities for new observations
# ============================================================
print("\n" + "=" * 60)
print("  Step 4: morp_predict — New Observation Probabilities")
print("=" * 60)

# 5 new hypothetical observations
X_new_raw = np.array([
    [1.0, -0.5, 0.3],   # High income, young, some education
    [-0.5, 1.0, -0.2],  # Low income, old, low education
    [0.0, 0.0, 0.0],    # Average everything
    [2.0, 0.5, 1.0],    # Very high income
    [-1.0, -1.0, -1.0], # Low everything
])

N_new = X_new_raw.shape[0]
X_new = np.zeros((N_new, n_dims, n_beta))
for d in range(n_dims):
    X_new[:, d, :] = X_new_raw

probs = morp_predict(results, X_new, n_dims, n_categories, n_beta)

for d in range(n_dims):
    dim_name = ["Satisfaction", "Recommendation"][d]
    print(f"\n  {dim_name} predicted probabilities:")
    print(f"  {'Obs':>5s} {'P(cat=0)':>10s} {'P(cat=1)':>10s} {'P(cat=2)':>10s}")
    print(f"  {'-'*37}")
    for q in range(N_new):
        print(f"  {q+1:>5d} {probs[d][q,0]:>10.4f} {probs[d][q,1]:>10.4f} {probs[d][q,2]:>10.4f}")

# ============================================================
#  Step 5: morp_predict_category — Most likely categories
# ============================================================
print("\n" + "=" * 60)
print("  Step 5: morp_predict_category — Most Likely")
print("=" * 60)

categories = morp_predict_category(results, X_new, n_dims, n_categories, n_beta)

print(f"\n  {'Obs':>5s} {'Satisfaction':>14s} {'Recommendation':>16s}")
print(f"  {'-'*37}")
for q in range(N_new):
    print(f"  {q+1:>5d} {categories[q,0]:>14d} {categories[q,1]:>16d}")

print(f"""
  Category labels:
    0 = low (below first threshold)
    1 = medium (between thresholds)
    2 = high (above second threshold)

  The MORP model jointly predicts ordinal outcomes across dimensions.
  When error correlation is nonzero (indep=False), predictions in one
  dimension are informed by the other dimension.
""")

print(f"  Next: t06a_backend_switching.py — NumPy vs PyTorch backends")
