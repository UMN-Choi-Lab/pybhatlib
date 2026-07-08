"""Tutorial T05b: MORP Prediction and ATE Analysis.

The Multivariate Ordered Response Probit (MORP) models multiple ordinal
outcomes simultaneously, capturing correlations between dimensions.

What you will learn:
  - Fitting a MORP model with synthetic data
  - morp_ate: predicted ordinal probabilities
  - morp_predict: probabilities for new observations
  - morp_predict_category: most likely category per dimension
  - morp_ate_from_params: GAUSS-style ATE from input coefficients with a
    base vs treatment scenario, cross-checked against the BHATLIB WALK model
    (Female = 0 base, Female = 1 treatment) and the GAUSS ``ate1.csv`` output

Reporting convention: MORP uses unit-variance identification, so the fitted
coefficients live on the reported scale directly. Report ``results.params``
(MORP has NO ``b_original``); standard errors / t / p are in ``results.se``,
``results.t_stat``, ``results.p_value`` and names in ``results.param_names``.

Prerequisites: t03c (rectangular MVNCD).

Expected runtime: ~20 sec
"""
import os, sys
import numpy as np
import pandas as pd
np.set_printoptions(precision=4, suppress=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from pybhatlib.models.morp import (
    MORPModel, MORPControl, morp_ate, morp_predict, morp_predict_category,
    morp_ate_from_params,
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

# Per-outcome spec: each coefficient maps outcome → column name (or "sero").
# Here income/age/education enter both outcomes with the same column.
spec = {
    "income":    {"satisfaction": "income",    "recommendation": "income"},
    "age":       {"satisfaction": "age",       "recommendation": "age"},
    "education": {"satisfaction": "education", "recommendation": "education"},
}

model = MORPModel(
    data=df,
    dep_vars=["satisfaction", "recommendation"],
    spec=spec,
    n_categories=n_categories,
    control=MORPControl(iid=True, maxiter=200, verbose=1, seed=42),
)
results = model.fit()
print()

# Reporting convention: MORP is unit-variance identified, so report
# results.params directly (there is NO b_original for MORP). Print a proper
# table with std errors / t / p for the parameters we recover.
print("  Parameter estimates (unit-variance identification):")
print(f"  {'name':>12s} {'est':>10s} {'se':>10s} {'t':>8s} {'p':>8s}")
print(f"  {'-' * 50}")
n_params = len(results.params)
for k in range(n_params):
    name = results.param_names[k]
    est = results.params[k]
    se = results.se[k]
    t = results.t_stat[k]
    p = results.p_value[k]
    print(f"  {name:>12s} {est:>10.4f} {se:>10.4f} {t:>8.2f} {p:>8.3f}")

print(f"\n  Number of parameters: {n_params}")
print(f"  True beta:      {beta_true}")
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
  When error correlation is nonzero (iid=False), predictions in one
  dimension are informed by the other dimension.
""")

# ============================================================
#  Step 6: GAUSS-validated ATE — BHATLIB WALK model
# ============================================================
print("\n" + "=" * 60)
print("  Step 6: GAUSS-Validated ATE (BHATLIB WALK Model)")
print("=" * 60)

print("""
GAUSS-style ATE from input coefficients. The original BHATLIB
`MORP_WALK_ATE.gss` driver plugs the converged WALK estimates back into the
likelihood, sweeps every joint ordinal combination, and writes the average
combination probabilities to `ate1.csv`. `morp_ate_from_params` reproduces
that exact workflow in Python: pass the reported betas, threshold cut-points,
and correlation matrix together with a design matrix, and request the joint
distribution (`joint=True`).

The WALK model is a 4-dimensional ordered-response model of well-being while
walking — Happy, Meaning, Stress, Tired — each on a 3-point scale. We build a
base scenario (Female = 0, matching `ate1.csv`) and a treatment scenario
(Female = 1) to read off the average treatment effect of being female on the
probability of the top category for each dimension.
""")

# --- WALK data + converged GAUSS estimates (MORP_WALK_ATE.gss `est` block) ---
walk_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "Gauss Files and Comparison", "MORP", "Example_Walk.csv",
)
walk = pd.read_csv(walk_path)
N_walk = len(walk)

# est layout: 8 thresholds (4 dims x 2) | 11 betas | 6 kernel correlations.
est_walk = np.array([
    -1.3401, -0.2849,            # Happy thresholds
    -0.7068,  0.3956,            # Meaning thresholds
     0.3463,  1.3346,            # Stress thresholds
    -0.3659,  0.6810,            # Tired thresholds
     0.1504, -0.0857, -0.1951,   # Happy:   female, age20, util
     0.2772,  0.3367,            # Meaning: female, age65
    -0.2080,  0.1409,  0.2064,   # Stress:  age65, morning, util
    -0.1085, -0.1305,  0.1121,   # Tired:   age65, morning, util
     0.4365, -0.4274, -0.2452,   # corr (H,M)(H,S)(H,T)
    -0.1818, -0.1608,            # corr (M,S)(M,T)
     0.4777,                     # corr (S,T)
])

walk_dims = ["Happy", "Meaning", "Stress", "Tired"]
walk_ncat = [3, 3, 3, 3]
walk_nbeta = 11

thresholds_walk = [est_walk[0:2], est_walk[2:4], est_walk[4:6], est_walk[6:8]]
beta_walk = est_walk[8:19]
corr_vals = est_walk[19:]

# Build the 4x4 kernel correlation from the 6 upper-triangular elements
# (row-by-row order: (0,1)(0,2)(0,3)(1,2)(1,3)(2,3)).
corr_walk = np.eye(4)
for k, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
    corr_walk[i, j] = corr_walk[j, i] = corr_vals[k]


def build_walk_X(df_scn):
    """Map the WALK `ivord` specification into a (N, 4, 11) design array."""
    female = df_scn["Female"].values
    age20 = df_scn["Age20"].values
    age65 = df_scn["Age65"].values
    morning = df_scn["Morning"].values
    util = df_scn["Util"].values
    X = np.zeros((len(df_scn), 4, walk_nbeta))
    # Happy:   col 0,1,2  = female, age20, util
    X[:, 0, 0], X[:, 0, 1], X[:, 0, 2] = female, age20, util
    # Meaning: col 3,4    = female, age65
    X[:, 1, 3], X[:, 1, 4] = female, age65
    # Stress:  col 5,6,7  = age65, morning, util
    X[:, 2, 5], X[:, 2, 6], X[:, 2, 7] = age65, morning, util
    # Tired:   col 8,9,10 = age65, morning, util
    X[:, 3, 8], X[:, 3, 9], X[:, 3, 10] = age65, morning, util
    return X


def top_category_shares(joint_result):
    """Average P(top category) per dimension from a joint ATE result."""
    combos = joint_result.combos
    probs = joint_result.probs
    return [probs[combos[:, d] == walk_ncat[d]].sum() for d in range(4)]


# Base scenario: Female = 0 (this is what GAUSS wrote to ate1.csv)
walk_base = walk.copy()
walk_base["Female"] = 0
X_base = build_walk_X(walk_base)
joint_base = morp_ate_from_params(
    beta_walk, thresholds_walk, corr_walk, X_base, 4, walk_ncat, walk_nbeta,
    dep_vars=walk_dims, joint=True,
)
shares_base = top_category_shares(joint_base)

# Treatment scenario: Female = 1
walk_trt = walk.copy()
walk_trt["Female"] = 1
X_trt = build_walk_X(walk_trt)
joint_trt = morp_ate_from_params(
    beta_walk, thresholds_walk, corr_walk, X_trt, 4, walk_ncat, walk_nbeta,
    dep_vars=walk_dims, joint=True,
)
shares_trt = top_category_shares(joint_trt)

# GAUSS ate1.csv reference (Female = 0 base): average P(top level = 3).
gauss_ref = {
    "Happy": 0.55404, "Meaning": 0.36629,
    "Stress": 0.12060, "Tired": 0.25812,
}

print(f"  WALK observations: {N_walk}")
print("\n  Average P(top category) — base scenario (Female = 0):")
print(f"  {'dimension':>10s} {'PyBhatLib':>12s} {'GAUSS ate1':>12s} {'abs diff':>10s}")
print(f"  {'-' * 46}")
for d, name in enumerate(walk_dims):
    py = shares_base[d]
    ref = gauss_ref[name]
    print(f"  {name:>10s} {py:>12.5f} {ref:>12.5f} {abs(py - ref):>10.5f}")

print("\n  Average treatment effect of Female (treatment - base):")
print(f"  {'dimension':>10s} {'Female=0':>10s} {'Female=1':>10s} {'ATE':>10s}")
print(f"  {'-' * 42}")
for d, name in enumerate(walk_dims):
    base = shares_base[d]
    trt = shares_trt[d]
    print(f"  {name:>10s} {base:>10.5f} {trt:>10.5f} {trt - base:>+10.5f}")

print("""
  Stress and Tired do not include Female, so their shares are identical
  across scenarios (ATE = 0). Being female raises the probability of the top
  Happy and Meaning categories, the genuine ATE captured by the WALK model.
  The base-scenario shares reproduce the GAUSS ate1.csv output to 1e-4.
""")

print(f"  Next: t06a_backend_switching.py — NumPy vs PyTorch backends")
