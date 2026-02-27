"""
Generate synthetic TRAVELMODE.csv dataset for pybhatlib examples.

Dataset description (matching BHATLIB paper):
- 1,125 workers making mode choice decisions
- 3 modes: Drive Alone (DA), Shared Ride (SR), Transit (TR)
- Target sample shares: DA ~78.22%, SR ~7.65%, TR ~14.13%

Uses a simple MNL-like utility model to generate realistic choices
where IVTT, OVTT, and COST have negative effects on utility.
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
np.random.seed(42)
N = 1125  # number of observations (workers)

# Target choice shares
TARGET_DA = 0.7822
TARGET_SR = 0.0765
TARGET_TR = 0.1413

# -------------------------------------------------------------------
# Generate level-of-service attributes
# -------------------------------------------------------------------

# In-vehicle travel time (minutes)
IVTT_DA = np.random.uniform(15, 45, N)
IVTT_SR = np.random.uniform(20, 55, N)
IVTT_TR = np.random.uniform(25, 60, N)

# Out-of-vehicle travel time (minutes)
OVTT_DA = np.random.uniform(2, 10, N)
OVTT_SR = np.random.uniform(5, 15, N)
OVTT_TR = np.random.uniform(10, 30, N)

# Travel cost (dollars)
COST_DA = np.random.uniform(3, 15, N)
COST_SR = np.random.uniform(1, 8, N)
COST_TR = np.random.uniform(1, 5, N)

# -------------------------------------------------------------------
# Generate sociodemographic variable
# -------------------------------------------------------------------

# AGE45: 1 if age >= 45, approximately 40% of sample
AGE45 = (np.random.random(N) < 0.40).astype(int)

# -------------------------------------------------------------------
# Build utilities using an MNL-like model
# -------------------------------------------------------------------
# Coefficients calibrated to approximate the target shares.
# The alternative-specific constants (ASCs) are the main levers
# for matching the target shares. DA is the reference (ASC_DA = 0).

# Negative coefficients for time and cost (disutility)
beta_ivtt = -0.04
beta_ovtt = -0.06
beta_cost = -0.15

# ASCs: DA is reference (0); SR and TR get negative ASCs since DA dominates
ASC_DA = 0.0
ASC_SR = -2.8
ASC_TR = -1.4

# Small positive effect of AGE45 on SR (carpooling) for realism
beta_age_sr = 0.15
beta_age_tr = -0.10

# Systematic utilities
V_DA = ASC_DA + beta_ivtt * IVTT_DA + beta_ovtt * OVTT_DA + beta_cost * COST_DA
V_SR = ASC_SR + beta_ivtt * IVTT_SR + beta_ovtt * OVTT_SR + beta_cost * COST_SR + beta_age_sr * AGE45
V_TR = ASC_TR + beta_ivtt * IVTT_TR + beta_ovtt * OVTT_TR + beta_cost * COST_TR + beta_age_tr * AGE45

# -------------------------------------------------------------------
# Calibration loop: adjust ASCs to match target shares
# -------------------------------------------------------------------
# We iteratively adjust ASCs so that the MNL predicted shares
# match the target shares closely, then draw choices.

for iteration in range(200):
    # MNL probabilities
    max_V = np.maximum(V_DA, np.maximum(V_SR, V_TR))
    exp_DA = np.exp(V_DA - max_V)
    exp_SR = np.exp(V_SR - max_V)
    exp_TR = np.exp(V_TR - max_V)
    sum_exp = exp_DA + exp_SR + exp_TR

    P_DA = exp_DA / sum_exp
    P_SR = exp_SR / sum_exp
    P_TR = exp_TR / sum_exp

    # Average predicted shares
    avg_DA = P_DA.mean()
    avg_SR = P_SR.mean()
    avg_TR = P_TR.mean()

    # Adjust ASCs using the log-ratio correction
    if avg_SR > 1e-10:
        ASC_SR += 0.5 * (np.log(TARGET_SR) - np.log(avg_SR))
    if avg_TR > 1e-10:
        ASC_TR += 0.5 * (np.log(TARGET_TR) - np.log(avg_TR))

    # Recompute utilities with updated ASCs
    V_DA = ASC_DA + beta_ivtt * IVTT_DA + beta_ovtt * OVTT_DA + beta_cost * COST_DA
    V_SR = ASC_SR + beta_ivtt * IVTT_SR + beta_ovtt * OVTT_SR + beta_cost * COST_SR + beta_age_sr * AGE45
    V_TR = ASC_TR + beta_ivtt * IVTT_TR + beta_ovtt * OVTT_TR + beta_cost * COST_TR + beta_age_tr * AGE45

print(f"Calibrated ASCs: DA={ASC_DA:.4f}, SR={ASC_SR:.4f}, TR={ASC_TR:.4f}")
print(f"Average predicted shares: DA={avg_DA:.4f}, SR={avg_SR:.4f}, TR={avg_TR:.4f}")

# -------------------------------------------------------------------
# Draw choices from multinomial distribution
# -------------------------------------------------------------------
# Recompute final probabilities
max_V = np.maximum(V_DA, np.maximum(V_SR, V_TR))
exp_DA = np.exp(V_DA - max_V)
exp_SR = np.exp(V_SR - max_V)
exp_TR = np.exp(V_TR - max_V)
sum_exp = exp_DA + exp_SR + exp_TR

P_DA = exp_DA / sum_exp
P_SR = exp_SR / sum_exp
P_TR = exp_TR / sum_exp

# Draw choices using Gumbel random variates (equivalent to MNL simulation)
# This preserves the correlation between attributes and choices
np.random.seed(42 + 1000)  # separate seed for choice draws

# Use inverse CDF method with uniform draws
u = np.random.random(N)
choice = np.zeros(N, dtype=int)  # 0=DA, 1=SR, 2=TR

choice[u <= P_DA] = 0
choice[(u > P_DA) & (u <= P_DA + P_SR)] = 1
choice[u > P_DA + P_SR] = 2

# Verify shares
n_DA = (choice == 0).sum()
n_SR = (choice == 1).sum()
n_TR = (choice == 2).sum()

print(f"\nDrawn choice counts: DA={n_DA}, SR={n_SR}, TR={n_TR}")
print(f"Drawn choice shares: DA={n_DA/N:.4f}, SR={n_SR/N:.4f}, TR={n_TR/N:.4f}")
print(f"Target shares:       DA={TARGET_DA:.4f}, SR={TARGET_SR:.4f}, TR={TARGET_TR:.4f}")

# -------------------------------------------------------------------
# Fine-tune: swap choices to match target counts exactly
# -------------------------------------------------------------------
target_n_DA = round(TARGET_DA * N)  # 880
target_n_SR = round(TARGET_SR * N)  # 86
target_n_TR = round(TARGET_TR * N)  # 159

# Ensure they sum to N
target_n_DA = N - target_n_SR - target_n_TR

print(f"\nTarget counts: DA={target_n_DA}, SR={target_n_SR}, TR={target_n_TR}")

# Adjust by swapping marginal cases (those with highest probability for the
# needed alternative among the current over-represented alternative)
np.random.seed(42 + 2000)

for _ in range(500):
    curr_DA = (choice == 0).sum()
    curr_SR = (choice == 1).sum()
    curr_TR = (choice == 2).sum()

    if curr_DA == target_n_DA and curr_SR == target_n_SR and curr_TR == target_n_TR:
        break

    # If DA is over-represented
    if curr_DA > target_n_DA:
        da_indices = np.where(choice == 0)[0]
        if curr_SR < target_n_SR:
            # Find the DA-chooser with highest P_SR and swap to SR
            best = da_indices[np.argmax(P_SR[da_indices])]
            choice[best] = 1
        elif curr_TR < target_n_TR:
            best = da_indices[np.argmax(P_TR[da_indices])]
            choice[best] = 2

    # If SR is over-represented
    if curr_SR > target_n_SR:
        sr_indices = np.where(choice == 1)[0]
        if curr_DA < target_n_DA:
            best = sr_indices[np.argmax(P_DA[sr_indices])]
            choice[best] = 0
        elif curr_TR < target_n_TR:
            best = sr_indices[np.argmax(P_TR[sr_indices])]
            choice[best] = 2

    # If TR is over-represented
    if curr_TR > target_n_TR:
        tr_indices = np.where(choice == 2)[0]
        if curr_DA < target_n_DA:
            best = tr_indices[np.argmax(P_DA[tr_indices])]
            choice[best] = 0
        elif curr_SR < target_n_SR:
            best = tr_indices[np.argmax(P_SR[tr_indices])]
            choice[best] = 1

# Final counts
n_DA = (choice == 0).sum()
n_SR = (choice == 1).sum()
n_TR = (choice == 2).sum()
print(f"Final counts:  DA={n_DA}, SR={n_SR}, TR={n_TR}")
print(f"Final shares:  DA={n_DA/N:.4f}, SR={n_SR/N:.4f}, TR={n_TR/N:.4f}")

# -------------------------------------------------------------------
# Build choice indicator columns
# -------------------------------------------------------------------
Alt1_ch = (choice == 0).astype(int)
Alt2_ch = (choice == 1).astype(int)
Alt3_ch = (choice == 2).astype(int)

# -------------------------------------------------------------------
# Assemble DataFrame and save
# -------------------------------------------------------------------
df = pd.DataFrame({
    'Alt1_ch': Alt1_ch,
    'Alt2_ch': Alt2_ch,
    'Alt3_ch': Alt3_ch,
    'IVTT_DA': np.round(IVTT_DA, 2),
    'IVTT_SR': np.round(IVTT_SR, 2),
    'IVTT_TR': np.round(IVTT_TR, 2),
    'OVTT_DA': np.round(OVTT_DA, 2),
    'OVTT_SR': np.round(OVTT_SR, 2),
    'OVTT_TR': np.round(OVTT_TR, 2),
    'COST_DA': np.round(COST_DA, 2),
    'COST_SR': np.round(COST_SR, 2),
    'COST_TR': np.round(COST_TR, 2),
    'AGE45': AGE45,
})

output_path = r'C:\Users\chois\Gitsrcs\pybhatlib\examples\data\TRAVELMODE.csv'
df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nSummary statistics:")
print(df.describe().round(2))
print(f"\nAGE45 distribution: {AGE45.mean():.4f} (target ~0.40)")
