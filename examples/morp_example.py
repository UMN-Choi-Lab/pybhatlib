"""MORP (Multivariate Ordered Response Probit) example with synthetic data.

Demonstrates fitting a bivariate ordered probit model with 2 ordinal
dimensions (e.g., satisfaction and recommendation), each with 3 categories.
"""

import numpy as np
import pandas as pd

from pybhatlib.models.morp import MORPControl, MORPModel


def generate_synthetic_morp_data(
    n: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Generate synthetic data for MORP example.

    Parameters
    ----------
    n : int
        Number of observations.
    seed : int
        Random seed.

    Returns
    -------
    df : pd.DataFrame
        Synthetic dataset.
    true_params : dict
        True parameter values used for data generation.
    """
    rng = np.random.default_rng(seed)

    # True parameters
    beta_true = np.array([0.5, -0.3, 0.2])  # income, age, education

    # True thresholds (2 dimensions, 3 categories each -> 2 thresholds per dim)
    tau_true = [
        np.array([-0.5, 0.8]),   # dimension 1 (satisfaction)
        np.array([-0.3, 1.0]),   # dimension 2 (recommendation)
    ]

    # True error correlation
    rho_true = 0.4
    sigma_true = np.array([[1.0, rho_true], [rho_true, 1.0]])

    # Generate covariates
    income = rng.standard_normal(n)
    age = rng.standard_normal(n)
    education = rng.standard_normal(n)

    X_vars = np.column_stack([income, age, education])

    # Generate latent utilities
    eps = rng.multivariate_normal(np.zeros(2), sigma_true, size=n)
    Y_star = np.column_stack([
        X_vars @ beta_true + eps[:, 0],
        X_vars @ beta_true + eps[:, 1],
    ])

    # Map to ordinal categories
    y1 = np.zeros(n, dtype=int)
    y2 = np.zeros(n, dtype=int)
    for i in range(n):
        # Dimension 1
        if Y_star[i, 0] <= tau_true[0][0]:
            y1[i] = 0
        elif Y_star[i, 0] <= tau_true[0][1]:
            y1[i] = 1
        else:
            y1[i] = 2

        # Dimension 2
        if Y_star[i, 1] <= tau_true[1][0]:
            y2[i] = 0
        elif Y_star[i, 1] <= tau_true[1][1]:
            y2[i] = 1
        else:
            y2[i] = 2

    df = pd.DataFrame({
        "income": income,
        "age": age,
        "education": education,
        "satisfaction": y1,
        "recommendation": y2,
    })

    true_params = {
        "beta": beta_true,
        "thresholds": tau_true,
        "sigma": sigma_true,
    }

    return df, true_params


def main():
    print("=" * 60)
    print("  MORP Example: Bivariate Ordered Probit")
    print("=" * 60)
    print()

    # Generate data
    df, true_params = generate_synthetic_morp_data(n=500, seed=42)

    print(f"Generated {len(df)} observations")
    print(f"True beta: {true_params['beta']}")
    print(f"True thresholds dim 1: {true_params['thresholds'][0]}")
    print(f"True thresholds dim 2: {true_params['thresholds'][1]}")
    print(f"True correlation: {true_params['sigma'][0, 1]:.2f}")
    print()

    # Category distribution
    for col in ["satisfaction", "recommendation"]:
        counts = df[col].value_counts().sort_index()
        print(f"  {col}: {dict(counts)}")
    print()

    # ------------------------------------------------------------------
    # Model 1: Independent errors
    # ------------------------------------------------------------------
    print("-" * 60)
    print("  Model 1: Independent errors")
    print("-" * 60)

    model_indep = MORPModel(
        data=df,
        dep_vars=["satisfaction", "recommendation"],
        indep_vars=["income", "age", "education"],
        n_categories=[3, 3],
        control=MORPControl(indep=True, verbose=1, seed=42),
    )
    results_indep = model_indep.fit()
    results_indep.summary()

    # ------------------------------------------------------------------
    # Model 2: Full covariance
    # ------------------------------------------------------------------
    print()
    print("-" * 60)
    print("  Model 2: Full covariance")
    print("-" * 60)

    model_full = MORPModel(
        data=df,
        dep_vars=["satisfaction", "recommendation"],
        indep_vars=["income", "age", "education"],
        n_categories=[3, 3],
        control=MORPControl(
            indep=False,
            method="ovus",
            verbose=1,
            seed=42,
        ),
    )
    results_full = model_full.fit()
    results_full.summary()

    print()
    print("True vs. Estimated parameters:")
    print(f"  beta:  true={true_params['beta']}")
    print(f"         est ={results_full.params[:3]}")
    if results_full.correlation_matrix is not None:
        print(f"  corr:  true={true_params['sigma'][0, 1]:.3f}")
        print(f"         est ={results_full.correlation_matrix[0, 1]:.3f}")


if __name__ == "__main__":
    main()
