# pybhatlib

Python reimplementation of **BHATLIB** — an open-source library for statistical and
econometric matrix-based inference methods.

BHATLIB (Bhat, Clower, Haddad, Jones; UT Austin / Aptech Systems) provides efficient
matrix operations, gradient-enabled routines for multivariate distribution evaluation
(including Bhat's 2018 MVNCD analytic approximation), and pre-built econometric models.

## Installation

```bash
pip install pybhatlib              # NumPy backend (default)
pip install pybhatlib[torch]       # With optional PyTorch backend
```

For development:

```bash
git clone https://github.com/UMN-Choi-Lab/pybhatlib.git
cd pybhatlib
pip install -e ".[dev]"
```

## Quick Start

```python
from pybhatlib.models.mnp import MNPModel, MNPControl

model = MNPModel(
    data="TRAVELMODE.csv",
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec={
        "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
        "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero", "Alt3_ch": "uno"},
        "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
        "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
        "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
    },
    control=MNPControl(iid=True),
)
results = model.fit()
results.summary()
```

## Features

- **Core matrix operations** (vecup): vecdup, matdupfull, LDLT decomposition, truncated MVN moments
- **Matrix gradients** (matgradient): gradcovcor, gomegxomegax, spherical parameterization
- **Distributions & gradients** (gradmvn): Bhat (2018) MVNCD analytic approximation with gradients
- **Multinomial Probit**: IID, flexible covariance, random coefficients, mixture-of-normals
- **Dual backend**: NumPy (default) + optional PyTorch with GPU support
- **Post-estimation**: Average Treatment Effects (ATE), forecasting

## References

1. Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation of the
   Multivariate Cumulative Normal Distribution Function. *Transportation Research Part B*, 109: 238-256.
2. Bhat, C. R., Clower, E., Haddad, A. J., Jones, J. BHATLIB: An Open-Source Library for
   Statistical and Econometric Matrix-Based Inference Methods in GAUSS.

## License

MIT
