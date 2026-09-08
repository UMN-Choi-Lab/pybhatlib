# pybhatlib

[![PyPI](https://img.shields.io/pypi/v/pybhatlib)](https://pypi.org/project/pybhatlib/)
[![Python](https://img.shields.io/pypi/pyversions/pybhatlib)](https://pypi.org/project/pybhatlib/)
[![Tests](https://img.shields.io/github/actions/workflow/status/UMN-Choi-Lab/pybhatlib/tests.yml?branch=main&label=tests)](https://github.com/UMN-Choi-Lab/pybhatlib/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/pypi/l/pybhatlib)](https://github.com/UMN-Choi-Lab/pybhatlib/blob/main/LICENSE)

Python reimplementation of **BHATLIB** — an open-source library for statistical and
econometric matrix-based inference methods.

BHATLIB (Bhat, Clower, Haddad, Jones; UT Austin / Aptech Systems) provides efficient
matrix operations, gradient-enabled routines for multivariate distribution evaluation
(including Bhat's 2018 MVNCD analytic approximation), and pre-built econometric models.

## Installation

```bash
git clone https://github.com/UMN-Choi-Lab/pybhatlib.git
cd pybhatlib
pip install -e .                  # core (NumPy + SciPy + Numba)
pip install -e ".[torch]"         # add PyTorch backend (optional GPU)
pip install -e ".[dev]"           # add pytest, ruff, mypy
pip install -e ".[all]"           # everything (torch + dev)
```

## Quick Start — Multinomial Probit (MNP)

Sample data: `examples/data/TRAVELMODE.csv` (3 modes — DA / SR / TR, 1125 observations).

```python
from pybhatlib.models.mnp import MNPModel, MNPControl

model = MNPModel(
    data="examples/data/TRAVELMODE.csv",
    alternatives=["Alt1_ch", "Alt2_ch", "Alt3_ch"],
    availability="none",
    spec={
        "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno",     "Alt3_ch": "sero"},
        "CON_TR": {"Alt1_ch": "sero", "Alt2_ch": "sero",    "Alt3_ch": "uno"},
        "IVTT":   {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
        "OVTT":   {"Alt1_ch": "OVTT_DA", "Alt2_ch": "OVTT_SR", "Alt3_ch": "OVTT_TR"},
        "COST":   {"Alt1_ch": "COST_DA", "Alt2_ch": "COST_SR", "Alt3_ch": "COST_TR"},
    },
    control=MNPControl(iid=True),
)
results = model.fit()
results.summary()
```

## Quick Start — Multivariate Ordered Response Probit (MORP)

```python
from pybhatlib.models.morp import MORPModel, MORPControl

model = MORPModel(
    data=df,                                     # DataFrame or CSV path
    dep_vars=["satisfaction", "recommendation"], # ordinal outcome columns
    spec={
        "income":    {"satisfaction": "income",    "recommendation": "income"},
        "age":       {"satisfaction": "age",       "recommendation": "age"},
        "education": {"satisfaction": "education", "recommendation": "education"},
    },
    n_categories=[3, 3],
    control=MORPControl(iid=True, seed=42),
)
results = model.fit()
results.summary()
```

A runnable end-to-end example (with `morp_ate`, `morp_predict`,
`morp_predict_category`) is at
[`examples/tutorials/t05b_morp_ate_predict.ipynb`](https://github.com/UMN-Choi-Lab/pybhatlib/blob/main/examples/tutorials/t05b_morp_ate_predict.ipynb).

## Features

**Models**
- **Multinomial Probit (MNP)** — IID, flexible covariance, heteroscedastic-only,
  random coefficients, mixture-of-normals
- **Multivariate Ordered Response Probit (MORP)** — multiple ordinal outcomes
  with shared covariance; per-outcome `spec` mapping
- **Multiple Discrete-Continuous Extreme Value (MDCEV)** — traditional
  (Bhat 2008) and linear (Bhat 2018) outside-good utility specifications,
  selected via `MDCEVControl.utility`

**Numerical core**
- `vecup` — vecdup, matdupfull, LDLT decomposition, truncated MVN moments
- `matgradient` — gradcovcor, gomegxomegax, spherical / Cholesky parameterizations
- `gradmvn` — Bhat (2018) MVNCD analytic approximation with analytic gradients

**Estimation**
- Multiple SE estimators (`se_method="bhhh" | "hessian" | "sandwich"`); BHHH is
  the default to match GAUSS BHATLIB's `_max_CovPar=2`. All three are computed
  at fit time and exposed as `se_bhhh` / `se_hessian` / `se_sandwich`;
  `results.summary()` prints a side-by-side diagnostic block (a large
  Hessian/BHHH divergence is a misspecification signal).
- `MNPControl.active_mask` to freeze a subset of parameters at their starting
  values without recoding the model.
- `verbose` levels: `0` silent, `1` summary, `2` per-iteration NLL,
  `3` per-iteration parameter / gradient / relative-gradient table.

**Backend**
- NumPy by default; optional PyTorch backend with GPU support (install with
  `[torch]`). All numerical functions take an optional `xp` kwarg.

**Post-estimation**
- Average Treatment Effects (ATE) with scenario-matrix support, forecasting,
  per-category MORP probability prediction.
- MORP ATEs from supplied coefficients without re-fitting:
  `MORPResults.from_estimates(beta, thresholds, correlation)` rebuilds a results
  object from natural-space estimates, and `morp_ate_from_params` /
  `morp_joint_probs` (mean joint category-combination probabilities, the GAUSS
  `ate1.csv` equivalent) compute effects directly from them.

## Verification

pybhatlib reproduces Table 1 from the BHATLIB paper (Bhat 2018) using the
TRAVELMODE dataset (3 modes — DA, SR, TR; 1125 observations):

| Model | Specification | Target LL | Achieved LL | Status |
|-------|--------------|-----------|-------------|--------|
| (a)(i)  | IID errors                  | -670.956 | -670.956 | exact match |
| (a)(ii) | Flexible covariance         | -661.111 | -661.111 | exact match |
| (b)     | + AGE45 demographics        | -659.285 | -659.284 | exact match |
| (c)     | + Random coeff. OVTT        | -635.871 | -635.871 | exact match |
| (d)     | 2-segment mixture           | -634.975 | -632.912 | close (multi-modal) |

Models (a)–(c) reproduce the published estimates and BHHH standard errors to
≤0.001 on every parameter (verified end-to-end against GAUSS 26.1.1 + MaxLik
5.0.9). Model (d) is documented as multi-modal — the Python optimum is a
slightly better local mode than the published one.

For MORP, `iid=False` now uses GAUSS BHATLIB's unit-variance identification by
default (`MORPControl.fix_scales=True`): the latent-utility variances are fixed
at 1 and only correlations are estimated. The full-covariance `MORP_DINING` /
`MORP_WALK` models reproduce the GAUSS mean log-likelihoods (−4.6598 / −3.7591)
and correlation matrices. `summary()` reports the actual threshold cut-points
(with delta-method standard errors) and a gradient column, matching GAUSS's
output. See `docs/plans/MORP_BHATLIB_PARITY.md`.

The driving notebooks are
[`t04h_bhatlib_table1.ipynb`](https://github.com/UMN-Choi-Lab/pybhatlib/blob/main/examples/tutorials/t04h_bhatlib_table1.ipynb) and
[`t04i_bhat2018_table2.ipynb`](https://github.com/UMN-Choi-Lab/pybhatlib/blob/main/examples/tutorials/t04i_bhat2018_table2.ipynb).

## Tutorials

A complete tutorial series lives under
[`examples/tutorials/`](https://github.com/UMN-Choi-Lab/pybhatlib/tree/main/examples/tutorials/) as Jupyter notebooks (each with a
matching `.py` script under `python_scripts/`):

| Track | Notebooks |
|-------|-----------|
| Foundations | `t00_quickstart`, `t01a_vectorization`, `t01b_ldlt`, `t01c_truncated_mvn` |
| Matrix gradients | `t02a_gradcovcor`, `t02b_spherical`, `t02c_chain_rules` |
| MVNCD | `t03a_mvncd_methods`, `t03b_mvncd_gradients`, `t03c_mvncd_rect`, `t03d_univariate_cdfs`, `t03e_bhat2018_table1` |
| MNP | `t04a_mnp_iid` … `t04g_mnp_forecasting`, `t04h_bhatlib_table1`, `t04i_bhat2018_table2` |
| MORP | `t05b_morp_ate_predict` |
| MDCEV | `t07a_mdcev_trad`, `t07b_mdcev_lin` |
| Backends & verification | `t06a_backend_switching`, `t06b_custom_specs`, `t06c_gradient_verification` |

## Contributing

See [`CONTRIBUTING.md`](https://github.com/UMN-Choi-Lab/pybhatlib/blob/main/CONTRIBUTING.md) for development setup, the PR-test
workflow, and CODEOWNERS conventions. Key tests:

```bash
pytest tests/                 # full suite
pytest tests/ -m "not slow"   # skip integration tests
pytest tests/ -m torch        # PyTorch backend tests only
```

## References

1. Bhat, C. R. (2018). New Matrix-Based Methods for the Analytic Evaluation of the
   Multivariate Cumulative Normal Distribution Function. *Transportation Research
   Part B*, 109: 238–256.
2. Bhat, C. R., Clower, E., Haddad, A. J., Jones, J. BHATLIB: An Open-Source
   Library for Statistical and Econometric Matrix-Based Inference Methods in GAUSS.

## License

MIT
