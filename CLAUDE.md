# pybhatlib

Python reimplementation of BHATLIB (Bhat, Clower, Haddad, Jones) for statistical
and econometric matrix-based inference methods.

## Build & Test

```bash
pip install -e ".[dev]"       # Install in editable mode with dev deps
pip install -e ".[torch]"     # Optional PyTorch backend
pytest tests/                 # Run all tests
pytest tests/ -m "not slow"   # Skip integration tests
pytest tests/ -m torch        # Only PyTorch backend tests
```

## Architecture

Build dependency chain: `backend → vecup → matgradient → gradmvn → optim/io → models/mnp`

- `backend/` — Array backend abstraction (NumPy default, optional PyTorch)
- `vecup/` — Low-level matrix ops (GAUSS Vecup.src): vecdup, matdupfull, LDLT, truncated MVN
- `matgradient/` — Matrix gradients (GAUSS Matgradient.src): gradcovcor, gomegxomegax, spherical param
- `gradmvn/` — Distributions & gradients (GAUSS Gradmvn.src): MVNCD (Bhat 2018), truncated, partial CDF
- `models/mnp/` — Multinomial Probit: IID, flexible covariance, random coefficients, mixture-of-normals
- `optim/` — Optimizer wrappers (scipy.optimize + optional PyTorch)
- `io/` — Data loading and spec parsing
- `utils/` — QMC sequences, seeds, validation

## Key Conventions

1. **Row-based arrangement**: All matrix vectorization proceeds row-by-row (matching BHATLIB)
2. **Upper triangular**: Symmetric matrices stored as vectors of upper diagonal elements
3. **Covariance parameterization**: Ω = ω Ω* ω (std devs × correlation × std devs)
4. **`xp` parameter pattern**: Numerical functions accept optional `xp` kwarg for backend selection
5. **Analytic gradients**: Primary approach; autograd is secondary (PyTorch only)

## Coding Standards

- PEP 8, type hints on all public functions
- Docstrings (NumPy style) on all public functions
- Private modules prefixed with underscore (e.g., `_vec_ops.py`)
- All tests use numerical gradient verification via finite differences where applicable
