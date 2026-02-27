# pybhatlib — Full Implementation Plan

## Context

**BHATLIB** is an open-source GAUSS library by Bhat, Clower, Haddad, and Jones (UT Austin / Aptech Systems) for statistical and econometric matrix-based inference methods. It provides:
- Efficient matrix operations and gradient-enabled routines for multivariate distribution evaluation
- Bhat's (2018) analytic approximation to the Multivariate Normal CDF (MVNCD)
- Pre-built models: Multinomial Probit (MNP), Multivariate Ordered Probit (MORP), MDCEV

**pybhatlib** is a Python reimplementation making these methods accessible beyond the GAUSS ecosystem.

### User Decisions
- **Scope**: MNP model first + all 3 core computational libraries
- **Optimizer**: Both scipy.optimize AND optional PyTorch backend
- **Array backend**: NumPy by default, optional PyTorch tensor support
- **Packaging**: pip-installable from the start (pyproject.toml)

---

## BHATLIB Technical Summary (from paper)

### Core Computational Libraries

**1. Vecup.src** — Low-level matrix manipulations and gradient functions:
- `vecdup(r)`: Extract upper triangular elements (incl. diagonal) → column vector. For 3×3 → 6×1.
- `vecndup(r)`: Extract upper diagonal elements (excl. diagonal) → column vector. For 3×3 → 3×1.
- `matdupfull(r)`: Expand column vector of upper diagonal elements → full symmetric matrix (inverse of vecdup).
- `matdupdiagonefull(r)`: Convert column vector → symmetric matrix with unit diagonal.
- `nondiag`: Extract non-diagonal elements of a matrix into a vector.
- `vecsymmetry`: Takes square symmetric matrix, produces matrix where each row unrolls the symmetric elements.
- Tools for truncated MVN mean/covariance computation
- LDLT factorization utilities
- Mask matrix operations for mixed models

**2. Matgradient.src** — Higher-level matrix operations and matrix gradients:
- `gradcovcor(CAPOMEGA)`: For Ω = ω Ω* ω (covariance = stddev × correlation × stddev), computes:
  - `glitomega`: K × [K×(K+1)/2] gradient of Ω elements w.r.t. K std dev elements
  - `gomegastar`: [K×(K-1)/2] × [K×(K+1)/2] gradient of Ω elements w.r.t. correlation elements
- `gomegxomegax`: Gradient of A = XΩX' w.r.t. symmetric matrix Ω
- Matrix chain rule operations following row-based arrangement convention

**3. Gradmvn.src** — Probability distributions and their gradients:
- MVNCD using Bhat's (2018) analytic ME approximation with LDLT decomposition
- Truncated (both-end) density and CDF via combinatorial methods
- Partial cumulative normal distribution functions (mixed point/interval)
- Gradient procedures for all distribution functions
- Additional distributions: logistic, skew-normal, skew-t, Gumbel, reverse-Gumbel

### Key Conventions
1. **Row-based matrix arrangement**: Vectorizing proceeds row by row
2. **Symmetric matrices**: Only upper diagonal elements stored as vectors
3. **Covariance parameterization**: Ω = ω Ω* ω (std devs × correlation × std devs)
4. **Positive-definiteness**: Spherical/radial parameterization Ω* = f(Θ) for unconstrained optimization
5. **Gradient chain rules**: dA/dω = dA/dΩ × dΩ/dω (row-based ordering)

### MNP Model
- **Theory**: U_qi = V_qi + ξ_qi, V_qi = β'x_qi, ξ_q ~ MVN(0, Λ)
- **Generalized MNP**: β_q = b + β̃_q, β̃_q ~ MVN(0, Ω), Ω = LL'
- **Mixture-of-normals**: β_q = Σ π_h β_qh, β_qh ~ MVN(b_h, Ω_qh), π₁ < π₂ < ... < π_H
- **Key procedures**: mnpFit, mnpATEFit, mnpControlCreate
- **Control fields**: IID, mix, indep, correst, heteronly, rannddiag, nseg
- **Output**: coefficients, SE, t-stats, p-values, gradient, log-likelihood, covariance matrix, convergence info

### BHATLIB Estimation Workflow
1. Load library and prepare environment
2. Specify data file and key variables (dvunordname, davunordname)
3. Define independent variables matrix (ivunord) with "sero"/"uno" keywords
4. Set coefficient names (var_unordnames)
5. Configure control structure (mCtl.IID, mix, ranvars, etc.)
6. Call fit procedure (mnpFit) → returns results structure
7. Post-estimation: ATE analysis (mnpATEFit), goodness-of-fit

### Validation Data (Table 1 from BHATLIB paper)
Using TRAVELMODE.csv (1,125 workers, 3 modes: DA 78.22%, SR 7.65%, TR 14.13%):

| Model | Description | Log-likelihood |
|-------|------------|----------------|
| (a)(i) | IID errors | -670.956 |
| (a)(ii) | Flexible covariance | -661.111 |
| (b) | + AGE45 variable | -659.285 |
| (c) | + Random coeff on OVTT | -635.871 |
| (d) | 2-segment mixture-of-normals | -634.975 |

ATE output (Figure 12): predicted shares at base level = [0.692, 0.141, 0.167]

---

## Package Structure

```
C:\Users\chois\Gitsrcs\pybhatlib\
├── pyproject.toml
├── README.md
├── LICENSE
├── CLAUDE.md
├── .gitignore
├── examples/
│   ├── data/TRAVELMODE.csv
│   ├── mnp_iid.py
│   ├── mnp_flexible_cov.py
│   ├── mnp_random_coefficients.py
│   └── mnp_ate_analysis.py
├── src/pybhatlib/
│   ├── __init__.py
│   ├── _version.py
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── _array_api.py          # get_backend(), set_backend(), array_namespace()
│   │   ├── _numpy_backend.py      # NumpyBackend: numpy + scipy.linalg + scipy.stats
│   │   └── _torch_backend.py      # TorchBackend: torch + torch.linalg (optional)
│   ├── vecup/                     # Low-level matrix ops (GAUSS Vecup.src)
│   │   ├── __init__.py
│   │   ├── _vec_ops.py            # vecdup, vecndup, matdupfull, matdupdiagonefull, vecsymmetry
│   │   ├── _nondiag.py            # nondiag
│   │   ├── _ldlt.py               # ldlt_decompose, ldlt_rank1_update
│   │   ├── _truncnorm.py          # truncated_mvn_moments
│   │   └── _mask.py               # Mask matrix ops for mixed models
│   ├── matgradient/               # Matrix gradients (GAUSS Matgradient.src)
│   │   ├── __init__.py
│   │   ├── _gradcovcor.py         # gradcovcor: dΩ/dω, dΩ/dΩ*
│   │   ├── _gomegxomegax.py       # dA/dΩ for A=XΩX'
│   │   ├── _spherical.py          # theta_to_corr, grad_corr_theta (PD parameterization)
│   │   └── _chain_rules.py        # Matrix chain rule helpers
│   ├── gradmvn/                   # Distributions & gradients (GAUSS Gradmvn.src)
│   │   ├── __init__.py
│   │   ├── _mvncd.py              # Bhat (2018) MVNCD analytic approximation
│   │   ├── _mvncd_grad.py         # Gradients of MVNCD
│   │   ├── _truncated.py          # Truncated MVN density/CDF
│   │   ├── _partial_cdf.py        # Partial cumulative normal (mixed point/interval)
│   │   ├── _univariate.py         # Standard normal PDF/CDF wrappers
│   │   └── _other_dists.py        # Logistic, skew-normal, skew-t, Gumbel
│   ├── models/
│   │   ├── __init__.py
│   │   ├── _base.py               # BaseModel ABC
│   │   └── mnp/
│   │       ├── __init__.py
│   │       ├── _mnp_control.py    # MNPControl dataclass
│   │       ├── _mnp_results.py    # MNPResults dataclass + summary()
│   │       ├── _mnp_model.py      # MNPModel class with fit()
│   │       ├── _mnp_loglik.py     # Log-likelihood & gradient
│   │       ├── _mnp_ate.py        # ATE post-estimation
│   │       └── _mnp_forecast.py   # Prediction
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── _scipy_optim.py        # scipy.optimize.minimize wrapper (BFGS, L-BFGS-B)
│   │   ├── _torch_optim.py        # PyTorch optimizer wrapper (L-BFGS, Adam)
│   │   └── _convergence.py        # Convergence diagnostics
│   ├── io/
│   │   ├── __init__.py
│   │   ├── _data_loader.py        # CSV/DAT/XLSX via pandas
│   │   └── _spec_parser.py        # Parse ivunord-style specs ("sero"/"uno")
│   └── utils/
│       ├── __init__.py
│       ├── _qmc.py                # Quasi-Monte Carlo sequences
│       ├── _seeds.py              # Random seed management
│       └── _validation.py         # Input validation
└── tests/
    ├── conftest.py                # Backend fixtures, test matrices, TRAVELMODE data
    ├── test_backend/
    ├── test_vecup/
    ├── test_matgradient/
    ├── test_gradmvn/
    ├── test_models/
    └── test_integration/          # End-to-end against BHATLIB paper Table 1
```

---

## Detailed Function Signatures

### Backend Abstraction (`backend/`)

```python
# _array_api.py
BackendName = Literal["numpy", "torch"]

class ArrayNamespace(Protocol):
    """Protocol: zeros, ones, eye, array, arange, concatenate, stack, diag,
    triu_indices, sqrt, exp, log, abs, sum, dot, matmul, transpose,
    solve, cholesky, det, inv, eigh, normal_pdf, normal_cdf, normal_ppf"""

def get_backend(name: BackendName | None = None) -> ArrayNamespace: ...
def set_backend(name: BackendName) -> None: ...
def array_namespace(*arrays) -> ArrayNamespace: ...

# _numpy_backend.py
class NumpyBackend:
    """Wraps numpy + scipy.linalg + scipy.stats."""
    float64 = np.float64
    # All standard ops + solve, cholesky, ldlt, normal_pdf/cdf/ppf

# _torch_backend.py
class TorchBackend:
    """Wraps torch + torch.linalg. Lazy import. Device support."""
    def __init__(self, device="cpu", dtype=None): ...
```

### vecup Module

```python
# _vec_ops.py — All accept optional xp kwarg
def vecdup(r: NDArray, *, xp=None) -> NDArray:
    """(K,K) → (K*(K+1)//2, 1) upper triangular elements incl diagonal."""

def vecndup(r: NDArray, *, xp=None) -> NDArray:
    """(K,K) → (K*(K-1)//2, 1) upper triangular elements excl diagonal."""

def matdupfull(r: NDArray, *, xp=None) -> NDArray:
    """(K*(K+1)//2,) → (P,P) full symmetric matrix. Inverse of vecdup."""

def matdupdiagonefull(r: NDArray, *, xp=None) -> NDArray:
    """(K*(K-1)//2,) → (P,P) symmetric matrix with unit diagonal."""

def vecsymmetry(r: NDArray, *, xp=None) -> NDArray:
    """(K,K) → (K*(K+1)//2, K*K) position pattern matrix."""

# _nondiag.py
def nondiag(r: NDArray, *, xp=None) -> NDArray:
    """(K,K) → (K*(K-1),1) non-diagonal elements row-by-row."""

# _ldlt.py
def ldlt_decompose(A: NDArray, *, xp=None) -> tuple[NDArray, NDArray]:
    """A = LDL^T. Returns (L, D)."""

def ldlt_rank1_update(L, D, v, alpha=1.0, *, xp=None) -> tuple[NDArray, NDArray]:
    """Rank-1 update: LDL^T + alpha*vv^T."""

# _truncnorm.py
def truncated_mvn_moments(mu, sigma, lower, upper, *, xp=None) -> tuple[NDArray, NDArray]:
    """Returns (mu_trunc, sigma_trunc)."""
```

### matgradient Module

```python
# _gradcovcor.py
@dataclass
class GradCovCorResult:
    glitomega: NDArray   # K × [K×(K+1)/2]
    gomegastar: NDArray  # [K×(K-1)/2] × [K×(K+1)/2]

def gradcovcor(capomega: NDArray, *, xp=None) -> GradCovCorResult:
    """For Ω = ω Ω* ω, compute dΩ/dω and dΩ/dΩ*."""

# _gomegxomegax.py
def gomegxomegax(X: NDArray, omega: NDArray, *, xp=None) -> NDArray:
    """dA/dΩ for A=XΩX'. Returns [K(K+1)/2, N(N+1)/2]."""

# _spherical.py
def theta_to_corr(theta: NDArray, K: int, *, xp=None) -> NDArray:
    """Unconstrained Θ → PD correlation matrix Ω*."""

def grad_corr_theta(theta: NDArray, K: int, *, xp=None) -> NDArray:
    """dΩ*/dΘ Jacobian."""

# _chain_rules.py
def chain_grad(dA_dOmega: NDArray, dOmega_dparam: NDArray, *, xp=None) -> NDArray:
    """dA/dparam = dA/dOmega @ dOmega/dparam (row-based order)."""
```

### gradmvn Module

```python
# _mvncd.py
def mvncd(a: NDArray, sigma: NDArray, *, method="me", xp=None) -> float:
    """P(X₁≤a₁,...,X_K≤a_K) for X~MVN(0,Σ). Bhat (2018) ME approximation."""

def mvncd_batch(a: NDArray, sigma: NDArray, *, method="me", xp=None) -> NDArray:
    """Vectorized MVNCD for N observations."""

# _mvncd_grad.py
@dataclass
class MVNCDGradResult:
    prob: float
    grad_a: NDArray       # (K,)
    grad_sigma: NDArray   # (K*(K-1)//2,)

def mvncd_grad(a, sigma, *, method="me", xp=None) -> MVNCDGradResult: ...

# _truncated.py
def truncated_mvn_pdf(x, mu, sigma, lower, upper, *, xp=None) -> float: ...
def truncated_mvn_cdf(x, mu, sigma, lower, upper, *, xp=None) -> float: ...
def truncated_mvn_pdf_grad(x, mu, sigma, lower, upper, *, xp=None) -> tuple: ...

# _partial_cdf.py
def partial_mvn_cdf(points, lower, upper, mu, sigma, point_indices=None, range_indices=None, *, xp=None) -> float: ...

# _other_dists.py
def mv_logistic_cdf(x, sigma, *, xp=None) -> float: ...
def skew_normal_pdf(x, alpha, *, xp=None) -> float: ...
def skew_normal_cdf(x, alpha, *, xp=None) -> float: ...
def skew_t_pdf(x, alpha, nu, *, xp=None) -> float: ...
def gumbel_pdf(x, mu=0.0, beta=1.0, *, xp=None) -> float: ...
def gumbel_cdf(x, mu=0.0, beta=1.0, *, xp=None) -> float: ...
def reverse_gumbel_pdf(x, mu=0.0, beta=1.0, *, xp=None) -> float: ...
def reverse_gumbel_cdf(x, mu=0.0, beta=1.0, *, xp=None) -> float: ...
```

### MNP Model

```python
# _mnp_control.py
@dataclass
class MNPControl:
    iid: bool = False
    mix: bool = False
    indep: bool = False
    correst: NDArray | None = None
    heteronly: bool = False
    rannddiag: bool = False
    nseg: int = 1
    maxiter: int = 200
    tol: float = 1e-5
    optimizer: Literal["bfgs", "lbfgsb", "torch_adam", "torch_lbfgs"] = "bfgs"
    verbose: int = 1
    seed: int | None = None

# _mnp_results.py
@dataclass
class MNPResults:
    b: NDArray                    # Estimated coefficients (parametrized)
    b_original: NDArray           # Unparametrized coefficients
    se: NDArray                   # Standard errors
    t_stat: NDArray               # t-statistics
    p_value: NDArray              # p-values
    gradient: NDArray             # Gradient at convergence
    ll: float                     # Mean log-likelihood
    ll_total: float               # Total log-likelihood
    n_obs: int                    # Number of observations
    param_names: list[str]        # Parameter names
    corr_matrix: NDArray          # Correlation matrix of parameters
    cov_matrix: NDArray           # Var-cov matrix of parameters
    n_iterations: int
    convergence_time: float
    converged: bool
    return_code: int
    lambda_hat: NDArray | None    # Kernel error covariance (if IID=False)
    omega_hat: NDArray | None     # Random coeff covariance (if mix=True)
    cholesky_L: NDArray | None    # Cholesky of Omega (if mix=True)
    segment_probs: NDArray | None # Mixture probabilities (if nseg>1)
    segment_means: list[NDArray] | None
    segment_covs: list[NDArray] | None
    control: MNPControl
    data_path: str

    def summary(self) -> str: ...
    def to_dataframe(self) -> pd.DataFrame: ...

# _mnp_model.py
class MNPModel:
    def __init__(self, data, alternatives, availability="none", spec=None,
                 var_names=None, mix=False, ranvars=None, control=None): ...
    def fit(self) -> MNPResults: ...

# _mnp_loglik.py
def mnp_loglik(theta, X, y, avail, control, *, return_gradient=False, xp=None) -> float | tuple: ...

# _mnp_ate.py
@dataclass
class ATEResult:
    n_obs: int
    predicted_shares: NDArray
    base_shares: NDArray | None
    treatment_shares: NDArray | None
    pct_ate: NDArray | None

def mnp_ate(results: MNPResults, changevar=None, changeval=None) -> ATEResult: ...
```

### Optimization

```python
# _scipy_optim.py
@dataclass
class OptimResult:
    x: NDArray; fun: float; grad: NDArray; hess_inv: NDArray
    n_iter: int; converged: bool; return_code: int; message: str

def minimize_scipy(func, x0, method="BFGS", maxiter=200, tol=1e-5, verbose=1, jac=True) -> OptimResult: ...

# _torch_optim.py
def minimize_torch(func, x0, method="lbfgs", maxiter=200, tol=1e-5, verbose=1, device="cpu") -> OptimResult: ...
```

### I/O

```python
# _data_loader.py
def load_data(path, *, file_type=None) -> pd.DataFrame: ...

# _spec_parser.py
def parse_spec(spec, data, alternatives, nseg=1) -> tuple[NDArray, list[str]]: ...
```

---

## pyproject.toml

```toml
[build-system]
requires = ["hatchling", "hatch-vcs"]
build-backend = "hatchling.build"

[project]
name = "pybhatlib"
dynamic = ["version"]
description = "Python implementation of BHATLIB: matrix-based inference for advanced econometric models"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [{ name = "Seongjin Choi", email = "choi@umn.edu" }]
keywords = ["discrete choice", "multinomial probit", "econometrics", "MVNCD", "covariance matrix"]
dependencies = ["numpy>=1.24", "scipy>=1.10", "pandas>=2.0"]

[project.optional-dependencies]
torch = ["torch>=2.0"]
dev = ["pytest>=7.0", "pytest-cov", "ruff", "mypy", "pre-commit"]
docs = ["sphinx", "sphinx-rtd-theme", "numpydoc"]
all = ["pybhatlib[torch,dev,docs]"]

[tool.hatch.version]
source = "vcs"

[tool.hatch.build.targets.wheel]
packages = ["src/pybhatlib"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["torch: tests requiring PyTorch", "slow: long-running integration tests"]

[tool.ruff]
line-length = 88
target-version = "py310"

[tool.mypy]
python_version = "3.10"
warn_return_any = true
```

---

## Implementation Phases

### Phase 0: Project Scaffolding
- Create pyproject.toml, README.md, LICENSE, .gitignore, CLAUDE.md
- Create all directories and `__init__.py` files
- Add TRAVELMODE.csv to examples/data/
- Initialize git, set up tests/conftest.py
- Verify `pip install -e .` works

### Phase 1: Backend Abstraction
- _array_api.py, _numpy_backend.py, _torch_backend.py
- Tests verifying both backends produce identical results

### Phase 2: vecup Module
- _vec_ops.py, _nondiag.py, _ldlt.py, _truncnorm.py, _mask.py
- Tests against paper examples (p. 6)

### Phase 3: matgradient Module
- _gradcovcor.py, _gomegxomegax.py, _spherical.py, _chain_rules.py
- Numerical gradient verification via finite differences

### Phase 4: gradmvn Module
- _mvncd.py (Bhat 2018 ME approximation), _mvncd_grad.py
- _truncated.py, _partial_cdf.py, _univariate.py, _other_dists.py
- Validate against scipy.stats for K≤3

### Phase 5: Optimization & I/O
- _scipy_optim.py, _torch_optim.py, _convergence.py
- _data_loader.py, _spec_parser.py

### Phase 6: MNP Model
- _mnp_control.py, _mnp_results.py, _mnp_model.py
- _mnp_loglik.py (IID, flexible cov, random coefficients, mixture-of-normals)
- _mnp_ate.py, _mnp_forecast.py

### Phase 7: Integration Testing & Validation
- Replicate Table 1 results from BHATLIB paper
- ATE validation against Figure 12
- Cross-backend verification (NumPy ≈ PyTorch)

---

## Build Order (dependency graph)

```
backend → vecup → matgradient → gradmvn → optim/io → models/mnp
```

## Key References

1. Bhat, C. R. (2018). "New matrix-based methods for the analytic evaluation of the MVNCD." TR Part B, 109: 238–256.
2. Bhat, C. R. (2015). "A new GHDM to jointly model mixed types of dependent variables." TR Part B, 79: 50–77.
3. Bhat, C. R. (2024). "Transformation-based flexible error structures for choice modeling." J. Choice Modelling, 53: 100522.
4. Higham, N. J. (2009). "Cholesky factorization." WIREs Comp. Stats., 1(2): 251–254.
5. Bhat, C. R. (2014). "The CML inference approach." Found. Trends Econometrics, 7(1): 1–117.
6. Saxena, S., Bhat, C. R., Pinjari, A. R. (2023). "Separation-based parameterization strategies." J. Choice Modelling, 47: 100411.
