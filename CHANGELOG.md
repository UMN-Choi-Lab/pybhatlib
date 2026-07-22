# Changelog

All notable changes to pybhatlib are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2026-07-22

First version available on PyPI. Identical in code to 0.3.1; released under a new
patch number only because 0.3.1's filenames had already been reserved on PyPI (an
earlier 0.3.1 upload was deleted, and PyPI never permits reusing a deleted
filename). See the 0.3.1 entry below for the full feature list.

## [0.3.1] - 2026-07-22

Prepared as the first PyPI release but withdrawn before general availability; never
installable from PyPI. Superseded by 0.3.2 (same code). Contents:

### Models
- **Multinomial Probit (MNP)** — IID, flexible covariance, heteroscedastic-only,
  random coefficients, and mixture-of-normals specifications.
- **Multivariate Ordered Response Probit (MORP)** — multiple ordinal outcomes
  with shared covariance and per-outcome `spec` mapping.
- **Multiple Discrete-Continuous Extreme Value (MDCEV)** — traditional
  (Bhat 2008) and linear (Bhat 2018) outside-good utility specifications.
- **Multinomial Logit (MNL)**.

### Numerical core
- `vecup` — vecdup, matdupfull, LDLT decomposition, truncated MVN moments.
- `matgradient` — gradcovcor, gomegxomegax, spherical / Cholesky parameterizations.
- `gradmvn` — Bhat (2018) MVNCD analytic approximation with analytic gradients.

### Estimation & post-estimation
- BHHH / Hessian / sandwich standard-error estimators, computed at fit time.
- Average Treatment Effects (ATE) with scenario-matrix support, forecasting,
  per-category MORP probability prediction, and ATEs computed directly from
  supplied coefficients without re-fitting.
- Optional PyTorch backend (`pip install pybhatlib[torch]`) with GPU support;
  all numerical functions accept an optional `xp` backend kwarg.

### Verification
- Reproduces Table 1 of Bhat (2018) on the TRAVELMODE dataset: models (a)-(c)
  match published log-likelihoods and BHHH standard errors to =0.001; model (d)
  is documented as multi-modal.

[0.3.2]: https://github.com/UMN-Choi-Lab/pybhatlib/releases/tag/v0.3.2
[0.3.1]: https://github.com/UMN-Choi-Lab/pybhatlib/releases/tag/v0.3.1
