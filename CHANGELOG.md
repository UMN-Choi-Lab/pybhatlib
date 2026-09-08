# Changelog

All notable changes to pybhatlib are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **MDCEV `gamma_spec` no longer includes the outside good.** Satiation specs
  list the inside goods only; the outside good's gamma is pinned to
  `MDCEVControl.outside_good_gamma` (GAUSS `u[.,1] = -1000`) and is no longer a
  fixed entry of `params` / `b_reported` / `param_names`. Old-style dicts with an
  outside-good row and column still parse (the column is dropped). A GAUSS
  `bmdcev` vector must have its `-1000` placeholder removed before it is passed
  to `MDCEVResults.from_estimates` / `mdcev_forecast`. (#58)
- `mdcev_ate` reports participation rates (share of simulated allocations with
  positive consumption, from `mdcev_forecast`) instead of the row-normalised
  participation returned by `mdcev_predict`. (#58)
- `mdcev_mixed_ate` / `MDCEVMixedModel.ate()` follow the same participation-rate
  definition (new `make_mdcev_mixed_participation_predict` hook over
  `mdcev_forecast`; `budget_col` accepted), so an `nrndcoef == 0` mixed model
  still collapses onto `mdcev_ate` value-for-value. The mixed prediction /
  forecast hooks now forward the model's `control.utility`, so a linear mixed
  MDCEV forecasts with the linear allocation. (#58 harmonization)

### Added
- Linear (Bhat 2018) outside-good forecasting: `mdcev_forecast` /
  `mdcev_predict` honour `utility="linear"` (from the results' control or the
  new `utility=` argument) with the allocation recursion of the GAUSS
  `Forecasting_LinMDCEV.gss` driver; validated against its `fout1.xlsx`
  reference (`tests/test_models/test_mdcev_gauss_forecast_parity.py`). (#58)
- `prepare_mdcev_forecast_data` / `mdcev_ate(scenarios=...)` accept string
  overrides that copy another column, as the MNP and MORP scenario APIs do. (#58)

### Fixed
- `simmdcev` / `simtradmdcev` clip uniform draws away from 0 and 1 so the
  double-log Gumbel transform stays finite. (#58)
- The MDCEV satiation and gradient helpers accept both gamma index layouts --
  inside-goods-only (fixed-coefficient model) and the full GAUSS layout kept by
  the mixed MDCEV kernel -- and raise on any other length instead of silently
  reading the wrong columns. (#58 harmonization)

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
