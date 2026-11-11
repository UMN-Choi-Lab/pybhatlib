"""Basic MNP uses the mixed-MNP names and natural distribution tables."""

import numpy as np
import pandas as pd
import pytest

from pybhatlib.models.mnp import MNPControl, MNPModel
from pybhatlib.models.mnp._mnp_loglik import _param_to_unpar


def _model(*, mix=True, randdiag=False, nseg=1, scaling="sumsq", iid=False):
    rng = np.random.default_rng(37)
    n = 60
    y = np.arange(n) % 3
    alts = ["a", "b", "c"]
    data = pd.DataFrame({a: (y == i).astype(float) for i, a in enumerate(alts)})
    for name in ("x", "z"):
        for a in alts:
            data[f"{name}_{a}"] = rng.normal(scale=0.4, size=n)
    spec = {name: {a: f"{name}_{a}" for a in alts} for name in ("x", "z")}
    control = MNPControl(mix=mix, randdiag=randdiag, nseg=nseg, scaling=scaling,
                         iid=iid, method="me", verbose=0, maxiter=0)
    model = MNPModel(data=data, alternatives=alts, spec=spec, control=control,
                     ranvars=["x", "z"] if mix else None)
    control.startb = np.linspace(0.1, 0.5, model.n_params)
    return model


@pytest.mark.parametrize("randdiag", [False, True])
@pytest.mark.parametrize("nseg", [1, 2])
@pytest.mark.parametrize("scaling", ["first", "sumsq"])
def test_random_coefficient_tables_and_names(randdiag, nseg, scaling):
    model = _model(randdiag=randdiag, nseg=nseg, scaling=scaling)
    result = model.fit()
    # The reporting update never changes optimizer-space estimates or Omega.
    np.testing.assert_array_equal(result.params, model.control.startb)
    np.testing.assert_allclose(result.omega_hat, result.cholesky_L @ result.cholesky_L.T)
    assert "scal[x]" in result.param_names and "scal[z]" in result.param_names
    assert not any("CovCOv" in name for name in result.param_names)
    assert ("corr[x,z]" in result.param_names) == (not randdiag)
    for i, name in enumerate(("x", "z")):
        value = result.b_original[result.param_names.index(f"scal[{name}]")]
        assert value == pytest.approx(np.sqrt(result.omega_hat[i, i]))
    if not randdiag:
        expected = result.omega_hat[0, 1] / np.sqrt(np.prod(np.diag(result.omega_hat)))
        assert result.b_original[result.param_names.index("corr[x,z]")] == pytest.approx(expected)

    summary = result.summary()
    assert summary.count("Random-coefficient locations and scales") == 1
    normalization = ("sum of squared differenced scales fixed to 1.000" if scaling == "sumsq"
                     else "first differenced scale fixed to 1.000")
    kernel_title = f"Kernel-error scales (differenced; {normalization})"
    assert summary.count(kernel_title) == 1
    assert "shape" not in summary.lower()
    assert "ker[b-a]" in summary and "ker[c-a]" in summary
    subject = ("Random-coefficient / kernel-error (differenced)" if nseg > 1
               else "Joint random-coefficient / kernel-error (differenced)")
    assert summary.count(f"{subject} correlation matrix") == 1
    assert summary.count(f"{subject} implied covariance matrix") == 1
    assert summary.index("Correlation matrix of the parameters") > summary.rindex("implied covariance matrix")
    kernel_table = summary.split(kernel_title, 1)[1].split(subject, 1)[0]
    assert "Location" not in kernel_table
    assert "Scale" in kernel_table
    kernel_row = next(line for line in kernel_table.splitlines() if line.strip().startswith("ker[b-a]"))
    assert len(kernel_row.split()) == 2
    assert float(kernel_row.split()[1]) == pytest.approx(np.sqrt(result.lambda_hat[0, 0]), abs=5e-6)
    assert "Estimated error correlation matrix" not in summary
    assert "Estimated scales" not in summary
    if nseg == 2:
        assert "scal[x]_s2" in result.param_names
        assert "(segment 1)" not in summary and "(segment 2)" not in summary
        matrix = summary.split(f"{subject} correlation matrix", 1)[1].split("implied covariance matrix", 1)[0]
        first_row = next(line.split() for line in matrix.splitlines()
                         if line.split()[:1] == ["x"] and len(line.split()) == 7)
        assert first_row[3:5] == ["--", "--"]
        table = summary.split("Random-coefficient locations and scales", 1)[1].split(kernel_title, 1)[0]
        for name in ("x", "z"):
            assert any(line.split()[:1] == [name] for line in table.splitlines())
            assert any(line.split()[:1] == [f"{name}_s2"] for line in table.splitlines())
        assert len(result.segment_covs) == 2 and len(result.segment_means) == 2
        for i, name in enumerate(("x", "z")):
            scale = result.b_original[result.param_names.index(f"scal[{name}]_s2")]
            assert scale == pytest.approx(np.sqrt(result.segment_covs[1][i, i]))


@pytest.mark.parametrize("nseg", [2, 3])
def test_segment_reporting_uses_probabilities_and_transforms_uncertainty(nseg):
    model = _model(nseg=nseg)
    theta = model.control.startb.copy()
    optimizer_names = model._build_param_names()
    logits = np.array([-1.2, 0.8])[:nseg - 1]
    for h, value in enumerate(logits, start=2):
        theta[optimizer_names.index(f"segment_logit[s{h}]")] = value
    model.control.startb = theta
    result = model.fit()
    weights = np.exp(np.r_[0.0, logits])
    probabilities = weights / weights.sum()
    np.testing.assert_allclose(result.segment_probs, probabilities)
    probability_names = ["segunpar"] if nseg == 2 else [f"segunpar_s{h}" for h in range(2, nseg + 1)]
    positions = [result.param_names.index(name) for name in probability_names]
    np.testing.assert_allclose(result.b_original[positions], probabilities[1:])
    assert np.all((result.b_original[positions] > 0) & (result.b_original[positions] < 1))
    summary = result.summary()
    assert "Segment probabilities" in summary
    coefficient_table = summary.split("Random-coefficient locations and scales", 1)[1].split("Kernel-error scales", 1)[0]
    for h in range(nseg):
        suffix = f"_s{h + 1}" if h else ""
        for name in ("x", "z"):
            row = next(line.split() for line in coefficient_table.splitlines()
                       if line.split()[:1] == [name + suffix])
            assert float(row[1]) == pytest.approx(result.segment_means[h][model.var_names.index(name)], abs=5e-6)
            assert float(row[2]) == pytest.approx(np.sqrt(result.segment_covs[h][model.var_names.index(name), model.var_names.index(name)]), abs=5e-6)
    probability_table = summary.split("Segment probabilities", 1)[1].split("Random-coefficient", 1)[0]
    displayed = [float(line.split()[1]) for line in probability_table.splitlines()
                 if line.split() and line.split()[0] in {f"s{h}" for h in range(1, nseg + 1)}]
    assert len(displayed) == nseg
    assert sum(displayed) == pytest.approx(1, abs=2e-5)

    # With identity optimizer covariance, probability covariance must be J J'.
    model.control.se_method = "hessian"
    _, se, _, _, cov, *_ = model._normalize_for_reporting(
        theta, None, None, hess_observed=np.eye(len(theta)),
    )
    p = probabilities[1:]
    jacobian = np.diag(p) - np.outer(p, p)
    expected = jacobian @ jacobian.T
    np.testing.assert_allclose(cov[np.ix_(positions, positions)], expected, atol=1e-8)
    np.testing.assert_allclose(se[positions], np.sqrt(np.diag(expected)), atol=1e-8)


@pytest.mark.parametrize("iid", [False, True])
@pytest.mark.parametrize("scaling", ["first", "sumsq"])
def test_no_random_coefficients_uses_labeled_kernel_tables(iid, scaling):
    result = _model(mix=False, iid=iid, scaling=scaling).fit()
    summary = result.summary()
    assert "random-coefficient" not in summary.lower()
    normalization = ("sum of squared differenced scales fixed to 1.000" if scaling == "sumsq"
                     else "first differenced scale fixed to 1.000")
    assert f"Kernel-error scales (differenced; {normalization})" in summary
    assert "Kernel-error (differenced) correlation matrix" in summary
    assert "Kernel-error (differenced) implied covariance matrix" in summary
    assert "ker[b-a]" in summary and "ker[c-a]" in summary
    assert "shape" not in summary.lower()
    assert "Location" not in summary
    assert summary.index("Correlation matrix of the parameters") > summary.index("implied covariance matrix")


def test_random_scale_standard_error_is_for_sd_not_variance():
    model = _model(randdiag=True)
    theta = model.control.startb
    natural = _param_to_unpar(theta, model.n_beta, model.n_alts, model.control, model.ranvar_indices)
    scores = model._bhhh_scores_unpar(theta, natural)
    variance = np.linalg.inv(scores.T @ scores)
    _, se, *_ = model._normalize_for_reporting(theta, None, None)
    sd = np.exp(theta[-1])
    index = model._build_report_names().index("scal[z]")
    assert se[index] == pytest.approx(np.sqrt(variance[-1, -1]) / (2 * sd), rel=1e-5)


def test_frozen_correlation_does_not_freeze_random_scales():
    model = _model(iid=True)
    # beta(2), then correlation, log SD(x), log SD(z).
    mask = np.array([True, True, False, True, True])
    model.control.active_mask = mask
    _, se, *_ = model._normalize_for_reporting(
        model.control.startb, None, None, active_mask=mask,
    )
    assert np.isfinite(se[model._build_report_names().index("scal[z]")])
    assert np.isfinite(se[model._build_report_names().index("scal[x]")])
    assert np.isnan(se[model._build_report_names().index("corr[x,z]")])


@pytest.mark.parametrize("scaling", ["first", "sumsq"])
@pytest.mark.parametrize("n_random", [0, 1])
def test_mixed_mnp_kernel_header_identifies_normalization(scaling, n_random):
    from pybhatlib.models.mnpkercp import MNPKerCPControl
    from pybhatlib.models.mnpkercp._mnpkercp_results import MNPKerCPResults, MixingSummary

    result = MNPKerCPResults.from_estimates([0.0], control=MNPKerCPControl(scaling=scaling))
    names = (["x"] if n_random else []) + ["ker[b-a]", "ker[c-a]"]
    result.mixing_summary = MixingSummary(
        names=names, n_random=n_random, means=np.zeros(len(names)),
        scales=np.ones(len(names)), shapes=np.ones(len(names)), corr_matrix=np.eye(len(names)),
    )
    summary = result.summary()
    normalization = ("sum of squared differenced scales fixed to 1.000" if scaling == "sumsq"
                     else "first differenced scale fixed to 1.000")
    assert f"Kernel-error scales (differenced; {normalization})" in summary
    assert "Shape" in summary  # Shape remains applicable to mixed MNP.
    assert ("Random-coefficient locations" in summary) == bool(n_random)
