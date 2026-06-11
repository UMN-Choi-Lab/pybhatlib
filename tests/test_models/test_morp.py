"""Tests for MORP (Multivariate Ordered Response Probit) model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pybhatlib.backend import get_backend
from pybhatlib.gradmvn._mvncd import mvncd_rect
from pybhatlib.models.morp import MORPControl, MORPModel
from pybhatlib.models.morp._morp_loglik import (
    _unpack_morp_params,
    count_morp_params,
    morp_loglik,
)


@pytest.fixture
def xp():
    return get_backend("numpy")


@pytest.fixture
def synthetic_morp_data():
    """Generate small synthetic MORP dataset."""
    rng = np.random.default_rng(42)
    n = 100

    beta_true = np.array([0.5, -0.3])
    tau_true = [np.array([-0.5, 0.5]), np.array([-0.3, 0.7])]
    rho_true = 0.3
    sigma_true = np.array([[1.0, rho_true], [rho_true, 1.0]])

    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X_vars = np.column_stack([x1, x2])

    eps = rng.multivariate_normal(np.zeros(2), sigma_true, size=n)
    Y_star = np.column_stack([
        X_vars @ beta_true + eps[:, 0],
        X_vars @ beta_true + eps[:, 1],
    ])

    y1 = np.digitize(Y_star[:, 0], tau_true[0])
    y2 = np.digitize(Y_star[:, 1], tau_true[1])

    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2,
    })

    return df, beta_true, tau_true, sigma_true


class TestMORPControl:
    def test_defaults(self):
        ctrl = MORPControl()
        assert ctrl.method == "ovus"
        assert ctrl.iid is False
        assert ctrl.spherical is True

    def test_independent(self):
        ctrl = MORPControl(iid=True)
        assert ctrl.iid is True


class TestCountMORPParams:
    def test_independent_2d_3cat(self):
        ctrl = MORPControl(iid=True)
        n = count_morp_params(3, 2, [3, 3], ctrl)
        # 3 betas + 2*2 thresholds = 7
        assert n == 7

    def test_full_cov_2d_3cat(self):
        # Default is now unit-variance identification (fix_scales=True), so the
        # (D-1) free scales are dropped — matching GAUSS BHATLIB MORP.
        ctrl = MORPControl(iid=False)
        n = count_morp_params(3, 2, [3, 3], ctrl)
        # 3 betas + 2*2 thresholds + 0 scales + 1 corr = 8
        assert n == 8

    def test_full_cov_2d_3cat_free_scales(self):
        # Legacy (unidentified) free-scale layout, opt-in via fix_scales=False.
        ctrl = MORPControl(iid=False, fix_scales=False)
        n = count_morp_params(3, 2, [3, 3], ctrl)
        # 3 betas + 2*2 thresholds + 1 scale + 1 corr = 9
        assert n == 9

    def test_heteronly_3d(self):
        ctrl = MORPControl(heteronly=True)
        n = count_morp_params(2, 3, [3, 4, 3], ctrl)
        # 2 betas + (2+3+2) thresholds + 2 scales = 11
        assert n == 11


class TestUnpackMORPParams:
    def test_threshold_ordering(self):
        """Thresholds should be strictly increasing."""
        ctrl = MORPControl(iid=True)
        # 2 betas + 2 thresholds for dim1 + 2 thresholds for dim2
        theta = np.array([0.5, -0.3, -0.5, 0.2, -0.3, 0.3])
        beta, thresholds, sigma = _unpack_morp_params(
            theta, 2, 2, [3, 3], ctrl
        )
        # tau[0] = theta[2] = -0.5
        # tau[1] = tau[0] + exp(theta[3]) = -0.5 + exp(0.2)
        assert thresholds[0][1] > thresholds[0][0]
        assert thresholds[1][1] > thresholds[1][0]

    def test_sigma_identity_for_indep(self):
        ctrl = MORPControl(iid=True)
        theta = np.zeros(7)
        _, _, sigma = _unpack_morp_params(theta, 3, 2, [3, 3], ctrl)
        np.testing.assert_allclose(sigma, np.eye(2))


class TestMORPLoglik:
    def test_loglik_finite(self, xp):
        """Log-likelihood should be finite for valid parameters."""
        ctrl = MORPControl(iid=True, method="scipy", verbose=0)

        X = np.random.randn(20, 2, 2)
        y = np.random.randint(0, 3, size=(20, 2))

        # 2 betas + 2*2 thresholds = 6 params
        theta = np.array([0.1, -0.1, -0.5, 0.2, -0.3, 0.3])

        nll = morp_loglik(theta, X, y, 2, [3, 3], 2, ctrl, xp=xp)
        assert np.isfinite(nll)
        assert nll > 0  # negative mean log-likelihood should be positive

    def test_loglik_with_gradient(self, xp):
        """Gradient computation should work."""
        ctrl = MORPControl(iid=True, method="scipy", verbose=0)

        X = np.random.randn(10, 2, 2)
        y = np.random.randint(0, 3, size=(10, 2))
        theta = np.array([0.1, -0.1, -0.5, 0.2, -0.3, 0.3])

        nll, grad = morp_loglik(
            theta, X, y, 2, [3, 3], 2, ctrl,
            return_gradient=True, xp=xp,
        )
        assert np.isfinite(nll)
        assert len(grad) == len(theta)
        assert np.all(np.isfinite(grad))


class TestMORPModel:
    def test_model_construction(self, synthetic_morp_data):
        df, _, _, _ = synthetic_morp_data

        spec = {
            "x1": {"y1": "x1", "y2": "x1"},
            "x2": {"y1": "x2", "y2": "x2"},
        }
        model = MORPModel(
            data=df,
            dep_vars=["y1", "y2"],
            spec=spec,
            n_categories=[3, 3],
            control=MORPControl(iid=True, verbose=0),
        )

        assert model.N == 100
        assert model.n_dims == 2
        assert model.n_beta == 2

    def test_model_fit_indep(self, synthetic_morp_data):
        """Test that independent MORP model converges."""
        df, _, _, _ = synthetic_morp_data

        spec = {
            "x1": {"y1": "x1", "y2": "x1"},
            "x2": {"y1": "x2", "y2": "x2"},
        }
        model = MORPModel(
            data=df,
            dep_vars=["y1", "y2"],
            spec=spec,
            n_categories=[3, 3],
            control=MORPControl(
                iid=True,
                method="scipy",
                verbose=0,
                seed=42,
                maxiter=50,
            ),
        )

        results = model.fit()
        assert np.isfinite(results.loglik)
        assert len(results.params) == model.n_params
        assert len(results.thresholds) == 2
        # Thresholds should be ordered
        for tau_d in results.thresholds:
            if len(tau_d) > 1:
                assert np.all(np.diff(tau_d) > 0)

    def test_mismatched_categories_raises(self):
        df = pd.DataFrame({
            "x1": [1.0, 2.0],
            "y1": [0, 1],
            "y2": [0, 2],
        })
        with pytest.raises(ValueError, match="n_categories"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec={"x1": {"y1": "x1", "y2": "x1"}},
                n_categories=[3],  # only 1 but 2 dep_vars
                control=MORPControl(verbose=0),
            )

    def test_missing_column_raises(self):
        df = pd.DataFrame({"x1": [1.0], "y1": [0], "y2": [1]})
        with pytest.raises(ValueError, match="not found"):
            MORPModel(
                data=df,
                dep_vars=["y1", "y2"],
                spec={"x1": {"y1": "x1", "y2": "x1"}, "missing_col": {"y1": "missing_col", "y2": "missing_col"}},
                n_categories=[3, 3],
            )


# ---------------------------------------------------------------------------
# UTA report (2026-06): reporting-space output and SE-diagnostic gating.
# ---------------------------------------------------------------------------

_REPORT_SPEC = {
    "x1": {"y1": "x1", "y2": "x1"},
    "x2": {"y1": "x2", "y2": "x2"},
}


def _make_report_df():
    rng = np.random.default_rng(7)
    n = 150
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    beta = np.array([0.5, -0.3])
    eps = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.3], [0.3, 1.0]], size=n)
    base = x1 * beta[0] + x2 * beta[1]
    y1 = np.digitize(base + eps[:, 0], [-0.5, 0.5])
    y2 = np.digitize(base + eps[:, 1], [-0.3, 0.7])
    return pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})


def _fit_synth(df, **ctrl_kw):
    model = MORPModel(
        data=df, dep_vars=["y1", "y2"], spec=_REPORT_SPEC,
        n_categories=[3, 3],
        control=MORPControl(iid=True, verbose=0, seed=42, maxiter=80, **ctrl_kw),
    )
    return model.fit()


@pytest.fixture(scope="module")
def report_fit():
    return _fit_synth(_make_report_df())


class TestMORPReportTable:
    """Issue 1+2: summary shows threshold cut-points (not raw tau/delta slots)
    with delta-method SEs and a gradient column (GAUSS BHATLIB parity)."""

    @pytest.fixture(scope="class")
    def fitted(self, report_fit):
        return report_fit

    def test_report_present_and_relabelled(self, fitted):
        assert fitted.report is not None
        assert any(n.startswith("thresh_y1_") for n in fitted.report.names)
        # raw tau_* labels must not leak into the display table
        assert not any(n.startswith("tau_") for n in fitted.report.names)

    def test_report_thresholds_match_unpack(self, fitted):
        # report estimates in each threshold block == cumulative thresholds
        idx = 2  # two betas precede the threshold blocks
        for tau_d in fitted.thresholds:
            m = len(tau_d)
            np.testing.assert_allclose(fitted.report.estimate[idx:idx + m], tau_d)
            idx += m

    def test_first_threshold_se_equals_raw(self, fitted):
        # the first threshold per dimension is the identity row of the
        # delta-method Jacobian, so its reported SE equals the raw tau_1 SE
        assert np.isfinite(fitted.report.se[2])
        np.testing.assert_allclose(fitted.report.se[2], fitted.se[2], rtol=1e-6)

    def test_gradient_column_present_and_small(self, fitted):
        assert len(fitted.report.gradient) == len(fitted.report.names)
        assert np.nanmax(np.abs(fitted.report.gradient)) < 0.1

    def test_summary_and_dataframe_show_thresholds(self, fitted):
        txt = fitted.summary()
        assert "Gradient" in txt and "thresh_y1_1" in txt and "tau_" not in txt
        dfres = fitted.to_dataframe()
        assert "Gradient" in dfres.columns
        assert any(str(ix).startswith("thresh_") for ix in dfres.index)


class TestMORPSEDiagnosticOptional:
    """Issue 4: the 3-estimator SE diagnostic is opt-in via se_diagnostic."""

    def test_diagnostic_off_by_default(self, report_fit):
        r = report_fit  # default se_diagnostic=False
        assert r.se_bhhh is not None          # primary still computed
        assert r.se_hessian is None and r.se_sandwich is None
        assert "alternative estimators" not in r.summary()

    def test_diagnostic_on_populates_all_three(self):
        r = _fit_synth(_make_report_df(), se_diagnostic=True)
        assert r.se_bhhh is not None
        assert r.se_hessian is not None
        assert r.se_sandwich is not None
        assert "alternative estimators" in r.summary()


class TestMORPFixScalesDefault:
    """Issue 3: full-covariance MORP defaults to unit-variance identification
    (fix_scales=True), matching GAUSS BHATLIB MORP (no free latent scales)."""

    def test_default_is_unit_variance(self):
        assert MORPControl(iid=False).fix_scales is True

    def test_full_cov_param_count_matches_gauss(self):
        # 2 betas + (2+2) thresholds + 0 scales + 1 corr = 7 (not 8 w/ a scale)
        n = count_morp_params(2, 2, [3, 3], MORPControl(iid=False))
        assert n == 7
        n_free = count_morp_params(2, 2, [3, 3],
                                   MORPControl(iid=False, fix_scales=False))
        assert n_free == 8  # legacy layout keeps the (D-1)=1 free scale


# ---------------------------------------------------------------------------
# UTA follow-up report (2026-06): corr_* rows must equal the printed
# correlation matrix (not the raw spherical-angle / atanh parameters).
# ---------------------------------------------------------------------------

def _make_corr_report_df():
    rng = np.random.default_rng(7)
    n = 250
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    beta = np.array([0.6, -0.4])
    rho = np.array([[1.0, 0.45, 0.2], [0.45, 1.0, -0.3], [0.2, -0.3, 1.0]])
    eps = (np.linalg.cholesky(rho) @ rng.standard_normal((3, n))).T
    base = x1 * beta[0] + x2 * beta[1]
    taus = [[-0.6, 0.7], [-0.4, 0.9], [-0.5, 0.6]]
    cols = {"x1": x1, "x2": x2}
    for d, t in enumerate(taus):
        cols[f"d{d + 1}"] = np.digitize(base + eps[:, d], t)
    return pd.DataFrame(cols)


def _fit_corr(spherical=True):
    df = _make_corr_report_df()
    spec = {"x1": {f"d{d}": "x1" for d in (1, 2, 3)},
            "x2": {f"d{d}": "x2" for d in (1, 2, 3)}}
    model = MORPModel(
        data=df, dep_vars=["d1", "d2", "d3"], spec=spec, n_categories=[3, 3, 3],
        control=MORPControl(iid=False, spherical=spherical, verbose=0,
                            seed=1, maxiter=150),
    )
    return model.fit()


class TestMORPCorrReporting:
    """corr_* report rows show the actual correlation entries with finite
    delta-method SEs, matching the 'Estimated error correlation matrix' block."""

    @pytest.fixture(scope="class")
    def fitted(self):
        return _fit_corr(spherical=True)

    def _corr_rows(self, r):
        return [(k, nm) for k, nm in enumerate(r.report.names)
                if nm.startswith("corr_")]

    def test_corr_rows_equal_correlation_matrix(self, fitted):
        rows = self._corr_rows(fitted)
        assert len(rows) == 3  # 3 dims -> 3 off-diagonal correlations
        C = fitted.correlation_matrix
        offdiag = [C[i, j] for i in range(3) for j in range(i + 1, 3)]
        reported = [fitted.report.estimate[k] for k, _ in rows]
        # exact match: both come from the same theta_to_corr transform
        np.testing.assert_allclose(reported, offdiag, atol=1e-12)

    def test_corr_rows_are_valid_correlations(self, fitted):
        for k, _ in self._corr_rows(fitted):
            est = fitted.report.estimate[k]
            assert -1.0 < est < 1.0          # a correlation, not a raw angle
            assert np.isfinite(fitted.report.se[k]) and fitted.report.se[k] > 0

    def test_gradient_small_at_optimum(self, fitted):
        assert np.nanmax(np.abs(fitted.report.gradient)) < 0.1

    def test_direct_parameterization_also_matches(self):
        r = _fit_corr(spherical=False)
        C = r.correlation_matrix
        offdiag = [C[i, j] for i in range(3) for j in range(i + 1, 3)]
        reported = [r.report.estimate[k] for k, nm in enumerate(r.report.names)
                    if nm.startswith("corr_")]
        np.testing.assert_allclose(reported, offdiag, atol=1e-12)


# ---------------------------------------------------------------------------
# UTA follow-up report (2026-06), issue #3: compute ATE from user-input final
# coefficients (betas / threshold cut-points / correlation) without re-fitting,
# matching the GAUSS "plug in est" workflow.
# ---------------------------------------------------------------------------

def _ate_X(df, n_dims, var_cols):
    n = len(df)
    X = np.zeros((n, n_dims, len(var_cols)))
    block = df[var_cols].to_numpy(dtype=float)
    for d in range(n_dims):
        X[:, d, :] = block
    return X


class TestMORPFromEstimatesATE:
    """from_estimates + morp_ate / morp_joint_probs reproduce a fit's ATE from
    the reported coefficients alone (no re-optimisation)."""

    @pytest.fixture(scope="class")
    def setup(self):
        from pybhatlib.models.morp import morp_ate, morp_joint_probs
        r = _fit_corr(spherical=True)
        df = _make_corr_report_df()
        X = _ate_X(df, 3, ["x1", "x2"])
        return r, X, morp_ate, morp_joint_probs

    def test_round_trip_marginal_ate_is_exact(self, setup):
        from pybhatlib.models.morp import MORPResults
        r, X, morp_ate, _ = setup
        r2 = MORPResults.from_estimates(
            r.params[:2], r.thresholds, r.correlation_matrix,
            dep_vars=["d1", "d2", "d3"], n_categories=[3, 3, 3],
        )
        a1 = morp_ate(r, X, 3, [3, 3, 3], 2)
        a2 = morp_ate(r2, X, 3, [3, 3, 3], 2)
        for d in range(3):
            np.testing.assert_allclose(
                a1.predicted_probs[d], a2.predicted_probs[d], atol=1e-12
            )

    def test_from_estimates_recovers_thresholds_and_corr(self, setup):
        from pybhatlib.models.morp import MORPResults
        r, _, _, _ = setup
        r2 = MORPResults.from_estimates(
            r.params[:2], r.thresholds, r.correlation_matrix,
            dep_vars=["d1", "d2", "d3"], n_categories=[3, 3, 3],
        )
        for a, b in zip(r2.thresholds, r.thresholds):
            np.testing.assert_allclose(a, b, atol=1e-10)
        np.testing.assert_allclose(
            r2.correlation_matrix, r.correlation_matrix, atol=1e-10
        )

    def test_joint_probs_sum_to_one(self, setup):
        r, X, _, morp_joint_probs = setup
        J = morp_joint_probs(r, X, 3, [3, 3, 3], 2)
        assert J.combos.shape == (27, 3)
        # MVNCD approximation -> close to 1, not exact
        np.testing.assert_allclose(J.probs.sum(), 1.0, atol=1e-2)

    def test_joint_marginal_matches_direct_marginal(self, setup):
        r, X, morp_ate, morp_joint_probs = setup
        a = morp_ate(r, X, 3, [3, 3, 3], 2)
        J = morp_joint_probs(r, X, 3, [3, 3, 3], 2)
        for d in range(3):
            np.testing.assert_allclose(J.marginal(d), a.predicted_probs[d], atol=1e-2)

    def test_from_params_convenience_matches(self, setup):
        from pybhatlib.models.morp import morp_ate_from_params
        r, X, morp_ate, _ = setup
        a1 = morp_ate(r, X, 3, [3, 3, 3], 2)
        a3 = morp_ate_from_params(
            r.params[:2], r.thresholds, r.correlation_matrix, X, 3, [3, 3, 3], 2,
            dep_vars=["d1", "d2", "d3"],
        )
        for d in range(3):
            np.testing.assert_allclose(
                a1.predicted_probs[d], a3.predicted_probs[d], atol=1e-12
            )


class TestMORPFromEstimatesValidation:
    """from_estimates input validation and the IID path."""

    def test_iid_when_correlation_none(self):
        from pybhatlib.models.morp import MORPResults
        r = MORPResults.from_estimates(
            [0.5], [np.array([-0.4, 0.6])], None,
            dep_vars=["y1"], n_categories=[3],
        )
        assert r.control.iid is True
        assert r.correlation_matrix is None
        np.testing.assert_allclose(r.thresholds[0], [-0.4, 0.6])

    def test_non_increasing_thresholds_raise(self):
        from pybhatlib.models.morp import MORPResults
        with pytest.raises(ValueError, match="strictly increasing"):
            MORPResults.from_estimates(
                [0.5], [np.array([0.6, -0.4])], None,
                dep_vars=["y1"], n_categories=[3],
            )

    def test_out_of_range_correlation_raises(self):
        from pybhatlib.models.morp import MORPResults
        with pytest.raises(ValueError, match="must be in"):
            MORPResults.from_estimates(
                [0.5, 0.3], [np.array([0.0]), np.array([0.0])],
                np.array([[1.0, 1.4], [1.4, 1.0]]),
                dep_vars=["y1", "y2"], n_categories=[2, 2],
            )

    def test_asymmetric_correlation_raises(self):
        """A non-symmetric correlation matrix must raise, not silently read the
        upper triangle (PR #30: a sign-flipped lower-triangle entry produced
        wrong joint ATE probabilities while the marginals looked fine)."""
        from pybhatlib.models.morp import MORPResults
        # Upper [0,2]=+0.4 disagrees with lower [2,0]=-0.4 (Anna's exact bug).
        corr = np.array([
            [1.0,  0.3,  0.4],
            [0.3,  1.0, -0.2],
            [-0.4, -0.2, 1.0],
        ])
        with pytest.raises(ValueError, match="symmetric"):
            MORPResults.from_estimates(
                [0.5, 0.3, 0.1],
                [np.array([0.0]), np.array([0.0]), np.array([0.0])],
                corr, dep_vars=["y1", "y2", "y3"], n_categories=[2, 2, 2],
            )
        # The error message should name the offending [i, j] entry.
        with pytest.raises(ValueError, match=r"\[0\s*,\s*2\]|\[2\s*,\s*0\]"):
            MORPResults.from_estimates(
                [0.5, 0.3, 0.1],
                [np.array([0.0]), np.array([0.0]), np.array([0.0])],
                corr, dep_vars=["y1", "y2", "y3"], n_categories=[2, 2, 2],
            )

    def test_symmetric_correlation_accepted(self):
        """A properly symmetric matrix passes the guard unchanged."""
        from pybhatlib.models.morp import MORPResults
        corr = np.array([
            [1.0,  0.3,  -0.4],
            [0.3,  1.0,  -0.2],
            [-0.4, -0.2,  1.0],
        ])
        r = MORPResults.from_estimates(
            [0.5, 0.3, 0.1],
            [np.array([0.0]), np.array([0.0]), np.array([0.0])],
            corr, dep_vars=["y1", "y2", "y3"], n_categories=[2, 2, 2],
        )
        np.testing.assert_allclose(r.correlation_matrix, corr, atol=1e-9)


# ---------------------------------------------------------------------------
# UTA follow-up report (2026-06), issue #2: BHHH per-observation scores are
# computed analytically (single pass) instead of by 2*n_params finite-difference
# passes — the dominant post-convergence cost for non-independent MORP.
# ---------------------------------------------------------------------------

class TestMORPAnalyticScores:
    """Analytic per-obs scores match finite differences and yield identical
    BHHH standard errors, at a fraction of the cost."""

    @pytest.fixture(scope="class")
    def model_and_theta(self):
        df = _make_corr_report_df()
        spec = {"x1": {f"d{d}": "x1" for d in (1, 2, 3)},
                "x2": {f"d{d}": "x2" for d in (1, 2, 3)}}
        model = MORPModel(
            data=df, dep_vars=["d1", "d2", "d3"], spec=spec,
            n_categories=[3, 3, 3],
            control=MORPControl(iid=False, verbose=0, seed=1, maxiter=120),
        )
        r = model.fit()
        return model, r.params

    def _fd_scores(self, model, theta, eps=1e-6):
        from pybhatlib.models.morp._morp_loglik import _per_obs_loglik
        from pybhatlib.backend._array_api import get_backend
        xp = get_backend("numpy")
        S = np.zeros((model.N, len(theta)))
        for k in range(len(theta)):
            tp = theta.copy(); tp[k] += eps
            tm = theta.copy(); tm[k] -= eps
            args = (model.X, model.y, model.n_dims, model.n_categories,
                    model.n_beta, model.control, xp)
            S[:, k] = (_per_obs_loglik(tp, *args) - _per_obs_loglik(tm, *args)) / (2 * eps)
        return S

    def test_analytic_scores_match_fd(self, model_and_theta):
        model, theta = model_and_theta
        S_an = model._per_obs_scores(theta)        # analytic fast path
        S_fd = self._fd_scores(model, theta)
        assert S_an.shape == (model.N, len(theta))
        np.testing.assert_allclose(S_an, S_fd, atol=1e-5)

    def test_bhhh_se_unchanged(self, model_and_theta):
        model, theta = model_and_theta
        se = lambda S: np.sqrt(np.diag(np.linalg.inv(S.T @ S)))
        se_an = se(model._per_obs_scores(theta))
        se_fd = se(self._fd_scores(model, theta))
        np.testing.assert_allclose(se_an, se_fd, rtol=1e-5)

    def test_analytic_grad_per_obs_sums_to_total(self):
        # per-obs scores sum to the total (un-normalised) log-likelihood gradient
        from pybhatlib.models.morp._morp_grad_analytic import morp_analytic_gradient
        model, theta = None, None
        df = _make_corr_report_df()
        spec = {"x1": {f"d{d}": "x1" for d in (1, 2, 3)},
                "x2": {f"d{d}": "x2" for d in (1, 2, 3)}}
        m = MORPModel(data=df, dep_vars=["d1", "d2", "d3"], spec=spec,
                      n_categories=[3, 3, 3],
                      control=MORPControl(iid=False, verbose=0, seed=1, maxiter=1))
        theta = m._default_start_values()
        nll, grad, scores = morp_analytic_gradient(
            theta, m.X, m.y, m.n_dims, m.n_categories, m.n_beta, m.control,
            return_per_obs=True,
        )
        # grad is the mean NLL gradient = -(1/N) sum_i score_i
        np.testing.assert_allclose(grad, -scores.mean(axis=0), atol=1e-10)
