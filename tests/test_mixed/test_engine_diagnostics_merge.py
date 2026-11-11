"""Log-space stability and prediction diagnostics survive the merge together."""

from dataclasses import replace

import numpy as np

from pybhatlib.mixed._kernel import KernelObsResult
from tests.test_mixed.test_engine_master import _build_estimator


def test_collecting_probabilities_preserves_log_space_likelihood_and_score(monkeypatch):
    est, layout = _build_estimator()
    est.config = replace(est.config, floor_pcomp=0.0, floor_z=0.0)

    def tiny_density(v, obs, state, *, rc_draw, want_grad):
        log_p = v[:, 0] - 1000.0
        score_v = np.zeros_like(v)
        score_v[:, 0] = 1.0
        return KernelObsResult(
            p_obs=np.exp(log_p), log_p_obs=log_p,
            dlogp_dV=np.zeros_like(v), score_dV=score_v,
            dlogp_dkparams=np.empty((len(v), 0)),
            score_dkparams=np.empty((len(v), 0)),
            dlogp_drc=np.zeros_like(rc_draw),
        )

    monkeypatch.setattr(est.kernel, "probability", tiny_density)
    theta = np.zeros(layout.n_theta)
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        expected_ll, expected_score = est.simulated_loglik(theta, want_grad=True)
        ll, score = est.simulated_loglik(theta, want_grad=True, collect_probabilities=True)
    assert np.all(np.isfinite(ll)) and np.all(np.isfinite(score))
    assert np.all(ll < -900)
    np.testing.assert_array_equal(ll, expected_ll)
    np.testing.assert_array_equal(score, expected_score)
    assert est.average_correct_probability == 0.0  # raw densities underflow, logs do not

    eps = 1e-5
    tp, tm = theta.copy(), theta.copy()
    tp[0] += eps
    tm[0] -= eps
    lp, _ = est.simulated_loglik(tp, want_grad=False)
    lm, _ = est.simulated_loglik(tm, want_grad=False)
    np.testing.assert_allclose(score[:, 0].sum(), (lp.sum() - lm.sum()) / (2 * eps), atol=1e-6)
