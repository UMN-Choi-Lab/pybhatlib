"""GAUSS relative-gradient stop for summed (mixed-model) objectives."""
import numpy as np
from scipy.optimize import rosen, rosen_der

from pybhatlib.optim._scipy_optim import minimize_scipy

N = 1000  # units summed in the objective, like individuals in the mixed engine


def _summed(x, digits=None):
    f = 1.0 + rosen(x)
    if digits is not None:  # quantize f: a noise floor like simulated LL
        f = round(f, digits)
    return N * f, N * rosen_der(x)


def test_relgrad_stop_on_per_unit_scale():
    x0 = np.array([-1.2, 1.0, 0.8])
    res = minimize_scipy(_summed, x0, tol=1e-5, verbose=0, obj_scale=N, maxiter=500)
    f, g = res.fun / N, res.grad / N
    rel = np.max(np.abs(g) * np.maximum(np.abs(res.x), 1.0)) / max(abs(f), 1.0)
    assert res.status == 99 and res.converged
    assert rel < 1e-5
    np.testing.assert_allclose(res.x, 1.0, atol=1e-3)


def test_line_search_stall_far_from_tol_is_not_converged():
    # f noise at 1e-6 per unit: the line search stalls (status 2) at a relative
    # gradient ~1e-3, 100x above tol, so it must not be reported as converged.
    res = minimize_scipy(lambda x: _summed(x, digits=6), np.zeros(3), tol=1e-5,
                         verbose=0, obj_scale=N, maxiter=500)
    assert res.status == 2
    assert not res.converged


def _scripted(monkeypatch, results):
    """Replace scipy minimize with a stub returning ``results`` in order."""
    from scipy.optimize import OptimizeResult

    import pybhatlib.optim._scipy_optim as so
    starts = []

    def fake(fun, x0, **kw):
        starts.append(np.array(x0, dtype=float))
        x, f = results[len(starts) - 1]
        return OptimizeResult(x=np.array(x, dtype=float), fun=f, jac=np.zeros(2),
                              status=0, success=True, nit=1, message="")
    monkeypatch.setattr(so, "minimize", fake)
    return starts


def test_collapsed_sd_is_reset_and_refit(monkeypatch):
    # first fit stops with log-SD at -9; the refit from the start value is better
    starts = _scripted(monkeypatch, [([2.0, -9.0], 1.0), ([2.1, 0.2], 0.5)])
    res = minimize_scipy(lambda x: (0.0, np.zeros(2)), np.array([0.0, -0.3]),
                         verbose=0, log_sd_idx=[1])
    np.testing.assert_array_equal(starts[1], [2.0, -0.3])  # only the SD is reset
    assert res.fun == 0.5 and res.x[1] == 0.2


def test_real_collapse_keeps_first_fit_and_warns(monkeypatch):
    import pytest
    _scripted(monkeypatch, [([2.0, -9.0], 1.0), ([2.0, -11.0], 1.0)])
    with pytest.warns(RuntimeWarning, match=r"SD collapsed.*\['sd\[x\]'\]"):
        res = minimize_scipy(lambda x: (0.0, np.zeros(2)), np.zeros(2), verbose=0,
                             log_sd_idx=[1], param_names=["b", "sd[x]"])
    assert res.x[1] == -9.0


def test_reset_can_be_disabled(monkeypatch):
    import pytest
    starts = _scripted(monkeypatch, [([2.0, -9.0], 1.0)])
    with pytest.warns(RuntimeWarning):
        minimize_scipy(lambda x: (0.0, np.zeros(2)), np.zeros(2), verbose=0,
                       log_sd_idx=[1], reset_collapsed_sd=False)
    assert len(starts) == 1
