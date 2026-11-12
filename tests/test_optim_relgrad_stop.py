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
