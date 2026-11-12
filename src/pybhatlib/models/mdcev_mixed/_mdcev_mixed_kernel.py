"""Stable logit-Jacobian likelihood and analytical scores for mixed MDCEV.

Log likelihoods and log derivatives are passed to the simulation engine without
exponentiating. Raw densities/derivatives remain available for existing callers.
Satiation terms are cached across draws of each parameter evaluation.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import gammaln, logsumexp

from pybhatlib.mixed._kernel import KernelObsResult, KernelState
from pybhatlib.models.mdcev._mdcev_control import MDCEVControl


_GAUSS_ZERO_FLOOR = 1e-4  # value GAUSS substitutes for a per-draw z <= 0


class LogitJacobianKernel:
    """MDCEV kernel with inside-good gamma parameters and a log-sigma scale.

    ``eqmatgam`` maps the supplied gamma parameters to satiation coefficients.
    Outside-good satiation is fixed by ``control.outside_good_gamma``.

    ``gauss_zero_floor=True`` reproduces GAUSS's ``z <= 0 -> 1e-4`` rule: a
    per-draw probability that underflows to zero in double precision is set
    to ``1e-4`` (with zero score), so the likelihood matches GAUSS output.
    ``False`` keeps the exact log-space probability.
    """

    def __init__(self, n_util: int, nvargam: int, *,
                 control: MDCEVControl | None = None,
                 eqmatgam: NDArray | None = None,
                 gauss_zero_floor: bool = True) -> None:
        self.n_util = int(n_util)
        self.gauss_zero_floor = bool(gauss_zero_floor)
        self.nvargam = int(nvargam)
        self.control = control or MDCEVControl(utility="trad")
        self.eqmatgam = np.asarray(
            np.eye(nvargam) if eqmatgam is None else eqmatgam, dtype=np.float64,
        )
        if self.eqmatgam.shape != (nvargam, nvargam):
            raise ValueError(f"eqmatgam must be ({nvargam}, {nvargam})")
        self._sat_cache_kstate = None
        self._sat_cache_obs = None
        self._sat_all = None

    def kernel_param_names(self) -> list[str]:
        return [f"gamma{j}" for j in range(self.nvargam)] + ["logsig_ker"]

    def prepare(self, theta: NDArray, layout: Any) -> KernelState:
        theta = np.asarray(theta, dtype=np.float64)
        sl = layout.slices()
        return SimpleNamespace(
            gamma_raw=theta[sl.get("gamma", slice(0, 0))].copy(),
            log_sigma=float(theta[sl["kern"]][0]),
        )

    def _get_sat_all(self, state: KernelState, obs: Any) -> dict:
        if self._sat_cache_kstate is state and self._sat_cache_obs is obs:
            return self._sat_all
        qty = np.asarray(obs.consumption, dtype=np.float64)
        price = np.asarray(obs.price, dtype=np.float64)
        if not np.all(np.isfinite(qty)) or not np.all(np.isfinite(price)):
            raise ValueError("MDCEV consumption and prices must be finite")
        if np.any(qty < 0) or np.any(price <= 0):
            raise ValueError("MDCEV requires nonnegative consumption and positive prices")
        if self.control.utility == "trad" and np.any(qty[:, 0] <= 0):
            raise ValueError("Traditional MDCEV requires positive outside consumption")
        gamma = self.eqmatgam.T @ state.gamma_raw
        u = np.einsum("qkj,j->qk", obs.gamma_design, gamma)
        u[:, 0] = self.control.outside_good_gamma
        log_qty = np.full_like(qty, -np.inf)
        np.log(qty, out=log_qty, where=qty > 0)
        log_total = np.logaddexp(log_qty, u)
        log_price = np.log(price)
        consumed = qty > 0
        consumed[:, 0] = True
        con = consumed[:, 1:]
        frac_qty = np.exp(log_qty[:, 1:] - log_total[:, 1:])
        frac_gamma = np.exp(u[:, 1:] - log_total[:, 1:])
        if self.control.utility == "trad":
            terms = np.where(consumed, log_price + log_total, -np.inf)
            log_sum = logsumexp(terms, axis=1)
            log_c = (-log_price[:, 0]
                     - np.where(consumed, log_total, 0.0).sum(axis=1) + log_sum)
            shares = np.exp(terms - log_sum[:, None])
            jac_gamma = (shares[:, 1:] - con) * frac_gamma
        else:
            log_c = -np.where(
                con, log_total[:, 1:] + log_price[:, 1:], 0.0,
            ).sum(axis=1)
            jac_gamma = -(con * frac_gamma)
        sat = dict(
            con=con, log_c=log_c, adjustment=log_total[:, 1:] - u[:, 1:],
            log_price=log_price, log_qty_out=log_qty[:, 0],
            frac_qty=frac_qty, jac_gamma=jac_gamma,
        )
        self._sat_cache_kstate = state
        self._sat_cache_obs = obs
        self._sat_all = sat
        return sat

    def probability(self, Vsub: NDArray, obs: Any, kstate: KernelState, *,
                    rc_draw: NDArray, want_grad: bool) -> KernelObsResult:
        """Return stable log likelihoods/scores plus compatible raw derivatives."""
        v = np.asarray(Vsub, dtype=np.float64)
        n_obs, nc = v.shape
        if nc != self.n_util:
            raise ValueError(f"Vsub has {nc} alternatives; expected {self.n_util}")
        ng = self.nvargam
        rc_arr = np.asarray(rc_draw)
        n_rnd = rc_arr.shape[1] if rc_arr.ndim == 2 else 0
        zeros_rc = np.zeros((n_obs, n_rnd))
        sat = self._get_sat_all(kstate, obs)
        con = sat["con"]
        log_sigma = float(kstate.log_sigma)
        inv_sigma = np.exp(-log_sigma)
        base = v - sat["log_price"]
        vdisc = base[:, 0:1] - base[:, 1:]
        if self.control.utility == "trad":
            vdisc = vdisc - sat["log_qty_out"][:, None]
        vcont = vdisc + sat["adjustment"]
        selected = np.where(con, vcont, vdisc) * inv_sigma
        args = -selected
        log_d = logsumexp(np.column_stack([np.zeros(n_obs), args]), axis=1)
        shares = np.exp(args - log_d[:, None])
        n_con = con.sum(axis=1)
        log_l = (-n_con * log_sigma + gammaln(n_con + 1)
                 - np.where(con, selected, 0.0).sum(axis=1)
                 - (n_con + 1.0) * log_d)
        log_p = log_l + np.where(n_con > 0, sat["log_c"], 0.0)
        p = np.exp(log_p)
        zeroed = (p <= 0.0) if self.gauss_zero_floor else None
        if zeroed is not None and zeroed.any():
            log_p = np.where(zeroed, np.log(_GAUSS_ZERO_FLOOR), log_p)
            p = np.exp(log_p)
        if not want_grad:
            return KernelObsResult(
                p_obs=p, log_p_obs=log_p, dlogp_dV=np.zeros_like(v),
                dlogp_dkparams=np.zeros((n_obs, ng + 1)), dlogp_drc=zeros_rc,
            )
        slope = ((n_con + 1.0)[:, None] * shares - con) * inv_sigma
        score_v = np.column_stack([slope.sum(axis=1), -slope])
        score_sigma = (-n_con + np.where(con, selected, 0.0).sum(axis=1)
                       - (n_con + 1.0) * (shares * selected).sum(axis=1))
        gamma_util = (-np.where(con, slope, 0.0) * sat["frac_qty"]
                      + sat["jac_gamma"])
        gamma_util = np.where(n_con[:, None] > 0, gamma_util, 0.0)
        score_gamma = np.einsum(
            "qk,qkj->qj", gamma_util, np.asarray(obs.gamma_design)[:, 1:, :],
        ) @ self.eqmatgam.T
        score_kernel = np.column_stack([score_gamma, score_sigma])
        if zeroed is not None and zeroed.any():
            # The floored value is a constant: no gradient flows through it.
            score_v[zeroed] = 0.0
            score_kernel[zeroed] = 0.0
        return KernelObsResult(
            p_obs=p, log_p_obs=log_p, score_dV=score_v,
            score_dkparams=score_kernel, dlogp_dV=p[:, None] * score_v,
            dlogp_dkparams=p[:, None] * score_kernel, dlogp_drc=zeros_rc,
        )
