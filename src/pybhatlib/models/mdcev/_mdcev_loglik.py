"""MDCEV log-likelihood and analytic gradient.

Direct Python translation of the ``lpr``, ``lgd``, ``lpr1``, and ``lgd1``
procedures from estimation_TradMDCEV.gss and Estimation_LinearMDCEV.gss
(Bhat, 2008 / 2018).

The two specifications differ in exactly two places, controlled by
``MDCEVControl.utility``:

Traditional (``utility="trad"``)
    - Jacobian scalar ``c`` is computed over all nc alternatives
      (including the outside good) using the full quantity+gamma term.
    - ``vdisc`` and ``vcont`` subtract ``ln(qty_outside)`` to form the
      differenced utility ratios.

Linear (``utility="linear"``)
    - Jacobian scalar ``c`` is computed over inside goods only
      (nc-1 alternatives) using consumed quantities and gammas.
    - ``vdisc`` and ``vcont`` are **not** adjusted by ``ln(qty_outside)``.

All mvlogit probability calls (``nonpdfcdfmvlogit``, ``noncdfmvlogit``,
``nonpdfmvlogit``) and the entire pdisc/pcont loop are identical across
both specifications.

Parameter vector layout
-----------------------
    x = [ beta (nvarm,) | gamma_raw (nvargam,) | log_sigma (1,) ]

``eqmatgam`` (shape nvargam x nvargam, identity by default) maps raw
gamma parameters to per-alternative gamma values:

    xgam = eqmatgam.T @ x[nvarm : nvarm + nvargam]

The outside-good gamma is forced to ``MDCEVControl.outside_good_gamma``
(default -1000) so the outside good carries no satiation parameter.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pybhatlib.models.mdcev._mdcev_control import MDCEVControl
from pybhatlib.gradmvn._logit_dist import (
    nonpdfmvlogit,
    gradnonpdfmvlogit,
    nonpdfcdfmvlogit,
    gradnonpdfcdfmvlogit,
    noncdfmvlogit,
    gradnoncdfmvlogit,
)


# ---------------------------------------------------------------------------
# Shared intermediate quantities
# ---------------------------------------------------------------------------

def _compute_utility_terms(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> dict:
    """Compute shared intermediate quantities for lpr/lgd.

    Implements the ``v``, ``u``, ``f``, ``b``, ``m``, ``h``, ``c``,
    ``vdisc``, ``vcont`` blocks that appear identically in both
    (the only difference between trad and
    linear is the ``c`` and ``vdisc``/``vcont`` computations, handled
    here via ``control.utility``).

    Parameters
    ----------
    x : NDArray, shape (nvarm + nvargam + 1,)
        Parameter vector; last element is log(sigma).
    dta : NDArray, shape (e1, n_cols)
        Data batch.
    ivm : NDArray, shape (nc * nvarm,)
        Column indices into ``dta`` for baseline utility variables,
        row-major over alternatives.
    ivg : NDArray, shape (nc * nvargam,)
        Column indices into ``dta`` for satiation (gamma) variables,
        row-major over alternatives.
    flagchm : NDArray, shape (nc,)
        Column indices of consumption quantities.
    flagprcm : NDArray, shape (nc,)
        Column indices of price variables.
    wtind : int
        Column index of observation weights.
    nvarm : int
        Number of baseline utility variables.
    nvargam : int
        Number of satiation (gamma) variables.
    nc : int
        Total number of alternatives including outside good.
    eqmatgam : NDArray, shape (nvargam, nvargam)
        Equality/restriction matrix for gamma parameters (identity by
        default, matching ``eqmatgam = eye(nvargam)`` in GAUSS).
    control : MDCEVControl

    Returns
    -------
    dict
        All intermediate arrays needed by the log-likelihood and gradient.
    """
    e1 = dta.shape[0]

    wt    = dta[:, wtind]                                       # (e1,)
    xgam  = eqmatgam.T @ x[nvarm: nvarm + nvargam]             # (nvargam,)
    xsigm = np.exp(x[nvarm + nvargam])                         # scalar

    # Baseline utilities v[q, k] = X_{qk} @ beta, shape (e1, nc)
    # GAUSS: v2 = (ones(nc,1) .*. x[1:nvarm]) *~ (dta[.,ivm])'
    # ivm is column-major: [cols for param 0 all alts, cols for param 1 all alts, ...]
    beta = x[:nvarm]
    v = np.zeros((e1, nc), dtype=np.float64)
    for j in range(nvarm):
        cols_j = ivm[j * nc: (j + 1) * nc]
        v[:, :] += dta[:, cols_j] * beta[j]

    # Satiation utility u[q, k] = X_gam_{qk} @ xgam, shape (e1, nc)
    # ivg is column-major: [cols for param 0 all alts, cols for param 1 all alts, ...]
    u = np.zeros((e1, nc), dtype=np.float64)
    for j in range(nvargam):
        cols_j = ivg[j * nc: (j + 1) * nc]
        u[:, :] += dta[:, cols_j] * xgam[j]

    # Outside good gamma forced to large negative value.
    # GAUSS: u[.,1] = -1000*ones(e1,1)
    u[:, 0] = control.outside_good_gamma

    # gamma_k = exp(u_k) for inside goods, shape (e1, nc-1)
    f    = np.exp(u[:, 1:])                                     # (e1, nc-1)

    # Consumption quantities and indicator for inside goods
    qty_inside = dta[:, flagchm[1:]]                            # (e1, nc-1)
    b    = (qty_inside > 0).astype(np.float64)                  # (e1, nc-1)
    newf = qty_inside.copy()                                    # (e1, nc-1)

    m = 1 + b.sum(axis=1)                                       # (e1,)  consumed alts
    h = nc - m                                                  # (e1,)  non-consumed inside

    price_all    = dta[:, flagprcm]                             # (e1, nc)
    price_inside = price_all[:, 1:]                             # (e1, nc-1)

    # ------------------------------------------------------------------
    # Jacobian scalar c — the only computation that differs by utility
    # ------------------------------------------------------------------
    if control.utility == "trad":
        # GAUSS trad lpr:
        #   c = 1/((dta[.,flagchm]+exp(u)))           shape (e1, nc)
        #   c1 = sumc(((price./c).*(1~b))')
        #   c = substute(c, (1~b)==0, 1); c = prodc(c')
        #   c = (1/p_0) * c * c1
        qty_all = dta[:, flagchm]                               # (e1, nc)
        c_raw   = 1.0 / (qty_all + np.exp(u))                  # (e1, nc)
        consumed_mask = np.column_stack([np.ones(e1), b])       # (e1, nc)  1 for outside + consumed
        c1      = (price_all / c_raw * consumed_mask).sum(axis=1)  # (e1,)
        c_prod  = np.where(consumed_mask == 0, 1.0, c_raw).prod(axis=1)  # (e1,)
        c       = (1.0 / price_all[:, 0]) * c_prod * c1        # (e1,)
        c3      = c_raw[:, 1:]                                  # (e1, nc-1)  for gradient
        c2      = c_prod / price_all[:, 0]                      # (e1,)       for gradient

    else:  # "linear"
        # GAUSS linear lpr:
        #   c = b/((newf+f))                          shape (e1, nc-1)
        #   c = c / price_inside
        #   c = substute(c, b==0, 1); c = prodc(c')
        c_inside = b / (newf + f)                               # (e1, nc-1)
        c_inside = c_inside / price_inside
        c_prod   = np.where(b == 0, 1.0, c_inside).prod(axis=1)  # (e1,)
        c        = c_prod                                       # (e1,)
        c3       = None                                         # unused for linear
        c2       = None                                         # unused for linear

    # ------------------------------------------------------------------
    # Price-adjusted utility differences — vdisc and vcont
    # The only other difference: trad subtracts ln(qty_outside)
    # ------------------------------------------------------------------
    qty_out = dta[:, flagchm[0]]                                # (e1,)

    vdisc_raw = v[:, 1:] - np.log(price_inside)                # (e1, nc-1)

    v_adj = v.copy()
    v_adj[:, 0] -= np.log(price_all[:, 0])
    v_adj[:, 1:] -= (
        np.log((qty_inside + f) / f) + np.log(price_inside)
    )

    if control.utility == "trad":
        # GAUSS trad: vdisc = v[.,1] - vdisc_raw - ln(qty_outside)
        #             vcont = v[.,1] - v[.,2:nc] - ln(qty_outside)
        log_q0   = np.log(qty_out)[:, np.newaxis]               # (e1, 1)
        vdisc    = v_adj[:, 0:1] - vdisc_raw - log_q0          # (e1, nc-1)
        vcont    = v_adj[:, 0:1] - v_adj[:, 1:] - log_q0       # (e1, nc-1)
    else:
        # GAUSS linear: vdisc = v[.,1] - vdisc_raw  (no log_q0)
        vdisc    = v_adj[:, 0:1] - vdisc_raw                   # (e1, nc-1)
        vcont    = v_adj[:, 0:1] - v_adj[:, 1:]                # (e1, nc-1)

    return dict(
        e1=e1, wt=wt, xsigm=xsigm,
        f=f, b=b, newf=newf,
        m=m, h=h, c=c, c2=c2, c3=c3,
        vdisc=vdisc, vcont=vcont,
        price_inside=price_inside,
    )


# ---------------------------------------------------------------------------
# Log-likelihood  (GAUSS: proc lpr)
# ---------------------------------------------------------------------------

def mdcev_loglik(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> NDArray:
    """Per-observation weighted log-likelihood for the MDCEV model.

    Implements the ``lpr`` procedure from both GAUSS source files.
    The ``control.utility`` flag selects between the traditional and
    linear outside-good specifications.

    Parameters
    ----------
    x : NDArray, shape (nvarm + nvargam + 1,)
        Parameter vector; last element is log(sigma).
    dta : NDArray, shape (e1, n_cols)
        Data batch.
    ivm : NDArray, shape (nc * nvarm,)
        Baseline utility variable column indices, row-major over alts.
    ivg : NDArray, shape (nc * nvargam,)
        Satiation variable column indices, row-major over alts.
    flagchm : NDArray, shape (nc,)
        Column indices of consumption quantities.
    flagprcm : NDArray, shape (nc,)
        Column indices of price variables.
    wtind : int
        Column index of observation weights.
    nvarm : int
        Number of baseline utility variables.
    nvargam : int
        Number of satiation variables.
    nc : int
        Total number of alternatives including outside good.
    eqmatgam : NDArray, shape (nvargam, nvargam)
        Gamma restriction/equality matrix.
    control : MDCEVControl

    Returns
    -------
    ll_obs : NDArray, shape (e1,)
        Weighted log-likelihood contribution for each observation.
    """
    terms  = _compute_utility_terms(
        x, dta, ivm, ivg, flagchm, flagprcm, wtind,
        nvarm, nvargam, nc, eqmatgam, control,
    )
    e1     = terms["e1"]
    wt     = terms["wt"]
    xsigm  = terms["xsigm"]
    b      = terms["b"]
    m      = terms["m"]
    h      = terms["h"]
    c      = terms["c"]
    vdisc  = terms["vdisc"]
    vcont  = terms["vcont"]

    pdisc = np.zeros(e1, dtype=np.float64)
    pcont = np.zeros(e1, dtype=np.float64)

    for i in range(e1):
        con_mask  = b[i].astype(bool)
        nonc_mask = ~con_mask
        n_con  = int(m[i] - 1)   # number of consumed inside goods
        n_nonc = int(h[i])        # number of non-consumed inside goods

        if n_nonc > 0:
            vtildek1_noncon = vdisc[i, nonc_mask].reshape(-1, 1)
            mu_noncon  = np.zeros((n_nonc, 1), dtype=np.float64)
            sig_noncon = xsigm * np.ones((n_nonc, 1), dtype=np.float64)
        if n_con > 0:
            vtildek1_consum = vdisc[i, con_mask].reshape(-1, 1)
            vdarrok1_consum = vcont[i, con_mask].reshape(-1, 1)
            mu_con  = np.zeros((n_con, 1), dtype=np.float64)
            sig_con = xsigm * np.ones((n_con, 1), dtype=np.float64)

        if n_nonc > 0 and n_con > 0:
            # Some inside goods consumed, some not
            mu_mixed  = np.zeros((n_con + n_nonc, 1), dtype=np.float64)
            sig_mixed = xsigm * np.ones((n_con + n_nonc, 1), dtype=np.float64)
            pdisc[i] = 1.0
            pcont[i] = c[i] * nonpdfcdfmvlogit(
                vdarrok1_consum, vtildek1_noncon, mu_mixed, sig_mixed,
            ).item()
        elif n_con == 0:
            # No inside goods consumed — pure outside-good observation
            pdisc[i] = noncdfmvlogit(vtildek1_noncon, mu_noncon, sig_noncon).item()
            pcont[i] = 1.0
        else:
            # All inside goods consumed (h[i] == 0)
            pdisc[i] = 1.0
            pcont[i] = c[i] * nonpdfmvlogit(vdarrok1_consum, mu_con, sig_con).item()

    z = pdisc * pcont
    z = np.where(z <= 0.0, 1e-4, z)
    return wt * np.log(z)


# ---------------------------------------------------------------------------
# Analytic gradient  (GAUSS: proc lgd)
# ---------------------------------------------------------------------------

def mdcev_gradient(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> NDArray:
    """Per-observation weighted analytic gradient of the MDCEV log-likelihood.

    Implements the ``lgd`` / ``lgd1`` procedures from both GAUSS source
    files.  The sigma derivative is computed in log-space (chain rule:
    d/d log_sigma = sigma * d/d sigma).

    The only branching between traditional and linear specifications is
    the ``ggam2`` computation: trad uses the fuller ``c2``/``c3``
    intermediates while linear uses the simpler single-term formula.

    Parameters
    ----------
    x : NDArray, shape (nvarm + nvargam + 1,)
    dta : NDArray, shape (e1, n_cols)
    ivm, ivg, flagchm, flagprcm, wtind : see ``mdcev_loglik``
    nvarm, nvargam, nc : see ``mdcev_loglik``
    eqmatgam : NDArray, shape (nvargam, nvargam)
    control : MDCEVControl

    Returns
    -------
    grad_obs : NDArray, shape (e1, nvarm + nvargam + 1)
        Weighted per-observation gradient contributions.
    """
    terms         = _compute_utility_terms(
        x, dta, ivm, ivg, flagchm, flagprcm, wtind,
        nvarm, nvargam, nc, eqmatgam, control,
    )
    e1            = terms["e1"]
    wt            = terms["wt"]
    xsigm         = terms["xsigm"]
    f             = terms["f"]
    b             = terms["b"]
    newf          = terms["newf"]
    m             = terms["m"]
    h             = terms["h"]
    c             = terms["c"]
    c2            = terms["c2"]                                 # trad only
    c3            = terms["c3"]                                 # trad only
    vdisc         = terms["vdisc"]
    vcont         = terms["vcont"]
    price_inside  = terms["price_inside"]

    mu_zero = np.zeros((nc - 1, 1), dtype=np.float64)   # kept for reference; not used directly
    sig_vec = xsigm * np.ones((nc - 1, 1), dtype=np.float64)

    pdisc    = np.zeros(e1)
    pcont    = np.zeros(e1)
    gcont    = np.zeros((e1, nc - 1))
    gdisc    = np.zeros((e1, nc - 1))
    gsigcont = np.zeros((e1, nc - 1))
    gsigdisc = np.zeros((e1, nc - 1))
    ggam2    = np.zeros((e1, nc - 1))

    for i in range(e1):
        con_mask  = b[i].astype(bool)
        nonc_mask = ~con_mask
        idx_con   = np.where(con_mask)[0]
        idx_nonc  = np.where(nonc_mask)[0]
        n_con     = idx_con.size
        n_nonc    = idx_nonc.size

        if n_nonc > 0:
            vtildek1_noncon = vdisc[i, nonc_mask].reshape(-1, 1)
            mu_noncon  = np.zeros((n_nonc, 1), dtype=np.float64)
            sig_noncon = xsigm * np.ones((n_nonc, 1), dtype=np.float64)
        if n_con > 0:
            vtildek1_consum = vdisc[i, con_mask].reshape(-1, 1)
            vdarrok1_consum = vcont[i, con_mask].reshape(-1, 1)
            mu_con  = np.zeros((n_con, 1), dtype=np.float64)
            sig_con = xsigm * np.ones((n_con, 1), dtype=np.float64)

        if n_nonc > 0 and n_con > 0:
            mu_mixed  = np.zeros((n_con + n_nonc, 1), dtype=np.float64)
            sig_mixed = xsigm * np.ones((n_con + n_nonc, 1), dtype=np.float64)
            pdisc[i] = 1.0
            pcont[i] = c[i] * nonpdfcdfmvlogit(
                vdarrok1_consum, vtildek1_noncon, mu_mixed, sig_mixed,
            ).item()

            # ggam2: gradient of pcont w.r.t. gamma utility index
            # trad GAUSS: (pcont.*(-c3) + c2.*prr.*(pcont/c)).*f
            # linear GAUSS: (-pcont.*((newf+f)^-1)).*f
            if control.utility == "trad":
                ggam2[i, idx_con] = (
                    pcont[i] * (-c3[i, idx_con])
                    + c2[i] * price_inside[i, idx_con] * (pcont[i] / c[i])
                ) * f[i, idx_con]
            else:
                ggam2[i, idx_con] = (
                    -pcont[i] / (newf[i, idx_con] + f[i, idx_con])
                ) * f[i, idx_con]

            # GAUSS: gsigdisc[i, indxcon]  = gsigconti  (from gradnonpdfcdfmvlogit, output 5)
            #        gsigdisc[i, indxnoncon] = gsigdisci  (from gradnonpdfcdfmvlogit, output 6)
            gconti, gdisci, _, _, gsigconti, gsigdisci = gradnonpdfcdfmvlogit(
                vdarrok1_consum, vtildek1_noncon, mu_mixed, sig_mixed,
            )
            gcont[i, idx_con]    = gconti.ravel()
            gdisc[i, idx_nonc]   = gdisci.ravel()
            gsigdisc[i, idx_con] = gsigconti.ravel()
            gsigdisc[i, idx_nonc]= gsigdisci.ravel()

        elif n_con == 0:
            # No inside goods consumed — pure outside-good observation
            pdisc[i] = noncdfmvlogit(vtildek1_noncon, mu_noncon, sig_noncon).item()
            pcont[i] = 1.0
            gdisci, _, gsigdisci = gradnoncdfmvlogit(
                vtildek1_noncon, mu_noncon, sig_noncon,
            )
            gdisc[i, idx_nonc]    = gdisci.ravel()
            gsigdisc[i, idx_nonc] = gsigdisci.ravel()

        else:
            # h[i] == 0: all inside goods consumed
            pdisc[i] = 1.0
            pcont[i] = c[i] * nonpdfmvlogit(vdarrok1_consum, mu_con, sig_con).item()

            if control.utility == "trad":
                ggam2[i, :] = (
                    pcont[i] * (-c3[i, :])
                    + c2[i] * price_inside[i, :] * (pcont[i] / c[i])
                ) * f[i, :]
            else:
                ggam2[i, :] = (
                    -pcont[i] / (newf[i, :] + f[i, :])
                ) * f[i, :]

            gconti, _, gsigconti = gradnonpdfmvlogit(
                vdarrok1_consum, mu_con, sig_con,
            )
            gcont[i, idx_con]    = gconti.ravel()
            gsigdisc[i, idx_con] = gsigconti.ravel()

    # ------------------------------------------------------------------
    # Aggregate gradients to parameter space
    # GAUSS: ggv = c.*((sumc(gdisc')+sumc(gcont'))~(-(gdisc+gcont)))
    #        gsiggg = (c.*(sumc(gsigdisc')+sumc(gsigcont')))*xsigm
    #        ggam = (zeros(e1,1))~(c.*(-(newf./f)./(newf+f)).*gcont.*f+ggam2)
    # ------------------------------------------------------------------
    sum_gd_gc = (gdisc + gcont).sum(axis=1, keepdims=True)     # (e1, 1)
    ggv = c[:, np.newaxis] * np.hstack([sum_gd_gc, -(gdisc + gcont)])  # (e1, nc)

    gsiggg = c * (gsigdisc + gsigcont).sum(axis=1) * xsigm              # (e1,)

    gamma_cont_term = (
        c[:, np.newaxis]
        * (-(newf / f) / (newf + f))
        * gcont
    ) * f + ggam2                                               # (e1, nc-1)
    ggam = np.hstack([np.zeros((e1, 1)), gamma_cont_term])      # (e1, nc)

    # Map back to beta parameter space via design matrices
    # GAUSS: g2v = ones(1,nvarm) .*. ggv'; gv = reshape(sumc(...))'
    # ivm is column-major: [cols for param 0 all alts, cols for param 1 all alts, ...]
    gv = np.zeros((e1, nvarm), dtype=np.float64)
    for j in range(nvarm):
        cols_j = ivm[j * nc: (j + 1) * nc]
        # For each parameter j, sum contribution from all alternatives
        for k in range(nc):
            gv[:, j] += ggv[:, k] * dta[:, cols_j[k]]

    # Map back to gamma parameter space; apply eqmatgam chain rule
    # GAUSS: g2g = ones(1,nvargam) .*. ggam'; gg = reshape(sumc(...))'
    #        return ... gg*eqmatgam' ...
    # ivg is column-major: [cols for param 0 all alts, cols for param 1 all alts, ...]
    gg_raw = np.zeros((e1, nvargam), dtype=np.float64)
    for j in range(nvargam):
        cols_j = ivg[j * nc: (j + 1) * nc]
        # For each parameter j, sum contribution from all alternatives
        for k in range(nc):
            gg_raw[:, j] += ggam[:, k] * dta[:, cols_j[k]]
    gg = gg_raw @ eqmatgam.T                                    # (e1, nvargam)

    z = pdisc * pcont
    z = np.where(z <= 0.0, 1e-4, z)

    grad_raw = np.hstack([gv, gg, gsiggg[:, np.newaxis]])       # (e1, nvarm+nvargam+1)
    return wt[:, np.newaxis] * (grad_raw / z[:, np.newaxis])


# ---------------------------------------------------------------------------
# Unparameterized versions (sigma instead of log_sigma)
# ---------------------------------------------------------------------------


def _compute_utility_terms_unpar(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> dict:
    """Compute shared intermediate quantities for unparameterized lpr1/lgd1.

    This wrapper is identical to ``_compute_utility_terms`` except that
    ``x[-1]`` is interpreted as sigma directly rather than log(sigma).
    """
    x_log = x.copy()
    x_log[nvarm + nvargam] = np.log(x[nvarm + nvargam])
    return _compute_utility_terms(
        x_log, dta, ivm, ivg, flagchm, flagprcm, wtind,
        nvarm, nvargam, nc, eqmatgam, control,
    )


def mdcev_loglik_unpar(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> NDArray:
    """Per-observation weighted log-likelihood for the MDCEV model (unparameterized).

    This wrapper is identical to ``mdcev_loglik`` except that ``x[-1]`` is
    interpreted as sigma directly rather than log(sigma).
    """
    x_log = x.copy()
    x_log[nvarm + nvargam] = np.log(x[nvarm + nvargam])
    return mdcev_loglik(
        x_log, dta, ivm, ivg, flagchm, flagprcm, wtind,
        nvarm, nvargam, nc, eqmatgam, control,
    )


def mdcev_gradient_unpar(
    x: NDArray,
    dta: NDArray,
    ivm: NDArray,
    ivg: NDArray,
    flagchm: NDArray,
    flagprcm: NDArray,
    wtind: int,
    nvarm: int,
    nvargam: int,
    nc: int,
    eqmatgam: NDArray,
    control: MDCEVControl,
) -> NDArray:
    """Per-observation gradient for the MDCEV model (unparameterized).

    This wrapper is identical to ``mdcev_gradient`` except that ``x[-1]`` is
    interpreted as sigma directly rather than log(sigma), and the final
    sigma gradient is converted from d/d log(sigma) to d/d sigma.
    """
    x_log = x.copy()
    x_log[nvarm + nvargam] = np.log(x[nvarm + nvargam])
    grad_obs = mdcev_gradient(
        x_log, dta, ivm, ivg, flagchm, flagprcm, wtind,
        nvarm, nvargam, nc, eqmatgam, control,
    )
    grad_obs[:, -1] /= x[nvarm + nvargam]
    return grad_obs


def numerical_hessian(f, x, eps=1e-7):
    """Compute numerical Hessian via central finite differences.

    Parameters
    ----------
    f : callable
        Scalar-valued function f(x).
    x : ndarray
        Point at which to evaluate the Hessian.
    eps : float
        Perturbation size.

    Returns
    -------
    hess : ndarray
        Numerical Hessian, shape (n, n).
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    hess = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i, n):  # Only compute upper triangle
            eps_i = eps * (1.0 + abs(x[i]))
            eps_j = eps * (1.0 + abs(x[j]))

            # Central difference for H[i,j]
            x_pp = x.copy(); x_pp[i] += eps_i; x_pp[j] += eps_j
            x_pm = x.copy(); x_pm[i] += eps_i; x_pm[j] -= eps_j
            x_mp = x.copy(); x_mp[i] -= eps_i; x_mp[j] += eps_j
            x_mm = x.copy(); x_mm[i] -= eps_i; x_mm[j] -= eps_j

            hess[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps_i * eps_j)
            hess[j,i] = hess[i,j]  # Symmetric

    return hess