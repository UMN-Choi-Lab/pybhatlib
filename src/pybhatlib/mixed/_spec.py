"""Mixing specification: variable names to index masks and parameter counts.

Ports the spec-setup block of the GAUSS mixed-logit driver ``MIXMNL.gss``
(lines ~140-291), which turns the user's variable-name lists
(``var_unordnames``, ``normvar``, ``logvar``, ``yjvar``, ``varneg``,
``varpos``) into the integer counts and 0/1 index masks that the log-likelihood
procedures ``lpr``/``lgd``/``lpr1`` consume.

The random-coefficient block is ordered ``[normal | log-normal | Yeo-Johnson]``
throughout (matching GAUSS ``mixpos = mixposn|mixposlg|mixposyj``); every mask
whose columns index the random coefficients follows that order.

Notation map (GAUSS name -> field on :class:`MixingSpec`)
--------------------------------------------------------
====================  =====================  ==================================
GAUSS                 field                  meaning
====================  =====================  ==================================
``nvarm``             ``n_beta``             number of exogenous variables
``nrndnor``           ``nrndnor``            # normal random coefficients
``nrndlog``           ``nrndlog``            # log-normal random coefficients
``nrndyj``            ``nrndyj``             # Yeo-Johnson random coefficients
``nrndcoef``          ``nrndcoef``           total # random coefficients
``nrndtcor``          ``nrndtcor``           # correlation params, ``k(k-1)/2``
``nscale``            ``nscale``             # scale params (``== nrndcoef``)
``numlam``            ``numlam``             # lambda params (``== nrndcoef``)
``mixposn/lg/yj``     ``mixposn/lg/yj``      0-based var positions per type
``mixpos``            ``mixpos``             0-based var positions, all types
``indxrndvar``        ``indxrndvar``         ``(n_beta, nrndcoef)`` selection
``indxvarnonegposlog``  ``indxvarnonegposlog``  additive-inject mask ``(n_beta,)``
``indxvarnegposlog``  ``indxvarnegposlog``   multiplicative-inject mask
``poslog``/``posnolog``  ``poslog``/``posnolog``  log / non-log selectors in rc space
====================  =====================  ==================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def _indcv(names: Sequence[str], universe: Sequence[str]) -> NDArray:
    """Return the 0-based positions of ``names`` within ``universe``.

    Order-preserving port of GAUSS ``indcv(names', universe)``: for each entry
    of ``names`` the index of its (first) occurrence in ``universe``.

    Parameters
    ----------
    names : sequence of str
        Names to locate.
    universe : sequence of str
        Reference name list (the full variable list).

    Returns
    -------
    ndarray of intp, shape (len(names),)
        0-based positions.

    Raises
    ------
    ValueError
        If any name is absent from ``universe``.
    """
    lookup = {nm: i for i, nm in enumerate(universe)}
    out = np.empty(len(names), dtype=np.intp)
    for k, nm in enumerate(names):
        if nm not in lookup:
            raise ValueError(f"variable {nm!r} not found in var_names {list(universe)!r}")
        out[k] = lookup[nm]
    return out


def _parttofull(pos: NDArray, n: int) -> NDArray:
    """Selection matrix of shape ``(n, len(pos))`` with a 1 at ``[pos[k], k]``.

    Port of GAUSS ``parttofull(pos, n)``: expands a partial (0-based) index
    vector into a full 0/1 selection matrix whose ``k``-th column picks out
    row ``pos[k]``.

    Parameters
    ----------
    pos : ndarray of int, shape (m,)
        0-based row positions.
    n : int
        Number of rows in the full space.

    Returns
    -------
    ndarray, shape (n, m)
        Selection matrix (float64).
    """
    pos = np.asarray(pos, dtype=np.intp)
    mat = np.zeros((n, pos.shape[0]), dtype=np.float64)
    if pos.shape[0]:
        mat[pos, np.arange(pos.shape[0])] = 1.0
    return mat


@dataclass(frozen=True)
class MixingSpec:
    """Index masks and parameter counts for a mixed random-coefficient model.

    Built by :meth:`from_var_names`. All array fields are NumPy ``float64``
    (masks / selection matrices) or ``intp`` (position vectors); every
    random-coefficient axis follows the ``[normal | log | yj]`` ordering.

    Attributes
    ----------
    var_names : tuple of str
        Full variable-name list (``var_unordnames``), length ``n_beta``.
    n_beta : int
        Number of exogenous variables (GAUSS ``nvarm``).
    nrndnor, nrndlog, nrndyj : int
        Counts of normal / log-normal / Yeo-Johnson random coefficients.
    nrndcoef : int
        Total random coefficients (``nrndnor + nrndlog + nrndyj``).
    nrndtot : int
        Total random elements. For the plain (MNL softmax) facade this equals
        ``nrndcoef``; for the MNP kernel-aware facade the correlation is over
        the **joint** random-coefficient + differenced-kernel space, so
        ``nrndtot == nrndcoef + kernel_dim`` (see ``kernel_dim``).
    nrndtcor : int
        Number of correlation parameters, ``nrndtot * (nrndtot - 1) // 2``.
    nscale : int
        Number of scale parameters (``== nrndcoef``).
    numlam : int
        Number of lambda parameters (``== nrndcoef``).
    kernel_dim : int
        Dimension of the differenced kernel-error space (GAUSS ``nc - 1``);
        ``0`` for the plain MNL softmax facade. The joint correlation block is
        sized over ``nrndcoef + kernel_dim`` elements.
    n_kern : int
        Number of free kernel-scale parameters (GAUSS ``nc - 2``, i.e.
        ``kernel_dim - 1``); ``0`` for the plain MNL softmax facade. These are
        the parameters feeding the sum-of-squares ``wker`` reparameterization
        (see :func:`pybhatlib.mixed._reparam.wker_reparam`).
    nvarneg, nvarpos : int
        Counts of strictly-negative / strictly-positive fixed components.
    mixposn, mixposlg, mixposyj : ndarray of intp
        0-based positions (into ``var_names``) of normal / log / yj coefficients.
    mixpos : ndarray of intp, shape (nrndcoef,)
        Concatenation ``mixposn | mixposlg | mixposyj``.
    indxrndvar : ndarray, shape (n_beta, nrndcoef)
        Selection matrix mapping a random-coef vector into full var space.
    indxvarneg, indxvarpos, indxvarnegpos, indxvarnonegpos : ndarray (n_beta,)
        0/1 sign masks for the fixed-component reparameterization.
    indxvarnonegposlog : ndarray, shape (n_beta,)
        Additive-injection mask (``1`` -> ``xmunew1``, additive random coef).
    indxvarnegposlog : ndarray, shape (n_beta,)
        Multiplicative-injection mask (``1`` -> ``xmunew2``, multiplicative).
    posvarneg, posvarpos : ndarray, shape (n_beta, k)
        Selection matrices for negative / positive fixed components.
    poslog, posnolog : ndarray, shape (nrndcoef, k)
        Selectors (in random-coef space) of log / non-log coefficients;
        empty second dimension when ``nrndlog == 0``.
    actlam : ndarray, shape (numlam,)
        ``_max_active`` flags for the lambda block (``0`` normal/log, ``1`` yj).
    nord : int
        Number of ordinal outcome dimensions (MORP kernel; GAUSS ``nord``).
        ``0`` for the MNL / MNP facades. When ``nord > 0`` the joint correlation
        is sized over ``nrndtot = nrndcoef + nord`` (the ``nord`` kernel error
        dimensions, not ``nc - 1``).
    nthresh : int
        Total number of ordinal threshold parameters, ``sum(n_categories[d] - 1)``
        over the ``nord`` ordinal dimensions (GAUSS ``nthresh``). The threshold
        block leads the MORP parameter vector. ``0`` for MNL / MNP.
    numthresh : tuple of int, shape (nord,)
        Per-dimension threshold counts (``n_categories[d] - 1``); partitions the
        ``nthresh`` block into per-ordinal increment sub-blocks.
    n_categories : tuple of int, shape (nord,)
        Number of observed ordinal categories per dimension (GAUSS ``ncord``).
    nkernlam : int
        Number of Yeo-Johnson kernel-error ``lambda`` parameters (GAUSS
        ``blamker``, one per ordinal dimension). Equals ``nord`` when
        ``normker`` is ``False`` (GAUSS ``_normker == 0``), else ``0``.
    nvargam : int
        Number of MDCEV translation / satiation (``gamma``) variables (GAUSS
        ``nvargam``). ``0`` for the MNL / MNP / MORP facades (which have no
        satiation block). Purely additive bookkeeping used by the MDCEV facade
        to size the ``gamma`` block of the :class:`~pybhatlib.mixed._reparam.ParamLayout`;
        it plays no role in the random-coefficient mixing itself (gamma is a
        kernel-owned, non-mixed parameter).
    normker : bool
        ``True`` (GAUSS ``_normker == 1``) keeps the ordinal kernel errors
        standard normal (no kernel-lam block); ``False`` activates the
        ``nkernlam == nord`` YJ kernel-lam block.
    """

    var_names: tuple[str, ...]
    n_beta: int
    nrndnor: int
    nrndlog: int
    nrndyj: int
    nrndcoef: int
    nrndtot: int
    nrndtcor: int
    nscale: int
    numlam: int
    nvarneg: int
    nvarpos: int
    mixposn: NDArray = field(repr=False)
    mixposlg: NDArray = field(repr=False)
    mixposyj: NDArray = field(repr=False)
    mixpos: NDArray = field(repr=False)
    indxrndvar: NDArray = field(repr=False)
    indxvarneg: NDArray = field(repr=False)
    indxvarpos: NDArray = field(repr=False)
    indxvarnegpos: NDArray = field(repr=False)
    indxvarnonegpos: NDArray = field(repr=False)
    indxvarnonegposlog: NDArray = field(repr=False)
    indxvarnegposlog: NDArray = field(repr=False)
    posvarneg: NDArray = field(repr=False)
    posvarpos: NDArray = field(repr=False)
    poslog: NDArray = field(repr=False)
    posnolog: NDArray = field(repr=False)
    actlam: NDArray = field(repr=False)
    kernel_dim: int = 0
    n_kern: int = 0
    nord: int = 0
    nthresh: int = 0
    numthresh: tuple[int, ...] = ()
    n_categories: tuple[int, ...] = ()
    nkernlam: int = 0
    normker: bool = True
    nvargam: int = 0

    @classmethod
    def from_var_names(
        cls,
        var_names: Sequence[str],
        normvar: Sequence[str] = (),
        logvar: Sequence[str] = (),
        yjvar: Sequence[str] = (),
        varneg: Sequence[str] = (),
        varpos: Sequence[str] = (),
        kernel_dim: int = 0,
        nord: int = 0,
        n_categories: Sequence[int] = (),
        normker: bool = True,
        nvargam: int = 0,
        randdiag: bool = False,
    ) -> "MixingSpec":
        """Build a :class:`MixingSpec` from the GAUSS variable-name lists.

        Verbatim port of the ``MIXMNL.gss`` spec-setup block (lines ~151-247):
        counts, sign masks (``indxvarneg``/``indxvarpos`` and the
        ``negpos``/``nonegpos`` complements), the log-vs-additive injection
        masks (``indxvarnegposlog``/``indxvarnonegposlog``), the per-type
        positions (``mixpos*``), and the ``indxrndvar`` / ``poslog`` /
        ``posnolog`` selection matrices.

        Parameters
        ----------
        var_names : sequence of str
            Full variable-name list (``var_unordnames``); defines
            ``n_beta = len(var_names)`` and the position universe.
        normvar : sequence of str, optional
            Variables carrying a normal random coefficient.
        logvar : sequence of str, optional
            Variables carrying a log-normal random coefficient. Each must also
            appear in ``varneg`` or ``varpos`` (GAUSS requirement).
        yjvar : sequence of str, optional
            Variables carrying a Yeo-Johnson-transformed random coefficient.
        varneg : sequence of str, optional
            Fixed components constrained to a strictly negative sign.
        varpos : sequence of str, optional
            Fixed components constrained to a strictly positive sign.
        kernel_dim : int, optional
            Dimension of the differenced kernel-error space (GAUSS ``nc - 1``).
            The default ``0`` reproduces the plain MNL softmax facade
            (``nrndtot == nrndcoef``, no kernel-scale block). For the MNP
            kernel, pass ``nc - 1``; the joint correlation is then sized over
            ``nrndtot = nrndcoef + kernel_dim`` and ``n_kern = kernel_dim - 1``
            free kernel-scale parameters are declared. Mutually exclusive with
            ``nord`` (MNP vs MORP kernel).
        nord : int, optional
            Number of ordinal outcome dimensions (MORP kernel; GAUSS ``nord``).
            The default ``0`` leaves the MNL / MNP facades unchanged. When
            ``nord > 0`` the joint correlation is sized over
            ``nrndtot = nrndcoef + nord`` (the ordinal kernel errors), a leading
            threshold block of ``nthresh = sum(n_categories[d] - 1)`` parameters
            is declared, and no kernel-scale block is created (``n_kern == 0``;
            the MORP ordinal kernel fixes ``wker`` to ones). ``n_categories``
            must then have length ``nord``.
        n_categories : sequence of int, optional
            Observed ordinal category counts per dimension (GAUSS ``ncord``),
            length ``nord``; each contributes ``n_categories[d] - 1`` threshold
            parameters. Required (and only used) when ``nord > 0``.
        normker : bool, optional
            ``True`` (default; GAUSS ``_normker == 1``) keeps the ordinal kernel
            errors standard normal. ``False`` (GAUSS ``_normker == 0``) activates
            the ``nkernlam == nord`` Yeo-Johnson kernel-lam block (only meaningful
            when ``nord > 0``).
        nvargam : int, optional
            Number of MDCEV translation / satiation (``gamma``) variables (GAUSS
            ``nvargam``); default ``0`` leaves the MNL / MNP / MORP facades
            unchanged. Stored verbatim for the MDCEV facade to size the
            ``gamma`` layout block; it does not affect any mask or count derived
            below.
        randdiag : bool, optional
            If ``True``, fix the random-coefficient correlation matrix to the
            identity and omit its free parameters (GAUSS ``_randdiag``).

        Returns
        -------
        MixingSpec

        Raises
        ------
        ValueError
            If any listed name is absent from ``var_names``; if ``kernel_dim``
            or ``nord`` is negative; if both ``kernel_dim`` and ``nord`` are
            positive (a spec is either MNP-kernel or MORP-kernel, not both); or
            if ``nord > 0`` and ``len(n_categories) != nord``.
        """
        if kernel_dim < 0:
            raise ValueError(f"kernel_dim must be non-negative, got {kernel_dim}")
        if nord < 0:
            raise ValueError(f"nord must be non-negative, got {nord}")
        if kernel_dim > 0 and nord > 0:
            raise ValueError(
                "kernel_dim (MNP) and nord (MORP) are mutually exclusive; "
                f"got kernel_dim={kernel_dim}, nord={nord}"
            )
        n_categories = tuple(int(c) for c in n_categories)
        if nord > 0:
            if len(n_categories) != nord:
                raise ValueError(
                    f"n_categories must have length nord={nord}, "
                    f"got {len(n_categories)}"
                )
            if any(c < 1 for c in n_categories):
                raise ValueError(
                    f"each n_categories entry must be >= 1, got {n_categories}"
                )
        var_names = tuple(str(v) for v in var_names)
        nvarm = len(var_names)

        nvarneg = len(varneg)
        nvarpos = len(varpos)

        nrndnor = len(normvar)
        nrndlog = len(logvar)
        nrndyj = len(yjvar)
        nrndcoef = nrndnor + nrndlog + nrndyj
        # Joint rc + kernel-error correlation space. Two kernels add extra
        # dimensions to the correlation block:
        #   MNP  (MNPKERCP.gss)  -- the kernel error is differenced from the
        #                           first alternative, adding nc-1 = kernel_dim;
        #   MORP (ordered-YJ)    -- one kernel error per ordinal dimension,
        #                           adding nord (GAUSS line 396: nrndtot =
        #                           nrndcoef + nord). Mutually exclusive, so
        #                           their sum is the extra-dimension count.
        n_extra = kernel_dim + nord
        nrndtot = nrndcoef + n_extra
        nrndtcor = 0 if randdiag else nrndtot * (nrndtot - 1) // 2
        # Free kernel-scale params (GAUSS MNP nc-2): one fewer than kernel_dim
        # because the sum-of-squares reparam normalizes wker to unit norm. MORP
        # fixes wker = ones (GAUSS line 617), so it declares no kernel-scale
        # block (kernel_dim == 0 -> n_kern == 0).
        n_kern = max(kernel_dim - 1, 0)
        nscale = nrndcoef

        # --- MORP ordinal threshold + kernel-lam blocks (ordered-YJ driver) --
        # Threshold block leads the parameter vector: per ordinal dimension d,
        # n_categories[d] - 1 increment parameters (GAUSS numthresh; b vector
        # line 477: b = thresh | b1 | startrandker | ...). Kernel-lam is the
        # optional YJ kernel-error lambda block (GAUSS blamker, nord params when
        # _normker == 0).
        numthresh = tuple(c - 1 for c in n_categories)
        nthresh = int(sum(numthresh))
        nkernlam = nord if (nord > 0 and not normker) else 0

        # --- fixed-component sign masks (MIXMNL 173-206) ---------------------
        indxvarneg = np.zeros(nvarm, dtype=np.float64)
        indxvarpos = np.zeros(nvarm, dtype=np.float64)
        indxvarnegpos = np.zeros(nvarm, dtype=np.float64)
        indxvarnonegpos = np.zeros(nvarm, dtype=np.float64)
        posvarneg = np.zeros((nvarm, 0), dtype=np.float64)
        posvarpos = np.zeros((nvarm, 0), dtype=np.float64)

        posvarneg1 = _indcv(varneg, var_names)
        posvarpos1 = _indcv(varpos, var_names)
        if nvarneg > 0 and nvarpos == 0:
            posvarneg = _parttofull(posvarneg1, nvarm)
            indxvarneg[posvarneg1] = 1.0
            indxvarnegpos = indxvarneg
            indxvarnonegpos = 1.0 - indxvarneg
        elif nvarpos > 0 and nvarneg == 0:
            posvarpos = _parttofull(posvarpos1, nvarm)
            indxvarpos[posvarpos1] = 1.0
            indxvarnegpos = indxvarpos
            indxvarnonegpos = 1.0 - indxvarpos
        elif nvarpos > 0 and nvarneg > 0:
            posvarneg = _parttofull(posvarneg1, nvarm)
            posvarpos = _parttofull(posvarpos1, nvarm)
            indxvarneg[posvarneg1] = 1.0
            indxvarpos[posvarpos1] = 1.0
            indxvarnegpos = indxvarneg + indxvarpos
            indxvarnonegpos = 1.0 - indxvarnegpos

        # Each log-normal random coefficient is injected multiplicatively onto
        # the sign-reparameterized fixed coefficient, so GAUSS requires every
        # ``logvar`` to also carry a strict sign (be in ``varneg`` or
        # ``varpos``). Enforce it: a silent mis-specification would otherwise
        # yield a wrong log-likelihood.
        if logvar and not (set(logvar) <= set(varneg) | set(varpos)):
            missing = sorted(set(logvar) - (set(varneg) | set(varpos)))
            raise ValueError(
                "each logvar variable must also appear in varneg or varpos "
                f"(GAUSS requirement); missing a strict sign: {missing}"
            )

        # --- additive-vs-multiplicative (log) injection masks (177-214) -----
        indxvarnonegposlog = np.ones(nvarm, dtype=np.float64)
        indxvarnegposlog = np.zeros(nvarm, dtype=np.float64)
        if nrndlog > 0:
            poslogvar1 = _indcv(logvar, var_names)
            indxvarlog = np.zeros(nvarm, dtype=np.float64)
            indxvarlog[poslogvar1] = 1.0
            indxvarnegposlog = indxvarnegpos * indxvarlog
            indxvarnonegposlog = 1.0 - indxvarnegposlog

        # --- per-type random-coefficient positions (216-231) ----------------
        mixposn = _indcv(normvar, var_names)
        mixposlg = _indcv(logvar, var_names)
        mixposyj = _indcv(yjvar, var_names)
        mixpos = np.concatenate([mixposn, mixposlg, mixposyj]).astype(np.intp)

        # actlam: max_active flag per coefficient (0 normal/log, 1 yj) (218-232)
        actlam = np.concatenate([
            np.zeros(nrndnor, dtype=np.float64),
            np.zeros(nrndlog, dtype=np.float64),
            np.ones(nrndyj, dtype=np.float64),
        ])
        numlam = actlam.shape[0]

        # --- log / non-log selectors in random-coef space (234-244) ---------
        poslog = np.zeros((nrndcoef, 0), dtype=np.float64)
        posnolog = np.zeros((nrndcoef, 0), dtype=np.float64)
        if nrndlog > 0:
            # position of log coefficients within the combined mixpos ordering
            poslog1 = _indnv(mixposlg, mixpos)
            if nrndcoef - nrndlog > 0:
                posnolog1 = _indnv(
                    np.concatenate([mixposn, mixposyj]).astype(np.intp), mixpos
                )
                poslog = _parttofull(poslog1, nrndcoef)
                posnolog = _parttofull(posnolog1, nrndcoef)
            else:
                poslog = np.eye(nrndcoef, dtype=np.float64)
                posnolog = np.zeros((nrndcoef, nrndcoef), dtype=np.float64)

        # --- indxrndvar: (nvarm, nrndcoef) selection matrix (245) -----------
        indxrndvar = _parttofull(mixpos, nvarm)

        return cls(
            var_names=var_names,
            n_beta=nvarm,
            nrndnor=nrndnor,
            nrndlog=nrndlog,
            nrndyj=nrndyj,
            nrndcoef=nrndcoef,
            nrndtot=nrndtot,
            nrndtcor=nrndtcor,
            nscale=nscale,
            numlam=numlam,
            nvarneg=nvarneg,
            nvarpos=nvarpos,
            mixposn=mixposn,
            mixposlg=mixposlg,
            mixposyj=mixposyj,
            mixpos=mixpos,
            indxrndvar=indxrndvar,
            indxvarneg=indxvarneg,
            indxvarpos=indxvarpos,
            indxvarnegpos=indxvarnegpos,
            indxvarnonegpos=indxvarnonegpos,
            indxvarnonegposlog=indxvarnonegposlog,
            indxvarnegposlog=indxvarnegposlog,
            posvarneg=posvarneg,
            posvarpos=posvarpos,
            poslog=poslog,
            posnolog=posnolog,
            actlam=actlam,
            kernel_dim=kernel_dim,
            n_kern=n_kern,
            nord=nord,
            nthresh=nthresh,
            numthresh=numthresh,
            n_categories=n_categories,
            nkernlam=nkernlam,
            normker=normker,
            nvargam=int(nvargam),
        )


def _indnv(needles: NDArray, haystack: NDArray) -> NDArray:
    """0-based positions of each ``needles`` value within ``haystack``.

    Port of GAUSS ``indnv(needles, haystack)`` (value lookup, order-preserving).

    Parameters
    ----------
    needles : ndarray
        Values to locate.
    haystack : ndarray
        Reference vector.

    Returns
    -------
    ndarray of intp, shape (len(needles),)
        0-based positions; ``-1`` for values not present.
    """
    haystack = np.asarray(haystack)
    lookup = {int(v): i for i, v in enumerate(haystack)}
    return np.array([lookup.get(int(v), -1) for v in np.asarray(needles)], dtype=np.intp)
