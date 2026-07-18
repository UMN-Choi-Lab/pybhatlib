"""Differencing (``dM``) matrix for the mixed-panel MNP kernel (MNPKERCP).

Port of the GAUSS ``proc dM(depvarq, availmatq)`` from ``MNPKERCP.gss`` (line
~1365). ``dM`` builds the ``(nc-1) x nc`` linear map that differences the latent
utility vector against the *chosen* alternative, then restricts the rows to the
*available* non-chosen alternatives.

In GAUSS the full map is assembled from two globals::

    iden_matrix  = eye(nc-1)
    one_negative = -1 * ones(nc-1, 1)

placed around the chosen column, so the chosen alternative's column holds ``-1``
and every other column holds an identity ``+1``. Consequently, for a full utility
vector ``V`` (length ``nc``),::

    (M @ V)[k] = V_other_k - V_chosen

i.e. each row is *(non-chosen alternative) minus (chosen alternative)* — the
standard MNP "difference against the chosen alternative" contrast used to form
the ``(nc-1)``-variate MVNCD probability. Rows for unavailable non-chosen
alternatives are dropped (the GAUSS ``delif``/``vectomat`` restriction), leaving
``A = (#available - 1)`` rows.

This module contains no fitting logic and holds no module-level mutable state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from numpy.typing import NDArray


def dm_matrix(
    chosen_idx: int,
    avail_mask: "NDArray[Any] | list[int] | tuple[int, ...]",
    nc: int,
    *,
    xp: Any = None,
) -> "NDArray[Any]":
    """Build the availability-restricted MNP differencing matrix ``M``.

    Ports GAUSS ``dM``: the full ``(nc-1) x nc`` map places ``-1`` in the chosen
    alternative's column and an identity ``+1`` in every other column, then keeps
    only the rows corresponding to *available* non-chosen alternatives.

    Parameters
    ----------
    chosen_idx : int
        Zero-based index of the chosen alternative, ``0 <= chosen_idx < nc``.
        (GAUSS ``Alt_chosen = maxindc(depvarq')`` is one-based; pass
        ``Alt_chosen - 1`` here.)
    avail_mask : array_like of shape (nc,)
        Availability indicator over all ``nc`` alternatives (nonzero / truthy =
        available). The chosen alternative is assumed available; its entry is
        ignored for row selection since the chosen row is never emitted.
    nc : int
        Total number of alternatives (GAUSS global ``nc``). Must satisfy
        ``nc >= 2``.
    xp : module, optional
        Array backend (e.g. ``numpy`` or a compatible namespace). Defaults to
        NumPy when omitted. Follows the repository ``xp`` kwarg convention.

    Returns
    -------
    NDArray
        Differencing matrix of shape ``(A, nc)`` where
        ``A = (# available alternatives) - 1``. For the all-available case this
        is the full ``(nc-1) x nc`` matrix. Each retained row ``k`` satisfies
        ``(M @ V)[k] = V_other_k - V_chosen``.

    Raises
    ------
    ValueError
        If ``nc < 2``, ``chosen_idx`` is out of range, ``avail_mask`` does not
        have length ``nc``, or the chosen alternative is marked unavailable.
    """
    if xp is None:
        import numpy as _np

        xp = _np

    if nc < 2:
        raise ValueError(f"nc must be >= 2, got {nc}")
    if not (0 <= chosen_idx < nc):
        raise ValueError(
            f"chosen_idx must satisfy 0 <= chosen_idx < nc={nc}, got {chosen_idx}"
        )

    avail = xp.asarray(avail_mask)
    if avail.shape != (nc,):
        raise ValueError(
            f"avail_mask must have shape ({nc},), got {tuple(avail.shape)}"
        )
    avail_bool = avail != 0
    if not bool(avail_bool[chosen_idx]):
        raise ValueError("chosen alternative must be available")

    # Full (nc-1) x nc map: identity (+1) in every column except the chosen
    # column, which holds -1. Row i's +1 sits at eye column i, which maps to the
    # full column i for i < chosen_idx and i+1 for i >= chosen_idx, i.e. row i
    # corresponds to the i-th non-chosen alternative (in original order).
    eye = xp.eye(nc - 1, dtype=float)
    one_negative = -xp.ones((nc - 1, 1), dtype=float)
    if chosen_idx == 0:
        full = xp.concatenate([one_negative, eye], axis=1)
    elif chosen_idx == nc - 1:
        full = xp.concatenate([eye, one_negative], axis=1)
    else:
        full = xp.concatenate(
            [eye[:, :chosen_idx], one_negative, eye[:, chosen_idx:]], axis=1
        )

    # Restrict to available non-chosen rows (GAUSS delif + vectomat selection).
    # Non-chosen alternatives, in original index order, are every index except
    # chosen_idx; keep the row iff that alternative is available.
    keep = xp.concatenate([avail_bool[:chosen_idx], avail_bool[chosen_idx + 1 :]])
    return full[keep]
