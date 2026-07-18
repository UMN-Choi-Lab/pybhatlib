"""Panel (person-index) primitive: the Dmask trick from BHATLIB.

Panel data has multiple observations per decision-maker ("person"). BHATLIB's
mixed-model / panel routines repeatedly need to broadcast person-level draws
out to the observation level, and to aggregate observation-level quantities
(log-likelihoods, weights) back up to the person level. Both operations are
expressed in GAUSS as multiplication by a 0/1 indicator matrix ``Dmask`` of
shape ``(n_obs, n_ind)`` with ``Dmask[o, i] = 1`` iff observation ``o``
belongs to person ``i``:

- broadcast:    ``obs_vals = Dmask @ person_vals``
- scatter-sum:  ``person_vals = Dmask.T @ obs_vals``

:class:`PanelIndex` builds this mapping once from a vector of person ids and
exposes the two operations (plus the panel log-likelihood product and a
weighted-mean helper) without ever materializing the dense ``(n_obs, n_ind)``
matrix on the hot path -- grouping is done via the ``np.unique`` inverse
index and ``np.add.reduceat`` / ``np.add.at``, which stays correct whether or
not a person's rows are contiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pybhatlib.backend._array_api import get_backend


@dataclass(frozen=True)
class PanelIndex:
    """Person-index mapping for panel data (the BHATLIB Dmask primitive).

    Stores, for ``n_obs`` observation rows belonging to ``n_ind`` distinct
    persons, the inverse index ``row_to_person`` (row ``o`` belongs to person
    ``row_to_person[o]``, 0-indexed into the unique ids) together with the
    sorted unique ids themselves. All public operations are expressed via
    this inverse index rather than a dense ``(n_obs, n_ind)`` mask, so cost
    is ``O(n_obs)`` per call even when ``n_ind`` is large (e.g. ~2341).

    Parameters
    ----------
    ids : ndarray, shape (n_ind,)
        Sorted unique person ids (as returned by ``np.unique``).
    row_to_person : ndarray of int, shape (n_obs,)
        For each observation row, the index (0..n_ind-1) of its person in
        ``ids``.

    Notes
    -----
    Rows belonging to the same person need not be contiguous: grouping uses
    ``np.add.at`` (scatter-add), which is correct for any row ordering.
    """

    ids: NDArray
    row_to_person: NDArray

    @classmethod
    def from_ids(cls, person_ids: NDArray) -> "PanelIndex":
        """Build a :class:`PanelIndex` from a vector of person ids.

        Parameters
        ----------
        person_ids : array-like, shape (n_obs,)
            Person identifier for each observation row (integer or label
            array; need not be sorted or contiguous).

        Returns
        -------
        panel : PanelIndex
            Panel index with ``n_ind`` unique persons and ``n_obs`` rows.

        Raises
        ------
        ValueError
            If ``person_ids`` is empty or not 1-D.
        """
        person_ids = np.asarray(person_ids)
        if person_ids.ndim != 1:
            raise ValueError(
                f"person_ids must be 1-D, got shape {person_ids.shape}"
            )
        if person_ids.size == 0:
            raise ValueError("person_ids must be non-empty")

        ids, row_to_person = np.unique(person_ids, return_inverse=True)
        row_to_person = np.asarray(row_to_person, dtype=np.intp).reshape(-1)
        return cls(ids=ids, row_to_person=row_to_person)

    @property
    def n_obs(self) -> int:
        """Number of observation rows."""
        return int(self.row_to_person.shape[0])

    @property
    def n_ind(self) -> int:
        """Number of distinct persons."""
        return int(self.ids.shape[0])

    def mask(self, *, xp=None) -> NDArray:
        """Return the dense ``(n_obs, n_ind)`` 0/1 Dmask matrix.

        Intended for testing/inspection only -- prefer :meth:`broadcast`,
        :meth:`scatter_sum`, :meth:`logprod`, :meth:`weightind` for actual
        computation since they avoid materializing this matrix.

        Parameters
        ----------
        xp : backend, optional
            Array backend. Defaults to NumPy.

        Returns
        -------
        dmask : ndarray, shape (n_obs, n_ind)
            ``dmask[o, i] = 1`` iff observation ``o`` belongs to person ``i``.
        """
        if xp is None:
            xp = get_backend("numpy")
        dmask = xp.zeros((self.n_obs, self.n_ind), dtype=xp.float64)
        dmask[np.arange(self.n_obs), self.row_to_person] = 1.0
        return dmask

    def broadcast(self, draws: NDArray, *, xp=None) -> NDArray:
        """Expand person-level values out to observation rows.

        Equivalent to ``Dmask @ draws`` (BHATLIB ``errbeta1``-style
        broadcast of person-level random draws to each of their obs rows).

        Parameters
        ----------
        draws : ndarray, shape (n_ind, ...)
            Person-level values (any number of trailing dims, e.g. draws or
            parameters).
        xp : backend, optional
            Array backend (unused for the numpy path; kept for interface
            symmetry with other panel ops).

        Returns
        -------
        obs_vals : ndarray, shape (n_obs, ...)
            ``obs_vals[o, ...] = draws[row_to_person[o], ...]``.
        """
        del xp  # fancy indexing is backend-agnostic for ndarray inputs
        draws = np.asarray(draws)
        if draws.shape[0] != self.n_ind:
            raise ValueError(
                f"draws.shape[0]={draws.shape[0]} must equal n_ind={self.n_ind}"
            )
        return draws[self.row_to_person, ...]

    def scatter_sum(self, obs_vals: NDArray, *, xp=None) -> NDArray:
        """Sum observation-level values within each person.

        Equivalent to ``Dmask.T @ obs_vals``.

        Parameters
        ----------
        obs_vals : ndarray, shape (n_obs, ...)
            Observation-level values (any number of trailing dims).
        xp : backend, optional
            Array backend. Defaults to NumPy (used to allocate the output).

        Returns
        -------
        person_sums : ndarray, shape (n_ind, ...)
            ``person_sums[i, ...] = sum_{o: row_to_person[o]==i} obs_vals[o, ...]``.
        """
        if xp is None:
            xp = get_backend("numpy")
        obs_vals = np.asarray(obs_vals)
        if obs_vals.shape[0] != self.n_obs:
            raise ValueError(
                f"obs_vals.shape[0]={obs_vals.shape[0]} must equal "
                f"n_obs={self.n_obs}"
            )
        out_shape = (self.n_ind,) + obs_vals.shape[1:]
        person_sums = xp.zeros(out_shape, dtype=xp.float64)
        np.add.at(person_sums, self.row_to_person, obs_vals)
        return person_sums

    def logprod(self, log_p_obs: NDArray, *, xp=None) -> NDArray:
        """Panel (within-person) product of per-observation probabilities.

        Computes ``Pprod = exp(Dmask.T @ log_p_obs)``, i.e. for each person
        the product over their observations of the per-observation
        probability -- the standard panel-probit / mixed-logit likelihood
        contribution.

        Parameters
        ----------
        log_p_obs : ndarray, shape (n_obs,) or (n_obs, ndraw)
            Per-observation log-probabilities (optionally per Monte Carlo
            draw as an extra trailing axis).
        xp : backend, optional
            Array backend. Defaults to NumPy.

        Returns
        -------
        p_prod : ndarray, shape (n_ind,) or (n_ind, ndraw)
            Per-person product of probabilities across their observations.
        """
        if xp is None:
            xp = get_backend("numpy")
        log_p_obs = np.asarray(log_p_obs)
        log_sum = self.scatter_sum(log_p_obs, xp=xp)
        return xp.exp(log_sum)

    def weightind(self, weights: NDArray, *, xp=None) -> NDArray:
        """Average observation-level weights within each person.

        Equivalent to ``(Dmask.T @ w) / (Dmask.T @ 1)``, i.e. the mean
        weight over a person's observation rows.

        Parameters
        ----------
        weights : ndarray, shape (n_obs,)
            Observation-level weights.
        xp : backend, optional
            Array backend. Defaults to NumPy.

        Returns
        -------
        person_weights : ndarray, shape (n_ind,)
            Per-person mean of ``weights`` over their observation rows.
        """
        if xp is None:
            xp = get_backend("numpy")
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {weights.shape}")

        weight_sums = self.scatter_sum(weights, xp=xp)
        counts = self.scatter_sum(xp.ones((self.n_obs,), dtype=xp.float64), xp=xp)
        return weight_sums / counts
