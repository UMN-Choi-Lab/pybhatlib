"""Parse variable specifications into design matrices.

Converts the Pythonic dict-based specification or GAUSS-style ivunord matrix
into numeric design matrices suitable for MNP estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def parse_spec(
    spec: dict,
    data: pd.DataFrame,
    alternatives: list[str],
    *,
    nseg: int = 1,
) -> tuple[NDArray, list[str]]:
    """Parse variable specification dict into design matrix.

    Parameters
    ----------
    spec : dict
        Maps variable names to alternative-specific column names or keywords.
        Format: {var_name: {alt_name: col_or_keyword, ...}, ...}
        Keywords:
        - "sero": zero (variable not in this alternative's utility)
        - "uno": one (alternative-specific constant = 1)
        - column name: use that column from data
    data : pd.DataFrame
        Dataset.
    alternatives : list of str
        Alternative column names (choice indicators).
    nseg : int
        Number of mixture segments. If >1, spec is duplicated for each segment.

    Returns
    -------
    X : ndarray, shape (N, n_alts, n_vars)
        Design matrix.
    var_names : list of str
        Variable names in order.

    Examples
    --------
    >>> spec = {
    ...     "CON_SR": {"Alt1_ch": "sero", "Alt2_ch": "uno", "Alt3_ch": "sero"},
    ...     "IVTT": {"Alt1_ch": "IVTT_DA", "Alt2_ch": "IVTT_SR", "Alt3_ch": "IVTT_TR"},
    ... }
    >>> X, names = parse_spec(spec, data, ["Alt1_ch", "Alt2_ch", "Alt3_ch"])
    """
    N = len(data)
    n_alts = len(alternatives)
    var_names = list(spec.keys())
    n_vars = len(var_names)

    # Total variables including segment duplicates
    if nseg > 1:
        total_vars = n_vars * nseg
        all_var_names = []
        for h in range(nseg):
            for vn in var_names:
                suffix = f"_s{h + 1}" if h > 0 else ""
                all_var_names.append(f"{vn}{suffix}")
    else:
        total_vars = n_vars
        all_var_names = var_names

    X = np.zeros((N, n_alts, total_vars), dtype=np.float64)

    for v_idx, var_name in enumerate(var_names):
        alt_spec = spec[var_name]

        for a_idx, alt_name in enumerate(alternatives):
            col_or_kw = alt_spec.get(alt_name, "sero")

            if isinstance(col_or_kw, str):
                col_or_kw_lower = col_or_kw.strip().lower()
                if col_or_kw_lower == "sero":
                    # Zero for all observations
                    X[:, a_idx, v_idx] = 0.0
                elif col_or_kw_lower == "uno":
                    # One for all observations (constant)
                    X[:, a_idx, v_idx] = 1.0
                else:
                    # Column name
                    if col_or_kw in data.columns:
                        X[:, a_idx, v_idx] = data[col_or_kw].values.astype(np.float64)
                    else:
                        raise ValueError(
                            f"Column '{col_or_kw}' not found in data for "
                            f"variable '{var_name}', alternative '{alt_name}'"
                        )
            elif isinstance(col_or_kw, (int, float)):
                X[:, a_idx, v_idx] = float(col_or_kw)

        # Duplicate for additional segments
        if nseg > 1:
            for h in range(1, nseg):
                seg_v_idx = h * n_vars + v_idx
                X[:, :, seg_v_idx] = X[:, :, v_idx]

    return X, all_var_names


def parse_ivunord(
    ivunord: list[list[str]],
    data: pd.DataFrame,
    alternatives: list[str],
    var_names: list[str] | None = None,
    *,
    nseg: int = 1,
) -> tuple[NDArray, list[str]]:
    """Parse GAUSS-style ivunord specification matrix.

    Parameters
    ----------
    ivunord : list of lists
        (n_alts x n_vars) matrix of strings. Each string is "sero", "uno",
        or a column name in data.
    data : pd.DataFrame
        Dataset.
    alternatives : list of str
        Alternative column names.
    var_names : list of str or None
        Variable names. Auto-generated if None.
    nseg : int
        Number of mixture segments.

    Returns
    -------
    X : ndarray, shape (N, n_alts, n_vars)
        Design matrix.
    var_names : list of str
        Variable names.

    Examples
    --------
    >>> ivunord = [
    ...     ["sero", "sero", "IVTT_DA", "OVTT_DA", "COST_DA"],
    ...     ["uno",  "sero", "IVTT_SR", "OVTT_SR", "COST_SR"],
    ...     ["sero", "uno",  "IVTT_TR", "OVTT_TR", "COST_TR"],
    ... ]
    """
    n_alts = len(alternatives)
    n_vars = len(ivunord[0]) if ivunord else 0

    if len(ivunord) != n_alts:
        raise ValueError(
            f"ivunord has {len(ivunord)} rows but {n_alts} alternatives specified"
        )

    if var_names is None:
        var_names = [f"var{i + 1}" for i in range(n_vars)]

    # Convert ivunord to dict-based spec
    spec = {}
    for v_idx, vn in enumerate(var_names):
        alt_dict = {}
        for a_idx, alt_name in enumerate(alternatives):
            alt_dict[alt_name] = ivunord[a_idx][v_idx]
        spec[vn] = alt_dict

    return parse_spec(spec, data, alternatives, nseg=nseg)
