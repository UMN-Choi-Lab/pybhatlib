"""Data loading utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Design keywords that are never real data columns.  ``"uno"`` maps to a column
#: of ones and ``"sero"`` to a column of zeros; they are synthesized by the
#: models (or already present as constant helper columns) rather than read from
#: the file, so they must not be requested by name when column-pruning a load.
_SPECIAL_COLS = ("uno", "sero")


def load_data(
    path: str | Path,
    *,
    file_type: str | None = None,
    usecols: Iterable[str] | Callable[[Any], bool] | None = None,
) -> pd.DataFrame:
    """Load data from CSV, DAT, or XLSX file.

    Parameters
    ----------
    path : str or Path
        Path to data file.
    file_type : str or None
        Force file type ("csv", "dat", "xlsx"). Auto-detected if None.
    usecols : iterable of str, callable, or None
        Subset of columns to read (passed straight through to the underlying
        pandas reader).  ``None`` (default) reads every column and is fully
        backward compatible.  A callable predicate ``col -> bool`` is the safest
        form for column pruning: pandas only evaluates it against columns that
        actually exist, so an over-broad request never raises on a missing name.

    Returns
    -------
    df : pd.DataFrame
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if file_type is None:
        file_type = path.suffix.lower().lstrip(".")

    if file_type in ("csv",):
        return pd.read_csv(path, usecols=usecols)
    elif file_type in ("dat", "txt"):
        # Try whitespace-delimited first, then comma
        try:
            return pd.read_csv(path, sep=r"\s+", usecols=usecols)
        except Exception:
            return pd.read_csv(path, usecols=usecols)
    elif file_type in ("xlsx", "xls"):
        return pd.read_excel(path, usecols=usecols)
    else:
        # Default to CSV
        return pd.read_csv(path, usecols=usecols)


def _iter_spec_entries(spec: Any) -> Iterator[str]:
    """Yield every design entry from a variable spec, in any supported form.

    Handles the dict-of-dicts spec (``{var: {alt: col_or_keyword}}`` used by
    MNL/MNP/MORP), a flat ``{var: col_or_keyword}`` mapping, a GAUSS-style 2-D
    string array / list-of-lists (``ivunord`` / ``ivmt`` / ``ivgt``), and a flat
    iterable of names.  Entries are yielded verbatim (keywords included); the
    caller is responsible for filtering the ``"uno"`` / ``"sero"`` keywords.
    """
    if spec is None:
        return
    if isinstance(spec, dict):
        for val in spec.values():
            if isinstance(val, dict):
                for v in val.values():
                    yield str(v)
            else:
                yield str(val)
        return
    if isinstance(spec, np.ndarray):
        for v in spec.ravel():
            yield str(v)
        return
    # list-of-lists (ivunord) or a flat iterable of names
    for row in spec:
        if isinstance(row, (list, tuple, np.ndarray)):
            for v in row:
                yield str(v)
        else:
            yield str(row)


def used_columns_selector(
    *,
    value_cols: Iterable[str] = (),
    avail_cols: Iterable[str] | None = None,
    id_cols: Iterable[str | None] = (),
    specs: Iterable[Any] = (),
) -> Callable[[Any], bool]:
    """Build a pandas ``usecols`` predicate over a model's referenced columns.

    Collects the columns a model actually touches so a wide panel can be loaded
    without materializing hundreds of unused columns:

    * ``value_cols`` -- the alternatives / dependent-variable / price columns.
    * ``avail_cols`` -- the availability columns (``None`` when every
      alternative is always available).
    * ``id_cols`` -- the panel ``person_id`` and observation-id columns
      (``None`` entries are ignored).
    * ``specs`` -- the variable specs (dict-of-dicts, GAUSS ``ivunord`` /
      ``ivmt`` / ``ivgt`` arrays, ...); every entry that names a real column is
      collected, while the ``"uno"`` / ``"sero"`` keywords (case-insensitive)
      are dropped.

    The ``"uno"`` / ``"sero"`` helper columns are *always* kept when the file
    actually carries such a column (case-insensitive), matching the GAUSS
    convention of constant ``uno``/``sero`` columns.

    Returns
    -------
    Callable[[Any], bool]
        A ``usecols`` predicate.  Because pandas evaluates it only against
        columns present in the file, over-inclusion is harmless and a requested
        column that is absent never raises -- so the collector errs toward
        keeping columns (a dropped-but-needed column would break estimation).
    """
    wanted: set[str] = set()

    def _add(name: Any) -> None:
        if name is None:
            return
        text = str(name)
        if text.strip().lower() in _SPECIAL_COLS:
            return
        wanted.add(text)

    for c in value_cols:
        _add(c)
    if avail_cols is not None:
        for c in avail_cols:
            _add(c)
    for c in id_cols:
        _add(c)
    for spec in specs:
        for entry in _iter_spec_entries(spec):
            _add(entry)

    def _predicate(col: Any) -> bool:
        text = str(col)
        return text in wanted or text.strip().lower() in _SPECIAL_COLS

    return _predicate
