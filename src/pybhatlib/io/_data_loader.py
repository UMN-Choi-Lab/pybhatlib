"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path, *, file_type: str | None = None) -> pd.DataFrame:
    """Load data from CSV, DAT, or XLSX file.

    Parameters
    ----------
    path : str or Path
        Path to data file.
    file_type : str or None
        Force file type ("csv", "dat", "xlsx"). Auto-detected if None.

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
        return pd.read_csv(path)
    elif file_type in ("dat", "txt"):
        # Try whitespace-delimited first, then comma
        try:
            return pd.read_csv(path, sep=r"\s+")
        except Exception:
            return pd.read_csv(path)
    elif file_type in ("xlsx", "xls"):
        return pd.read_excel(path)
    else:
        # Default to CSV
        return pd.read_csv(path)
