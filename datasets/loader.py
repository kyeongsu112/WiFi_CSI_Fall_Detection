"""CSI data loading utilities for the WiFall dataset.

WiFall CSI format:
  - 28-column CSV with a header row
  - The 'data' column contains a string-encoded list of 104 integers
  - 104 integers = 52 I/Q pairs → 52 subcarrier amplitudes
  - Amplitude = sqrt(I^2 + Q^2) per subcarrier
"""
from __future__ import annotations

import ast
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

N_SUBCARRIERS: int = 52
IQ_VECTOR_LEN: int = N_SUBCARRIERS * 2  # 104


def iq_to_amplitude(flat_iq: list[int] | np.ndarray) -> np.ndarray:
    """Convert a flat I/Q vector to per-subcarrier amplitude values.

    Args:
        flat_iq: 104 integers ordered as [I0, Q0, I1, Q1, ..., I51, Q51].

    Returns:
        np.ndarray of shape (52,), dtype float32.
        Each element is sqrt(I_k^2 + Q_k^2) for subcarrier k.

    Raises:
        ValueError: if len(flat_iq) != 104.
    """
    arr = np.asarray(flat_iq, dtype=np.float32)
    if arr.size != IQ_VECTOR_LEN:
        raise ValueError(
            f"Expected {IQ_VECTOR_LEN} I/Q values, got {arr.size}. "
            "Each WiFall CSI row must have exactly 104 integers."
        )
    iq = arr.reshape(N_SUBCARRIERS, 2)          # (52, 2)
    return np.sqrt(iq[:, 0] ** 2 + iq[:, 1] ** 2)  # (52,)


def load_csi_from_fileobj(fileobj) -> np.ndarray:
    """Parse a WiFall CSV file object into a CSI amplitude array.

    Reads the 'data' column, parses the string-encoded I/Q list per row,
    and converts each row to 52 amplitude values.

    Args:
        fileobj: binary file-like object (e.g. from zipfile.ZipFile.open()).

    Returns:
        np.ndarray of shape (n_valid_rows, 52), dtype float32.

    Raises:
        ValueError: if the CSV has no 'data' column.
        ValueError: if any 'data' cell cannot be parsed or has wrong length.
    """
    df = pd.read_csv(io.TextIOWrapper(fileobj, encoding="utf-8"))

    if "data" not in df.columns:
        raise ValueError(
            "'data' column not found. "
            f"Columns present: {list(df.columns)}"
        )

    rows: list[np.ndarray] = []
    for idx, raw in enumerate(df["data"]):
        try:
            flat = ast.literal_eval(str(raw))
        except (ValueError, SyntaxError) as exc:
            raise ValueError(
                f"Row {idx}: failed to parse 'data' cell: {str(raw)[:60]!r}"
            ) from exc
        rows.append(iq_to_amplitude(flat))

    if not rows:
        return np.empty((0, N_SUBCARRIERS), dtype=np.float32)

    return np.stack(rows, axis=0)  # (n_rows, 52)


def count_csi_rows(fileobj) -> int:
    """Count data rows in a WiFall CSV without parsing the 'data' column.

    Significantly faster than load_csi_from_fileobj when only the row count
    is needed (e.g. to compute window boundaries during manifest generation).

    Args:
        fileobj: binary file-like object (e.g. from zipfile.ZipFile.open()).

    Returns:
        Number of data rows, excluding the header. 0 for header-only files.
    """
    # WiFall CSVs use \r\r\n line endings. TextIOWrapper (universal newlines)
    # splits both \r as line terminators, producing spurious empty rows after
    # every real data row. Filter them out by only counting non-empty rows.
    reader = csv.reader(io.TextIOWrapper(fileobj, encoding="utf-8"))
    try:
        next(reader)        # skip header row
    except StopIteration:
        return 0
    return sum(1 for row in reader if row)


def load_csi_file(
    zip_entry: str,
    zip_path: str | Path = "data/WiFall.zip",
) -> np.ndarray:
    """Load CSI amplitude data for a single session from the WiFall zip.

    Args:
        zip_entry: internal zip path, e.g. 'WiFall/ID0/fall/xxx.csv'.
        zip_path:  path to WiFall.zip on disk.

    Returns:
        np.ndarray of shape (n_samples, 52), dtype float32.
    """
    with zipfile.ZipFile(Path(zip_path)) as zf:
        with zf.open(zip_entry) as f:
            return load_csi_from_fileobj(f)


def load_csi_window(
    zip_entry: str,
    start_row: int,
    end_row: int,
    zip_path: str | Path = "data/WiFall.zip",
) -> np.ndarray:
    """Load a single CSI window (rows [start_row, end_row)) from the zip.

    Returns ndarray of shape (end_row - start_row, 52), dtype float32.
    end_row is exclusive: matches manifest columns start_row / end_row.
    """
    arr = load_csi_file(zip_entry, zip_path)
    return arr[start_row:end_row]


def subject_window_distribution(manifest_path: str | Path) -> pd.DataFrame:
    """Return a DataFrame with per-subject, per-label window counts.

    Columns: subject_id, binary_label, n_windows.
    Sorted by subject_id ascending.
    """
    df = pd.read_csv(Path(manifest_path))
    counts = (
        df.groupby(["subject_id", "binary_label"])
        .size()
        .reset_index(name="n_windows")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    return counts
