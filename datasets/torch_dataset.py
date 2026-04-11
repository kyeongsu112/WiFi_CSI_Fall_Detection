"""PyTorch-specific dataset for WiFall.

Separated from datasets.loader so that Step 1 data-prep code
(scripts/prepare_wifall.py and its tests) can import datasets.loader
without requiring torch to be installed.

Usage (Step 2 training only):
    from datasets.torch_dataset import WifallDataset, LABEL_MAP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets.loader import load_csi_file

LABEL_MAP: dict[str, int] = {"fall": 1, "non_fall": 0}


class WifallDataset(torch.utils.data.Dataset):
    """Window-level PyTorch Dataset over the WiFall manifest.

    Each sample is a (tensor, label) pair:
      - tensor: shape (52, 100), dtype float32  — (n_subcarriers, window_size)
      - label:  int — 1 for fall, 0 for non_fall  (from LABEL_MAP)

    Parameters
    ----------
    manifest_path : str or Path
        Path to wifall_manifest.csv produced by scripts/prepare_wifall.py.
    zip_path : str or Path
        Path to WiFall.zip.
    subjects : list[str] or None
        If given, only rows whose subject_id is in this list are included.
        Pass None to include all subjects.
    repair_shape : bool
        If False (default) a mismatch between the number of rows the manifest
        claims a window has and the number actually returned by the CSV parser
        raises ``ValueError``.  This surfaces manifest/parser disagreements
        loudly so they can be investigated and corrected at the source.
        Set True only when you have inspected the mismatch and want the
        dataset to pad (short windows) or truncate (long windows) silently.

    Notes
    -----
    - end_row is treated as exclusive (Python slice convention).
    - CSI arrays are cached per source_file after first load to avoid
      re-opening the zip on every __getitem__ call.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        zip_path: str | Path = "data/WiFall.zip",
        subjects: list[str] | None = None,
        repair_shape: bool = False,
    ) -> None:
        df = pd.read_csv(Path(manifest_path))
        if subjects is not None:
            df = df[df["subject_id"].isin(subjects)]
        self._rows: list[dict] = df.to_dict(orient="records")
        self._zip_path: Path = Path(zip_path)
        self._cache: dict[str, np.ndarray] = {}
        self._repair_shape: bool = repair_shape

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int):
        row = self._rows[idx]
        source_file: str = row["source_file"]
        start_row: int = int(row["start_row"])
        end_row: int = int(row["end_row"])
        binary_label: str = row["binary_label"]

        if source_file not in self._cache:
            self._cache[source_file] = load_csi_file(source_file, self._zip_path)

        arr = self._cache[source_file]
        expected_rows = end_row - start_row
        window = arr[start_row:end_row]   # (window_len, 52)
        actual_rows = window.shape[0]
        if actual_rows != expected_rows:
            if not self._repair_shape:
                raise ValueError(
                    f"Shape mismatch at manifest row {idx} "
                    f"(source_file={source_file!r}, "
                    f"start_row={start_row}, end_row={end_row}): "
                    f"manifest expects {expected_rows} rows but the CSV "
                    f"parser returned {actual_rows}. "
                    "This indicates a disagreement between count_csi_rows "
                    "(used during manifest build) and load_csi_from_fileobj "
                    "(used at training time), likely caused by \\r\\r\\n "
                    "line-ending differences. Re-run scripts/prepare_wifall.py "
                    "to regenerate the manifest, or pass repair_shape=True to "
                    "the dataset constructor to pad/truncate (not recommended)."
                )
            # repair_shape=True: pad or truncate to keep DataLoader stable.
            if actual_rows < expected_rows:
                window = np.pad(window, ((0, expected_rows - actual_rows), (0, 0)))
            else:
                window = window[:expected_rows]
        window = window.T                 # (52, window_len)
        if binary_label not in LABEL_MAP:
            raise KeyError(
                f"Unknown binary_label {binary_label!r} at manifest row {idx}. "
                f"Expected one of {list(LABEL_MAP)}."
            )
        return (
            torch.tensor(window, dtype=torch.float32),
            LABEL_MAP[binary_label],
        )
