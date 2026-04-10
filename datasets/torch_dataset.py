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
    ) -> None:
        df = pd.read_csv(Path(manifest_path))
        if subjects is not None:
            df = df[df["subject_id"].isin(subjects)]
        self._rows: list[dict] = df.to_dict(orient="records")
        self._zip_path: Path = Path(zip_path)
        self._cache: dict[str, np.ndarray] = {}

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
        window = arr[start_row:end_row]   # (window_len, 52)
        window = window.T                 # (52, window_len)
        return (
            torch.tensor(window, dtype=torch.float32),
            LABEL_MAP[binary_label],
        )
