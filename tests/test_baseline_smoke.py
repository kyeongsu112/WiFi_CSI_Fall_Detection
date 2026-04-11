from __future__ import annotations

import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import pytest

from training.model import FallDetector, save_model, load_model
from datasets.torch_dataset import WifallDataset, LABEL_MAP


# ---------------------------------------------------------------------------
# Minimal fake dataset (no WiFall.zip or manifest required)
# ---------------------------------------------------------------------------

class _FakeDataset(torch.utils.data.Dataset):
    """Synthetic dataset: 2 fall + 2 non_fall windows of shape (52, 100)."""

    def __init__(self) -> None:
        self._items: list[tuple[torch.Tensor, int]] = []
        for label in [1, 1, 0, 0]:
            tensor = torch.randn(52, 100)
            self._items.append((tensor, label))
        # Expose a minimal manifest-like attribute for evaluate()
        import pandas as pd
        self.manifest = pd.DataFrame({
            "source_file": ["fake.csv"] * 4,
            "window_index": [0, 1, 0, 1],
        })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._items[idx]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model() -> FallDetector:
    return FallDetector(
        n_subcarriers=52,
        conv_channels=[64, 128, 128],
        kernel_sizes=[5, 3, 3],
        dropout=0.5,
    )


@pytest.fixture
def fake_dataset() -> _FakeDataset:
    return _FakeDataset()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_output_shape(default_model: FallDetector) -> None:
    """Output shape must be (B, 1) for any batch size."""
    batch_sizes = [1, 4, 8]
    default_model.eval()
    with torch.no_grad():
        for b in batch_sizes:
            x = torch.randn(b, 52, 100)
            out = default_model(x)
            assert out.shape == (b, 1), f"Expected ({b}, 1), got {out.shape}"


def test_output_is_raw_logit(default_model: FallDetector) -> None:
    """forward() must return raw logits — no sigmoid applied internally.

    Verified structurally: the final layer is nn.Linear and no nn.Sigmoid
    appears anywhere in the module tree.  This is the contract required by
    BCEWithLogitsLoss used in the training loop.
    """
    # Final layer must be a plain Linear (not Sigmoid or anything else)
    assert isinstance(default_model.fc, nn.Linear), (
        f"Expected self.fc to be nn.Linear, got {type(default_model.fc).__name__}"
    )
    # No sigmoid anywhere in the full module tree
    sigmoid_modules = [
        type(m).__name__
        for m in default_model.modules()
        if isinstance(m, nn.Sigmoid)
    ]
    assert not sigmoid_modules, (
        f"Found nn.Sigmoid in module tree: {sigmoid_modules}. "
        "forward() must return raw logits for BCEWithLogitsLoss."
    )


def test_sigmoid_bounds_output(default_model: FallDetector) -> None:
    """After sigmoid the output must be in [0, 1]."""
    default_model.eval()
    x = torch.randn(8, 52, 100)
    with torch.no_grad():
        logits = default_model(x)
        probs = torch.sigmoid(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), \
        "sigmoid output out of [0, 1] range"


def test_save_load_roundtrip(default_model: FallDetector, tmp_path: Path) -> None:
    """save_model / load_model round-trip must preserve weights and metadata."""
    config = {
        "model": {
            "conv_channels": [64, 128, 128],
            "kernel_sizes": [5, 3, 3],
            "dropout": 0.5,
        },
        "data": {
            "n_subcarriers": 52,
            "window_size": 100,
        },
    }
    ckpt_path = tmp_path / "models" / "test_model.pt"

    save_model(default_model, ckpt_path, config)
    assert ckpt_path.exists(), "Checkpoint file was not created"

    loaded_model, metadata = load_model(ckpt_path)

    # Model must be in eval mode
    assert not loaded_model.training, "Loaded model should be in eval mode"

    # Weights must match
    for (name_orig, p_orig), (name_loaded, p_loaded) in zip(
        default_model.state_dict().items(),
        loaded_model.state_dict().items(),
    ):
        assert name_orig == name_loaded
        assert torch.allclose(p_orig, p_loaded), f"Weight mismatch at {name_orig}"

    # Metadata checks
    assert metadata["n_subcarriers"] == 52
    assert metadata["window_size"] == 100
    assert metadata["input_shape"] == [1, 52, 100]


def test_forward_deterministic_eval(default_model: FallDetector) -> None:
    """In eval mode, two forward passes with the same input must be identical."""
    default_model.eval()
    x = torch.randn(4, 52, 100)
    with torch.no_grad():
        out1 = default_model(x)
        out2 = default_model(x)
    assert torch.allclose(out1, out2), "Forward pass is not deterministic in eval mode"


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    """load_model must raise FileNotFoundError with a clear message when the
    checkpoint does not exist — not a cryptic torch error."""
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="does_not_exist.pt"):
        load_model(missing)


def test_load_model_missing_state_dict_key_raises(tmp_path: Path) -> None:
    """A checkpoint lacking 'model_state_dict' must raise ValueError, not KeyError."""
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"model_config": {}, "n_subcarriers": 52, "window_size": 100}, bad_ckpt)
    with pytest.raises(ValueError, match="model_state_dict"):
        load_model(bad_ckpt)


def test_load_model_incompatible_state_dict_raises(
    default_model: FallDetector, tmp_path: Path
) -> None:
    """A state dict that doesn't match the architecture must raise ValueError
    with a human-readable message, not a bare RuntimeError from PyTorch."""
    # Save a model with different architecture (1 conv channel instead of 64)
    tiny = FallDetector(n_subcarriers=52, conv_channels=[1, 2, 2], kernel_sizes=[3, 3, 3])
    ckpt = tmp_path / "tiny.pt"
    # Manually craft a checkpoint that claims 64 channels but has the tiny weights
    checkpoint = {
        "model_state_dict": tiny.state_dict(),
        "model_config": {"model": {"conv_channels": [64, 128, 128]}},
        "n_subcarriers": 52,
        "window_size": 100,
        "input_shape": [1, 52, 100],
    }
    torch.save(checkpoint, ckpt)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(ckpt)


# ---------------------------------------------------------------------------
# WifallDataset shape-enforcement tests (no real ZIP required)
# ---------------------------------------------------------------------------

class _FakeZip:
    """Minimal stand-in for datasets.loader.load_csi_file that returns a fixed array."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def __call__(self, source_file: str, zip_path: Path) -> np.ndarray:
        return self._arr


def _make_dataset_from_manifest_rows(
    rows: list[dict],
    csi_array: np.ndarray,
    tmp_path: Path,
    *,
    repair_shape: bool = False,
) -> WifallDataset:
    """Build a WifallDataset backed by a synthetic in-memory CSV manifest
    and a patched load_csi_file that returns *csi_array* for every source_file.
    """
    import csv
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ds = WifallDataset(
        manifest_path=manifest_path,
        zip_path=tmp_path / "fake.zip",
        repair_shape=repair_shape,
    )
    # Patch the cache so no real zip I/O happens
    ds._cache["WiFall/ID0/fall/fake.csv"] = csi_array
    return ds


def test_wifall_dataset_returns_correct_shape(tmp_path: Path) -> None:
    """__getitem__ must return a (52, 100) tensor when the CSV has exactly 100 rows."""
    arr = np.random.default_rng(0).random((200, 52), dtype=np.float32)
    rows = [
        {"source_file": "WiFall/ID0/fall/fake.csv",
         "start_row": 0, "end_row": 100,
         "binary_label": "fall", "window_index": 0},
        {"source_file": "WiFall/ID0/fall/fake.csv",
         "start_row": 100, "end_row": 200,
         "binary_label": "non_fall", "window_index": 1},
    ]
    ds = _make_dataset_from_manifest_rows(rows, arr, tmp_path)
    x, y = ds[0]
    assert x.shape == (52, 100), f"Expected (52, 100), got {x.shape}"
    assert y == LABEL_MAP["fall"]


def test_wifall_dataset_shape_mismatch_raises_by_default(tmp_path: Path) -> None:
    """By default a manifest/parser row-count mismatch raises ValueError.
    This surfaces disagreements loudly rather than silently masking them."""
    arr = np.ones((95, 52), dtype=np.float32)   # only 95 rows available
    rows = [{"source_file": "WiFall/ID0/fall/fake.csv",
             "start_row": 0, "end_row": 100,     # manifest claims 100 rows
             "binary_label": "non_fall", "window_index": 0}]
    ds = _make_dataset_from_manifest_rows(rows, arr, tmp_path)
    with pytest.raises(ValueError, match="95"):
        ds[0]


def test_wifall_dataset_pads_short_window_when_repair_enabled(tmp_path: Path) -> None:
    """repair_shape=True: fewer rows than expected → zero-padded to expected size."""
    arr = np.ones((95, 52), dtype=np.float32)
    rows = [{"source_file": "WiFall/ID0/fall/fake.csv",
             "start_row": 0, "end_row": 100,
             "binary_label": "non_fall", "window_index": 0}]
    ds = _make_dataset_from_manifest_rows(rows, arr, tmp_path, repair_shape=True)
    x, _ = ds[0]
    assert x.shape == (52, 100), f"Expected (52, 100) after padding, got {x.shape}"
    assert x[:, 95:].sum().item() == 0.0  # padded columns are zero


def test_wifall_dataset_truncates_long_window_when_repair_enabled(tmp_path: Path) -> None:
    """repair_shape=True: more rows than expected → truncated to expected size."""
    arr = np.ones((150, 52), dtype=np.float32)
    rows = [{"source_file": "WiFall/ID0/fall/fake.csv",
             "start_row": 0, "end_row": 100,
             "binary_label": "fall", "window_index": 0}]
    ds = _make_dataset_from_manifest_rows(rows, arr, tmp_path, repair_shape=True)
    x, _ = ds[0]
    assert x.shape == (52, 100)


def test_wifall_dataset_unknown_label_raises(tmp_path: Path) -> None:
    """__getitem__ must raise KeyError for an unrecognised binary_label."""
    arr = np.zeros((100, 52), dtype=np.float32)
    rows = [{"source_file": "WiFall/ID0/fall/fake.csv",
             "start_row": 0, "end_row": 100,
             "binary_label": "unknown_xyz", "window_index": 0}]
    ds = _make_dataset_from_manifest_rows(rows, arr, tmp_path)
    with pytest.raises(KeyError, match="unknown_xyz"):
        ds[0]
