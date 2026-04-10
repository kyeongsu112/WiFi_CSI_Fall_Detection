from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import pytest

from training.model import FallDetector, save_model, load_model


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
