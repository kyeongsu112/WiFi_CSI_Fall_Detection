from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class FallDetector(nn.Module):
    """1D-CNN binary fall detector.

    Input:  (B, 52, 100)  — (batch, n_subcarriers, window_size)
    Output: (B, 1)        — raw logit (NOT sigmoid; use BCEWithLogitsLoss)
    """

    def __init__(
        self,
        n_subcarriers: int = 52,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 128]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]

        in_ch = n_subcarriers

        # Block 1: Conv1d → BN → ReLU → MaxPool1d(2)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, conv_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Block 2: Conv1d → BN → ReLU → MaxPool1d(2)
        self.block2 = nn.Sequential(
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Block 3: Conv1d → BN → ReLU → AdaptiveAvgPool1d(1)
        self.block3 = nn.Sequential(
            nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2),
            nn.BatchNorm1d(conv_channels[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(conv_channels[2], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_subcarriers, window_size)
        x = self.block1(x)   # (B, 64, 50)
        x = self.block2(x)   # (B, 128, 25)
        x = self.block3(x)   # (B, 128, 1)
        x = x.squeeze(-1)    # (B, 128)
        x = self.dropout(x)
        x = self.fc(x)       # (B, 1)
        return x


def save_model(model: FallDetector, path: str | Path, config: dict) -> None:
    """Save checkpoint with model weights and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": config,
        "input_shape": [1, 52, 100],
        "n_subcarriers": 52,
        "window_size": 100,
    }
    torch.save(checkpoint, path)


def load_model(path: str | Path) -> tuple[FallDetector, dict]:
    """Load checkpoint. Returns (model_in_eval_mode, metadata_dict)."""
    path = Path(path)
    checkpoint = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model_section = model_config.get("model", {})
    model = FallDetector(
        n_subcarriers=checkpoint.get("n_subcarriers", 52),
        conv_channels=model_section.get("conv_channels", [64, 128, 128]),
        kernel_sizes=model_section.get("kernel_sizes", [5, 3, 3]),
        dropout=model_section.get("dropout", 0.5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        "model_config": model_config,
        "input_shape": checkpoint.get("input_shape", [1, 52, 100]),
        "n_subcarriers": checkpoint.get("n_subcarriers", 52),
        "window_size": checkpoint.get("window_size", 100),
    }
    return model, metadata
