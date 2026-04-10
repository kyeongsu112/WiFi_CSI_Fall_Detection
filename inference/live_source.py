"""Unified CSI source abstraction for the WiFall inference pipeline.

Step 4 introduces a common interface so the dashboard can run with any of:

    replay    — windows from the WiFall manifest + zip (Step 3 behaviour, unchanged)
    mock_live — synthetic random windows for end-to-end pipeline testing without hardware
    esp32     — placeholder for future live ESP32 UDP integration

Window contract (matches FallDetector in training/model.py):
    CsiWindow.data  shape (52, 100)  float32
                    axis-0 = n_subcarriers, axis-1 = window_size

Model contract (unchanged from Step 2 / Step 3):
    Model returns raw logits.
    Sigmoid is applied here in InferencePipeline, NOT inside the model.

Usage
-----
    from inference.live_source import build_source, InferencePipeline

    source   = build_source(source_mode="replay", manifest_path=..., zip_path=...,
                            config_path=..., step_delay=None)
    pipeline = InferencePipeline.from_config("configs/inference.yaml")

    stop = threading.Event()
    for step_idx, window in enumerate(source.windows(stop)):
        event = pipeline.step(window, step_idx)
        ...   # ReplayEvent with .alert_state, .probability, etc.
"""
from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from inference.replay import (
    ReplayConfig,
    ReplayEvent,
    SimpleConfirmationEngine,
    load_replay_config,
)
from training.model import load_model


# ---------------------------------------------------------------------------
# Window wrapper
# ---------------------------------------------------------------------------

@dataclass
class CsiWindow:
    """One normalised CSI window ready for model inference.

    Attributes:
        data:         float32 ndarray of shape (52, 100).
        source_file:  Human-readable origin label ("" for generated windows).
        window_index: Position within the originating session or stream.
    """
    data: np.ndarray   # (n_subcarriers=52, window_size=100), float32
    source_file: str = ""
    window_index: int = 0


# ---------------------------------------------------------------------------
# Abstract source
# ---------------------------------------------------------------------------

class CsiSource(ABC):
    """Abstract CSI window source.

    All sources honour stop_event so the SSE server can cancel the worker
    thread cleanly when a browser client disconnects.
    """

    @abstractmethod
    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield normalised CsiWindow objects until exhausted or stop_event is set."""


# ---------------------------------------------------------------------------
# Replay source
# ---------------------------------------------------------------------------

_WINDOW_SIZE = 100      # canonical; matches FallDetector checkpoint
_N_SUBCARRIERS = 52     # canonical; matches FallDetector checkpoint


class ReplaySource(CsiSource):
    """CSI source that reads windows from the WiFall manifest (replay mode).

    Loads windows from the WiFall zip in manifest order.  Padding/trimming
    to the canonical (52, 100) shape is applied here; the model and alert
    state machine are handled by InferencePipeline.

    The underlying inference/replay.py (replay_manifest function) is left
    completely unchanged — its Step 3 interface and tests remain valid.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        zip_path: str | Path = "data/WiFall.zip",
        config_path: str | Path = "configs/inference.yaml",
        step_delay: float | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.zip_path = zip_path
        self.config_path = config_path
        self.step_delay = step_delay

    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield one CsiWindow per manifest row, in manifest order."""
        import pandas as pd
        from datasets.loader import load_csi_window

        cfg = load_replay_config(self.config_path)
        delay = self.step_delay if self.step_delay is not None else cfg.step_delay_seconds

        manifest = pd.read_csv(self.manifest_path)

        for row in manifest.itertuples(index=False):
            if stop_event is not None and stop_event.is_set():
                return

            raw = load_csi_window(
                str(row.source_file),
                int(row.start_row),
                int(row.end_row),
                self.zip_path,
            )  # (T, 52)

            t = raw.shape[0]
            if t < _WINDOW_SIZE:
                raw = np.pad(raw, ((0, _WINDOW_SIZE - t), (0, 0)))
            else:
                raw = raw[:_WINDOW_SIZE]

            # (window_size, n_subcarriers) → (n_subcarriers, window_size)
            yield CsiWindow(
                data=raw.T.astype(np.float32),
                source_file=str(row.source_file),
                window_index=int(row.window_index),
            )

            if delay > 0:
                time.sleep(delay)

            if stop_event is not None and stop_event.is_set():
                return


# ---------------------------------------------------------------------------
# Mock live source
# ---------------------------------------------------------------------------

class MockLiveSource(CsiSource):
    """CSI source that generates synthetic random windows for pipeline testing.

    Produces white-noise arrays scaled to a realistic CSI amplitude range.
    The model will score these as non-fall (low probability) since they lack
    the structured amplitude patterns present in WiFall data.

    Purpose: exercise the full inference pipeline (source → model → state
    machine → SSE) without WiFall.zip or ESP32 hardware.  The source runs
    indefinitely until stop_event is set.
    """

    def __init__(
        self,
        step_delay_seconds: float = 0.05,
        seed: int | None = None,
    ) -> None:
        self.step_delay_seconds = step_delay_seconds
        self._rng = np.random.default_rng(seed)

    def windows(
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        """Yield random (52, 100) windows indefinitely at step_delay_seconds rate."""
        step = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            # Gaussian noise scaled to typical WiFall CSI amplitude range
            data = self._rng.normal(
                loc=50.0, scale=15.0, size=(_N_SUBCARRIERS, _WINDOW_SIZE)
            ).astype(np.float32)
            data = np.clip(data, 0.0, None)

            yield CsiWindow(
                data=data,
                source_file="mock://live",
                window_index=step,
            )

            step += 1

            if self.step_delay_seconds > 0:
                time.sleep(self.step_delay_seconds)

            if stop_event is not None and stop_event.is_set():
                return


# ---------------------------------------------------------------------------
# ESP32 source (placeholder)
# ---------------------------------------------------------------------------

class Esp32Source(CsiSource):
    """Placeholder for future live ESP32 UDP source.

    Not yet implemented.  Raises NotImplementedError at construction time.

    Future implementation should:
      1. Open a UDP socket on the configured host/port.
      2. Receive esp32_adr018-format packets (see collector/ for parsing logic).
      3. Buffer incoming CSI rows into 100-row windows (stride configurable).
      4. Yield CsiWindow(data=(52,100), source_file="esp32://...", window_index=N).

    See configs/inference.live_esp32.example.yaml for the expected transport
    parameters, window shape, and model path.
    """

    def __init__(self, **kwargs: object) -> None:
        raise NotImplementedError(
            "Esp32Source is not yet implemented.\n"
            "To use live ESP32 input:\n"
            "  1. Implement UDP reception in inference/live_source.py (Esp32Source).\n"
            "  2. Set source_mode: esp32 in configs/inference.yaml.\n"
            "  3. See configs/inference.live_esp32.example.yaml for transport details."
        )

    def windows(  # pragma: no cover
        self, stop_event: threading.Event | None = None
    ) -> Generator[CsiWindow, None, None]:
        raise NotImplementedError
        yield  # make mypy happy


# ---------------------------------------------------------------------------
# Inference pipeline (model + sigmoid + state machine)
# ---------------------------------------------------------------------------

class InferencePipeline:
    """Runs model inference and alert-state tracking on CsiWindow objects.

    Owns the model, threshold, and SimpleConfirmationEngine.  Consumes one
    CsiWindow per call to step() and returns a fully-populated ReplayEvent.

    The model contract is unchanged from Step 2/3:
      - FallDetector returns raw logits.
      - Sigmoid is applied here, not inside the model.
    """

    def __init__(self, cfg: ReplayConfig) -> None:
        self.cfg = cfg
        self._model, _ = load_model(cfg.model_path)
        self._engine = SimpleConfirmationEngine(
            threshold=cfg.candidate_threshold,
            confirm_n=cfg.confirm_n_windows,
            cooldown_n=cfg.cooldown_windows,
        )

    @classmethod
    def from_config(cls, config_path: str | Path) -> "InferencePipeline":
        """Construct from an inference.yaml path."""
        return cls(load_replay_config(config_path))

    def step(self, window: CsiWindow, step_idx: int) -> ReplayEvent:
        """Score one window and return a ReplayEvent with current alert state.

        Args:
            window:   A CsiWindow with .data of shape (52, 100).
            step_idx: Monotonically increasing step counter (set by caller).

        Returns:
            ReplayEvent with probability, predicted_label, and alert_state.
        """
        # (52, 100) → (1, 52, 100)
        x = torch.tensor(window.data[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            logit: float = self._model(x).item()

        prob: float = float(torch.sigmoid(torch.tensor(logit)))
        predicted_label = "fall" if prob >= self.cfg.candidate_threshold else "non_fall"
        alert_state = self._engine.step(prob)

        return ReplayEvent(
            step=step_idx,
            source_file=window.source_file,
            window_index=window.window_index,
            probability=prob,
            predicted_label=predicted_label,
            alert_state=alert_state,
        )


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------

def build_source(
    source_mode: str,
    *,
    manifest_path: str | Path = "artifacts/processed/wifall_manifest.csv",
    zip_path: str | Path = "data/WiFall.zip",
    config_path: str | Path = "configs/inference.yaml",
    step_delay: float | None = None,
) -> CsiSource:
    """Instantiate the correct CsiSource for the given mode.

    Args:
        source_mode:   One of "replay", "mock_live", "esp32".
        manifest_path: Path to wifall_manifest.csv (replay only).
        zip_path:      Path to WiFall.zip (replay only).
        config_path:   Path to inference.yaml (all modes).
        step_delay:    Override inter-window sleep (None = use config value).

    Returns:
        A CsiSource ready for use in the inference pipeline.

    Raises:
        ValueError:          Unknown source_mode.
        NotImplementedError: source_mode == "esp32" (not yet implemented).
    """
    if source_mode == "replay":
        return ReplaySource(
            manifest_path=manifest_path,
            zip_path=zip_path,
            config_path=config_path,
            step_delay=step_delay,
        )
    if source_mode == "mock_live":
        cfg = load_replay_config(config_path)
        delay = step_delay if step_delay is not None else cfg.step_delay_seconds
        return MockLiveSource(step_delay_seconds=delay)
    if source_mode == "esp32":
        return Esp32Source()
    raise ValueError(
        f"Unknown source_mode: {source_mode!r}. "
        "Valid choices: 'replay', 'mock_live', 'esp32'."
    )


# ---------------------------------------------------------------------------
# Source mode resolution (precedence helper used by script and server)
# ---------------------------------------------------------------------------

_DEFAULT_SOURCE_MODE = "replay"


def resolve_source_mode(
    cli_source: str | None,
    config_path: str | Path = "configs/inference.yaml",
) -> str:
    """Return the effective source mode, applying the canonical precedence rule.

    Precedence (highest → lowest):
      1. cli_source  — explicit --source CLI argument (not None).
      2. source_mode field in the inference config file.
      3. "replay"    — safe hardcoded default only when the config file is
                       absent (FileNotFoundError / OSError).

    Silent fallback is intentionally narrow.  The following always propagate
    so that misconfiguration is visible immediately:
      - Malformed YAML             → yaml.YAMLError (or a subclass)
      - Invalid source_mode value  → pydantic.ValidationError
      - Any other unexpected error → re-raised as-is

    Args:
        cli_source:   Value of --source from argparse, or
                      os.environ.get("WIFALL_SOURCE_MODE").
                      Pass None when the flag / env var was not supplied.
        config_path:  Path to inference.yaml.

    Returns:
        One of "replay", "mock_live", or "esp32".

    Raises:
        yaml.YAMLError:          config file exists but is not valid YAML.
        pydantic.ValidationError: config parses but contains an invalid
                                  source_mode value.
    """
    if cli_source is not None:
        return cli_source

    try:
        from shared.config import load_inference_config
        return load_inference_config(Path(config_path)).source_mode
    except (FileNotFoundError, OSError):
        # Config file is genuinely absent — use the safe default.
        return _DEFAULT_SOURCE_MODE
