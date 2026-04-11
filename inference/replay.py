"""Replay engine for WiFall fall-detection demo.

Loads the trained baseline model and iterates through all windows in the
manifest CSV, simulating pseudo real-time input.  Each window is scored by
the model; a simple consecutive-window state machine drives the alert states:

    idle  →  candidate  →  confirmed  →  cooldown  →  idle

Model contract (unchanged from Step 2):
  - Model returns raw logits.
  - Sigmoid is applied HERE, not inside the model.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import torch
import yaml

from inference.confirmation import (
    ConfirmationConfig,
    ConfirmationEngine,
    compute_motion_score,
)
from training.model import load_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ReplayConfig:
    model_path: str = "artifacts/models/wifall_baseline.pt"
    candidate_threshold: float = 0.01
    post_fall_inactivity_seconds: float = 6.0
    motion_floor_threshold: float = 0.15
    confirm_window_seconds: float = 8.0
    cooldown_seconds: float = 5.0
    window_duration_seconds: float = 1.0
    step_delay_seconds: float = 0.05 # inter-window sleep for demo pacing
    # confirm_n_windows / cooldown_windows were read here historically but are
    # not wired to the active ConfirmationEngine — removed to avoid confusion.


def load_replay_config(config_path: str | Path = "configs/inference.yaml") -> ReplayConfig:
    """Load ReplayConfig from inference.yaml."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ReplayConfig(
        model_path=raw.get("model_path", "artifacts/models/wifall_baseline.pt"),
        candidate_threshold=float(raw.get("candidate_threshold", 0.01)),
        post_fall_inactivity_seconds=float(raw.get("post_fall_inactivity_seconds", 6.0)),
        motion_floor_threshold=float(raw.get("motion_floor_threshold", 0.15)),
        confirm_window_seconds=float(raw.get("confirm_window_seconds", 8.0)),
        cooldown_seconds=float(raw.get("cooldown_seconds", 5.0)),
        window_duration_seconds=float(raw.get("window_duration_seconds", 1.0)),
        step_delay_seconds=float(raw.get("step_delay_seconds", 0.05)),
    )


# ---------------------------------------------------------------------------
# Event
# ---------------------------------------------------------------------------

@dataclass
class ReplayEvent:
    step: int
    source_file: str
    window_index: int
    probability: float
    predicted_label: str   # "fall" | "non_fall"
    alert_state: str       # "idle" | "candidate" | "confirmed" | "cooldown"
    motion_score: float | None = None
    source_status: dict[str, object] | None = None

    def to_dict(self) -> dict:
        payload = {
            "step": self.step,
            "source_file": self.source_file,
            "window_index": self.window_index,
            "probability": round(self.probability, 6),
            "motion_score": (
                round(self.motion_score, 6) if self.motion_score is not None else None
            ),
            "predicted_label": self.predicted_label,
            "alert_state": self.alert_state,
        }
        if self.source_status is not None:
            payload["source_status"] = dict(self.source_status)
        return payload


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class SimpleConfirmationEngine:
    """Consecutive-window state machine — NOT used by the live inference pipeline.

    The active server path (InferencePipeline in inference/live_source.py) and
    replay_manifest() both use the time-based ConfirmationEngine from
    inference/confirmation.py.  This class is retained for reference; it is
    not instantiated anywhere in the running codebase.

    Consecutive-window state machine for replay demo.

    Transitions:
      idle      → candidate  : prob >= threshold (first candidate window)
      candidate → candidate  : prob >= threshold, consecutive_count < confirm_n
      candidate → confirmed  : prob >= threshold, consecutive_count >= confirm_n
      candidate → idle       : prob < threshold (streak broken)
      confirmed → cooldown   : automatic on next step
      cooldown  → idle       : after cooldown_n windows
    """

    def __init__(self, threshold: float, confirm_n: int, cooldown_n: int) -> None:
        self.threshold = threshold
        self.confirm_n = confirm_n
        self.cooldown_n = cooldown_n
        self.state = "idle"
        self._consecutive = 0
        self._cooldown_left = 0

    def step(self, probability: float) -> str:
        # --- cooldown ---
        if self.state == "cooldown":
            self._cooldown_left -= 1
            if self._cooldown_left <= 0:
                self.state = "idle"
                self._consecutive = 0
            return self.state

        is_above = probability >= self.threshold

        # --- idle ---
        if self.state == "idle":
            if is_above:
                self.state = "candidate"
                self._consecutive = 1
            return self.state

        # --- candidate ---
        if self.state == "candidate":
            if is_above:
                self._consecutive += 1
                if self._consecutive >= self.confirm_n:
                    self.state = "confirmed"
            else:
                self.state = "idle"
                self._consecutive = 0
            return self.state

        # --- confirmed (transition to cooldown on very next step) ---
        if self.state == "confirmed":
            self.state = "cooldown"
            self._cooldown_left = self.cooldown_n
            self._consecutive = 0
            return self.state

        return self.state  # should not reach


# ---------------------------------------------------------------------------
# Replay generator
# ---------------------------------------------------------------------------

def replay_manifest(
    manifest_path: str | Path,
    zip_path: str | Path = "data/WiFall.zip",
    config_path: str | Path = "configs/inference.yaml",
    step_delay: float | None = None,
    stop_event: threading.Event | None = None,
) -> Generator[ReplayEvent, None, None]:
    """Iterate every window in the manifest and yield a ReplayEvent per step.

    Args:
        manifest_path:  Path to wifall_manifest.csv.
        zip_path:       Path to WiFall.zip (used by the CSI loader).
        config_path:    Path to inference.yaml (source of truth).
        step_delay:     Seconds to sleep between windows.  None → use config value.
        stop_event:     If set, the generator returns early on the next window
                        boundary.  Used by the SSE server to cancel replay when
                        the client disconnects.

    Yields:
        ReplayEvent for each window in manifest order.
    """
    from datasets.loader import load_csi_window

    cfg = load_replay_config(config_path)
    delay = step_delay if step_delay is not None else cfg.step_delay_seconds

    model, meta = load_model(cfg.model_path)
    window_size: int = meta.get("window_size", 100)
    n_subcarriers: int = meta.get("n_subcarriers", 52)

    engine = ConfirmationEngine(
        ConfirmationConfig(
            candidate_threshold=cfg.candidate_threshold,
            inactivity_seconds=cfg.post_fall_inactivity_seconds,
            motion_threshold=cfg.motion_floor_threshold,
            confirm_window_seconds=cfg.confirm_window_seconds,
            cooldown_seconds=cfg.cooldown_seconds,
        )
    )
    previous_window: np.ndarray | None = None

    manifest = pd.read_csv(manifest_path)

    for step, row in enumerate(manifest.itertuples(index=False)):
        source_file: str = row.source_file
        window_index: int = int(row.window_index)
        start_row: int = int(row.start_row)
        end_row: int = int(row.end_row)

        # Load CSI window → (T, 52)
        window = load_csi_window(source_file, start_row, end_row, zip_path)

        # Normalise to exactly (window_size, n_subcarriers)
        t = window.shape[0]
        if t < window_size:
            window = np.pad(window, ((0, window_size - t), (0, 0)))
        else:
            window = window[:window_size]

        # (window_size, n_subcarriers) → transpose → (n_subcarriers, window_size)
        # then add batch dim → (1, n_subcarriers, window_size)
        window_data = window.T.astype(np.float32)
        x = torch.tensor(window_data[np.newaxis], dtype=torch.float32)

        with torch.no_grad():
            logit: float = model(x).item()

        prob: float = float(torch.sigmoid(torch.tensor(logit)))
        motion_score = compute_motion_score(window_data, previous_window)
        previous_window = window_data

        predicted_label = "fall" if prob >= cfg.candidate_threshold else "non_fall"
        alert_state = engine.step(prob, motion_score, cfg.window_duration_seconds)

        yield ReplayEvent(
            step=step,
            source_file=source_file,
            window_index=window_index,
            probability=prob,
            motion_score=motion_score,
            predicted_label=predicted_label,
            alert_state=alert_state,
        )

        if delay > 0:
            time.sleep(delay)

        if stop_event is not None and stop_event.is_set():
            return
