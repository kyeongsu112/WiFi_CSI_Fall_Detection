"""Cross-module integration smoke test: FallDetector → checkpoint → InferencePipeline.

Exercises the real forward-pass chain without needing WiFall.zip or a
pre-trained checkpoint.  Catches boundary mismatches between:
  - training/model.py  (FallDetector, save_model, load_model)
  - inference/live_source.py  (InferencePipeline, _get_cached_model, CsiWindow)
  - inference/confirmation.py  (ConfirmationEngine state transitions)
  - inference/replay.py  (ReplayConfig, ReplayEvent)

This test is the first that exercises the complete inference stack end-to-end
with real tensor computation — all prior tests either mock the model or stop
short of the full pipeline.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.live_source import CsiWindow, InferencePipeline
from inference.replay import ReplayConfig, ReplayEvent
from training.model import FallDetector, save_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_tiny_checkpoint(path: Path) -> None:
    """Save a randomly-initialised FallDetector checkpoint to *path*."""
    model = FallDetector(
        n_subcarriers=52,
        conv_channels=[64, 128, 128],
        kernel_sizes=[5, 3, 3],
        dropout=0.5,
    )
    model.eval()
    config = {
        "model": {
            "conv_channels": [64, 128, 128],
            "kernel_sizes": [5, 3, 3],
            "dropout": 0.5,
        },
        "data": {"n_subcarriers": 52, "window_size": 100},
    }
    save_model(model, path, config)


def _make_pipeline(model_path: str) -> InferencePipeline:
    cfg = ReplayConfig(
        model_path=model_path,
        candidate_threshold=0.50,
        post_fall_inactivity_seconds=2.0,
        motion_floor_threshold=0.15,
        confirm_window_seconds=4.0,
        cooldown_seconds=3.0,
        step_delay_seconds=0.0,
    )
    return InferencePipeline(cfg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pipeline_produces_replay_event_with_correct_types(tmp_path: Path) -> None:
    """InferencePipeline.step() must return a well-formed ReplayEvent.

    Verifies the full chain: checkpoint save → load → forward pass → event
    without any mocks.  Uses a randomly-initialised model (weights are not
    meaningful — we only check the interface contract).
    """
    ckpt = tmp_path / "models" / "smoke.pt"
    _save_tiny_checkpoint(ckpt)

    pipeline = _make_pipeline(str(ckpt))
    window = CsiWindow(
        data=np.zeros((52, 100), dtype=np.float32),
        source_file="synthetic://smoke",
        window_index=3,
    )

    event = pipeline.step(window, step_idx=3)

    assert isinstance(event, ReplayEvent)
    assert event.step == 3
    assert event.source_file == "synthetic://smoke"
    assert event.window_index == 3
    assert isinstance(event.probability, float)
    assert 0.0 <= event.probability <= 1.0
    assert event.predicted_label in ("fall", "non_fall")
    assert event.alert_state in ("idle", "candidate", "confirmed", "cooldown")
    assert event.motion_score is not None  # first window → returns 1.0


def test_pipeline_motion_score_first_window_is_one(tmp_path: Path) -> None:
    """The first window has no previous window, so motion_score must be 1.0."""
    ckpt = tmp_path / "models" / "smoke.pt"
    _save_tiny_checkpoint(ckpt)

    pipeline = _make_pipeline(str(ckpt))
    window = CsiWindow(data=np.ones((52, 100), dtype=np.float32))
    event = pipeline.step(window, step_idx=0)

    assert event.motion_score == pytest.approx(1.0)


def test_pipeline_label_matches_threshold(tmp_path: Path) -> None:
    """predicted_label must be 'fall' iff probability >= candidate_threshold."""
    ckpt = tmp_path / "models" / "smoke.pt"
    _save_tiny_checkpoint(ckpt)

    pipeline = _make_pipeline(str(ckpt))
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))
    event = pipeline.step(window, step_idx=0)

    if event.probability >= pipeline.cfg.candidate_threshold:
        assert event.predicted_label == "fall"
    else:
        assert event.predicted_label == "non_fall"


def test_pipeline_state_machine_reachable(tmp_path: Path) -> None:
    """ConfirmationEngine must transition beyond idle over repeated steps.

    Uses candidate_threshold=0.0 so that any probability (sigmoid is always
    positive) triggers a candidate transition regardless of how the random
    model weights initialise.  This makes the test deterministic without
    requiring a trained checkpoint.
    """
    ckpt = tmp_path / "models" / "smoke.pt"
    _save_tiny_checkpoint(ckpt)

    # threshold=0.0 guarantees prob >= threshold for every forward pass
    cfg = ReplayConfig(
        model_path=str(ckpt),
        candidate_threshold=0.0,
        post_fall_inactivity_seconds=2.0,
        motion_floor_threshold=0.15,
        confirm_window_seconds=4.0,
        cooldown_seconds=3.0,
        step_delay_seconds=0.0,
    )
    pipeline = InferencePipeline(cfg)
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))

    states: list[str] = []
    for i in range(20):
        event = pipeline.step(window, step_idx=i)
        states.append(event.alert_state)

    assert "candidate" in states, (
        "State machine never entered 'candidate' even with threshold=0.0"
    )
    assert len(set(states)) > 1, (
        "State machine produced only one state over 20 steps — "
        "ConfirmationEngine is not transitioning"
    )


def test_model_cache_shared_across_pipeline_instances(tmp_path: Path) -> None:
    """Two InferencePipeline instances pointing to the same checkpoint must
    share the same FallDetector object (identity check, not just equality).

    Verifies the _get_cached_model mechanism works across instances.
    """
    ckpt = tmp_path / "models" / "shared.pt"
    _save_tiny_checkpoint(ckpt)

    p1 = _make_pipeline(str(ckpt))
    p2 = _make_pipeline(str(ckpt))

    assert p1._model is p2._model, (
        "Expected both pipelines to share the same FallDetector object via "
        "_get_cached_model, but got two distinct instances"
    )


def test_checkpoint_metadata_matches_inference_constants(tmp_path: Path) -> None:
    """Saved checkpoint must record n_subcarriers=52 and window_size=100.

    These values must match the hardcoded _WINDOW_SIZE / _N_SUBCARRIERS
    constants in live_source.py that drive the ReplaySource padding logic.
    A mismatch here would silently corrupt model inputs at runtime.
    """
    from training.model import load_model
    from inference.live_source import _WINDOW_SIZE, _N_SUBCARRIERS

    ckpt = tmp_path / "models" / "meta_check.pt"
    _save_tiny_checkpoint(ckpt)

    _, metadata = load_model(ckpt)

    assert metadata["n_subcarriers"] == _N_SUBCARRIERS, (
        f"Checkpoint n_subcarriers={metadata['n_subcarriers']} != "
        f"live_source._N_SUBCARRIERS={_N_SUBCARRIERS}"
    )
    assert metadata["window_size"] == _WINDOW_SIZE, (
        f"Checkpoint window_size={metadata['window_size']} != "
        f"live_source._WINDOW_SIZE={_WINDOW_SIZE}"
    )
