"""Tests for the Step 4 CSI source abstraction (inference/live_source.py).

Covers:
  - build_source() factory: correct type returned per mode, error on unknown
  - MockLiveSource: window shape, dtype, source label, stop_event cancellation
  - ReplaySource: class instantiation (integration skipped without WiFall.zip)
  - Esp32Source: raises NotImplementedError
  - InferencePipeline: step() returns ReplayEvent with correct structure
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from inference.live_source import (
    CsiWindow,
    Esp32Source,
    InferencePipeline,
    MockLiveSource,
    ReplaySource,
    build_source,
    resolve_source_mode,
)
from inference.replay import ReplayConfig, ReplayEvent


# ---------------------------------------------------------------------------
# resolve_source_mode precedence
# ---------------------------------------------------------------------------

def _minimal_config(tmp_path, source_mode: str) -> str:
    """Write the minimal valid InferenceConfig YAML and return its path."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text(f"source_mode: {source_mode}\n", encoding="utf-8")
    return str(cfg)


def test_resolve_cli_overrides_config(tmp_path) -> None:
    """Explicit --source beats whatever is in the config file."""
    cfg_path = _minimal_config(tmp_path, "mock_live")
    result = resolve_source_mode("replay", config_path=cfg_path)
    assert result == "replay"


def test_resolve_config_used_when_cli_is_none(tmp_path) -> None:
    """When no CLI flag is given, source_mode from config is used."""
    cfg_path = _minimal_config(tmp_path, "mock_live")
    result = resolve_source_mode(None, config_path=cfg_path)
    assert result == "mock_live"


def test_resolve_all_config_source_modes(tmp_path) -> None:
    """All three valid source_mode values round-trip through resolve."""
    for mode in ("replay", "mock_live", "esp32"):
        cfg_path = _minimal_config(tmp_path, mode)
        assert resolve_source_mode(None, config_path=cfg_path) == mode


def test_resolve_falls_back_to_replay_when_config_file_missing() -> None:
    """A genuinely absent config file yields 'replay' — not an error."""
    result = resolve_source_mode(None, config_path="no_such_file.yaml")
    assert result == "replay"


def test_resolve_falls_back_to_replay_when_source_mode_absent_from_yaml(
    tmp_path,
) -> None:
    """A config that omits source_mode uses the Pydantic default ('replay')."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text("candidate_threshold: 0.50\n", encoding="utf-8")
    result = resolve_source_mode(None, config_path=str(cfg))
    assert result == "replay"


def test_resolve_raises_on_invalid_source_mode_in_config(tmp_path) -> None:
    """An invalid source_mode value must propagate as ValidationError, not fall back."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text("source_mode: bluetooth\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        resolve_source_mode(None, config_path=str(cfg))


def test_resolve_raises_on_malformed_yaml(tmp_path) -> None:
    """A malformed YAML file must propagate, not fall back to replay."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text("{invalid: yaml: {{{broken\n", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        resolve_source_mode(None, config_path=str(cfg))


def test_resolve_cli_none_string_is_not_treated_as_absent() -> None:
    """Only Python None means 'absent'; any other value is used as-is."""
    result = resolve_source_mode("mock_live", config_path="no_such_file.yaml")
    assert result == "mock_live"


# ---------------------------------------------------------------------------
# build_source factory
# ---------------------------------------------------------------------------

def test_build_source_replay_returns_replay_source() -> None:
    src = build_source("replay", manifest_path="does_not_matter.csv")
    assert isinstance(src, ReplaySource)


def test_build_source_mock_live_returns_mock_source() -> None:
    src = build_source("mock_live")
    assert isinstance(src, MockLiveSource)


def test_build_source_esp32_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        build_source("esp32")


def test_build_source_unknown_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown source_mode"):
        build_source("bluetooth")


def test_build_source_mock_live_inherits_config_delay(tmp_path) -> None:
    """build_source reads step_delay_seconds from config when delay is None."""
    cfg_file = tmp_path / "inference.yaml"
    cfg_file.write_text(
        "model_path: artifacts/models/wifall_baseline.pt\n"
        "candidate_threshold: 0.50\n"
        "source_mode: mock_live\n"
        "post_fall_inactivity_seconds: 6\n"
        "motion_floor_threshold: 0.15\n"
        "confirm_window_seconds: 8\n"
        "cooldown_seconds: 5\n"
        "health_timeout_seconds: 5\n"
        "confirm_n_windows: 3\n"
        "cooldown_windows: 10\n"
        "step_delay_seconds: 0.123\n",
        encoding="utf-8",
    )
    src = build_source("mock_live", config_path=str(cfg_file))
    assert isinstance(src, MockLiveSource)
    assert src.step_delay_seconds == pytest.approx(0.123)


def test_build_source_step_delay_override() -> None:
    src = build_source("mock_live", step_delay=0.42)
    assert isinstance(src, MockLiveSource)
    assert src.step_delay_seconds == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# MockLiveSource
# ---------------------------------------------------------------------------

def test_mock_live_yields_correct_shape() -> None:
    stop = threading.Event()
    src = MockLiveSource(step_delay_seconds=0, seed=0)
    windows: list[CsiWindow] = []
    for w in src.windows(stop):
        windows.append(w)
        if len(windows) >= 5:
            stop.set()

    assert len(windows) == 5
    for w in windows:
        assert w.data.shape == (52, 100), f"bad shape: {w.data.shape}"
        assert w.data.dtype == np.float32
        assert np.all(w.data >= 0), "negative amplitudes in mock window"


def test_mock_live_source_file_label() -> None:
    stop = threading.Event()
    src = MockLiveSource(step_delay_seconds=0, seed=1)
    for w in src.windows(stop):
        assert w.source_file == "mock://live"
        stop.set()


def test_mock_live_window_index_increments() -> None:
    stop = threading.Event()
    src = MockLiveSource(step_delay_seconds=0, seed=2)
    indices = []
    for w in src.windows(stop):
        indices.append(w.window_index)
        if len(indices) >= 4:
            stop.set()
    assert indices == [0, 1, 2, 3]


def test_mock_live_respects_stop_event() -> None:
    stop = threading.Event()
    src = MockLiveSource(step_delay_seconds=0, seed=3)
    count = 0
    for _ in src.windows(stop):
        count += 1
        if count == 3:
            stop.set()
    assert count == 3


def test_mock_live_no_stop_event_runs_n_steps() -> None:
    """Without stop_event, MockLiveSource can be iterated with islice."""
    import itertools
    src = MockLiveSource(step_delay_seconds=0, seed=99)
    windows = list(itertools.islice(src.windows(None), 10))
    assert len(windows) == 10
    assert all(w.data.shape == (52, 100) for w in windows)


# ---------------------------------------------------------------------------
# ReplaySource (unit — no WiFall.zip required)
# ---------------------------------------------------------------------------

def test_replay_source_instantiation() -> None:
    """ReplaySource can be constructed without accessing the filesystem."""
    src = ReplaySource(
        manifest_path="artifacts/processed/wifall_manifest.csv",
        zip_path="data/WiFall.zip",
    )
    assert isinstance(src, ReplaySource)


# ---------------------------------------------------------------------------
# Esp32Source
# ---------------------------------------------------------------------------

def test_esp32_source_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Esp32Source"):
        Esp32Source()


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

def _make_pipeline() -> InferencePipeline:
    """Build an InferencePipeline with a mocked model (no .pt file needed)."""
    cfg = ReplayConfig(
        model_path="artifacts/models/wifall_baseline.pt",
        candidate_threshold=0.50,
        confirm_n_windows=3,
        cooldown_windows=10,
        step_delay_seconds=0.0,
    )
    pipeline = object.__new__(InferencePipeline)
    pipeline.cfg = cfg

    # Mock model: always returns logit=2.0 → sigmoid≈0.88 → "fall"
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.item.return_value = 2.0
    pipeline._model = mock_model

    from inference.replay import SimpleConfirmationEngine
    pipeline._engine = SimpleConfirmationEngine(
        threshold=cfg.candidate_threshold,
        confirm_n=cfg.confirm_n_windows,
        cooldown_n=cfg.cooldown_windows,
    )
    return pipeline


def test_inference_pipeline_step_returns_replay_event() -> None:
    pipeline = _make_pipeline()
    window = CsiWindow(
        data=np.zeros((52, 100), dtype=np.float32),
        source_file="test.csv",
        window_index=7,
    )
    event = pipeline.step(window, step_idx=7)
    assert isinstance(event, ReplayEvent)
    assert event.step == 7
    assert event.source_file == "test.csv"
    assert event.window_index == 7


def test_inference_pipeline_probability_above_threshold_labels_fall() -> None:
    pipeline = _make_pipeline()  # model returns logit=2.0 → prob≈0.88 > 0.50
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))
    event = pipeline.step(window, step_idx=0)
    assert event.probability > 0.50
    assert event.predicted_label == "fall"


def test_inference_pipeline_low_logit_labels_non_fall() -> None:
    pipeline = _make_pipeline()
    # Override model to return logit=-5.0 → prob≈0.007 < 0.50
    pipeline._model.return_value.item.return_value = -5.0
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))
    event = pipeline.step(window, step_idx=0)
    assert event.probability < 0.50
    assert event.predicted_label == "non_fall"


def test_inference_pipeline_state_machine_transitions() -> None:
    """Three consecutive high-prob windows should produce confirmed state."""
    pipeline = _make_pipeline()  # model returns logit=2.0 (above threshold)
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))

    states = [pipeline.step(window, i).alert_state for i in range(4)]
    # step 0: idle → candidate
    # step 1: candidate (consecutive=2)
    # step 2: confirmed (consecutive=3 >= confirm_n=3)
    # step 3: confirmed → cooldown (auto-transition)
    assert states[0] == "candidate"
    assert states[1] == "candidate"
    assert states[2] == "confirmed"
    assert states[3] == "cooldown"


def test_inference_pipeline_event_dict_has_all_fields() -> None:
    pipeline = _make_pipeline()
    window = CsiWindow(data=np.zeros((52, 100), dtype=np.float32))
    event = pipeline.step(window, 0)
    d = event.to_dict()
    for field in ("step", "source_file", "window_index", "probability",
                  "predicted_label", "alert_state"):
        assert field in d, f"missing field: {field}"
