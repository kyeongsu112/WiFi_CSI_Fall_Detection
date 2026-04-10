"""Smoke tests for the FastAPI dashboard routes.

Tests GET /, GET /health, and GET /stream without starting a real server or
loading the WiFall model/manifest.  All /stream tests patch
app.server.generate_events with a lightweight stub generator, avoiding real
model/manifest/zip I/O.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import app
from inference.replay import ReplayEvent

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    step: int = 0,
    alert_state: str = "idle",
    probability: float = 0.001,
    predicted_label: str = "non_fall",
    motion_score: float | None = None,
    source_status: dict | None = None,
) -> ReplayEvent:
    return ReplayEvent(
        step=step,
        source_file="WiFall/ID0/fall/test.csv",
        window_index=step,
        probability=probability,
        predicted_label=predicted_label,
        alert_state=alert_state,
        motion_score=motion_score,
        source_status=source_status,
    )


def _fake_generate(*events: ReplayEvent):
    """Return a stub for generate_events that yields the given ReplayEvents."""
    def _inner(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        yield from events
    return _inner


def _collect_sse_lines(url: str = "/stream") -> list[str]:
    """Stream GET *url* and return all non-empty lines received."""
    with client.stream("GET", url) as resp:
        return [line for line in resp.iter_lines() if line]


# ---------------------------------------------------------------------------
# Basic route tests
# ---------------------------------------------------------------------------

def test_health_returns_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_index_renders_html() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "WiFall" in resp.text


# ---------------------------------------------------------------------------
# SSE /stream tests
# ---------------------------------------------------------------------------

def test_stream_returns_event_stream_content_type() -> None:
    with patch("app.server.generate_events", _fake_generate()):
        with client.stream("GET", "/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


def test_stream_yields_at_least_one_data_event() -> None:
    event = _make_event(step=0, alert_state="candidate", probability=0.95, predicted_label="fall")
    with patch("app.server.generate_events", _fake_generate(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert len(data_lines) >= 1

    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["step"] == 0
    assert payload["alert_state"] == "candidate"
    assert payload["predicted_label"] == "fall"
    assert abs(payload["probability"] - 0.95) < 1e-4


def test_stream_data_event_fields_complete() -> None:
    """Every data payload must carry all six required dashboard fields."""
    event = _make_event(step=7, alert_state="confirmed", probability=0.99, predicted_label="fall")
    with patch("app.server.generate_events", _fake_generate(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert data_lines, "no data events received"
    payload = json.loads(data_lines[0].removeprefix("data: "))

    for field in ("step", "source_file", "window_index", "probability",
                  "predicted_label", "alert_state"):
        assert field in payload, f"missing field: {field}"


def test_stream_serializes_source_status_and_motion_score() -> None:
    event = _make_event(
        step=3,
        alert_state="candidate",
        probability=0.88,
        motion_score=0.04,
        source_status={
            "mode": "esp32",
            "transport_state": "streaming",
            "active_sid": "dev-01",
            "packets_received": 120,
            "packets_dropped": 2,
            "windows_emitted": 1,
        },
    )
    with patch("app.server.generate_events", _fake_generate(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["motion_score"] == pytest.approx(0.04)
    assert payload["source_status"]["transport_state"] == "streaming"
    assert payload["source_status"]["active_sid"] == "dev-01"


def test_stream_emits_done_event_when_replay_ends() -> None:
    with patch("app.server.generate_events", _fake_generate(_make_event())):
        lines = _collect_sse_lines()

    assert "event: done" in lines


def test_stream_emits_error_payload_on_generate_exception() -> None:
    def _error_generate(source_mode, manifest_path, zip_path, config_path,
                        step_delay, stop_event):
        raise RuntimeError("simulated generate failure")
        yield  # make it a generator

    with patch("app.server.generate_events", _error_generate):
        lines = _collect_sse_lines()

    error_lines = [l for l in lines if l.startswith("data:")]
    assert error_lines, "expected at least one data line"
    payload = json.loads(error_lines[0].removeprefix("data: "))
    assert "error" in payload
    assert "simulated generate failure" in payload["error"]


# ---------------------------------------------------------------------------
# Source mode precedence tests (server-level)
# ---------------------------------------------------------------------------

def _capture_generate(*events):
    """Return a generate_events stub that records which source_mode it received."""
    captured: dict = {}

    def _inner(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        captured["source_mode"] = source_mode
        yield from events

    return _inner, captured


def test_stream_uses_env_source_mode_when_set(monkeypatch) -> None:
    """WIFALL_SOURCE_MODE env var is honoured when explicitly set."""
    monkeypatch.setenv("WIFALL_SOURCE_MODE", "mock_live")
    stub, captured = _capture_generate(_make_event())
    with patch("app.server.generate_events", stub):
        _collect_sse_lines()
    assert captured.get("source_mode") == "mock_live"


def test_stream_reads_source_mode_from_config_when_env_absent(
    monkeypatch, tmp_path
) -> None:
    """When WIFALL_SOURCE_MODE is not set, source_mode from config is used."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text("source_mode: mock_live\n", encoding="utf-8")

    monkeypatch.delenv("WIFALL_SOURCE_MODE", raising=False)
    monkeypatch.setenv("WIFALL_CONFIG_PATH", str(cfg))

    stub, captured = _capture_generate(_make_event())
    with patch("app.server.generate_events", stub):
        _collect_sse_lines()
    assert captured.get("source_mode") == "mock_live"


def test_stream_env_overrides_config_source_mode(monkeypatch, tmp_path) -> None:
    """WIFALL_SOURCE_MODE overrides source_mode in the config file."""
    cfg = tmp_path / "inference.yaml"
    cfg.write_text("source_mode: mock_live\n", encoding="utf-8")

    monkeypatch.setenv("WIFALL_SOURCE_MODE", "replay")
    monkeypatch.setenv("WIFALL_CONFIG_PATH", str(cfg))

    stub, captured = _capture_generate(_make_event())
    with patch("app.server.generate_events", stub):
        _collect_sse_lines()
    assert captured.get("source_mode") == "replay"


def test_stream_label_is_non_fall_not_no_fall() -> None:
    """Verify the canonical 'non_fall' label (not the old 'no_fall' spelling)."""
    event = _make_event(step=0, alert_state="idle", probability=0.001, predicted_label="non_fall")
    with patch("app.server.generate_events", _fake_generate(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert data_lines
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["predicted_label"] == "non_fall"
    assert payload["predicted_label"] != "no_fall"
