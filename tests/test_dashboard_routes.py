"""Smoke tests for the FastAPI dashboard routes.

Tests GET /, GET /health, and GET /stream without starting a real server or
loading the WiFall model/manifest.  The replay_manifest function is patched
with a lightweight stub generator for all /stream tests.
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
) -> ReplayEvent:
    return ReplayEvent(
        step=step,
        source_file="WiFall/ID0/fall/test.csv",
        window_index=step,
        probability=probability,
        predicted_label=predicted_label,
        alert_state=alert_state,
    )


def _fake_replay(*events: ReplayEvent):
    """Return a stub replay_manifest that yields the given events."""
    def _inner(manifest_path, zip_path, config_path, step_delay=None, stop_event=None):
        yield from events
    return _inner


def _collect_sse_lines(url: str = "/stream", **patch_kwargs) -> list[str]:
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
    with patch("app.server.replay_manifest", _fake_replay()):
        with client.stream("GET", "/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]


def test_stream_yields_at_least_one_data_event() -> None:
    event = _make_event(step=0, alert_state="candidate", probability=0.95, predicted_label="fall")
    with patch("app.server.replay_manifest", _fake_replay(event)):
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
    with patch("app.server.replay_manifest", _fake_replay(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert data_lines, "no data events received"
    payload = json.loads(data_lines[0].removeprefix("data: "))

    for field in ("step", "source_file", "window_index", "probability", "predicted_label", "alert_state"):
        assert field in payload, f"missing field: {field}"


def test_stream_emits_done_event_when_replay_ends() -> None:
    with patch("app.server.replay_manifest", _fake_replay(_make_event())):
        lines = _collect_sse_lines()

    assert "event: done" in lines


def test_stream_emits_error_payload_on_replay_exception() -> None:
    def _error_replay(manifest_path, zip_path, config_path, step_delay=None, stop_event=None):
        raise RuntimeError("simulated replay failure")
        yield  # make it a generator

    with patch("app.server.replay_manifest", _error_replay):
        lines = _collect_sse_lines()

    error_lines = [l for l in lines if l.startswith("data:")]
    assert error_lines, "expected at least one data line"
    payload = json.loads(error_lines[0].removeprefix("data: "))
    assert "error" in payload
    assert "simulated replay failure" in payload["error"]


def test_stream_label_is_non_fall_not_no_fall() -> None:
    """Verify the canonical 'non_fall' label (not the old 'no_fall' spelling)."""
    event = _make_event(step=0, alert_state="idle", probability=0.001, predicted_label="non_fall")
    with patch("app.server.replay_manifest", _fake_replay(event)):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert data_lines
    payload = json.loads(data_lines[0].removeprefix("data: "))
    assert payload["predicted_label"] == "non_fall"
    assert payload["predicted_label"] != "no_fall"
