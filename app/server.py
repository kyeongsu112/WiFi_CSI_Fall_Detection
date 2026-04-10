"""WiFall inference dashboard server.

FastAPI application that:
  - Serves the Jinja2 dashboard at GET /
  - Streams inference events via SSE at GET /stream
  - Exposes GET /health for smoke tests

The SSE stream can be driven by any of:
  source_mode=replay     — replays WiFall manifest windows (Step 3 default)
  source_mode=mock_live  — synthetic random windows (no hardware needed)
  source_mode=esp32      — JSON v1 UDP live source for ESP32/local senders

Configuration is passed via environment variables (set by scripts/replay_dashboard.py):

    WIFALL_SOURCE_MODE    source type: replay | mock_live | esp32  (default: replay)
    WIFALL_CONFIG_PATH    path to inference.yaml
    WIFALL_MANIFEST_PATH  path to wifall_manifest.csv  (replay mode only)
    WIFALL_ZIP_PATH       path to WiFall.zip            (replay mode only)
    WIFALL_STEP_DELAY     inter-window sleep override in seconds (optional)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from inference.live_source import resolve_source_mode
from inference.replay import ReplayEvent

app = FastAPI(title="WiFi Fall Alert Dashboard")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ---------------------------------------------------------------------------
# Event stream (module-level so tests can patch it)
# ---------------------------------------------------------------------------

def generate_events(
    source_mode: str,
    manifest_path: str,
    zip_path: str,
    config_path: str,
    step_delay: float | None,
    stop_event: threading.Event,
) -> Generator[ReplayEvent, None, None]:
    """Build the source + pipeline and yield ReplayEvents until done or cancelled.

    Defined at module level so tests can patch app.server.generate_events with a
    stub generator, avoiding real model/manifest I/O in unit tests.

    Args:
        source_mode:   "replay" | "mock_live" | "esp32"
        manifest_path: Path to wifall_manifest.csv (replay mode only).
        zip_path:      Path to WiFall.zip (replay mode only).
        config_path:   Path to inference.yaml.
        step_delay:    Override inter-window sleep (None = use config value).
        stop_event:    Set by the SSE handler when the client disconnects.
    """
    from inference.live_source import InferencePipeline, build_source

    source = build_source(
        source_mode=source_mode,
        manifest_path=manifest_path,
        zip_path=zip_path,
        config_path=config_path,
        step_delay=step_delay,
    )
    pipeline = InferencePipeline.from_config(config_path)

    for step_idx, csi_window in enumerate(source.windows(stop_event)):
        event = pipeline.step(csi_window, step_idx)
        source_status = source.get_runtime_status()
        if source_status is not None:
            event.source_status = source_status
        yield event


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/stream")
async def stream(request: Request):
    """SSE endpoint.  Each connection runs its own independent inference loop.

    The worker thread is stopped via stop_event when the client disconnects,
    preventing orphan threads on browser refresh/reconnect.
    """
    config_path   = _get_env("WIFALL_CONFIG_PATH",   "configs/inference.yaml")
    # Precedence: WIFALL_SOURCE_MODE env var > config source_mode > "replay"
    source_mode   = resolve_source_mode(os.environ.get("WIFALL_SOURCE_MODE"), config_path)
    manifest_path = _get_env("WIFALL_MANIFEST_PATH", "artifacts/processed/wifall_manifest.csv")
    zip_path      = _get_env("WIFALL_ZIP_PATH",      "data/WiFall.zip")
    delay_env     = _get_env("WIFALL_STEP_DELAY",    "")
    step_delay    = float(delay_env) if delay_env else None

    async def event_generator():
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_thread() -> None:
            try:
                for event in generate_events(
                    source_mode, manifest_path, zip_path, config_path,
                    step_delay, stop_event,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event.to_dict())
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=run_thread, daemon=True)
        thread.start()

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if payload is None:
                    yield f"event: done\ndata: {{}}\n\n"
                    break

                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
