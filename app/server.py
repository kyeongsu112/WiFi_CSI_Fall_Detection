"""WiFall replay dashboard server.

FastAPI application that:
  - Serves the Jinja2 dashboard at GET /
  - Streams replay events via SSE at GET /stream
  - Exposes GET /health for smoke tests

Configuration is passed in via environment variables set by
scripts/replay_dashboard.py before uvicorn starts:

    WIFALL_MANIFEST_PATH   path to wifall_manifest.csv
    WIFALL_ZIP_PATH        path to WiFall.zip
    WIFALL_CONFIG_PATH     path to inference.yaml
    WIFALL_STEP_DELAY      inter-window sleep in seconds (float, optional)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Module-level import so tests can patch app.server.replay_manifest
from inference.replay import replay_manifest

app = FastAPI(title="WiFi Fall Alert Dashboard")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


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
    """SSE endpoint.  Each connection triggers its own independent replay run.

    The replay worker thread is cancelled via a stop_event when the client
    disconnects (browser refresh / tab close), preventing orphan threads.
    """
    manifest_path = _get_env("WIFALL_MANIFEST_PATH", "artifacts/processed/wifall_manifest.csv")
    zip_path      = _get_env("WIFALL_ZIP_PATH",      "data/WiFall.zip")
    config_path   = _get_env("WIFALL_CONFIG_PATH",   "configs/inference.yaml")
    delay_env     = _get_env("WIFALL_STEP_DELAY",     "")
    step_delay    = float(delay_env) if delay_env else None

    async def event_generator():
        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_replay() -> None:
            try:
                for event in replay_manifest(
                    manifest_path=manifest_path,
                    zip_path=zip_path,
                    config_path=config_path,
                    step_delay=step_delay,
                    stop_event=stop_event,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event.to_dict())
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, {"error": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        thread = threading.Thread(target=run_replay, daemon=True)
        thread.start()

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Keep-alive comment keeps the browser from timing out
                    yield ": keep-alive\n\n"
                    continue

                if payload is None:
                    yield f"event: done\ndata: {{}}\n\n"
                    break

                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            # Signal the worker to stop at the next window boundary,
            # regardless of whether we exited via disconnect or normal end.
            stop_event.set()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
