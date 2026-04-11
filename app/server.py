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
import logging
import os
import threading
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from inference.live_source import resolve_source_mode
from inference.replay import ReplayEvent

_logger = logging.getLogger(__name__)

# Maximum number of inference events buffered between the worker thread and the
# SSE consumer.  Provides a memory ceiling while true backpressure (below)
# slows the producer rather than dropping events.
_SSE_QUEUE_MAXSIZE: int = 256

# How long (seconds) the inference thread will block waiting for queue space
# before emitting a warning and moving on.  Prevents indefinite stalls when
# a client stalls but the connection is not yet detected as disconnected.
_SSE_BACKPRESSURE_TIMEOUT: float = 5.0

app = FastAPI(title="WiFi Fall Alert Dashboard")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _blocking_put(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    item: object,
    *,
    timeout: float = _SSE_BACKPRESSURE_TIMEOUT,
) -> None:
    """Deliver a regular inference event, blocking up to *timeout* seconds.

    Uses ``run_coroutine_threadsafe`` so the asyncio ``queue.put`` coroutine
    runs in the event loop while this thread waits.  If the client is still
    connected but consuming slowly the inference thread slows to match — no
    events are dropped under normal backpressure.

    Timeout / cancellation semantics
    ---------------------------------
    If ``future.result()`` raises ``TimeoutError``, the underlying
    ``queue.put`` coroutine is still pending in the event loop.  We
    **cancel it explicitly** to prevent the item from being enqueued *after*
    the timeout — a late enqueue could corrupt stream ordering or deliver a
    "dropped" event to the consumer unexpectedly.  After cancellation the
    event is discarded and a warning is logged.

    Regular-event drop policy
    --------------------------
    Regular inference events (non-terminal) may be dropped when the queue
    stays full for longer than *timeout* seconds.  Terminal events (sentinel
    and error payload) use :func:`_blocking_put_terminal` instead, which
    retries until delivered or the consumer disconnects.
    """
    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()  # prevent late enqueue that would corrupt stream ordering
        _logger.warning(
            "SSE queue blocked for %.1f s; cancelling put and dropping one event "
            "(client may be stalled or connection is closing)",
            timeout,
        )


async def _put_with_retry(
    queue: asyncio.Queue,
    item: object,
    stop_event: threading.Event,
    retry_timeout: float,
) -> None:
    """Deliver *item* to *queue*, retrying on backpressure, from within the event loop.

    Uses ``queue.full()`` + ``queue.put_nowait()`` instead of
    ``asyncio.wait_for(queue.put(...))`` to avoid a duplicate-enqueue race:

    * ``asyncio.wait_for`` cancellation is NOT atomic with queue state.  When a
      slot opens simultaneously with the timeout firing, CPython's task machinery
      sets ``_must_cancel = True`` AFTER ``Queue._put(item)`` has already run.
      The task is then marked cancelled, ``wait_for`` raises ``TimeoutError``,
      and the retry loop enqueues the same item a second time.

    * ``queue.full()`` followed immediately by ``queue.put_nowait()`` with no
      ``await`` between them is race-free in a single-threaded asyncio event loop
      — no other coroutine can run between those two statements.

    One call to this coroutine → at most one enqueue of *item*.
    """
    while not stop_event.is_set():
        if not queue.full():
            queue.put_nowait(item)
            return  # delivered exactly once
        _logger.warning(
            "SSE queue full for %.1f s; retrying terminal event delivery "
            "(consumer is slow — will retry until delivered or client disconnects)",
            retry_timeout,
        )
        await asyncio.sleep(retry_timeout)
    _logger.debug(
        "Terminal event delivery abandoned: consumer already disconnected "
        "(stop_event is set)."
    )


def _blocking_put_terminal(
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    item: object,
    stop_event: threading.Event,
    *,
    retry_timeout: float = _SSE_BACKPRESSURE_TIMEOUT,
) -> None:
    """Deliver a terminal event by scheduling a single retry-capable coroutine.

    Schedules exactly one :func:`_put_with_retry` coroutine for the entire
    delivery of *item*.  The retry loop runs inside the event loop using a
    race-free check-then-put pattern — no duplicate-enqueue race.

    If *stop_event* is already set on entry the call returns immediately.
    The coroutine also checks *stop_event* before each retry, so a disconnect
    that occurs during delivery abandons the loop without a duplicate put.

    The sentinel ``None`` causes the consumer to emit ``event: done`` and
    exit its loop (which then sets *stop_event*).  The error payload
    ``{"error": ...}`` must be delivered before the sentinel so the client
    sees a clean completion sequence.
    """
    if stop_event.is_set():
        _logger.debug(
            "Terminal event delivery abandoned: consumer already disconnected "
            "(stop_event is set)."
        )
        return
    future = asyncio.run_coroutine_threadsafe(
        _put_with_retry(queue, item, stop_event, retry_timeout), loop
    )
    future.result()  # block until delivered or abandoned; timing managed inside


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
        queue: asyncio.Queue = asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)

        def run_thread() -> None:
            try:
                for event in generate_events(
                    source_mode, manifest_path, zip_path, config_path,
                    step_delay, stop_event,
                ):
                    _blocking_put(queue, loop, event.to_dict())
            except Exception:
                _logger.exception("Unhandled error in SSE inference worker thread")
                # Error payload is terminal: retry until delivered or disconnected.
                _blocking_put_terminal(queue, loop, {"error": "Internal server error"}, stop_event)
            finally:
                # Sentinel is terminal: retry until delivered or disconnected.
                # Delivery guarantees the consumer sends event: done and exits cleanly
                # rather than looping on keep-alives indefinitely.
                _blocking_put_terminal(queue, loop, None, stop_event)

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
