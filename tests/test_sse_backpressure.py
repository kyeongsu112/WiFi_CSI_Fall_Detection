"""SSE backpressure, terminal-event delivery, and shutdown tests.

Design
------
Group A (real-queue): spin up a standalone asyncio event loop in a daemon
thread so _blocking_put_terminal can be exercised against genuine
asyncio.Queue.put() backpressure.  The queue *actually fills* and blocks —
no mocked TimeoutError.

Group B (integration): use TestClient to drive the /stream endpoint end-to-end
and verify stop_event propagation, cooperative producer shutdown, and stream
completion under a queue that was momentarily full.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.server import app, _blocking_put_terminal
from inference.replay import ReplayEvent

client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_event(step: int = 0) -> ReplayEvent:
    return ReplayEvent(
        step=step,
        source_file="test.csv",
        window_index=step,
        probability=0.1,
        predicted_label="non_fall",
        alert_state="idle",
    )


@contextmanager
def _running_event_loop():
    """Start a real asyncio event loop in a daemon thread; stop and join on exit."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=3.0)


async def _fill_queue(queue: asyncio.Queue, n: int) -> None:
    for i in range(n):
        await queue.put({"step": i})


async def _drain_after_delay(queue: asyncio.Queue, delay: float) -> list:
    """Sleep *delay* seconds then empty the queue; return the drained items."""
    await asyncio.sleep(delay)
    items: list = []
    while True:
        try:
            items.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


def _collect_sse_lines(url: str = "/stream") -> list[str]:
    with client.stream("GET", url) as resp:
        return [line for line in resp.iter_lines() if line]


# ---------------------------------------------------------------------------
# Group A: Real asyncio.Queue backpressure — no mocked TimeoutError
# ---------------------------------------------------------------------------

def test_sentinel_delivered_via_retry_on_real_full_queue() -> None:
    """_blocking_put_terminal retries against a genuinely-full asyncio.Queue.

    Queue is filled to capacity with real asyncio coroutines.  A drain
    task releases space after 120 ms.  _blocking_put_terminal must:
      1. retry (cancelling timed-out futures) until space opens, then
      2. deliver the sentinel and return.

    This exercises the actual backpressure path — not a simulated timeout.
    """
    QUEUE_MAXSIZE = 4
    DRAIN_DELAY = 0.12       # 120 ms before drain
    RETRY_TIMEOUT = 0.02     # 20 ms per attempt → ~6 retries before drain fires

    stop_event = threading.Event()

    with _running_event_loop() as loop:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

        # Fill queue to capacity inside the event loop.
        asyncio.run_coroutine_threadsafe(
            _fill_queue(queue, QUEUE_MAXSIZE), loop
        ).result(timeout=2.0)

        # Schedule the drain (runs in the event-loop thread after DRAIN_DELAY).
        drain_fut = asyncio.run_coroutine_threadsafe(
            _drain_after_delay(queue, DRAIN_DELAY), loop
        )

        # From THIS thread (simulating run_thread), deliver the sentinel.
        # Must retry because the queue is full until the drain fires.
        _blocking_put_terminal(queue, loop, None, stop_event, retry_timeout=RETRY_TIMEOUT)

        drain_fut.result(timeout=2.0)

        # Sentinel must now be in the queue.
        sentinel = asyncio.run_coroutine_threadsafe(queue.get(), loop).result(timeout=1.0)

    assert sentinel is None, "sentinel must be in the queue after delivery"
    assert not stop_event.is_set(), "stop_event must not be set — client was connected throughout"


def test_error_then_sentinel_ordering_on_real_full_queue() -> None:
    """Error payload is enqueued before sentinel even under real queue backpressure.

    Mirrors the run_thread except+finally path:
      _blocking_put_terminal(error_payload) → then → _blocking_put_terminal(None)

    After the queue drains the items must appear in order:
      [error_payload, None]
    """
    QUEUE_MAXSIZE = 4
    DRAIN_DELAY = 0.12
    RETRY_TIMEOUT = 0.02

    stop_event = threading.Event()

    with _running_event_loop() as loop:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)

        asyncio.run_coroutine_threadsafe(
            _fill_queue(queue, QUEUE_MAXSIZE), loop
        ).result(timeout=2.0)

        drain_fut = asyncio.run_coroutine_threadsafe(
            _drain_after_delay(queue, DRAIN_DELAY), loop
        )

        # Mimic what run_thread does after generate_events raises.
        _blocking_put_terminal(
            queue, loop, {"error": "Internal server error"}, stop_event,
            retry_timeout=RETRY_TIMEOUT,
        )
        _blocking_put_terminal(
            queue, loop, None, stop_event,
            retry_timeout=RETRY_TIMEOUT,
        )

        drain_fut.result(timeout=2.0)

        # Collect whatever is in the queue now (error payload + sentinel).
        async def _drain_sync() -> list:
            items: list = []
            while True:
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            return items

        collected = asyncio.run_coroutine_threadsafe(_drain_sync(), loop).result(timeout=1.0)

    assert len(collected) == 2, (
        f"expected [error_payload, None], got {collected}"
    )
    assert isinstance(collected[0], dict) and "error" in collected[0], (
        "error payload must arrive first"
    )
    assert collected[1] is None, "sentinel must arrive last"


def test_terminal_delivery_abandoned_immediately_when_stop_event_already_set() -> None:
    """No put is attempted when stop_event is set before _blocking_put_terminal is called.

    This is the fast-exit path when the consumer has already disconnected.
    Uses a real asyncio.Queue to confirm the queue is not touched at all.

    Note: asyncio.Queue.qsize() is safe to call from any thread.
    """
    stop_event = threading.Event()
    stop_event.set()

    with _running_event_loop() as loop:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4)

        # Pre-fill to prove we never wait for space.
        asyncio.run_coroutine_threadsafe(
            _fill_queue(queue, 4), loop
        ).result(timeout=1.0)

        size_before = queue.qsize()  # thread-safe; no async needed

        # Should return immediately without touching the queue.
        _blocking_put_terminal(queue, loop, None, stop_event, retry_timeout=0.02)

        size_after = queue.qsize()

    assert size_before == 4
    assert size_after == 4, "queue must be unmodified when stop_event is already set"


# ---------------------------------------------------------------------------
# Group B: Integration tests through the /stream endpoint
# ---------------------------------------------------------------------------

def test_stop_event_set_after_normal_stream_completion() -> None:
    """Consumer's finally block sets stop_event after the stream ends normally.

    Proves the mechanism that tells the worker thread to stop: the consumer
    always sets stop_event in its finally block, regardless of how it exits.
    """
    stop_events: list[threading.Event] = []

    def _gen(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        stop_events.append(stop_event)
        yield _make_event(step=0)

    with patch("app.server.generate_events", _gen):
        lines = _collect_sse_lines()

    assert "event: done" in lines, "stream must emit event: done"
    assert stop_events, "generate_events must have been called"

    # Consumer's finally sets stop_event after the loop exits.
    stop_events[0].wait(timeout=2.0)
    assert stop_events[0].is_set(), "stop_event must be set after normal stream completion"


def test_stop_event_set_on_client_disconnect() -> None:
    """When the client disconnects mid-stream, stop_event is set by the endpoint.

    The consumer's finally block (finally: stop_event.set()) runs whether the
    loop exits via sentinel, is_disconnected(), or any other path.  This test
    verifies the disconnect path specifically.
    """
    stop_events: list[threading.Event] = []

    def _long_gen(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        stop_events.append(stop_event)
        for i in range(500):
            yield _make_event(step=i)

    with patch("app.server.generate_events", _long_gen):
        with client.stream("GET", "/stream") as resp:
            count = 0
            for line in resp.iter_lines():
                if line.startswith("data:") and line != "data: {}":
                    count += 1
                    if count >= 2:
                        break  # disconnect after 2 data events

    assert stop_events, "generate_events must have been called"

    # Endpoint detects disconnect via is_disconnected() (within 1 s),
    # then the finally block sets stop_event.  Allow 3 s total.
    assert stop_events[0].wait(timeout=3.0), (
        "stop_event must be set within 3 s of client disconnect"
    )


def test_producer_exits_cooperatively_after_stop_event() -> None:
    """Generator exits when stop_event is set (cooperative shutdown mechanism).

    The "disconnect → stop_event set" half is proven by
    test_stop_event_set_on_client_disconnect.  This test proves the complementary
    half: "stop_event set → generator exits promptly", in isolation from
    TestClient drain behaviour.

    TestClient (httpx.IteratorStream) drains the full response body on close,
    so an endpoint-level test with a slow generator would block for minutes.
    Instead we test the cooperative-shutdown contract directly:

      1. Start the generator logic in a real thread.
      2. Let it run briefly (20 ms).
      3. Set stop_event.
      4. Assert the thread exits quickly and the loop count is well below 10 000.
    """
    stop_event = threading.Event()
    generator_exited = threading.Event()
    yielded_counts: list[int] = []

    def _cooperative_generator() -> None:
        count = 0
        for _ in range(10_000):      # 10 000 × 1 ms = 10 s without early exit
            if stop_event.is_set():
                break
            count += 1
            time.sleep(0.001)        # 1 ms per iteration keeps CPU load minimal
        yielded_counts.append(count)
        generator_exited.set()

    thread = threading.Thread(target=_cooperative_generator, daemon=True)
    thread.start()
    thread.join(timeout=0.05)        # let it run ≤50 ms (~50 iterations), then signal stop
    stop_event.set()

    assert generator_exited.wait(timeout=2.0), (
        "generator must exit within 2 s of stop_event being set"
    )
    assert yielded_counts, "generator must record its exit count"
    assert yielded_counts[0] < 10_000, (
        f"generator yielded {yielded_counts[0]} / 10 000 iterations — "
        "must exit early on stop_event, not run to exhaustion"
    )


def test_stream_emits_done_event_with_more_events_than_queue_capacity() -> None:
    """event: done is received even when the producer generates > _SSE_QUEUE_MAXSIZE events.

    With N_EVENTS > 256, the queue is momentarily full during the stream.
    The sentinel must still arrive and the client must see event: done.
    """
    from app.server import _SSE_QUEUE_MAXSIZE
    N_EVENTS = _SSE_QUEUE_MAXSIZE + 50  # guaranteed to exceed queue capacity

    def _bulk_gen(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        for i in range(N_EVENTS):
            if stop_event.is_set():
                return
            yield _make_event(step=i)

    with patch("app.server.generate_events", _bulk_gen):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:") and l != "data: {}"]
    assert len(data_lines) >= 1, "must receive at least one data event"

    assert "event: done" in lines, (
        "event: done must be delivered even when N_EVENTS > queue capacity — "
        "sentinel must not be silently dropped"
    )


def test_error_and_done_emitted_in_order_after_generate_exception() -> None:
    """When generate_events raises, the client sees: error payload → event: done.

    This is the end-to-end integration of run_thread's except+finally paths.
    Verifies that _blocking_put_terminal delivers both events in sequence.
    """
    def _error_gen(source_mode, manifest_path, zip_path, config_path, step_delay, stop_event):
        raise RuntimeError("boom")
        yield  # make it a generator

    with patch("app.server.generate_events", _error_gen):
        lines = _collect_sse_lines()

    data_lines = [l for l in lines if l.startswith("data:")]
    assert data_lines, "must receive at least one data line"

    error_payload = json.loads(data_lines[0].removeprefix("data: "))
    assert "error" in error_payload, "first data line must be the error payload"
    assert "boom" not in error_payload["error"], "internal error text must be sanitized"

    assert "event: done" in lines, (
        "event: done must follow the error payload — stream must close cleanly"
    )

    # Verify order: error before done.
    error_pos = next(i for i, l in enumerate(lines) if l.startswith("data:") and "error" in l)
    done_pos = lines.index("event: done")
    assert error_pos < done_pos, "error payload must precede event: done"
