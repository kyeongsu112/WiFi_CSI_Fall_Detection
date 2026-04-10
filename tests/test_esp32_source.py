"""Tests for Esp32Source and the v1 UDP packet parser.

Covers:
  - _parse_esp32_packet: valid packets, missing/wrong fields, malformed JSON,
    non-UTF-8 bytes, non-numeric amplitudes, wrong amp length.
  - _parse_esp32_packet: non-finite amplitude rejection (NaN, +Inf, -Inf).
  - Esp32Source.windows(): window shape/dtype, source_file label,
    window_index incrementing, malformed packets dropped silently,
    stop_event cancellation, no-frames-before-full-window-size invariant.
  - Esp32Source.windows(): single-sender sid lock — packets from a different
    sid are dropped; the active sid is set on the first valid packet.

All network tests use loopback (127.0.0.1) with an OS-assigned free port so
they work on any machine with no configuration.
"""
from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from inference.live_source import Esp32Source, CsiWindow, _parse_esp32_packet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Ask the OS for a free UDP port and return it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_raw_packet(
    amp_len: int = 52,
    amp_value: float = 10.0,
    sid: str = "test",
    include_ts: bool = True,
    include_fi: bool = True,
) -> bytes:
    """Build a valid v1 JSON packet bytes."""
    obj: dict = {"amp": [float(amp_value)] * amp_len}
    if include_ts:
        obj["ts"] = 1_712_700_000.0
    if include_fi:
        obj["fi"] = 0
    obj["sid"] = sid
    return json.dumps(obj).encode("utf-8")


def _send_frames(
    port: int,
    count: int,
    amp_value: float = 10.0,
    sid: str = "test",
    n_subcarriers: int = 52,
    host: str = "127.0.0.1",
    delay: float = 0.0,
) -> None:
    """Send ``count`` valid UDP packets to 127.0.0.1:``port``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for i in range(count):
        pkt = json.dumps(
            {"ts": float(i), "fi": i, "amp": [float(amp_value)] * n_subcarriers, "sid": sid}
        ).encode("utf-8")
        sock.sendto(pkt, (host, port))
        if delay > 0:
            time.sleep(delay)
    sock.close()


# ---------------------------------------------------------------------------
# _parse_esp32_packet — unit tests (no socket needed)
# ---------------------------------------------------------------------------

class TestParseEsp32Packet:
    def test_valid_full_packet(self) -> None:
        raw = _make_raw_packet()
        result = _parse_esp32_packet(raw)
        assert result is not None
        arr, sid = result
        assert arr.shape == (52,)
        assert arr.dtype == np.float32
        assert sid == "test"

    def test_valid_minimal_packet_only_amp(self) -> None:
        """Only 'amp' is required; all other fields are optional."""
        raw = json.dumps({"amp": [1.0] * 52}).encode("utf-8")
        result = _parse_esp32_packet(raw)
        assert result is not None
        arr, sid = result
        assert arr.shape == (52,)
        assert sid == "esp32"  # default

    def test_amplitude_values_preserved(self) -> None:
        expected = list(range(52))
        raw = json.dumps({"amp": expected}).encode("utf-8")
        arr, _ = _parse_esp32_packet(raw)
        np.testing.assert_array_almost_equal(arr, np.array(expected, dtype=np.float32))

    def test_sid_defaults_to_esp32_when_absent(self) -> None:
        raw = json.dumps({"amp": [0.0] * 52}).encode("utf-8")
        _, sid = _parse_esp32_packet(raw)
        assert sid == "esp32"

    def test_custom_n_subcarriers(self) -> None:
        raw = json.dumps({"amp": [1.0] * 32}).encode("utf-8")
        result = _parse_esp32_packet(raw, n_subcarriers=32)
        assert result is not None
        arr, _ = result
        assert arr.shape == (32,)

    def test_wrong_amp_length_too_short(self) -> None:
        raw = _make_raw_packet(amp_len=51)
        assert _parse_esp32_packet(raw) is None

    def test_wrong_amp_length_too_long(self) -> None:
        raw = _make_raw_packet(amp_len=53)
        assert _parse_esp32_packet(raw) is None

    def test_missing_amp_key(self) -> None:
        raw = json.dumps({"ts": 1.0, "fi": 0, "sid": "x"}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_amp_is_not_a_list(self) -> None:
        raw = json.dumps({"amp": 42}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_amp_contains_non_numeric_strings(self) -> None:
        raw = json.dumps({"amp": ["a"] * 52}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_malformed_json(self) -> None:
        assert _parse_esp32_packet(b"{not valid json}") is None

    def test_non_utf8_bytes(self) -> None:
        assert _parse_esp32_packet(b"\xff\xfe\xfd\xfc") is None

    def test_empty_bytes(self) -> None:
        assert _parse_esp32_packet(b"") is None

    def test_json_array_not_object(self) -> None:
        raw = json.dumps([1.0] * 52).encode("utf-8")
        assert _parse_esp32_packet(raw) is None


# ---------------------------------------------------------------------------
# Esp32Source — integration tests over loopback UDP
# ---------------------------------------------------------------------------

class TestEsp32Source:
    """All tests use 127.0.0.1 with a free OS-assigned port."""

    def test_window_shape_and_dtype(self) -> None:
        """One full window must be shape (52, 100) float32."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=100)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 100)
        t.join(timeout=5.0)

        assert not t.is_alive(), "source did not stop within 5 s"
        assert len(windows) == 1
        assert windows[0].data.shape == (52, 100)
        assert windows[0].data.dtype == np.float32

    def test_source_file_label(self) -> None:
        """source_file must be 'esp32://<sid>'."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 10, sid="dev-01")
        t.join(timeout=5.0)

        assert windows
        assert windows[0].source_file == "esp32://dev-01"

    def test_window_index_increments(self) -> None:
        """window_index must be 0, 1, 2, … for consecutive windows."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                if len(windows) >= 3:
                    stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 30)  # 3 windows × 10 frames
        t.join(timeout=5.0)

        assert [w.window_index for w in windows] == [0, 1, 2]

    def test_no_window_before_buffer_full(self) -> None:
        """Fewer than window_size frames must produce no window."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 9)   # one short
        time.sleep(0.3)         # give source time to process
        stop.set()
        t.join(timeout=3.0)

        assert len(windows) == 0

    def test_amplitude_values_in_window(self) -> None:
        """Data values in the emitted window must match what was sent."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 10, amp_value=42.0)
        t.join(timeout=5.0)

        assert windows
        # All values should be close to 42.0
        np.testing.assert_allclose(windows[0].data, 42.0, atol=1e-3)

    def test_malformed_packets_dropped_silently(self) -> None:
        """Bad packets must be dropped; subsequent valid frames still emit windows."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Inject 5 bad packets
        for _ in range(5):
            sock.sendto(b"not json at all", ("127.0.0.1", port))
        sock.sendto(b'{"amp":"wrong_type"}', ("127.0.0.1", port))
        sock.sendto(b'{"amp":[1.0,2.0]}', ("127.0.0.1", port))  # wrong length
        sock.close()

        # Then 10 good packets
        _send_frames(port, 10)
        t.join(timeout=5.0)

        assert len(windows) == 1, "expected exactly one window from 10 valid frames"

    def test_stop_event_cancels_immediately(self) -> None:
        """Setting stop_event before any packets arrive terminates the generator."""
        port = _free_port()
        stop = threading.Event()
        stop.set()  # stopped before we even start
        src = Esp32Source(host="127.0.0.1", port=port)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        t.join(timeout=3.0)

        assert not t.is_alive(), "source did not stop on pre-set stop_event"
        assert len(windows) == 0

    def test_stop_event_cancels_mid_stream(self) -> None:
        """stop_event set after first window stops after at most one more window."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()  # stop after first window

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 50)  # plenty of frames
        t.join(timeout=5.0)

        # Should stop after the first (or at most second) window
        assert not t.is_alive()
        assert 1 <= len(windows) <= 2

    def test_no_overlap_between_windows(self) -> None:
        """v1 has no overlap: 20 frames with window_size=10 → exactly 2 windows."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                if len(windows) >= 2:
                    stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 20)
        t.join(timeout=5.0)

        assert len(windows) == 2
        assert windows[0].window_index == 0
        assert windows[1].window_index == 1


# ---------------------------------------------------------------------------
# _parse_esp32_packet — non-finite amplitude rejection
# ---------------------------------------------------------------------------

class TestParseEsp32PacketNonFinite:
    """NaN, +Infinity, and -Infinity in the amp array must all be rejected."""

    def test_nan_amplitude_rejected(self) -> None:
        # Python's json module encodes float('nan') as the bare token NaN,
        # which it also accepts back on loads().  Ensure we catch it.
        raw = json.dumps({"amp": [float("nan")] * 52}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_pos_inf_amplitude_rejected(self) -> None:
        # Bare token Infinity — non-standard JSON accepted by Python's parser.
        raw = json.dumps({"amp": [float("inf")] * 52}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_neg_inf_amplitude_rejected(self) -> None:
        raw = json.dumps({"amp": [float("-inf")] * 52}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_single_nan_in_otherwise_valid_packet_rejected(self) -> None:
        """One NaN among 51 valid values must still reject the whole packet."""
        amp = [10.0] * 51 + [float("nan")]
        raw = json.dumps({"amp": amp}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_overflow_to_inf_after_float32_cast_rejected(self) -> None:
        # 1e40 is a valid JSON number but overflows float32 → inf.
        # Firmware floats that overflow must not reach the model.
        amp = [1e40] * 52
        raw = json.dumps({"amp": amp}).encode("utf-8")
        assert _parse_esp32_packet(raw) is None

    def test_finite_amplitudes_still_accepted(self) -> None:
        """Sanity check: large-but-finite values must not be wrongly rejected."""
        amp = [3.4e38] * 52  # near float32 max, but still finite
        raw = json.dumps({"amp": amp}).encode("utf-8")
        result = _parse_esp32_packet(raw)
        assert result is not None
        arr, _ = result
        assert np.all(np.isfinite(arr))


# ---------------------------------------------------------------------------
# Esp32Source — single-sender sid lock policy
# ---------------------------------------------------------------------------

class TestEsp32SourceSidLock:
    """Packets from a non-active sid must be dropped; no mixed-sid windows."""

    def _send_mixed(
        self,
        port: int,
        frames_a: int,
        frames_b: int,
        frames_a2: int,
        *,
        sid_a: str = "dev-01",
        sid_b: str = "dev-02",
    ) -> None:
        """Send frames_a from sid_a, then frames_b from sid_b, then frames_a2 from sid_a."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        fi = 0
        for _ in range(frames_a):
            pkt = json.dumps({"fi": fi, "amp": [1.0] * 52, "sid": sid_a}).encode()
            sock.sendto(pkt, ("127.0.0.1", port))
            fi += 1
        for _ in range(frames_b):
            pkt = json.dumps({"fi": fi, "amp": [99.0] * 52, "sid": sid_b}).encode()
            sock.sendto(pkt, ("127.0.0.1", port))
            fi += 1
        for _ in range(frames_a2):
            pkt = json.dumps({"fi": fi, "amp": [1.0] * 52, "sid": sid_a}).encode()
            sock.sendto(pkt, ("127.0.0.1", port))
            fi += 1
        sock.close()

    def test_first_sid_becomes_active(self) -> None:
        """The first packet's sid is locked; the second sender's frames are dropped.

        Setup: window_size=10. Send 10 frames from dev-01, then 10 from dev-02.
        Expected: exactly 1 window emitted, all from dev-01 (amp≈1.0), none from dev-02 (amp≈99.0).
        """
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        self._send_mixed(port, frames_a=10, frames_b=10, frames_a2=0)
        t.join(timeout=5.0)

        assert len(windows) == 1
        # Window must contain only dev-01 frames (amp≈1.0), not dev-02 (amp≈99.0)
        np.testing.assert_allclose(windows[0].data, 1.0, atol=1e-3)
        assert windows[0].source_file == "esp32://dev-01"

    def test_non_active_sid_frames_do_not_pollute_window(self) -> None:
        """Interleaved packets from a second sender must not appear in any window.

        Setup: window_size=10. Send 5 from dev-01, 5 from dev-02 (dropped),
        then 5 more from dev-01.  The 10 dev-01 frames accumulate into one window.
        """
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        self._send_mixed(port, frames_a=5, frames_b=5, frames_a2=5)
        t.join(timeout=5.0)

        assert len(windows) == 1
        # All values must be 1.0 (dev-01 only), not 99.0 (dev-02)
        np.testing.assert_allclose(windows[0].data, 1.0, atol=1e-3)

    def test_non_active_sid_packets_counted_as_dropped(self) -> None:
        """Source must not silently swallow non-active sid packets without logging.

        We can't inspect internal counters directly, but we can assert that
        fewer windows are produced than would be expected if all frames were accepted.

        Setup: window_size=10. Send 10 dev-01 frames + 10 dev-02 frames.
        Without sid lock: 2 windows. With sid lock: 1 window (dev-02 dropped).
        """
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                if len(windows) >= 2:
                    stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        self._send_mixed(port, frames_a=10, frames_b=10, frames_a2=0)
        time.sleep(0.4)  # give source time to process all packets
        stop.set()
        t.join(timeout=3.0)

        # Only 1 window from dev-01; dev-02 frames were dropped
        assert len(windows) == 1

    def test_single_sender_with_consistent_sid_works_normally(self) -> None:
        """A single sender using the same sid throughout must work without issue."""
        port = _free_port()
        stop = threading.Event()
        src = Esp32Source(host="127.0.0.1", port=port, window_size=10)
        windows: list[CsiWindow] = []

        def collect() -> None:
            for w in src.windows(stop):
                windows.append(w)
                if len(windows) >= 3:
                    stop.set()

        t = threading.Thread(target=collect, daemon=True)
        t.start()
        time.sleep(0.1)
        _send_frames(port, 30, sid="dev-01")
        t.join(timeout=5.0)

        assert len(windows) == 3
        for w in windows:
            assert w.source_file == "esp32://dev-01"
